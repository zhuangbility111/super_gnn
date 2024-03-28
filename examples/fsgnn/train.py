from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from utils import set_random_seed, create_model_and_optimizer, gcn_normalize
from super_gnn.communicator import Communicator
from super_gnn.time_recorder import TimeRecorder
from super_gnn.aggregator import Aggregator
from super_gnn.data_manager import load_data
from logger import Logger
import uuid
import yaml
import copy


def avg_pred(output, labels, mask):
    num_correct_samples = (
            (output[mask].argmax(-1) == labels[mask]).sum() if mask.size(0) != 0 else 0
        )
    num_samples = mask.size(0)
    pred_result = torch.tensor([num_correct_samples, num_samples], dtype=torch.int64)
    if dist.get_world_size() > 1:
        dist.all_reduce(pred_result, op=dist.ReduceOp.SUM)
    avg_acc = float(pred_result[0]) / float(pred_result[1])
    return avg_acc


def avg_loss(loss, num_samples):
    # check if the loss is a tensor which is nan
    if torch.isnan(loss):
            loss = torch.tensor(0.0)
    loss_tensor = torch.tensor([loss.item(), num_samples])
    if dist.get_world_size() > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = float(loss_tensor[0] / loss_tensor[1])
    return avg_loss


def train_step(model, optimizer, labels, list_mat, train_mask, use_layer_norm):
    model.train()
    dist.barrier()
    train_begin = time.perf_counter()
    optimizer.zero_grad()
    output = model(list_mat, use_layer_norm)
    loss_train = F.nll_loss(output[train_mask], labels[train_mask])
    loss_train.backward()
    optimizer.step()
    train_end = time.perf_counter()
    epoch_time = (train_end - train_begin) * 1000.0 # ms

    # check accuracy
    acc_train = avg_pred(output, labels, train_mask)
    return loss_train, acc_train, epoch_time


def valid_step(model, labels, list_mat, valid_mask, use_layer_norm, best, bad_counter, checkpt_file):
    # get validation loss
    model.eval()
    with torch.no_grad():
        output = model(list_mat, use_layer_norm)
        loss_val = F.nll_loss(output[valid_mask], labels[valid_mask], reduction="sum")
        loss_val = avg_loss(loss_val, valid_mask.size(0))

        # check accuracy
        acc_val = avg_pred(output, labels, valid_mask)
    return loss_val, acc_val


def test_step(model, labels, list_mat, use_layer_norm, test_mask, checkpt_file):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(list_mat, use_layer_norm)

        # check accuracy
        acc_test = avg_pred(output, labels, test_mask)
    return acc_test

def train(config):
    # checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
    checkpt_file = f'pretrained/model_{config["graph_name"]}_{dist.get_world_size()}.pt'

    data = load_data(config)

    # precompute normalization
    graph = data['graph']

    graph_with_self_loop = copy.deepcopy(graph)
    graph_with_self_loop.local_adj_t = graph_with_self_loop.local_adj_t.fill_diag(1.0)

    gcn_normalize(graph)
    gcn_normalize(graph_with_self_loop)

    print("number of training samples = {}".format(data["nodes_train_masks"].shape[0]))
    print("number of valid samples = {}".format(data["nodes_valid_masks"].shape[0]))
    print("number of testing samples = {}".format(data["nodes_test_masks"].shape[0]))

    features = data['nodes_features']

    list_mat = []
    list_mat.append(features)

    no_loop_mat = features
    loop_mat = features

    for ii in range(config["num_layers"]):
        # aggregate
        no_loop_mat = Aggregator.apply(graph, no_loop_mat, 0, 32, False, False)
        loop_mat = Aggregator.apply(graph_with_self_loop, loop_mat, 0, 32, False, False)
        list_mat.append(no_loop_mat)
        list_mat.append(loop_mat)

    model, optimizer = create_model_and_optimizer(config, list_mat)

    bad_counter = 0
    best = 999999999
    for epoch in range(config["num_epochs"]):
        loss_train, acc_train, epoch_time = train_step(model, optimizer, data["nodes_labels"], list_mat, 
                                    data["nodes_train_masks"], config["layer_norm"])

        loss_val, acc_val = valid_step(model, data["nodes_labels"], list_mat, data["nodes_valid_masks"],
                                        config["layer_norm"], best, bad_counter, checkpt_file)

        # bad_counter, best = check_model(model, data["nodes_labels"], list_mat, data["nodes_valid_masks"], 
        #                                 config["layer_norm"], best, bad_counter, checkpt_file)

        if rank == 0:
            print('Rank: '.format(rank),
                    'World_size: '.format(world_size),
                    'Epoch:{:04d}'.format(epoch+1),
                    'train',
                    'loss:{:.3f}'.format(loss_train),
                    'acc:{:.2f}'.format(acc_train*100),
                    '| val',
                    'loss:{:.3f}'.format(loss_val),
                    'acc:{:.2f}'.format(acc_val*100),
                    '| time:{:.4f} ms'.format(epoch_time))


        if loss_val < best:
            best = loss_val
            if dist.get_rank() == 0:
                torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == config["patience"]:
            break

    # Testing
    if dist.get_world_size() > 1:
        dist.barrier()
    
    accuracy = test_step(model, data["nodes_labels"], list_mat, config["layer_norm"], data["nodes_test_masks"], checkpt_file)

    return accuracy * 100.0


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_bits", type=int, default=32)
    parser.add_argument("--is_pre_delay", type=str, default="false")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # config['is_fp16'] = True if args.is_fp16 == 'true' else False
    config["num_bits"] = args.num_bits
    config["is_pre_delay"] = True if args.is_pre_delay == "true" else False

    Communicator(config["num_bits"], config["is_async"])
    rank, world_size = Communicator.ctx.init_dist_group()

    if (
        config["graph_name"] != "arxiv"
        and config["graph_name"] != "products"
        and config["graph_name"] != "papers100M"
    ):
        config["input_dir"] += "{}_{}_part/".format(config["graph_name"], world_size)
    else:
        config["input_dir"] += "ogbn_{}_{}_part/".format(config["graph_name"], world_size)

    set_random_seed(config["random_seed"])

    TimeRecorder(config["num_layers"], config["num_epochs"])

    # print("finish data loading.", flush=True)

    TimeRecorder.ctx.print_total_time()
    TimeRecorder.ctx.save_time_to_file(config["graph_name"], world_size)

    Logger()

    print("==========================")
    print(f"Dataset: {config['graph_name']}")
    print(f"Dropout:{config['dropout']}, layer_norm: {config['layer_norm']}")
    print(f"w_att:{config['w_att']}, w_fc2:{config['w_fc2']}, w_fc1:{config['w_fc1']}, lr_fc:{config['lr_fc']}, lr_att:{config['lr_att']}")

    accuracy_list = []
    num_runs = 1
    for i in range(num_runs):
        accuracy = train(config)
        accuracy_list.append(accuracy)
        print("Run {} accuracy: {}".format(i, accuracy))

    print(f"Test accuracy: {np.mean(accuracy_list)}, {np.round(np.std(accuracy_list),2)}")
