import torch
import torch.distributed as dist
import torch.nn.functional as F
import argparse
import time
import yaml
from utils import create_model_and_optimizer, set_random_seed
from super_gnn.communicator import Communicator
from super_gnn.data_manager import load_data
from super_gnn.assigner import Assigner

from super_gnn.time_recorder import TimeRecorder

from super_gnn.quantizer import Quantizer_for_all_procs
from logger import Logger

from torch_geometric.utils import index_to_mask
from mask_label import MaskLabel


def train_step(model, data, optimizer, epoch, num_train_nodes, label_rate):
    model.train()

    propagation_mask = MaskLabel.ratio_mask(data["nodes_train_masks"], ratio=label_rate)
    supervision_mask = data["nodes_train_masks"] ^ propagation_mask

    forward_start = time.perf_counter()
    Assigner.ctx.reassign_node_dataformat(epoch)
    optimizer.zero_grad()
    out = model(data["graph"], data["nodes_features"], data["nodes_labels"], propagation_mask)
    backward_start = time.perf_counter()
    # loss = F.nll_loss(out[data["nodes_train_masks"]], data["nodes_labels"][data["nodes_train_masks"]], reduction="sum")
    loss = F.nll_loss(out[supervision_mask], data["nodes_labels"][supervision_mask], reduction="sum")
    loss = loss / num_train_nodes
    loss.backward()

    update_weight_start = time.perf_counter()
    optimizer.step()
    update_weight_end = time.perf_counter()
    # total_forward_dur += backward_start - forward_start
    # total_backward_dur += update_weight_start - backward_start
    # total_update_weight_dur += update_weight_end - update_weight_start
    # total_training_dur += update_weight_end - forward_start

    TimeRecorder.ctx.record_total_training_time(update_weight_end - forward_start)
    TimeRecorder.ctx.next_epoch()

    return loss

@torch.no_grad()
def test_step(model, data):
    model.eval()

    TimeRecorder.ctx.set_is_training(False)

    predict_result = []

    propagation_mask = data["nodes_train_masks"]
    # propagation_mask = torch.zeros(data["nodes_train_masks"].size(0)).bool()
    out = model(data["graph"], data["nodes_features"], data["nodes_labels"], propagation_mask)

    for mask in (data["nodes_train_masks"], data["nodes_valid_masks"], data["nodes_test_masks"]):
        num_correct_samples = (
            (out[mask].argmax(-1) == data["nodes_labels"][mask]).sum() if mask.size(0) != 0 else 0
        )
        # num_samples = mask.size(0)
        num_samples = mask.sum()
        predict_result.append(num_correct_samples)
        predict_result.append(num_samples)
    predict_result = torch.tensor(predict_result)
    if dist.get_world_size() > 1:
        dist.all_reduce(predict_result, op=dist.ReduceOp.SUM)

    train_acc = float(predict_result[0] / predict_result[1])
    val_acc = float(predict_result[2] / predict_result[3])
    test_acc = float(predict_result[4] / predict_result[5])

    TimeRecorder.ctx.set_is_training(True)

    return train_acc, val_acc, test_acc

def train(model, data, optimizer, num_epochs, num_bits):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # start training
    total_forward_dur = 0.0
    total_backward_dur = 0.0
    total_update_weight_dur = 0.0
    total_training_dur = 0.0

    label_rate = 0.62

    num_train_nodes = torch.tensor([data["nodes_train_masks"].size(0)], dtype=torch.int64)
    dist.all_reduce(num_train_nodes, op=dist.ReduceOp.SUM)
    num_train_nodes = float(num_train_nodes.item())
    num_train_nodes *= (1 - label_rate)

    # convert mask to bool
    data["nodes_train_masks"] = index_to_mask(data["nodes_train_masks"], data["nodes_features"].size(0))
    data["nodes_valid_masks"] = index_to_mask(data["nodes_valid_masks"], data["nodes_features"].size(0))
    data["nodes_test_masks"] = index_to_mask(data["nodes_test_masks"], data["nodes_features"].size(0))

    print(f'data[nodes_train_masks] = {data["nodes_train_masks"]}')

    # with profile(activities=[ProfilerActivity.CPU]) as prof:
    dist.barrier()
    for epoch in range(num_epochs):
        train_begin = time.perf_counter()
        loss = train_step(model, data, optimizer, epoch, num_train_nodes, label_rate)
        train_end = time.perf_counter()
        train_acc, val_acc, test_acc = test_step(model, data)
        
        print("Rank: {}, Epoch: {}, Loss: {:.5f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Time: {:.6f}".format(
            rank, epoch, loss, train_acc, val_acc, test_acc, train_end - train_begin), flush=True)

        # Logger.ctx.print_acc_and_perf(model, data, epoch, loss, update_weight_end - forward_start)
        
    # Logger.ctx.print_forward_backward_perf(total_forward_dur, total_backward_dur, total_update_weight_dur, total_training_dur)

if __name__ == "__main__":
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

    # print(config, flush=True)

    Communicator(config["num_bits"], config["is_async"])
    rank, world_size = Communicator.ctx.init_dist_group()

    Quantizer_for_all_procs(world_size, config["num_bits"])

    if (
        config["graph_name"] != "arxiv"
        and config["graph_name"] != "products"
        and config["graph_name"] != "papers100M"
    ):
        config["input_dir"] += "{}_{}_part/".format(config["graph_name"], world_size)
    else:
        config["input_dir"] += "ogbn_{}_{}_part/".format(config["graph_name"], world_size)

    set_random_seed(config["random_seed"])
    model, optimizer = create_model_and_optimizer(config)
    data = load_data(config)

    Assigner(
        config["num_bits"],
        config["num_layers"],
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        config["assign_period"],
        data["graph"].comm_buf.send_buf.size(0),
        data["graph"].comm_buf.recv_buf.size(0),
    )

    Logger()

    TimeRecorder(config["num_layers"], config["num_epochs"])

    print("config: {}".format(config), flush=True)

    # print("finish data loading.", flush=True)
    train(model, data, optimizer, config["num_epochs"], config["num_bits"])

    TimeRecorder.ctx.print_total_time()
    # TimeRecorder.ctx.save_time_to_file(config["graph_name"], world_size)
