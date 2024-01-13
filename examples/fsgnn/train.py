from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from fsgnn import FSGNN
from utils import set_random_seed, create_model_and_optimizer
from super_gnn.communicator import Communicator
from super_gnn.time_recorder import TimeRecorder
from super_gnn.data_manager import load_data
import uuid
import pickle
import yaml


def train_step(model,optimizer,labels,list_mat,idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat, layer_norm)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,labels,list_mat,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(model,labels,list_mat,idx_test):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        #print(mask_val)
        return loss_test.item(),acc_test.item()


def train(config):
    checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

    model, optimizer = create_model_and_optimizer(config)
    data = load_data(config)

    # precompute normalization
    adj_t = 

    adj = adj.to(device)
    adj_i = adj_i.to(device)
    list_mat = []
    list_mat.append(features)
    no_loop_mat = features
    loop_mat = features

    for ii in range(args.layer):
        no_loop_mat = torch.spmm(adj, no_loop_mat)
        loop_mat = torch.spmm(adj_i, loop_mat)
        list_mat.append(no_loop_mat)
        list_mat.append(loop_mat)

    # Select X and self-looped features 
    if feat_type == "homophily":
        select_idx = [0] + [2*ll for ll in range(1,num_layer+1)]
        list_mat = [list_mat[ll] for ll in select_idx]

    #Select X and no-loop features
    elif feat_type == "heterophily":
        select_idx = [0] + [2*ll-1 for ll in range(1,num_layer+1)]
        list_mat = [list_mat[ll] for ll in select_idx]
        
    #Otherwise all hop features are selected
    
    model = FSGNN(nfeat=num_features,
                nlayers=len(list_mat),
                nhidden=args.hidden,
                nclass=num_labels,
                dropout=args.dropout).to(device)


    optimizer_sett = [
        {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
        {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
        {'params': model.att, 'weight_decay': args.w_att, 'lr': args.lr_att},
    ]

    optimizer = optim.Adam(optimizer_sett)

    bad_counter = 0
    best = 999999999
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train_step(model,optimizer,labels,list_mat,idx_train)
        loss_val,acc_val = validate_step(model,labels,list_mat,idx_val)
        #Uncomment following lines to see loss and accuracy values
        '''
        if(epoch+1)%1 == 0:

            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        '''        

        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    test_out = test_step(model,labels,list_mat,idx_test)
    acc = test_out[1]


    return acc*100


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

    print("==========================")
    print(f"Dataset: {config['graph_name']}")
    print(f"Dropout:{config['dropout']}, layer_norm: {config['layer_norm']}")
    print(f"w_att:{config['w_att']}, w_fc2:{config['w_fc2']}, w_fc1:{config['w_fc1']}, lr_fc:{config['lr_fc']}, lr_att:{config['lr_att']}")

    train(config)

    # t_total = time.time()
    # acc_list = []

    # for i in range(10):
    # datastr = args.data
    # accuracy_data = train(config)
    # acc_list.append(accuracy_data)

        ##print(i,": {:.2f}".format(acc_list[-1]))

    # print("Train cost: {:.4f}s".format(time.time() - t_total))
    #print("Test acc.:{:.2f}".format(np.mean(acc_list)))
    # print(f"Test accuracy: {np.mean(acc_list)}, {np.round(np.std(acc_list),2)}")


