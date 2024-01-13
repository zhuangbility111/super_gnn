import torch
import torch.distributed as dist
import random
import numpy as np
from fsgnn import FSGNN
from super_gnn.data_manager import CommBuffer


def create_model_and_optimizer(config):
    model = FSGNN(config["in_channels"],
                  config["num_layers"],
                  config["hidden_channels"],
                  config["out_channels"],
                  config["dropout"])

    optimizer_setting = [
        {'params': model.fc2.parameters(), 'weight_decay': config["w_fc2"], 'lr': config["lr_fc"]},
        {'params': model.fc1.parameters(), 'weight_decay': config["w_fc1"], 'lr': config["lr_fc"]},
        {'params': model.att, 'weight_decay': config["w_att"], 'lr': config["lr_att"]},
    ]

    optimizer = torch.optim.Adam(optimizer_setting)
     
    model.reset_parameters()
    if dist.get_world_size() > 1:
        # wrap model with ddp
        model = torch.nn.parallel.DistributedDataParallel(model)

    return model, optimizer

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def gcn_normalize(dist_graph):
    # add dist_graph-loop
    dist_graph.local_adj_t = dist_graph.local_adj_t.fill_diag(1.0)
    # get in-degrees
    in_degrees = dist_graph.get_in_degrees()
    deg_inv_sqrt = in_degrees.pow(-0.5)

    # normalization for local graph
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    dist_graph.local_adj_t = deg_inv_sqrt.view(-1, 1) * dist_graph.local_adj_t * deg_inv_sqrt.view(1, -1)

    # normalization for remote graph
    comm_splits = dist_graph.comm_splits
    comm_buf = CommBuffer(comm_splits, 1, 32)

    torch.index_select(deg_inv_sqrt, 0, dist_graph.idx_nodes_send_to_others, out=comm_buf.send_buf)
    dist.all_to_all_single(comm_buf.recv_buf, comm_buf.send_buf, comm_splits.recv_splits, comm_splits.send_splits)

    remote_deg_inv_sqrt = comm_buf.recv_buf.view(-1)
    dist_graph.remote_adj_t = deg_inv_sqrt.view(-1, 1) * dist_graph.remote_adj_t * remote_deg_inv_sqrt.view(1, -1)

    return dist_graph