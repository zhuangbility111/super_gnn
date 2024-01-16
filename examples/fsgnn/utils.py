import torch
import torch.distributed as dist
import random
import numpy as np
from fsgnn import FSGNN
from super_gnn.data_manager import CommBuffer

def allreduce_hook(process_group: dist.ProcessGroup, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    grad_tensor = bucket.buffer()
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    return (
        dist.all_reduce(grad_tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


def create_model_and_optimizer(config, list_mat):
    model = FSGNN(nfeat=config["in_channels"],
                  nlayers=len(list_mat),
                  nhidden=config["hidden_channels"],
                  nclass=config["out_channels"],
                  dropout=config["dropout"])

    optimizer_setting = [
        {'params': model.fc2.parameters(), 'weight_decay': config["w_fc2"], 'lr': config["lr_fc"]},
        {'params': model.fc1.parameters(), 'weight_decay': config["w_fc1"], 'lr': config["lr_fc"]},
        {'params': model.att, 'weight_decay': config["w_att"], 'lr': config["lr_att"]},
    ]

    optimizer = torch.optim.Adam(optimizer_setting)
    
    if dist.get_world_size() > 1:
        # wrap model with ddp
        model = torch.nn.parallel.DistributedDataParallel(model)

        if config["use_defined_ddp"]:
            # define ddp communication hooks
            model.register_comm_hook(process_group=None, hook=allreduce_hook)

    return model, optimizer

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def gcn_normalize(dist_graph):
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
