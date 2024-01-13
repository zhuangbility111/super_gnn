import torch
import torch.distributed as dist
import random
import numpy as np
from fsgnn import FSGNN


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

