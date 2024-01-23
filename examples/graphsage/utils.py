import torch
import torch.distributed as dist
import random
import numpy as np
from sage import DistSAGE

def allreduce_hook(process_group: dist.ProcessGroup, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    grad_tensor = bucket.buffer()
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    return (
        dist.all_reduce(grad_tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def create_model_and_optimizer(config: dict):
    model = None
    optimizer = None
    if config["model_name"] == "sage":
        model = DistSAGE(
            config["in_channels"],
            config["hidden_channels"],
            config["out_channels"],
            config["num_layers"],
            config["dropout"],
            config["num_bits"],
            config["is_pre_delay"],
            config["norm_type"],
        )

        model.reset_parameters()
        if dist.get_world_size() > 1:
            # wrap model with ddp
            model = torch.nn.parallel.DistributedDataParallel(model)
            model.register_comm_hook(state=None, hook=allreduce_hook)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    return model, optimizer


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
