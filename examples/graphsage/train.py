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


def train(model, data, optimizer, num_epochs, num_bits):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # start training
    total_forward_dur = 0.0
    total_backward_dur = 0.0
    total_update_weight_dur = 0.0
    total_training_dur = 0.0

    # with profile(activities=[ProfilerActivity.CPU]) as prof:
    dist.barrier()
    for epoch in range(num_epochs):
        model.train()
        forward_start = time.perf_counter()
        Assigner.ctx.reassign_node_dataformat(epoch)
        optimizer.zero_grad()
        out = model(data["graph"], data["nodes_features"])
        backward_start = time.perf_counter()
        loss = F.nll_loss(out[data["nodes_train_masks"]], data["nodes_labels"][data["nodes_train_masks"]])
        loss.backward()

        update_weight_start = time.perf_counter()
        optimizer.step()
        update_weight_end = time.perf_counter()
        total_forward_dur += backward_start - forward_start
        total_backward_dur += update_weight_start - backward_start
        total_update_weight_dur += update_weight_end - update_weight_start
        total_training_dur += update_weight_end - forward_start

        Logger.ctx.print_acc_and_perf(model, data, epoch, loss, update_weight_end - forward_start)
        
        TimeRecorder.ctx.record_total_training_time(update_weight_end - forward_start)
        TimeRecorder.ctx.next_epoch()
    
    Logger.ctx.print_forward_backward_perf(total_forward_dur, total_backward_dur, total_update_weight_dur, total_training_dur)


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
    TimeRecorder.ctx.save_time_to_file(config["graph_name"], world_size)