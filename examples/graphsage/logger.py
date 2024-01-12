import torch
import torch.distributed as dist
from super_gnn.time_recorder import TimeRecorder


class Logger(object):
    def __init__(self):
        Logger.ctx = self

    def print_acc_and_perf(self, model, data, epoch, loss, epoch_time):
        # check accuracy
        TimeRecorder.ctx.set_is_training(False)
        model.eval()
        predict_result = []
        out = model(data["graph"], data["nodes_features"])
        for mask in (data["nodes_train_masks"], data["nodes_valid_masks"], data["nodes_test_masks"]):
            num_correct_samples = (
                (out[mask].argmax(-1) == data["nodes_labels"][mask]).sum() if mask.size(0) != 0 else 0
            )
            num_samples = mask.size(0)
            predict_result.append(num_correct_samples)
            predict_result.append(num_samples)
        predict_result = torch.tensor(predict_result)
        if dist.get_world_size() > 1:
            dist.all_reduce(predict_result, op=dist.ReduceOp.SUM)

        train_acc = float(predict_result[0] / predict_result[1])
        val_acc = float(predict_result[2] / predict_result[3])
        test_acc = float(predict_result[4] / predict_result[5])
        TimeRecorder.ctx.set_is_training(True)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            print(
                f"Rank: {rank}, World_size: {world_size}, Epoch: {epoch}, Loss: {loss}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Time: {epoch_time:.6f}"
            )
        

    def print_forward_backward_perf(self, total_forward_dur, total_backward_dur, total_update_weight_dur, total_training_dur):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        total_forward_dur = torch.tensor([total_forward_dur])
        total_backward_dur = torch.tensor([total_backward_dur])
        total_update_weight_dur = torch.tensor([total_update_weight_dur])
        ave_total_training_dur = torch.tensor([total_training_dur])
        max_total_training_dur = torch.tensor([total_training_dur])

        dist.reduce(total_forward_dur, 0, op=dist.ReduceOp.SUM)
        dist.reduce(total_backward_dur, 0, op=dist.ReduceOp.SUM)
        dist.reduce(total_update_weight_dur, 0, op=dist.ReduceOp.SUM)
        dist.reduce(ave_total_training_dur, 0, op=dist.ReduceOp.SUM)
        dist.reduce(max_total_training_dur, 0, op=dist.ReduceOp.MAX)

        if dist.get_rank() == 0:
            print("training end.")
            print("forward_time(ms): {}".format(total_forward_dur[0] / float(world_size) * 1000))
            print("backward_time(ms): {}".format(total_backward_dur[0] / float(world_size) * 1000))
            print("update_weight_time(ms): {}".format(total_update_weight_dur[0] / float(world_size) * 1000))
            print(
                "total_training_time(average)(ms): {}".format(
                    ave_total_training_dur[0] / float(world_size) * 1000
                )
            )
            print("total_training_time(max)(ms): {}".format(max_total_training_dur[0] * 1000.0))
