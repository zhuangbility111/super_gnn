import torch
import torch.distributed as dist
import torch.nn.functional as F


class Logger(object):
    def __init__(self):
        Logger.ctx = self

    def print_acc_and_perf(self, model, output, labels, train_mask, valid_mask, loss_train, epoch, epoch_time):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # check accuracy
        predict_result = []
        for mask in (train_mask, valid_mask):
            num_correct_samples = (
                (output[mask].argmax(-1) == labels[mask]).sum() if mask.size(0) != 0 else 0
            )
            num_samples = mask.size(0)
            predict_result.append(num_correct_samples)
            predict_result.append(num_samples)
        predict_result = torch.tensor(predict_result)

        if dist.get_world_size() > 1:
            dist.all_reduce(predict_result, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_train, op=dist.ReduceOp.SUM)

        train_acc = float(predict_result[0] / predict_result[1])
        val_acc = float(predict_result[2] / predict_result[3])
        avg_loss_train = float(loss_train.item() / world_size)

        if rank == 0:
            print(
                f"Rank: {rank}, World_size: {world_size}, Epoch: {epoch}, Avg Loss: {avg_loss_train}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Time (ms): {epoch_time:.6f}", flush=True
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
