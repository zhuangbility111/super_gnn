import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

import time

from super_gnn.time_recorder import TimeRecorder
from super_gnn.aggregator import Aggregator


class DistSAGEConvGrad(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bits: int,
        is_pre_delay: bool,
        add_self_loops: bool = False,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.num_bits = num_bits
        self.is_pre_delay = is_pre_delay

        self.lin_neigh = Linear(in_channels, out_channels, bias=False)
        self.lin_self = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.lin_neigh.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.lin_self.weight, gain=gain)
        # if self.bias is not None:
        #     torch.nn.init.zeros_(self.bias)
        # self.lin.reset_parameters()
        # zeros(self.bias)

    def propagate(self, graph, local_nodes_feat, layer):
        local_out = Aggregator.apply(
            graph, local_nodes_feat, layer, self.num_bits, self.is_pre_delay, self.training
        )
        return local_out

    def forward(self, graph, local_nodes_feat, layer) -> Tensor:
        # communication first
        # linear_first = self.in_channels > self.out_channels

        # if linear_first:
        # local_nodes_feat = self.lin(local_nodes_feat)

        with TimeRecorder.ctx.time_block("total_aggregator(outer)", is_record=False):
            out = self.propagate(graph, local_nodes_feat, layer)

        # print("outer propagate forward (ms): {}".format((add_bias_begin - propagate_begin) * 1000.0))
        # out += local_nodes_feat

        with TimeRecorder.ctx.time_block("Normalization", is_record=False):
            if self.normalize:
                # out /= graph.in_degrees + 1
                out /= graph.in_degrees

        with TimeRecorder.ctx.time_block("nn_operation", is_record=False):
            # if not linear_first:
            out = self.lin_neigh(out)
            out += self.lin_self(local_nodes_feat)

        with TimeRecorder.ctx.time_block("add_bias", is_record=False):
            if self.bias is not None:
                out += self.bias

        return out
