import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import LayerNorm
import time

# from torch_geometric.nn import DistSAGEConvGradWithPre
# from torch_geometric.nn import DistSAGEConvGrad
from super_gnn.layers.sageconv import DistSAGEConvGrad


from typing import Optional
from torch.nn import Parameter
from torch.nn import init
import torch.distributed as dist

from mask_label import MaskLabel

class DistributedBN1D(torch.nn.Module):
    """Distributed Batch normalization layer

    Normalizes a 2D feature tensor using the global mean and standard deviation calculated across all workers.


    :param n_feats: The second dimension (feature dimension) in the 2D input tensor
    :type n_feats: int
    :param eps:  a value added to the variance for numerical stability 
    :type eps: float
    :param affine: When ``True``, the module will use learnable affine parameter
    :type affine: bool
    :param distributed: Boolean speficying whether to run in distributed mode where normalizing\
    statistics are calculated across all workers, or local mode where the normalizing statistics\
    are calculated using only the local input feature tensor. If not specified, it will be set to\
    ``True`` if the user has called :func:`sar.initialize_comms`, and ``False`` otherwise
    :type distributed: Optional[bool]

    """
    def __init__(self, n_feats: int, eps: float = 1.0e-5, affine: bool = True, distributed: Optional[bool] = None):
        super().__init__()
        self.n_feats = n_feats
        self.weight: Optional[Parameter]
        self.bias: Optional[Parameter]
        self.affine = affine
        if affine:
            self.weight = Parameter(torch.ones(n_feats))
            self.bias = Parameter(torch.zeros(n_feats))
        else:
            self.weight = None
            self.bias = None

        self.eps = eps
        
        self.distributed = distributed

    def forward(self, inp):
        '''
        forward implementation of DistributedBN1D
        '''
        assert inp.ndim == 2, 'distributedBN1D must have a 2D input'
        if self.distributed:
            mean, var = mean_op(inp), var_op(inp)
            std = torch.sqrt(var - mean**2 + self.eps)
        else:
            mean = inp.mean(0)
            std = inp.std(0)
        normalized_x = (inp - mean.unsqueeze(0)) / std.unsqueeze(0)

        if self.weight is not None and self.bias is not None:
            result = normalized_x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        else:
            result = normalized_x
        return result
    
    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)



class MeanOp(torch.autograd.Function):  # pylint: disable = abstract-method
    @staticmethod
    # pylint: disable = arguments-differ
    def forward(ctx, x):
        own_sum = torch.empty(x.size(1)+1)
        own_sum[:-1] = x.sum(0).data
        own_sum[-1] = x.size(0)
        dist.all_reduce(own_sum, op=dist.ReduceOp.SUM)
        mean = (own_sum[:-1]/own_sum[-1]).to(x.device)
        ctx.n_points = torch.round(own_sum[-1]).long().item()
        ctx.inp_size = x.size(0)
        return mean

    @staticmethod
    # pylint: disable = arguments-differ
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad.repeat(ctx.inp_size, 1) / ctx.n_points


class VarOp(torch.autograd.Function):  # pylint: disable = abstract-method
    @staticmethod
    # pylint: disable = arguments-differ
    def forward(ctx, features):
        own_sum = torch.empty(features.size(1)+1)
        own_sum[:-1] = (features**2).sum(0).data
        own_sum[-1] = features.size(0)
        dist.all_reduce(own_sum, op=dist.ReduceOp.SUM)
        variance = (own_sum[:-1]/own_sum[-1]).to(features.device)

        ctx.n_points = torch.round(own_sum[-1]).long().item()
        ctx.save_for_backward(features)
        return variance

    @staticmethod
    # pylint: disable = arguments-differ
    def backward(ctx, grad):
        features,  = ctx.saved_tensors
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return (grad.unsqueeze(0) * 2 * features) / ctx.n_points


mean_op = MeanOp.apply
var_op = VarOp.apply


class DistSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0.5,
        num_bits=32,
        is_pre_delay=False,
        norm_type="LayerNorm",
        is_label_augment=False,
    ):
        super().__init__()

        if is_label_augment:
            self.label_emb = MaskLabel(out_channels, in_channels)
        else:
            self.label_emb = None

        self.convs = torch.nn.ModuleList()
        self.convs.append(DistSAGEConvGrad(in_channels, hidden_channels, num_bits, is_pre_delay))

        if norm_type != None:
            self.norms = torch.nn.ModuleList()
        
        if norm_type == "LayerNorm":
            self.norms.append(LayerNorm(hidden_channels))
        elif norm_type == "BatchNorm1d":
            self.norms.append(DistributedBN1D(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(DistSAGEConvGrad(hidden_channels, hidden_channels, num_bits, is_pre_delay))

            if norm_type == "LayerNorm":
                self.norms.append(LayerNorm(hidden_channels))
            elif norm_type == "BatchNorm1d":
                self.norms.append(DistributedBN1D(hidden_channels))

        self.convs.append(DistSAGEConvGrad(hidden_channels, out_channels, num_bits, is_pre_delay))

        self.dropout = dropout
        self.norm_type = norm_type

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.norm_type != None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, graph, nodes_feats, labels, label_mask):
        total_conv_time = 0.0
        total_relu_time = 0.0
        total_dropout_time = 0.0

        if self.label_emb != None:
            # label augmentation (add partial label information to nodes_feats)
            nodes_feats = self.label_emb(nodes_feats, labels, label_mask)

        for i, conv in enumerate(self.convs[:-1]):
            conv_begin = time.perf_counter()
            nodes_feats = conv(graph, nodes_feats, i)
            dropout_begin = time.perf_counter()
            nodes_feats = F.dropout(nodes_feats, p=self.dropout, training=self.training)
            if self.norm_type != None:
                nodes_feats = self.norms[i](nodes_feats)
            relu_begin = time.perf_counter()
            nodes_feats = F.relu(nodes_feats)
            relu_end = time.perf_counter()
            total_conv_time = dropout_begin - conv_begin
            total_relu_time = relu_end - relu_begin
            total_dropout_time = relu_begin - dropout_begin
            rank = dist.get_rank()
            if rank == 0:
                print("----------------------------------------")
                print("Time of conv(ms): {:.4f}".format(total_conv_time * 1000.0))
                print("Time of relu(ms): {:.4f}".format(total_relu_time * 1000.0))
                print("Time of dropout(ms): {:.4f}".format(total_dropout_time * 1000.0))
                print("----------------------------------------")

        conv_begin = time.perf_counter()
        nodes_feats = self.convs[-1](graph, nodes_feats, len(self.convs) - 1)
        return F.log_softmax(nodes_feats, dim=1)
