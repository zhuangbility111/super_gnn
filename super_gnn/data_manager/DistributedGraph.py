from torch import Tensor
import torch
from torch_sparse import SparseTensor
from typing import Optional
from super_gnn.data_manager import CommBuffer, CommBufferForQuantization
from super_gnn.data_manager import CommSplits


class DistributedGraph(object):
    def __init__(
        self,
        local_adj_t: SparseTensor,
        remote_adj_t: SparseTensor,
        idx_nodes_send_to_others: Tensor,
        num_nodes_send_to_others: list,
        num_nodes_recv_from_others: list,
        in_degrees: Tensor,
        comm_splits: CommSplits,
        comm_buf: CommBuffer,
        comm_buf_for_quantization: Optional[CommBufferForQuantization],
    ) -> None:
        self.local_adj_t = local_adj_t
        self.remote_adj_t = remote_adj_t

        self.idx_nodes_send_to_others = idx_nodes_send_to_others

        self.num_nodes_send_to_others = num_nodes_send_to_others
        self.num_nodes_recv_from_others = num_nodes_recv_from_others

        self.in_degrees = in_degrees

        self.comm_splits = comm_splits
        self.comm_buf = comm_buf
        self.comm_buf_for_quantization = comm_buf_for_quantization
    
    def get_in_degrees(self):
        local_edges_list = self.local_adj_t.coo()
        remote_edges_list = self.remote_adj_t.coo()
        num_local_nodes = self.local_adj_t.sparse_sizes()[0]
        local_degs = torch.zeros((num_local_nodes), dtype=torch.float32)
        source = torch.ones((local_edges_list[0].shape[0]), dtype=torch.float32)
        local_degs.index_add_(dim=0, index=local_edges_list[0], source=source)
        source = torch.ones((remote_edges_list[0].shape[0]), dtype=torch.float32)
        local_degs.index_add_(dim=0, index=remote_edges_list[0], source=source)
        # local_degs = local_degs.clamp(min=1).unsqueeze(-1)
        local_degs = local_degs.unsqueeze(-1)
        if local_degs is not None:
            self.in_degrees = local_degs
        return local_degs

class DistributedGraphForPre(object):
    def __init__(
        self,
        local_adj_t: SparseTensor,
        adj_t_pre_post_aggr_from: SparseTensor,
        adj_t_pre_post_aggr_to: SparseTensor,
        pre_post_aggr_from_splits: list,
        pre_post_aggr_to_splits: list,
        in_degrees: Tensor,
        comm_splits: CommSplits,
        comm_buf: CommBuffer,
        comm_buf_for_quantization: Optional[CommBufferForQuantization],
    ) -> None:
        self.local_adj_t = local_adj_t
        self.adj_t_pre_post_aggr_from = adj_t_pre_post_aggr_from
        self.adj_t_pre_post_aggr_to = adj_t_pre_post_aggr_to

        self.pre_post_aggr_from_splits = pre_post_aggr_from_splits
        self.pre_post_aggr_to_splits = pre_post_aggr_to_splits

        self.in_degrees = in_degrees

        self.comm_splits = comm_splits
        self.comm_buf = comm_buf
        self.comm_buf_for_quantization = comm_buf_for_quantization
    
    def get_in_degrees(self):
        local_edges_list = self.local_adj_t.coo()
        remote_edges_list = self.remote_adj_t.coo()
        num_local_nodes = self.local_adj_t.sparse_sizes()[0]
        local_degs = torch.zeros((num_local_nodes), dtype=torch.float32)
        source = torch.ones((local_edges_list[0].shape[0]), dtype=torch.float32)
        local_degs.index_add_(dim=0, index=local_edges_list[0], source=source)
        source = torch.ones((remote_edges_list[0].shape[0]), dtype=torch.float32)
        local_degs.index_add_(dim=0, index=remote_edges_list[0], source=source)
        local_degs = local_degs.clamp(min=1).unsqueeze(-1)
        if local_degs is not None:
            self.in_degrees = local_degs
        return local_degs
