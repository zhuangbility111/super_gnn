import torch
import time
import sys
import torch.distributed as dist
# from torch_geometric.nn.spmm_kernel import SPMM_forward, SPMM_backward
from super_gnn.ops.spmm_kernel import SPMM_forward, SPMM_backward


from super_gnn.communicator import Communicator
from super_gnn.time_recorder import TimeRecorder
from super_gnn.quantizer import QuantizerForMixedBits, Quantizer_for_all_procs
from super_gnn.data_manager import DistributedGraph, DistributedGraphForPre
from typing import Union

class Aggregator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                graph: Union[DistributedGraph, DistributedGraphForPre], 
                local_nodes_feat, 
                layer, 
                num_bits: int, 
                is_pre_delay: bool, 
                is_training: bool):
        ctx.graph = graph
        ctx.num_bits = num_bits
        ctx.is_pre_delay = is_pre_delay
        ctx.layer = layer

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        comm_handle = None
        with TimeRecorder.ctx.time_block("total_aggregator(inner)"):
            with TimeRecorder.ctx.time_block("resize_comm_buffer"):
                graph.comm_buf.resize_buffer(graph.comm_splits, local_nodes_feat.size(-1), num_bits)
                if num_bits != 32 and num_bits != 16 and graph.comm_buf_for_quantization is not None:
                    graph.comm_buf_for_quantization.resize_buffer(graph.comm_splits, local_nodes_feat.size(-1), num_bits)
            
                send_buf = graph.comm_buf.send_buf
                # zero the send buffer
                send_buf.zero_()
            
            if world_size > 1:
                with TimeRecorder.ctx.time_block("pre_aggregation"):
                    if is_pre_delay:  # pre aggregation
                        SPMM_forward(graph.adj_t_pre_post_aggr_to, local_nodes_feat, send_buf)
                    else:  # no pre aggregation
                        torch.index_select(local_nodes_feat, 0, graph.idx_nodes_send_to_others, out=send_buf)
                
                with TimeRecorder.ctx.time_block("total_communication(outer)"):
                    layer = f"forward{layer}"
                    comm_handle = Communicator.ctx.comm(
                        graph.comm_splits,
                        graph.comm_buf,
                        graph.comm_buf_for_quantization,
                        layer,
                        is_training,
                        "forward",
                    )

            with TimeRecorder.ctx.time_block("local_aggregation"):
                out = torch.zeros([graph.local_adj_t.sparse_size(0), local_nodes_feat.size(-1)], dtype=torch.float32)
                # aggregate message from local nodes
                # out = SPMM_forward(graph.local_adj_t, local_nodes_feat, out)
                SPMM_forward(graph.local_adj_t, local_nodes_feat, out)
            # out = SPMM_forward(graph.local_adj_t, local_nodes_feat, out)

            with TimeRecorder.ctx.time_block("wait_for_async_comm"):
                if comm_handle is not None:
                    comm_handle.wait()

            with TimeRecorder.ctx.time_block("dequantization"):
                num_recv_nodes = sum(graph.comm_splits.recv_splits)
                if world_size > 1 and num_recv_nodes != 0 and num_bits != 32 and num_bits != 16 and is_training:
                    if num_bits == 2 or num_bits == 4 or num_bits == 8:
                        Quantizer_for_all_procs.ctx.dequantize_intX_to_fp32(graph.comm_buf_for_quantization.quantized_recv_data_buf, 
                                                                            graph.comm_buf.recv_buf, 
                                                                            graph.comm_buf_for_quantization.quantized_recv_params_buf_fp32,
                                                                            graph.comm_buf_for_quantization.dequantized_work_range_per_proc)
                    # else:
                    #     Quantizer_v1.dequantize_intX_to_fp32(
                    #         quantized_recv_buf, recv_buf, dequantized_nodes_feat_range, dequantized_params
                    #     )

            with TimeRecorder.ctx.time_block("post_aggregation"):
                remote_nodes_feat = graph.comm_buf.recv_buf
                # aggregate message from remote nodes
                if world_size > 1 and remote_nodes_feat.size(0) != 0:
                    if is_pre_delay:  # post aggregation
                        SPMM_forward(graph.adj_t_pre_post_aggr_from, remote_nodes_feat, out)
                    else:
                        SPMM_forward(graph.remote_adj_t, remote_nodes_feat, out)
                        # out = SPMM_forward(graph.remote_adj_t, remote_nodes_feat, out)

        # return local_out
        return out

    @staticmethod
    def backward(ctx, local_out_grad):
        graph = ctx.graph
        num_bits = ctx.num_bits
        is_pre_delay = ctx.is_pre_delay
        layer = f"backward{ctx.layer}"

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # scatter gradient to remote nodes

        with TimeRecorder.ctx.time_block("total_aggregator(inner)"):
            with TimeRecorder.ctx.time_block("resize_comm_buffer"):
                # need to use the reverse buf for backward
                graph.comm_buf.resize_buffer(graph.comm_splits, local_out_grad.size(-1), num_bits)
                if num_bits != 32 and num_bits != 16 and graph.comm_buf_for_quantization is not None:
                    graph.comm_buf_for_quantization.resize_buffer(graph.comm_splits, local_out_grad.size(-1), num_bits)


            # need to use the reverse buf for backward
            remote_nodes_grad_buf = graph.comm_buf.recv_buf

            with TimeRecorder.ctx.time_block("pre_aggregation"):
                remote_nodes_grad_buf.zero_()
                if remote_nodes_grad_buf.size(0) != 0:
                    if is_pre_delay:  # pre aggregation
                        SPMM_backward(graph.adj_t_pre_post_aggr_from, local_out_grad, remote_nodes_grad_buf)
                    else:  # no pre aggregation
                        SPMM_backward(graph.remote_adj_t, local_out_grad, remote_nodes_grad_buf)

            comm_handle = None

            with TimeRecorder.ctx.time_block("total_communication(outer)"):
                # communicate to obtain the local node grads from other subgraph
                if world_size > 1:
                    comm_handle = Communicator.ctx.comm(
                        graph.comm_splits,
                        graph.comm_buf,
                        graph.comm_buf_for_quantization,
                        layer,
                        True,
                        "backward",
                    )

            with TimeRecorder.ctx.time_block("local_aggregation"):
                # scatter gradient to local nodes
                local_nodes_grad = torch.zeros(
                    [graph.local_adj_t.sparse_size(-1), local_out_grad.size(-1)], dtype=torch.float32
                )
                SPMM_backward(graph.local_adj_t, local_out_grad, local_nodes_grad)

            with TimeRecorder.ctx.time_block("wait_for_async_comm"):
                if comm_handle is not None:
                    comm_handle.wait()

            with TimeRecorder.ctx.time_block("dequantization"):
                num_recv_nodes = sum(graph.comm_splits.send_splits)
                # convert communication data to fp32
                if world_size > 1 and num_recv_nodes != 0 and num_bits != 32 and num_bits != 16:
                    if num_bits == 2 or num_bits == 4 or num_bits == 8:
                        Quantizer_for_all_procs.ctx.dequantize_intX_to_fp32(
                            graph.comm_buf_for_quantization.quantized_send_data_buf, 
                            graph.comm_buf.send_buf,
                            graph.comm_buf_for_quantization.quantized_send_params_buf_fp32,
                            graph.comm_buf_for_quantization.quantized_work_range_per_proc
                        )

            with TimeRecorder.ctx.time_block("post_aggregation"):
                local_nodes_grad_buf = graph.comm_buf.send_buf
                # then accumulate the local node grads
                if local_nodes_grad_buf.size(0) != 0:
                    if is_pre_delay:  # post aggregation
                        SPMM_backward(graph.adj_t_pre_post_aggr_to, local_nodes_grad_buf, local_nodes_grad)
                    else:
                        local_nodes_grad.index_add_(
                            dim=0, index=graph.idx_nodes_send_to_others, source=local_nodes_grad_buf
                        )

        return None, local_nodes_grad, None, None, None, None
