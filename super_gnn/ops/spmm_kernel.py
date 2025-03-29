import torch
from torch_scatter import segment_csr
from torch_sparse import SparseTensor
from supergnn_ops import spmm
import time


def distribute_work_for_row(rowptr: torch.Tensor, value: torch.Tensor, num_threads_on_row: int) -> torch.Tensor:
    flops_per_row = segment_csr(value, rowptr, None, "sum")
    num_rows = flops_per_row.shape[0]
    total_flops_on_rows = flops_per_row.sum()
    target_flops_on_rows = total_flops_on_rows // num_threads_on_row
    row_splits = torch.full((num_threads_on_row + 1,), num_rows, dtype=torch.int32)
    row_splits[0] = 0

    # use greedy algorithm to distribute work for row_splits
    cur_flops_sum = 0
    cur_tid = 0
    for i in range(flops_per_row.shape[0]):
        cur_flops_sum += flops_per_row[i]
        if cur_flops_sum > target_flops_on_rows and cur_tid < num_threads_on_row:
            # row_splits[cur_tid+1] = i+1
            row_splits[cur_tid+1] = i
            cur_tid += 1
            cur_flops_sum = flops_per_row[i]
    row_splits[-1] = num_rows

    # print(row_splits)
    return row_splits

def disribute_work_for_col(other: torch.Tensor,  num_threads_on_col: int):
    num_cols = other.shape[-1]
    total_flops_on_cols = num_cols
    target_flops_on_cols = total_flops_on_cols // num_threads_on_col
    col_splits = torch.full((num_threads_on_col + 1,), num_cols, dtype=torch.int32)
    col_splits[0] = 0

    # to distributed work evenly for col_splits
    for i in range(num_threads_on_col):
        col_splits[i] = i * target_flops_on_cols
    col_splits[-1] = num_cols

    # print(col_splits)
    return col_splits
    
def SPMM_forward_cpu(src: SparseTensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()
    row_splits, col_splits = src.get_work_range()
    if row_splits.size(0) == 0:
        num_threads = torch.get_num_threads()
        row_splits = distribute_work_for_row(rowptr, value, num_threads)
    col_splits = disribute_work_for_col(other, 1)
    src.set_work_range(row_splits, col_splits)
    if value is not None:
        value = value.to(other.dtype)
    sparse_row = rowptr.shape[0] - 1
    spmm(rowptr, col, value, other, out, sparse_row, row_splits, col_splits)
    return None

def SPMM_forward_gpu(src: SparseTensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    sparse_matrix = src.to_torch_sparse_coo_tensor()
    result = torch.sparse.mm(sparse_matrix, other)
    out.copy_(result)
    return None

def SPMM_forward(src: SparseTensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if src.device.type == 'cuda' and other.device.type == 'cuda':
        return SPMM_forward_gpu(src, other, out)
    return SPMM_forward_cpu(src, other, out)

def SPMM_backward_cpu(src: SparseTensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    colptr = src.storage.colptr()
    row_T = src.storage.row_T()
    value_T = src.storage.value_T()
    row_splits, col_splits = src.get_work_range_for_transpose()
    if row_splits.size(0) == 0:
        num_threads = torch.get_num_threads()
        row_splits = distribute_work_for_row(colptr, value_T, num_threads)
    col_splits = disribute_work_for_col(other, 1)
    src.set_work_range_for_transpose(row_splits, col_splits)
    sparse_row = colptr.shape[0] - 1
    return spmm(colptr, row_T, value_T, other, out, sparse_row, row_splits, col_splits)

def SPMM_backward_gpu(src: SparseTensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    colptr = src.storage.colptr()
    row_T = src.storage.row_T()
    value_T = src.storage.value_T()
    sparse_matrix = torch.sparse_csr_tensor(
        colptr, row_T, value_T, device=src.device, dtype=other.dtype
    )
    result = torch.sparse.mm(sparse_matrix, other)
    out.copy_(result)
    return None

def SPMM_backward(src: SparseTensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if src.device.type == 'cuda' and other.device.type == 'cuda':
        return SPMM_backward_gpu(src, other, out)
    return SPMM_backward_cpu(src, other, out)


