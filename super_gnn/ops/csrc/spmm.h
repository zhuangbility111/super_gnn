#include <torch/extension.h>

void spmm(
    torch::Tensor rowptr, torch::Tensor col, torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
    torch::Tensor out, int64_t sparse_rows, torch::Tensor parallel_row_split,
    torch::Tensor parallel_col_split);