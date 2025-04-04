#pragma once

#include <torch/extension.h>

// Function declarations
// void quantize_cuda_launcher(
//     torch::Tensor data_fp32,
//     torch::Tensor data_uint8,
//     torch::Tensor quantized_params,
//     torch::Tensor work_range_per_proc,
//     int num_procs,
//     int bits);

// void dequantize_cuda_launcher(
//     torch::Tensor data_uint8,
//     torch::Tensor data_fp32,
//     torch::Tensor quantized_params,
//     torch::Tensor work_range_per_proc,
//     int num_procs,
//     int bits);

void quantize_tensor_for_all_procs_cuda(
    torch::Tensor data_fp32,
    torch::Tensor data_uint8,
    torch::Tensor quantized_params,
    torch::Tensor work_range_per_proc,
    int num_procs,
    int bits);

void dequantize_tensor_for_all_procs_cuda(
    torch::Tensor data_uint8,
    torch::Tensor data_fp32,
    torch::Tensor quantized_params,
    torch::Tensor work_range_per_proc,
    int num_procs,
    int bits);
