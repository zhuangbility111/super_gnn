// quantize_cuda_pack.cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <stdexcept>
#include <algorithm>
#include <torch/extension.h> // For PyTorch C++ extensions
#include <ATen/cuda/CUDAContext.h> // For CUDA stream management
#include <ATen/ATen.h> // For tensor operations

// #define MAX_THREAD_BLOCKS ((1 << 16) - 1)
#define MAX_THREAD_BLOCKS INT_MAX
#define MAX_THREADS_PER_BLOCK 1024
#define QUANTIZED_PARAMS_SIZE 2

// 模板核函数：模板参数 num_per_pack 表示每个 int8_t 内打包的数目；boundary_check 为 true 时，对组内每行做边界判断
template <int ELEMS_PER_BYTE, bool BOUNDARY_CHECK>
__global__ void quantize_kernel(
    const float* __restrict__ data_fp32,         // [M, N]
    uint8_t* __restrict__ data_uint8,               // 打包后尺寸：[group, N]，group 数由计算区域内的组数决定
    const float* __restrict__ quantized_params,// [M, 2]，每行预先存好 zero_point 和 scale
    int data_fp32_row_start, int data_fp32_row_end, int data_uint8_row_start, int N)                 // 处理区域为 [start, end) 行；N 为列数
{
    // 每个 block 负责处理一个组
    int group = blockIdx.x;
    // 该组对应 data_fp32 内连续的 ELEMS_PER_BYTE 行
    int data_fp32_row_base = data_fp32_row_start + group * ELEMS_PER_BYTE;
    int data_uint8_row_base = data_uint8_row_start + group;

    const int BITS = 8 / ELEMS_PER_BYTE; // bits 只能是 2、4 或 8
    
    if (BOUNDARY_CHECK) {
        if (data_fp32_row_base >= data_fp32_row_end)
            return; // for the thread block that the start row is out of range
    }
    // 每个 block 内所有线程对列并行处理
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        uint8_t pack = 0;
#pragma unroll
        for (int i = 0; i < ELEMS_PER_BYTE; i++) {
            int row = data_fp32_row_base + i;
            int q = 0;
            if (BOUNDARY_CHECK) {
                if (row < data_fp32_row_end) {
                    float zero_point = quantized_params[row * QUANTIZED_PARAMS_SIZE + 0];
                    float scale = quantized_params[row * QUANTIZED_PARAMS_SIZE + 1];
                    float val = data_fp32[row * N + col];
                    q = (int)__float2int_rn((val - zero_point) * __frcp_rn(scale + 1e-20f));
                }
            } else {
                float zero_point = quantized_params[row * QUANTIZED_PARAMS_SIZE + 0];
                float scale = quantized_params[row * QUANTIZED_PARAMS_SIZE + 1];
                float val = data_fp32[row * N + col];
                q = (int)__float2int_rn((val - zero_point) * __frcp_rn(scale + 1e-20f));
            }
            pack |= (q << ((ELEMS_PER_BYTE - 1 - i) * BITS));
        }
        data_uint8[data_uint8_row_base * N + col] = pack;
    }
}

// 模板核函数：反量化
template <int ELEMS_PER_BYTE, bool BOUNDARY_CHECK>
__global__ void dequantize_kernel(
    const uint8_t* __restrict__ data_uint8,         // 打包后尺寸：[group, N]
    float* __restrict__ data_fp32,              // 解包后尺寸：[M, N]
    const float* __restrict__ quantized_params, // [M, 2]，每行预先存好 zero_point 和 scale
    int data_fp32_row_start, int data_fp32_row_end, int data_uint8_row_start, int N) // 处理区域为 [start, end) 行；N 为列数
{
    int group = blockIdx.x;
    int data_fp32_row_base = data_fp32_row_start + group * ELEMS_PER_BYTE;
    int data_uint8_row_base = data_uint8_row_start + group;

    const int BITS = 8 / ELEMS_PER_BYTE; // bits 只能是 2、4 或 8
    const int MASK = (1 << BITS) - 1;

    if (BOUNDARY_CHECK) {
        if (data_fp32_row_base >= data_fp32_row_end)
            return; // for the thread block that the start row is out of range
    }

    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        uint8_t pack = data_uint8[data_uint8_row_base * N + col];
#pragma unroll
        for (int i = 0; i < ELEMS_PER_BYTE; i++) {
            int row = data_fp32_row_base + i;
            if (BOUNDARY_CHECK) {
                if (row < data_fp32_row_end) {
                    float zero_point = quantized_params[row * QUANTIZED_PARAMS_SIZE + 0];
                    float scale = quantized_params[row * QUANTIZED_PARAMS_SIZE + 1];
                    int q = (pack >> ((ELEMS_PER_BYTE - 1 - i) * BITS)) & MASK;
                    data_fp32[row * N + col] = q * scale + zero_point;
                }
            } else {
                float zero_point = quantized_params[row * QUANTIZED_PARAMS_SIZE + 0];
                float scale = quantized_params[row * QUANTIZED_PARAMS_SIZE + 1];
                int q = (pack >> ((ELEMS_PER_BYTE - 1 - i) * BITS)) & MASK;
                data_fp32[row * N + col] = q * scale + zero_point;
            }
        }
    }
}

// =====================================================================
// Launcher：针对每个计算区域（proc），根据区域内行数 rows，计算完整组数和不足组数，然后分别 launch kernel
// 每个 int8_t 内打包的数目 elems_per_byte 就等于 ELEMS_PER_BYTE
// =====================================================================
void quantize_cuda_launcher(
    torch::Tensor data_fp32,            // fp32, [M, N]
    torch::Tensor data_uint8,           // int8, 尺寸取决于 bits：bits==8: [M, N]；bits==4: [ceil(M/2), N]；bits==2: [ceil(M/4), N]
    torch::Tensor quantized_params, // fp32, [M, 2]
    torch::Tensor work_range_per_proc, // int, 长度为 num_procs + 1，每对元素为一个计算区域的 [start, end)
    int num_procs,
    int bits)
{
    // check if data is on the gpu
    if (!data_fp32.is_cuda() || !data_uint8.is_cuda() || !quantized_params.is_cuda() || !work_range_per_proc.is_cpu()) {
        throw std::runtime_error("All tensors must be on the GPU or work_range_per_proc must be on CPU");
    }
    // check if data is contiguous
    if (!data_fp32.is_contiguous() || !data_uint8.is_contiguous() || !quantized_params.is_contiguous() || !work_range_per_proc.is_contiguous()) {
        throw std::runtime_error("All tensors must be contiguous");
    }
    // check if data is of the right type
    if (data_fp32.dtype() != at::kFloat || data_uint8.dtype() != at::kByte || quantized_params.dtype() != at::kFloat || work_range_per_proc.dtype() != at::kInt) {
        throw std::runtime_error("data_fp32 must be float, data_uint8 must be byte, quantized_params must be float, and work_range_per_proc must be int");
    }

    const int M = data_fp32.size(0);
    const int N = data_fp32.size(1);
    const float* data_fp32_ptr = data_fp32.data_ptr<float>();
    uint8_t* data_uint8_ptr = data_uint8.data_ptr<uint8_t>();
    const float* params_ptr = quantized_params.data_ptr<float>();
    const int* work_range_ptr = work_range_per_proc.data_ptr<int>();

    // 固定线程数（例如 N）负责列上的并行
    const int threads = std::min(N, MAX_THREADS_PER_BLOCK); // Updated to use N as the number of threads
    dim3 block_dim(threads);

    // get cuda stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 根据 bits 确定 ELEMS_PER_BYTE（即每个 int8_t 内打包的数目）
    const int ELEMS_PER_BYTE = 8 / bits; // bits 只能是 2、4 或 8

    int data_uint8_row_start = 0;
    // 针对每个计算区域
    for (int proc = 0; proc < num_procs; proc++) {
        int data_fp32_row_start = work_range_ptr[proc];
        int data_fp32_row_end   = work_range_ptr[proc + 1];
        if (data_fp32_row_start < 0 || data_fp32_row_end > M || data_fp32_row_end <= data_fp32_row_start)
            continue;
        int rows = data_fp32_row_end - data_fp32_row_start;
        // 完整组数，一个thread block 处理一个组
        int groups = (rows + ELEMS_PER_BYTE - 1) / ELEMS_PER_BYTE;
        // 如果groups太多导致thread block数超过最大值，则分多次处理
        int num_thread_blocks = std::min(groups, MAX_THREAD_BLOCKS);
        int iters = (groups + num_thread_blocks - 1) / num_thread_blocks;
        dim3 grid_dim(num_thread_blocks);

        for (int i = 0; i < iters; i++) {
            int group_start = data_fp32_row_start + i * num_thread_blocks * ELEMS_PER_BYTE;
            int group_end = std::min(group_start + num_thread_blocks * ELEMS_PER_BYTE, data_fp32_row_end);
            // printf("group_start: %d, group_end: %d, output_row_start: %d\n", group_start, group_end, data_uint8_row_start);
            bool boundary_check = (group_start + num_thread_blocks * ELEMS_PER_BYTE > data_fp32_row_end);
            if (boundary_check) {
                if (bits == 8) {
                    quantize_kernel<1, true>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_fp32_ptr,
                        data_uint8_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                } 
                else if (bits == 4) {
                    quantize_kernel<2, true>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_fp32_ptr,
                        data_uint8_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                } else if (bits == 2) {
                    quantize_kernel<4, true>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_fp32_ptr,
                        data_uint8_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                }
                
            } else {
                if (bits == 8) {
                    quantize_kernel<1, false>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_fp32_ptr,
                        data_uint8_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                } 
                else if (bits == 4) {
                    quantize_kernel<2, false>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_fp32_ptr,
                        data_uint8_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                } else if (bits == 2) {
                    quantize_kernel<4, false>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_fp32_ptr,
                        data_uint8_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                }
            }
            // 更新输出行起始位置
            data_uint8_row_start += (group_end - group_start + (ELEMS_PER_BYTE - 1)) / ELEMS_PER_BYTE;
        }
    }
}

// =====================================================================
// Launcher：针对每个计算区域（proc），根据区域内行数 rows，计算完整组数和不足组数，然后分别 launch kernel
// =====================================================================
void dequantize_cuda_launcher(
    torch::Tensor data_uint8,            // int8, 尺寸取决于 bits：bits==8: [M, N]；bits==4: [ceil(M/2), N]；bits==2: [ceil(M/4), N]
    torch::Tensor data_fp32,           // fp32, [M, N]
    torch::Tensor quantized_params, // fp32, [M, 2]
    torch::Tensor work_range_per_proc, // int, 长度为 num_procs + 1，每对元素为一个计算区域的 [start, end)
    int num_procs,
    int bits)
{
    // check if data is on the gpu
    if (!data_fp32.is_cuda() || !data_uint8.is_cuda() || !quantized_params.is_cuda() || !work_range_per_proc.is_cpu()) {
        throw std::runtime_error("All tensors must be on the GPU or work_range_per_proc must be on CPU");
    }
    // check if data is contiguous
    if (!data_fp32.is_contiguous() || !data_uint8.is_contiguous() || !quantized_params.is_contiguous() || !work_range_per_proc.is_contiguous()) {
        throw std::runtime_error("All tensors must be contiguous");
    }
    // check if data is of the right type
    if (data_fp32.dtype() != at::kFloat || data_uint8.dtype() != at::kByte || quantized_params.dtype() != at::kFloat || work_range_per_proc.dtype() != at::kInt) {
        throw std::runtime_error("data_fp32 must be float, data_uint8 must be byte, quantized_params must be float, and work_range_per_proc must be int");
    }

    const int M = data_fp32.size(0);
    const int N = data_fp32.size(1);
    const uint8_t* data_uint8_ptr = data_uint8.data_ptr<uint8_t>();
    float* data_fp32_ptr = data_fp32.data_ptr<float>();
    const float* params_ptr = quantized_params.data_ptr<float>();
    const int* work_range_ptr = work_range_per_proc.data_ptr<int>();

    // get cuda stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int threads = std::min(N, MAX_THREADS_PER_BLOCK);
    dim3 block_dim(threads);

    const int ELEMS_PER_BYTE = 8 / bits;

    int data_uint8_row_start = 0;
    for (int proc = 0; proc < num_procs; proc++) {
        int data_fp32_row_start = work_range_ptr[proc];
        int data_fp32_row_end   = work_range_ptr[proc + 1];
        if (data_fp32_row_start < 0 || data_fp32_row_end > M || data_fp32_row_end <= data_fp32_row_start)
            continue;
        int rows = data_fp32_row_end - data_fp32_row_start;
        int groups = (rows + ELEMS_PER_BYTE - 1) / ELEMS_PER_BYTE;
        int num_thread_blocks = std::min(groups, MAX_THREAD_BLOCKS);
        int iters = (groups + num_thread_blocks - 1) / num_thread_blocks;
        dim3 grid_dim(num_thread_blocks);

        for (int i = 0; i < iters; i++) {
            int group_start = data_fp32_row_start + i * num_thread_blocks * ELEMS_PER_BYTE;
            int group_end = std::min(group_start + num_thread_blocks * ELEMS_PER_BYTE, data_fp32_row_end);
            bool boundary_check = (group_start + num_thread_blocks * ELEMS_PER_BYTE > data_fp32_row_end);
            if (boundary_check) {
                if (bits == 8) {
                    dequantize_kernel<1, true>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_uint8_ptr,
                        data_fp32_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                } else if (bits == 4) {
                    dequantize_kernel<2, true>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_uint8_ptr,
                        data_fp32_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                } else if (bits == 2) {
                    dequantize_kernel<4, true>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_uint8_ptr,
                        data_fp32_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                }
            }
            // 否则不需要边界检查
            else {
                if (bits == 8) {
                    dequantize_kernel<1, false>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_uint8_ptr,
                        data_fp32_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                } else if (bits == 4) {
                    dequantize_kernel<2, false>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_uint8_ptr,
                        data_fp32_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                } else if (bits == 2) {
                    dequantize_kernel<4, false>
                    <<<grid_dim, block_dim, 0, stream>>>(
                        data_uint8_ptr,
                        data_fp32_ptr,
                        params_ptr,
                        group_start,
                        group_end,
                        data_uint8_row_start,
                        N);
                }
            }
            // 更新输出行起始位置
            data_uint8_row_start += (group_end - group_start + (ELEMS_PER_BYTE - 1)) / ELEMS_PER_BYTE;
        }
    }
}

void quantize_tensor_for_all_procs_cuda(
    torch::Tensor data_fp32,
    torch::Tensor data_uint8,
    torch::Tensor quantized_params,
    torch::Tensor work_range_per_proc,
    int num_procs,
    int bits)
{
    quantize_cuda_launcher(data_fp32, data_uint8, quantized_params, work_range_per_proc, num_procs, bits);
}

void dequantize_tensor_for_all_procs_cuda(
    torch::Tensor data_uint8,
    torch::Tensor data_fp32,
    torch::Tensor quantized_params,
    torch::Tensor work_range_per_proc,
    int num_procs,
    int bits)
{
    dequantize_cuda_launcher(data_uint8, data_fp32, quantized_params, work_range_per_proc, num_procs, bits);
}
