#include <torch/extension.h>
#include "spmm.h"

#ifdef __ARM_FEATURE_SVE /* __ARM_FEATURE_SVE */
#include "quantization_arm.h"
#elif __AVX512F__ /* AVX512 */
#include "quantization_x86.h"
#endif

// if cuda is enabled, include the quantization_cuda.h
#include "quantization_cuda.h"


#include "vertex_cover.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("quantize_tensor", &quantize_tensor);
    // m.def("dequantize_tensor", &dequantize_tensor);
    // m.def("quantize_tensor_v1", &quantize_tensor_v1);
    // m.def("dequantize_tensor_v1", &dequantize_tensor_v1);
#ifdef __ARM_FEATURE_SVE /* __ARM_FEATURE_SVE */
    m.def("quantize_tensor_for_all_procs", &quantize_tensor_for_all_procs);
    m.def("dequantize_tensor_for_all_procs", &dequantize_tensor_for_all_procs);
#elif __AVX512F__ /* AVX512 */
    m.def("quantize_tensor_v2_torch", &quantize_tensor_v2_torch);
    m.def("dequantize_tensor_v2_torch", &dequantize_tensor_v2_torch);
#endif

    m.def("quantize_tensor_for_all_procs_cuda", &quantize_tensor_for_all_procs_cuda, "Pack quantization (int2/int4/int8) with block-level grouping and template boundary check (CUDA)");
    m.def("dequantize_tensor_for_all_procs_cuda", &dequantize_tensor_for_all_procs_cuda, "Unpack dequantization (int2/int4/int8) with block-level grouping (CUDA)");

    // m.def("spmm", &spmm);

    // m.def("find_vertex_cover", &find_vertex_cover);
}