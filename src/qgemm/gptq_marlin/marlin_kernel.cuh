#pragma once

#include "marlin_device_ops.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"
#include "marlin.cuh"

namespace marlin {
template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const bool has_act_order,    // whether act_order is enabled
          const bool has_zp,           // whether zero-points are enabled
          const int group_blocks       // number of consecutive 16x16 blocks
                                       // with a separate quantization scale
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int num_groups,       // number of scale groups per output channel
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
);

}  // namespace marlin 
