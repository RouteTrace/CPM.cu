/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#include <cuda.h>
#include "marlin.cuh"
#include "core/scalar_type.hpp"
#include "gptq_marlin.cuh"
#include "gptq_marlin_mm.cuh"


// torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
//                                torch::Tensor& b_scales, torch::Tensor& b_zeros,
//                                torch::Tensor& g_idx, torch::Tensor& perm,
//                                torch::Tensor& workspace,
//                                vllm::ScalarType const& b_q_type, // init in linear
//                                int64_t size_m, int64_t size_n, int64_t size_k,
//                                bool is_k_full, bool has_zp,
//                                bool use_fp32_reduce, 
//                                // new input
//                                void* c
//                                ) {

template <typename T>
void gptq_marlin_gemm(T* a, int32_t* b_q_weight,
                               T* b_scales, int32_t* b_zeros,
                               int32_t* g_idx, int32_t* perm,
                               int32_t* workspace,
                               vllm::ScalarType const& b_q_type, // init in linear
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp,
                               bool use_fp32_reduce, 
                               // TODO: new args
                               T* c,
                               int num_groups, int group_size,
                               int b_q_weight_size1,
                               bool has_act_order,
                               cudaStream_t stream,
                               T* a_tmp,
                               float* c_tmp
                               ) {

  int pack_factor = 32 / b_q_type.size_bits();

  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel (can usually be left as auto -1)
  int sms = -1;

  // Verify workspace size
  TORCH_CHECK(size_n % marlin::min_thread_n == 0, "size_n = ", size_n,
              ", is not divisible by min_thread_n = ", marlin::min_thread_n);

  // int dev = a.get_device();
  int dev = 0;  // 选择第一个 GPU（设备 0）
  cudaSetDevice(dev);  // 设置当前 CUDA 设备
  marlin::marlin_mm<T>(
      a, b_q_weight, c,
      c_tmp, b_scales,
      b_zeros, g_idx, perm,
      a_tmp, size_m, size_n, size_k,
      workspace, b_q_type, has_act_order, is_k_full, has_zp,
      num_groups, group_size, dev, stream,
      thread_k, thread_n, sms, marlin::max_par, use_fp32_reduce);

  return;
}

// Explicit template instantiations
#ifdef ENABLE_DTYPE_FP16
template void gptq_marlin_gemm<__half>(__half* a, int32_t* b_q_weight,
                               __half* b_scales, int32_t* b_zeros,
                               int32_t* g_idx, int32_t* perm,
                               int32_t* workspace,
                               vllm::ScalarType const& b_q_type,
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp,
                               bool use_fp32_reduce, 
                               __half* c,
                               int num_groups, int group_size,
                               int b_q_weight_size1,
                               bool has_act_order,
                               cudaStream_t stream,
                               __half* a_tmp,
                               float* c_tmp);
#endif

#ifdef ENABLE_DTYPE_BF16
template void gptq_marlin_gemm<__nv_bfloat16>(__nv_bfloat16* a, int32_t* b_q_weight,
                               __nv_bfloat16* b_scales, int32_t* b_zeros,
                               int32_t* g_idx, int32_t* perm,
                               int32_t* workspace,
                               vllm::ScalarType const& b_q_type,
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp,
                               bool use_fp32_reduce, 
                               __nv_bfloat16* c,
                               int num_groups, int group_size,
                               int b_q_weight_size1,
                               bool has_act_order,
                               cudaStream_t stream,
                               __nv_bfloat16* a_tmp,
                               float* c_tmp);
#endif
