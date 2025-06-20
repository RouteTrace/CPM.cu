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

#include <cuda.h>
#include "marlin_kernel.cuh"
#include "marlin.cuh"
#include "core/scalar_type.hpp"
#include "gptq_marlin_mm.cuh"
#include "gptq_marlin_utils.cuh"


template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}

namespace marlin {

template <typename scalar_t>
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* s,
               void* zp, void* g_idx, void* perm, void* a_tmp, int prob_m,
               int prob_n, int prob_k, void* workspace,
               vllm::ScalarType const& q_type, bool has_act_order,
               bool is_k_full, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k, int thread_n,
               int sms, int max_par, bool use_fp32_reduce) {
  if (has_zp) {
    TORCH_CHECK(
        q_type == vllm::kU4 || q_type == vllm::kU8,
        "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());
  } else {
    TORCH_CHECK(
        q_type == vllm::kU4B8 || q_type == vllm::kU8B128,
        "q_type must be uint4b8 or uint8b128 when has_zp = False. Got = ",
        q_type.str());
  }

  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
              ", ", prob_n, ", ", prob_k, "]");

  // TODO: remove alias when we start supporting other 8bit types
  int num_bits = q_type.size_bits();
  int tot_m = prob_m;
  int tot_m_blocks = div_ceil(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  // Set thread config
  exec_config_t exec_cfg;
  if (thread_k != -1 && thread_n != -1) {
    // User-defined config
    exec_cfg =
        exec_config_t{4, thread_config_t{thread_k, thread_n, default_threads}};
  } else {
    // Auto config
    exec_cfg =
        determine_thread_config(prob_m, prob_n, prob_k, num_bits, group_size,
                                has_act_order, is_k_full, max_shared_mem);
  }

  TORCH_CHECK(exec_cfg.max_m_blocks > 0 &&
                  is_valid_config(exec_cfg.tb_cfg, exec_cfg.max_m_blocks,
                                  prob_m, prob_n, prob_k, num_bits, group_size,
                                  has_act_order, is_k_full, max_shared_mem),
              "Invalid thread config: max_m_blocks = ", exec_cfg.max_m_blocks,
              ", thread_k = ", exec_cfg.tb_cfg.thread_k,
              ", thread_n = ", exec_cfg.tb_cfg.thread_n,
              ", num_threads = ", exec_cfg.tb_cfg.num_threads, " for MKN = [",
              prob_m, ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
              ", group_size = ", group_size,
              ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
              ", max_shared_mem = ", max_shared_mem);

  int num_threads = exec_cfg.tb_cfg.num_threads;
  thread_k = exec_cfg.tb_cfg.thread_k;
  thread_n = exec_cfg.tb_cfg.thread_n;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  int blocks = sms;

  TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
              " is not divisible by thread_n = ", thread_n);
  TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
              " is not divisible by thread_k = ", thread_k);

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    } else {
      TORCH_CHECK(group_size == 0);
      group_blocks = 0;
    }

  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    }
  }

  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* s_ptr = (const int4*)s;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;

  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    int block_rows = div_ceil(prob_m, blocks);
    permute_cols_kernel<<<blocks, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, block_rows);
    A_ptr = a_tmp_ptr;
  }

  // If we have a full K, then we can run the non-act-order version of Marlin
  // (since the weight rows are reordered by increasing group ids, and by having
  // a full K, we have full original groups)
  if (is_k_full) {
    has_act_order = false;
  }

  // Main loop
  for (int i = 0; i < tot_m_blocks; i += exec_cfg.max_m_blocks) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > exec_cfg.max_m_blocks) {
      // Note that parallel > 1 currently only works for inputs without any
      // padding
      par = (16 * thread_m_blocks - pad) / (16 * exec_cfg.max_m_blocks);
      if (par > max_par) par = max_par;
      prob_m = (16 * exec_cfg.max_m_blocks) * par;
      i += exec_cfg.max_m_blocks * (par - 1);
      thread_m_blocks = exec_cfg.max_m_blocks;
    }

    if (false) {
    }
    GPTQ_CALL_IF(vllm::kU4B8, 16, 4, 256)
    GPTQ_CALL_IF(vllm::kU4B8, 8, 8, 256)
    GPTQ_CALL_IF(vllm::kU4B8, 8, 4, 128)
    GPTQ_CALL_IF(vllm::kU4B8, 4, 8, 128)
    GPTQ_CALL_IF(vllm::kU8B128, 16, 4, 256)
    GPTQ_CALL_IF(vllm::kU8B128, 8, 8, 256)
    GPTQ_CALL_IF(vllm::kU8B128, 8, 4, 128)
    GPTQ_CALL_IF(vllm::kU8B128, 4, 8, 128)

    AWQ_CALL_IF(vllm::kU4, 16, 4, 256)
    AWQ_CALL_IF(vllm::kU4, 8, 8, 256)
    AWQ_CALL_IF(vllm::kU4, 8, 4, 128)
    AWQ_CALL_IF(vllm::kU4, 4, 8, 128)
    AWQ_CALL_IF(vllm::kU8, 16, 4, 256)
    AWQ_CALL_IF(vllm::kU8, 8, 8, 256)
    AWQ_CALL_IF(vllm::kU8, 8, 4, 128)
    AWQ_CALL_IF(vllm::kU8, 4, 8, 128)
    else {
      TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                  ", ", prob_k, "]", ", has_act_order = ", has_act_order,
                  ", num_groups = ", num_groups, ", group_size = ", group_size,
                  ", thread_m_blocks = ", thread_m_blocks,
                  ", thread_n_blocks = ", thread_n_blocks,
                  ", thread_k_blocks = ", thread_k_blocks,
                  ", num_bits = ", num_bits);
    }

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }
}


} // namespace marlin 

// Explicit template instantiations
#ifdef ENABLE_DTYPE_FP16
template void marlin::marlin_mm<__half>(const void* A, const void* B, void* C, void* C_tmp, void* s,
                                void* zp, void* g_idx, void* perm, void* a_tmp, int prob_m,
                                int prob_n, int prob_k, void* workspace,
                                vllm::ScalarType const& q_type, bool has_act_order,
                                bool is_k_full, bool has_zp, int num_groups, int group_size,
                                int dev, cudaStream_t stream, int thread_k, int thread_n,
                                int sms, int max_par, bool use_fp32_reduce);
#endif

#ifdef ENABLE_DTYPE_BF16
template void marlin::marlin_mm<__nv_bfloat16>(const void* A, const void* B, void* C, void* C_tmp, void* s,
                                void* zp, void* g_idx, void* perm, void* a_tmp, int prob_m,
                                int prob_n, int prob_k, void* workspace,
                                vllm::ScalarType const& q_type, bool has_act_order,
                                bool is_k_full, bool has_zp, int num_groups, int group_size,
                                int dev, cudaStream_t stream, int thread_k, int thread_n,
                                int sms, int max_par, bool use_fp32_reduce);
#endif
