#pragma once

#include <cuda.h>
#include "core/scalar_type.hpp"

namespace marlin {

template <typename scalar_t>
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* s,
               void* zp, void* g_idx, void* perm, void* a_tmp, int prob_m,
               int prob_n, int prob_k, void* workspace,
               vllm::ScalarType const& q_type, bool has_act_order,
               bool is_k_full, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k, int thread_n,
               int sms, int max_par, bool use_fp32_reduce);

} // namespace marlin 