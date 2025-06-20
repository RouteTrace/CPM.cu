#pragma once

#include <cuda.h>
#include "core/scalar_type.hpp"

namespace marlin {

template <typename T>
inline std::string str(T x);

__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows);

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

typedef struct {
  int max_m_blocks;
  thread_config_t tb_cfg;
} exec_config_t;

extern thread_config_t small_batch_thread_configs[];
extern thread_config_t large_batch_thread_configs[];

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full);

bool is_valid_cache_size(thread_config_t const& th_config, int max_m_blocks,
                         int prob_m, int prob_n, int prob_k, int num_bits,
                         int scales_cache_size, int max_shared_mem);

bool is_valid_config(thread_config_t const& th_config, int max_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int max_shared_mem);

int determine_reduce_max_m(int prob_m, int max_par);

exec_config_t determine_thread_config(int prob_m, int prob_n, int prob_k,
                                      int num_bits, int group_size,
                                      bool has_act_order, bool is_k_full,
                                      int max_shared_mem);

#define __CALL_IF(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, \
                    HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS, NUM_THREADS)          \
    else if (q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS &&         \
             thread_n_blocks == THREAD_N_BLOCKS &&                             \
             thread_k_blocks == THREAD_K_BLOCKS &&                             \
             has_act_order == HAS_ACT_ORDER && has_zp == HAS_ZP &&             \
             group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS) {     \
      cudaFuncSetAttribute(                                                    \
          Marlin<scalar_t, W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS,          \
                 THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, HAS_ACT_ORDER, \
                 HAS_ZP, GROUP_BLOCKS>,                                        \
          cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);        \
      Marlin<scalar_t, W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS,              \
             THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, HAS_ACT_ORDER,     \
             HAS_ZP, GROUP_BLOCKS>                                             \
          <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(                   \
              A_ptr, B_ptr, C_ptr, C_tmp_ptr, s_ptr, zp_ptr, g_idx_ptr,        \
              num_groups, prob_m, prob_n, prob_k, locks, use_fp32_reduce);     \
    }

#define GPTQ_CALL_IF(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
                                                                            \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                            \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                            \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                            \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)

#define AWQ_CALL_IF(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                            \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                            \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                            \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)

} // namespace marlin