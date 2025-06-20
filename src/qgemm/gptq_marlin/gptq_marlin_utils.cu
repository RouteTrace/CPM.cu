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
#include "marlin.cuh"
#include "core/scalar_type.hpp"
#include "gptq_marlin_utils.cuh"

namespace marlin {

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows) {
  int start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = finish_row - start_row;

  int row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int offset = row * row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      int cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        int cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128},
};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128},
};

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * pipe_stages * 2;     // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;

  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * pipe_stages;
  }
}

bool is_valid_cache_size(thread_config_t const& th_config, int max_m_blocks,
                         int prob_m, int prob_n, int prob_k, int num_bits,
                         int scales_cache_size, int max_shared_mem) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;

  int b_size = (tb_k * tb_n / pack_factor) * 4;

  // Get A size
  int m_blocks = div_ceil(prob_m, 16);
  int tb_max_m = 16;

  while (true) {
    if (m_blocks >= max_m_blocks) {
      tb_max_m *= max_m_blocks;
      break;
    }

    max_m_blocks--;
    if (max_m_blocks == 0) {
      TORCH_CHECK(false, "Unexpected m_blocks = ", m_blocks);
    }
  }

  int a_size = (tb_max_m * tb_k) * 2;

  float pipe_size = (a_size + b_size) * pipe_stages;

  float reduce_size = max(th_config.num_threads * 32 * 4,
                          (tb_n / 64) * 32 * (tb_max_m / 16) * 4 * 2 * 4 * 2);

  TORCH_CHECK(max_shared_mem / 2 > scales_cache_size);  // Sanity

  return pipe_size + reduce_size < 0.95f * (max_shared_mem - scales_cache_size);
}

bool is_valid_config(thread_config_t const& th_config, int max_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  //  Determine cache for scales
  int scales_cache_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full);

  // Check that pipeline fits into cache
  if (!is_valid_cache_size(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                           num_bits, scales_cache_size, max_shared_mem)) {
    return false;
  }

  return true;
}

int determine_reduce_max_m(int prob_m, int max_par) {
  constexpr int tile_m_size = 16;

  if (prob_m <= tile_m_size) {
    return tile_m_size;

  } else if (prob_m <= tile_m_size * 2) {
    return tile_m_size * 2;

  } else if (prob_m <= tile_m_size * 3) {
    return tile_m_size * 3;

  } else if (prob_m <= tile_m_size * 4) {
    return tile_m_size * 4;

  } else {
    int cur_par = min(div_ceil(prob_m, tile_m_size * 4), max_par);
    return tile_m_size * 4 * cur_par;
  }
}

exec_config_t determine_thread_config(int prob_m, int prob_n, int prob_k,
                                      int num_bits, int group_size,
                                      bool has_act_order, bool is_k_full,
                                      int max_shared_mem) {
  int max_m_blocks = 4;
  while (max_m_blocks > 0) {
    if (prob_m <= 16) {
      for (auto th_config : small_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    } else {
      for (auto th_config : large_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    }

    max_m_blocks--;  // Process less M blocks per invocation to reduce cache
                     // usage
  }

  return exec_config_t{0, {-1, -1, -1}};
}


} // namespace marlin 
