#pragma once
#include "../utils.cuh"
#include "../trait.cuh"
#include <iostream>
/* 面向4090GPU的开发版本 
    不一定兼容top>64的情况
*/
namespace functions {
namespace {
template<typename T, int TOP_SIZE>
static __device__ inline void warpBitonicSort(T& v1, int& pos, bool asc) {  // asc:false->最终结果是每个warp中的元素是降序的
    /* N不能等于64，否则j步长为32，不能跨warp交换数据 */
    int K = TOP_SIZE < 32 ? TOP_SIZE : 32;
    int lane_id = threadIdx.x & (TOP_SIZE - 1); // equal to threadIdx.x % TOP_SIZE

    #pragma unroll
    for (int k = 2; k <= K; k *= 2) {
        bool desc = ((lane_id & k) == 0) ^ asc; 
        #pragma unroll
        for (int j = k / 2; j > 0; j /= 2) {
            T v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
            int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos, j);
            bool upper = (lane_id & j) != 0; 
            if (desc ^ (v1 > v2 || (v1 == v2 && pos < pos2)) ^ upper) {
                v1 = v2;
                pos = pos2;
            }
        }
    }
}
template<typename T, int TOP_SIZE>
static __device__ inline void warpBitonicMerge(T& v1, int& pos1, T& v2, int& pos2) {
    if (v1 < v2 || (v1 == v2 && pos1 > pos2)) {
        v1 = v2;
        pos1 = pos2;
    }
    int N = TOP_SIZE < 32 ? TOP_SIZE : 32;
    int lane_id = threadIdx.x & (N - 1);
    
    // resort
    #pragma unroll
    for (int j = N / 2; j > 0; j /= 2) {
        v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
        int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos1, j);
        bool upper = (lane_id & j) != 0;
        if ((v1 < v2 || (v1 == v2 && pos1 > pos2)) ^ upper) {
            v1 = v2;
            pos1 = pos2;
        }
    }
}

template<typename T, int TOP_SIZE>
static __device__ inline void blockBitonicReduce(T& v, int& pos) {
    __shared__ T shared_val[1024];
    __shared__ int shared_pos[1024]; 
    // block reduce
    shared_val[threadIdx.x] = v;
    shared_pos[threadIdx.x] = pos;
    int reduce_end = TOP_SIZE < 32 ? 32 : TOP_SIZE; // reduce_end = 32 or TOP_SIZE(64)
    // inter warp reduce
    #pragma unroll
    for (int i = 512; i >= reduce_end; i >>= 1) {
        if (blockDim.x > i) {
            __syncthreads();
            if (threadIdx.x < i) {
                int idx_next = (i << 1) - threadIdx.x - 1; 
                T nw_v = (idx_next < blockDim.x) ? shared_val[idx_next] : T(-TypeTraits<T>::inf());
                int nw_pos = (idx_next < blockDim.x) ? shared_pos[idx_next] : -1;
                warpBitonicMerge<T, TOP_SIZE>(v, pos, nw_v, nw_pos); // merge and rebuild in desc order
                shared_val[threadIdx.x] = v;
                shared_pos[threadIdx.x] = pos;
            }
        }
    }
    /*  至此, 前TOP_SIZE个线程包含了block中最大的TOP_SIZE个元素，但是无序；如果top != 2的幂，则需要排序才能选出确切的topk；
        Question: 如果TOP_SIZE =64, 是否可以reduce_end执行到32，然后top64的后32个元素默认已经小于前32个元素了？此时只需要对32~63tid进行一次warpBitonicSort即可实现top64排序？ 
    */

    // intra warp reduce
    if (reduce_end == 32 && threadIdx.x < 32) {
        warpBitonicSort<T, 32>(v, pos, false);
    }
    else if (reduce_end != 32 && threadIdx.x < TOP_SIZE)
    {   
        // 需要对多个warp进行排序（目前只有2个warp的情况）
        // 使用基于共享内存的Bitonic Sort 对前 TOP_SIZE 个元素排序
        __syncthreads();
        for (int k = 2; k <= TOP_SIZE; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                int ixj = threadIdx.x ^ j;
                if (ixj > threadIdx.x)
                { 
                    // 确定排序方向（升序还是降序）
                    bool descending = ((threadIdx.x & k) == 0);

                    T partner_v = shared_val[ixj];
                    int partner_pos = shared_pos[ixj];
                    T my_v = shared_val[threadIdx.x];
                    int my_pos = shared_pos[threadIdx.x];

                    if (descending && my_v < partner_v)
                    {
                        shared_val[threadIdx.x] = partner_v;
                        shared_pos[threadIdx.x] = partner_pos;
                        shared_val[ixj] = my_v;
                        shared_pos[ixj] = my_pos;
                    }

                    else if (!descending && my_v > partner_v)
                    {
                        shared_val[threadIdx.x] = partner_v;
                        shared_pos[threadIdx.x] = partner_pos;
                        shared_val[ixj] = my_v;
                        shared_pos[ixj] = my_pos;
                    }
                }


                __syncthreads();
            }
        }
        // 排序完成后，每个线程从共享内存中读回最终排好序的数据到自己的寄存器
        v = shared_val[threadIdx.x];
        pos = shared_pos[threadIdx.x];
    }
}

// intra-block topk
template<typename T, int TOP_SIZE, bool FIRST> 
static __global__ void kernel_bitonic_topk_multiblock(
    int n, // dim
    const T *inp,       // (batch, n)
    const int *idx_inp, // (batch, n)
    T *buf_out,     // (batch, n / 1024 * N)
    int *buf_idx,    // (batch, n / 1024 * N)
    int top, // top <= top_size
    T *out,// final_out (batch, top)
    int *idx// final_idx (batch, top)
) {
    int offset_col = blockIdx.y * blockDim.x + threadIdx.x;
    int offset_inp = blockIdx.x * n + offset_col;
    int offset_out = blockIdx.x * (gridDim.y * TOP_SIZE) + blockIdx.y * TOP_SIZE + threadIdx.x;
    T local_v = (offset_col < n) ? inp[offset_inp] : T(-TypeTraits<T>::inf()); // 搬运到寄存器上，超出范围的值为负无穷
    // first call this kernel, idx_inp is nullptr, odrdered is true
    int local_pos = (idx_inp == nullptr) ? offset_col : ((offset_col < n) ? idx_inp[offset_inp] : -1); // 用来记录数据在最初的输入x中的位置
    
    if (FIRST) warpBitonicSort<T, TOP_SIZE>(local_v, local_pos, false); // local sort in desc order
    // satge1 : Reduce干的两件事，1、top_val已经在 tid<=top_size的warp中，2、如果有多个warp暂时是分别降序(warp之间需要再次排序)
    blockBitonicReduce<T, TOP_SIZE>(local_v, local_pos); 

    // stage2: 排序之后就可以按照top的数量写入out pos 了
    bool is_over = (CEIL_DIV(n, 1024) * TOP_SIZE) <= TOP_SIZE; // ture:最后一次执行，要写入out pos
    
    if (!is_over && threadIdx.x < TOP_SIZE) {
        buf_out[offset_out] = local_v;
        buf_idx[offset_out] = local_pos;
    }
    if(is_over && threadIdx.x < top) {
        // 最后一次执行，写入out pos
        int result_offset = blockIdx.x * top + threadIdx.x; // 定位输出到out的位置
        out[result_offset] = local_v;
        idx[result_offset] = local_pos;
    }
}

#define TOPK_SIZE_DISPATCH(top, ...) \
    do { \
        const int &top_v = top; \
        if(top_v > 32){ \
           const int top_size = 64; \
            __VA_ARGS__ \
        }else if (top_v > 16) { \
            const int top_size = 32; \
            __VA_ARGS__ \
        } else if (top_v > 8) { \
            const int top_size = 16; \
            __VA_ARGS__ \
        } else if (top_v > 4) { \
            const int top_size = 8; \
            __VA_ARGS__ \
        } else if (top_v > 2) { \
            const int top_size = 4; \
            __VA_ARGS__ \
        } else if (top_v > 1) { \
            const int top_size = 2; \
            __VA_ARGS__ \
        } else { \
            const int top_size = 1; \
            __VA_ARGS__ \
        } \
    } while(0)
template <typename T>
void bitonic_topk(
    const Stream& stream,
    const int batch,
    const int n,
    const int top,
    const T* x, 
    T* out, 
    int* pos,	
    T* buf_val,
    int* buf_pos,
    T* nw_buf_val,
    int* nw_buf_pos
) {
    TOPK_SIZE_DISPATCH(top, {
        bool first = true;
        dim3 blockDim(1024, 1, 1);
        unsigned int tmp_n = n;
        do {
            dim3 gridDim(batch, CEIL_DIV(tmp_n, 1024), 1); // grid_size (batch, 4, 1)
            if (first) {  
                kernel_bitonic_topk_multiblock<T, top_size, true><<<gridDim, blockDim, 0, stream.stream>>>(
                    tmp_n,
                    x,
                    nullptr,
                    buf_val, // num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(T)
                    buf_pos, // num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(int)
                    top,
                    out,   // num_tokens * top * sizeof(T)
                    pos // num_tokens * top * sizeof(int)
                );
                first = false;
            } else {
                kernel_bitonic_topk_multiblock<T, top_size, false><<<gridDim, blockDim, 0, stream.stream>>>(
                    tmp_n,
                    buf_val,
                    buf_pos,
                    nw_buf_val,
                    nw_buf_pos,
                    top,
                    out,
                    pos
                );
                std::swap(buf_val, nw_buf_val);
                std::swap(buf_pos, nw_buf_pos);
            }
            tmp_n = CEIL_DIV(tmp_n, 1024) * top_size;
        } while (tmp_n > top_size);

    });
}
}

template <typename T>
struct TopK {
private:
    T *buf_val, *nw_buf_val;
    int *buf_pos, *nw_buf_pos;
public:
    int dim, top;
    T* topk_val;
    int* topk_pos;

    TopK(const int dim, const int top) {
        this->dim = dim; 
        this->top = top; 
    }
    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        TOPK_SIZE_DISPATCH(top, {
            offset = memory->allocate((void**)&buf_val, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(T));
            offset = memory->allocate((void**)&buf_pos, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(int));
            offset = memory->allocate((void**)&nw_buf_val, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(T));
            offset = memory->allocate((void**)&nw_buf_pos, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(int));
        });
        offset = memory->allocate((void**)&topk_val, offset, num_tokens * top * sizeof(T));
        offset = memory->allocate((void**)&topk_pos, offset, num_tokens * top * sizeof(int));
        return offset;
    }
    void prefill(
        const Stream& stream,
        int num_tokens,
        const T* input,
        int dim = -1,
        int top = -1
    ) {
        assert(dim == -1 || dim <= this->dim);
        assert(top == -1 || top <= this->top);
        if (dim == -1) dim = this->dim;
        if (top == -1) top = this->top;
        // Fix：把大于32的情况考虑进kernel，重复利用第一次bitonic_topk的结果；
        bitonic_topk<T>( 
            stream,
            num_tokens,
            dim, top,
            input,
            this->topk_val, this->topk_pos, // out, pos
            this->buf_val, this->buf_pos,
            this->nw_buf_val, this->nw_buf_pos
        );

    }
};
} // namespace functions