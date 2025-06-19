#pragma once
#include "../utils.cuh"
#include "../trait.cuh"
#include <iostream>
/* 面向4090GPU的开发版本 

*/
namespace functions {
namespace {
template<typename T, int N>
static __device__ inline void warpBitonicSort(T& v1, int& pos, bool asc) {  // asc:false->最终结果是每个warp中的元素是降序的
    /* N不能等于64，否则j步长为32，不能跨warp交换数据 */
    int lane_id = threadIdx.x & (N - 1); // equal to threadIdx.x % N 
    // 外层循环控制k定义要合并的长度,内层循环j决定步长
    #pragma unroll
    for (int k = 2; k <= N; k *= 2) {
        bool desc = ((lane_id & k) == 0) ^ asc; //以k为范围，交替升/降序: desc
        #pragma unroll
        for (int j = k / 2; j > 0; j /= 2) {
            T v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
            int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos, j);
            bool upper = (lane_id & j) != 0; // 每个k范围内的后半段，upper=true
            if (desc ^ (v1 > v2 || (v1 == v2 && pos < pos2)) ^ upper) {
                v1 = v2;
                pos = pos2;
            }
        }
    }
}
template<typename T, int N>
static __device__ inline void warpBitonicMerge(T& v1, int& pos1, T& v2, int& pos2) {//比较两个warp之间的伙伴线程，然后进行归并。结果是每两个伙伴warp选出top warp_size个元素，放到第一个warp中
    if (v1 < v2 || (v1 == v2 && pos1 > pos2)) {
        v1 = v2;
        pos1 = pos2;
    }
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
    __shared__ int shared_pos[1024]; //此处没有bank conflict
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
                int idx_next = (i << 1) - threadIdx.x - 1; // 伙伴线程id,eg. 1023 = 1024 - 0(tid) - 1
                T nw_v = (idx_next < blockDim.x) ? shared_val[idx_next] : T(-TypeTraits<T>::inf());
                int nw_pos = (idx_next < blockDim.x) ? shared_pos[idx_next] : -1;
                warpBitonicMerge<T, N>(v, pos, nw_v, nw_pos); // merge and rebuild in desc order
                shared_val[threadIdx.x] = v;
                shared_pos[threadIdx.x] = pos;
            }
        }
    }
    // 至此, 前TOP_SIZE个线程包含了block中最大的TOP_SIZE个元素，但是无序；如果top != 2的幂，则需要排序才能选出确切的topk；
    // TODO 因此对前TOP_SIZE个线程进行排序是有必要的，需要修改下面排序的逻辑
    // intra warp reduce
    if (reduce_end == 32 && threadIdx.x < 32) {
        warpBitonicSort<T, 32>(v, pos, false);
    }
    else if (reduce_end != 32 && threadIdx.x < TOP_SIZE)
    {
        // 使用基于共享内存的奇偶排序网络 (Bitonic Sort) 对前 TOP_SIZE 个元素排序
        __syncthreads();
        // 外层循环k控制已经排好序的子序列的长度 (2, 4, 8, 16, 32, 64...)
        for (int k = 2; k <= TOP_SIZE; k <<= 1)
        {
            // 内层循环j控制比较和交换的距离
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                // 确定要比较和交换的伙伴线程索引
                int ixj = threadIdx.x ^ j;
                // 只有当伙伴线程也在排序范围内时才进行操作
                if (ixj > threadIdx.x)
                { // 避免每个pair被处理两次，让tid较小的线程做主导
                    // 确定排序方向（升序还是降序）
                    bool descending = ((threadIdx.x & k) == 0);
                    // 从共享内存读取伙伴线程的数据
                    T partner_v = shared_val[ixj];
                    int partner_pos = shared_pos[ixj];
                    T my_v = shared_val[threadIdx.x];
                    int my_pos = shared_pos[threadIdx.x];
                    // 如果是降序要求，但我的值比伙伴小，则交换
                    if (descending && my_v < partner_v)
                    {
                        shared_val[threadIdx.x] = partner_v;
                        shared_pos[threadIdx.x] = partner_pos;
                        shared_val[ixj] = my_v;
                        shared_pos[ixj] = my_pos;
                    }
                    // 如果是升序要求，但我的值比伙伴大，则交换
                    else if (!descending && my_v > partner_v)
                    {
                        shared_val[threadIdx.x] = partner_v;
                        shared_pos[threadIdx.x] = partner_pos;
                        shared_val[ixj] = my_v;
                        shared_pos[ixj] = my_pos;
                    }
                }
                // 在每次比较-交换的子阶段完成后，必须进行同步！
                // 这确保了所有线程都完成了内存写入，才能进入下一个比较距离（j）的阶段
                // 这是跨Warp通信正确性的关键
                __syncthreads();
            }
        }
        // 排序完成后，每个线程从共享内存中读回最终排好序的数据到自己的寄存器
        v = shared_val[threadIdx.x];
        pos = shared_pos[threadIdx.x];
    }
}


template<typename T, int N>
static __global__ void kernel_bitonic_topk(
    int n, int top,
    T *inp,     // (batch, n)
    float *out,     // (batch, top)
    int *idx    // (batch, top)
) {
    int offset_inp = blockIdx.x * n;
    int offset_out = blockIdx.x * top;
    T local_v = threadIdx.x < n ? inp[offset_inp + threadIdx.x] : -TypeTraits<T>::inf();
    int local_pos = threadIdx.x;
    warpBitonicSort<T, N>(local_v, local_pos, false); // local sort in desc order
    for (int i = blockDim.x; i < n; i += blockDim.x) {
        T nw_v = (i + threadIdx.x) < n ? inp[offset_inp + i + threadIdx.x] : -TypeTraits<T>::inf();
        int nw_pos = i + threadIdx.x;
        // step.1: local sort
        warpBitonicSort<T, N>(nw_v, nw_pos, true); // local sort in asc order
        // step.2&3: merge and rebuild
        warpBitonicMerge<T, N>(local_v, local_pos, nw_v, nw_pos); // merge and rebuild in desc order
    }
    blockBitonicReduce<T, N>(local_v, local_pos);
    if (threadIdx.x < top) {
        out[offset_out + threadIdx.x] = local_v;
        idx[offset_out + threadIdx.x] = local_pos;
    }
}
// intra-block topk
// gridDim(batch, n / 1024, 1), threadDim(1024, 1, 1)
// N = top_size(16/32/64)
template<typename T, int TOP_SIZE, bool FIRST> 
static __global__ void kernel_bitonic_topk_multiblock(
    int n, // dim
    const T *inp,       // (batch, n)
    const int *idx_inp, // (batch, n)
    T *out,     // (batch, n / 1024 * N)
    int *idx    // (batch, n / 1024 * N)
) {
    int offset_col = blockIdx.y * blockDim.x + threadIdx.x;
    int offset_inp = blockIdx.x * n + offset_col;
    int offset_out = blockIdx.x * (gridDim.y * TOP_SIZE) + blockIdx.y * TOP_SIZE + threadIdx.x;
    T local_v = (offset_col < n) ? inp[offset_inp] : T(-TypeTraits<T>::inf()); // 搬运到寄存器上，超出范围的值为负无穷
    // first call this kernel, idx_inp is nullptr, odrdered is true
    int local_pos = (idx_inp == nullptr) ? offset_col : ((offset_col < n) ? idx_inp[offset_inp] : -1); // 用来记录数据在最初的输入x中的位置
    
    if (FIRST) warpBitonicSort<T, TOP_SIZE>(local_v, local_pos, false); // local sort in desc order
    blockBitonicReduce<T, N>(local_v, local_pos);
    // satge1 : TODO 第二次调用Reduce之后，top_val已经在 tid<=top_size的warp中，但是如果有多个warp暂时是分别降序(warp之间需要再次排序)

    // stage2: TODO 排序之后就可以按照top的数量写入out pos 了，需要按顺序
    if (threadIdx.x < N) {
        out[offset_out] = local_v;
        idx[offset_out] = local_pos;
    }
}
// copy kernel
// gridDim(batch, 1, 1),   blockDim(top, 1, 1)
template<typename T>
static __global__ void kernel_bitonic_topk_multiblock_copy (
    int n, int top,
    const T *inp,       // (batch, n)
    const int *idx_inp, // (batch, n)
    T *out,         // (batch, top)
    int *idx            // (batch, top)
) {
    int offset_inp = blockIdx.x * n + threadIdx.x;
    int offset_out = blockIdx.x * top + threadIdx.x;
    if (threadIdx.x < top) {
        out[offset_out] = inp[offset_inp];
        idx[offset_out] = idx_inp[offset_inp];
    }
}
#define ROUND_UP_POW2(x) ((x) <= 1 ? 1 : (1 << (32 - __clz((x) - 1)))) // round up to the next power of 2
#define TOPK_SIZE_DISPATCH(top, ...) \
    do { \
        const int &top_v = top; \
        const int top_size = ROUND_UP_POW2(top_v); \
        __VA_ARGS__ \
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
        unsigned int tmp_n = n;// dim
        std::cout<<"bitonic_topk:179--> tmp_n: " << tmp_n << ", n: " << n << ", top_size: " << top_size << ", batch:" << batch << std::endl;
        // tmp_n <= 32768 --> 1024 --> top_size
        do {
            dim3 gridDim(batch, CEIL_DIV(tmp_n, 1024), 1); // grid_size (batch, 4, 1)
            if (first) {  
                kernel_bitonic_topk_multiblock<T, top_size, first><<<gridDim, blockDim, 0, stream.stream>>>(
                    tmp_n,
                    x,
                    nullptr,
                    buf_val,
                    buf_pos
                );
                first = false;
            } else {
                kernel_bitonic_topk_multiblock<T, top_size, first><<<gridDim, blockDim, 0, stream.stream>>>(
                    tmp_n,
                    buf_val,
                    buf_pos,
                    nw_buf_val,
                    nw_buf_pos
                );
                buf_val = nw_buf_val; // may need to optimize
                buf_pos = nw_buf_pos;
            }
            tmp_n = CEIL_DIV(tmp_n, 1024) * top_size;
        } while (tmp_n > top_size);

        // copy to output tensor
        // TODO: kernel fusion(把copy操作放进其他kernel中避免启动核函数开销)
        {
            dim3 gridDim(batch, 1, 1);
            blockDim = dim3(top_size, 1, 1);
            kernel_bitonic_topk_multiblock_copy<T><<<gridDim, blockDim, 0, stream.stream>>>(
                top_size, top,
                buf_val,
                buf_pos,
                out,
                pos
            );
        }
    });
}

template<typename T>
static __global__ void set_topk_to_neg_inf_kernel(int dim, int top, T* x, const int* topk_pos) {
    x[blockIdx.x * dim + topk_pos[blockIdx.x * top + threadIdx.x]] = -TypeTraits<T>::inf();
}
} // namespace

template<typename T>
void set_topk_to_neg_inf(const Stream& stream, int num_tokens, int dim, int top, int num, T* x, const int* topk_pos) {
    set_topk_to_neg_inf_kernel<<<num_tokens, num, 0, stream.stream>>>(dim, top, x, topk_pos);
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
    T* tmp_x;

    TopK(const int dim, const int top) {
        this->dim = dim; // 4096
        this->top = top; // 32/64
    }
    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        TOPK_SIZE_DISPATCH(top, {
            offset = memory->allocate((void**)&buf_val, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(T));
            offset = memory->allocate((void**)&buf_pos, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(int));
            offset = memory->allocate((void**)&nw_buf_val, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(T));
            offset = memory->allocate((void**)&nw_buf_pos, offset, num_tokens * CEIL_DIV(dim, 1024) * top_size * sizeof(int));
        });
        if (top > 32) { // tmp fix
            offset = memory->allocate((void**)&tmp_x, offset, num_tokens * dim * sizeof(T));
        }
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
        // int batch, int n, int top, const T *x, T *out, int *pos, T *buf_val, int *buf_pos, T *nw_buf_val, int *nw_buf_pos
        std::cout<<"Location-->topk.cuh:270--> bitonic_topk: num_tokens: " << num_tokens << ", dim: " << dim << ", top: " << top << std::endl;
        bitonic_topk<T>( 
            stream,
            num_tokens,
            dim, top,
            input,
            this->topk_val, this->topk_pos, // out, pos
            this->buf_val, this->buf_pos,
            this->nw_buf_val, this->nw_buf_pos
        );
        // TODO：把大于32的情况考虑进kernel，重复利用第一次bitonic_topk的结果；
        // 目前这段代码相当于把算好的top_val设负无穷，再循环执行bitonic_topk 性能非常差
        if (top > 32) {
            for (int i = 32; i < top; i += 32) {
                cudaMemcpyAsync(this->tmp_x, input, num_tokens * dim * sizeof(T), cudaMemcpyDeviceToDevice, stream.stream);
                set_topk_to_neg_inf(stream, num_tokens, dim, top, 32, this->tmp_x, this->topk_pos + (i - 32));
                bitonic_topk<T>(
                    stream,
                    num_tokens,
                    dim, top,
                    this->tmp_x,
                    this->topk_val + i, this->topk_pos + i,
                    this->buf_val, this->buf_pos,
                    this->nw_buf_val, this->nw_buf_pos
                );
            }
        }
    }
};
} // namespace functions