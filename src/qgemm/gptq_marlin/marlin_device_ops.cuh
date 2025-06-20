#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"


#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace marlin {

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
template <typename scalar_t>
__device__ inline void mma(const typename ScalarType<scalar_t>::FragA& a_frag,
                           const typename ScalarType<scalar_t>::FragB& frag_b,
                           typename ScalarType<scalar_t>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  if constexpr (std::is_same<scalar_t, half>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else {
    STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
  }
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
template <typename scalar_t>
__device__ inline void ldsm4(typename ScalarType<scalar_t>::FragA& frag_a,
                             const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Constructs destination register by taking bytes from 2 sources (based on
// mask)
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

template <typename scalar_t, vllm::ScalarTypeId w_type_id>
__device__ inline typename ScalarType<scalar_t>::FragB dequant(int q);

//
// Efficiently dequantize 4bit values packed in an int32 value into a full
// B-fragment of 4 fp16 values. We mostly follow the strategy in the link below,
// with some small changes:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L215-L287
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L327-L385
//
template <>
__device__ inline typename ScalarType<half>::FragB
dequant<half, vllm::kU4B8.id()>(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant<nv_bfloat16, vllm::kU4B8.id()>(int q) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  typename ScalarType<nv_bfloat16>::FragB frag_b;
  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC308C308;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<half>::FragB
dequant<half, vllm::kU4.id()>(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;
  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant<nv_bfloat16, vllm::kU4.id()>(int q) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  typename ScalarType<nv_bfloat16>::FragB frag_b;
  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC300C300;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  return frag_b;
}

//
// Fast Int8ToFp16/Int8ToBf16: Efficiently dequantize 8bit int values to fp16 or
// bf16 Reference:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L53-L85
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L125-L175
//
template <>
__device__ inline typename ScalarType<half>::FragB
dequant<half, vllm::kU8B128.id()>(int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;

  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant<nv_bfloat16, vllm::kU8B128.id()>(int q) {
  typename ScalarType<nv_bfloat16>::FragB frag_b;

  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388736.f;
  fp32_intermediates[1] -= 8388736.f;
  fp32_intermediates[2] -= 8388736.f;
  fp32_intermediates[3] -= 8388736.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);

  return frag_b;
}

template <>
__device__ inline typename ScalarType<half>::FragB
dequant<half, vllm::kU8.id()>(int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;

  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant<nv_bfloat16, vllm::kU8.id()>(int q) {
  typename ScalarType<nv_bfloat16>::FragB frag_b;

  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388608.f;
  fp32_intermediates[1] -= 8388608.f;
  fp32_intermediates[2] -= 8388608.f;
  fp32_intermediates[3] -= 8388608.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);

  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
template <typename scalar_t>
__device__ inline void scale(typename ScalarType<scalar_t>::FragB& frag_b,
                             typename ScalarType<scalar_t>::FragS& frag_s,
                             int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s =
      ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <typename scalar_t>
__device__ inline void sub_zp(typename ScalarType<scalar_t>::FragB& frag_b,
                              typename ScalarType<scalar_t>::scalar_t2& frag_zp,
                              int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 zp =
      ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

// Same as above, but for act_order (each K is multiplied individually)
template <typename scalar_t>
__device__ inline void scale4(typename ScalarType<scalar_t>::FragB& frag_b,
                              typename ScalarType<scalar_t>::FragS& frag_s_1,
                              typename ScalarType<scalar_t>::FragS& frag_s_2,
                              typename ScalarType<scalar_t>::FragS& frag_s_3,
                              typename ScalarType<scalar_t>::FragS& frag_s_4,
                              int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s_val_1_2;
  s_val_1_2.x = reinterpret_cast<scalar_t*>(&frag_s_1)[i];
  s_val_1_2.y = reinterpret_cast<scalar_t*>(&frag_s_2)[i];

  scalar_t2 s_val_3_4;
  s_val_3_4.x = reinterpret_cast<scalar_t*>(&frag_s_3)[i];
  s_val_3_4.y = reinterpret_cast<scalar_t*>(&frag_s_4)[i];

  frag_b[0] = __hmul2(frag_b[0], s_val_1_2);
  frag_b[1] = __hmul2(frag_b[1], s_val_3_4);
}

// Given 2 floats multiply by 2 scales (halves)
template <typename scalar_t>
__device__ inline void scale_float(float* c,
                                   typename ScalarType<scalar_t>::FragS& s) {
  scalar_t* s_ptr = reinterpret_cast<scalar_t*>(&s);
  c[0] = __fmul_rn(c[0], ScalarType<scalar_t>::num2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], ScalarType<scalar_t>::num2float(s_ptr[1]));
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}

}  // namespace marlin 