// Shared device helpers for cckernel (sm_89 / Ada). Plain CUDA, no torch headers.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CCK_DEVICE __device__ __forceinline__

namespace cck {

// ---------------------------------------------------------------------------------------------
// numeric helpers
// ---------------------------------------------------------------------------------------------
CCK_DEVICE float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }  // round through bf16
CCK_DEVICE float silu(float x) { return x / (1.0f + __expf(-x)); }
CCK_DEVICE float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
CCK_DEVICE float softplusf_(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }  // torch threshold=20

CCK_DEVICE float bf16lo(uint32_t w) { return __uint_as_float(w << 16); }
CCK_DEVICE float bf16hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

template <int kWidth = 32>
CCK_DEVICE float warp_sum(float v) {
#pragma unroll
  for (int o = kWidth / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

CCK_DEVICE float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Block-wide sum for blockDim.x == NT (multiple of 32). `red` must hold NT/32 floats.
template <int NT>
CCK_DEVICE float block_sum(float v, float* red) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) red[w] = v;
  __syncthreads();
  float t = (l < NT / 32) ? red[l] : 0.0f;
  t = warp_sum(t);
  __syncthreads();
  return t;
}

// ---------------------------------------------------------------------------------------------
// memory helpers
// ---------------------------------------------------------------------------------------------
// Streaming 128-bit load: weights are read exactly once per token, keep them out of L1.
CCK_DEVICE uint4 ld_stream_v4(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
CCK_DEVICE uint2 ld_stream_v2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

// ---------------------------------------------------------------------------------------------
// packed-weight decode (see cckernel/quant.py for the layout and the permutation)
// ---------------------------------------------------------------------------------------------
// e-order <-> k_local permutation inside a 64-block: swap bit fields [1:2] and [4:5] (involution).
CCK_DEVICE __host__ int perm64(int x) { return (x & 0x9) | ((x >> 3) & 0x6) | ((x & 0x6) << 3); }

// lop3 helper: (a & b) | c
CCK_DEVICE uint32_t and_or(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Low-plane step t of word x -> bf16x2 (128 + u_{2t}, 128 + u_{2t+1}) with the high bits merged.
template <int BITS>
CCK_DEVICE uint32_t decode_pair_bf16magic(uint32_t lo, uint32_t hiw, int t, int h, int c) {
  uint32_t v = and_or(lo >> (4 * t), 0x000F000Fu, 0x43004300u);
  if constexpr (BITS == 6) {
    const int sh = 2 * (4 * h + t) - 4;
    const uint32_t hb = sh >= 0 ? (hiw >> sh) : (hiw << (-sh));
    v = and_or(hb, 0x00300030u, v);
  } else if constexpr (BITS == 5) {
    const int sh = 8 * (c & 1) + 4 * h + t - 4;
    const uint32_t hb = sh >= 0 ? (hiw >> sh) : (hiw << (-sh));
    v = and_or(hb, 0x00100010u, v);
  }
  return v;
}

// Decode the 8 weights of low word w (e-order 8w..8w+7) to fp32 (u - z).
template <int BITS>
CCK_DEVICE void decode8(const uint32_t* lo8, const uint32_t* hi4, int w, float* f) {
  constexpr float kOff = 128.0f + (float)(1 << (BITS - 1));
  const int c = w >> 1, h = w & 1;
  uint32_t hiw = 0;
  if constexpr (BITS == 6) hiw = hi4[c];
  if constexpr (BITS == 5) hiw = hi4[c >> 1];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const uint32_t v = decode_pair_bf16magic<BITS>(lo8[w], hiw, t, h, c);
    f[2 * t] = bf16lo(v) - kOff;
    f[2 * t + 1] = bf16hi(v) - kOff;
  }
}

// INT8: bytes -> fp32 via the 2^23 magic number (prmt + fsub, no I2F).
CCK_DEVICE float byte_to_f(uint32_t word, int b) {
  return __uint_as_float(__byte_perm(word, 0x4B000000u, 0x7650u | b)) - (8388608.0f + 128.0f);
}

}  // namespace cck
