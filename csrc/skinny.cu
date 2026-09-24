// Tensor-core "skinny" GEMM for 2..8 tokens (speculative verification): Y[M, N] = X[M, K] W^T.
//
// FlashDecoding++ (arXiv 2311.01282): pad M to 8, not 64. QuIP#-style operand swap: the weights are
// the mma A operand (16 rows x 16 k) and the activations the B operand (16 k x 8 tokens), so one
// mma.m16n8k16 covers all tokens and the kernel streams exactly the same weight bytes as the GEMV:
// verifying 8 tokens costs about one decode step.
//
// The packed layout (cckernel/quant.py) was designed so that lane (g = lane/4, c = lane%4) needs,
// per 64-block and row, exactly low words (2c, 2c+1) (8 contiguous bytes; a quad reads 32 B) and
// high word c (6 bit) / c>>1 (5 bit); extraction step t of those words gives the A-register pairs of
// k-step t. Per-(row, group-of-128) scales are applied to a per-group accumulator in fp32.
// CTA = 16 rows x (8 warps splitting K); partial tiles are reduced through shared memory.
#include "cck.h"
#include "common.cuh"

namespace cck {

constexpr int kSkWarps = 8;
constexpr int kSkThreads = kSkWarps * 32;

struct SkArgs {
  const uint8_t* lo;
  const uint8_t* hi;
  const __half* scales;
  int N, K;
  const __nv_bfloat16* x;  // [M, K]
  int x_stride;
  int M;
  void* y;  // [M, N] with row stride N
  int epi;
};

CCK_DEVICE void mma_bf16(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// bf16x2 (128+u0, 128+u1) -> (u0 - z, u1 - z) exactly (integers are exact in bf16 up to 256).
template <int BITS>
CCK_DEVICE uint32_t sub_zero(uint32_t v) {
  constexpr float off = 128.0f + (float)(1 << (BITS - 1));
  __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&v);
  const __nv_bfloat162 o = __floats2bfloat162_rn(off, off);
  __nv_bfloat162 r = __hsub2(x, o);
  return *reinterpret_cast<uint32_t*>(&r);
}

CCK_DEVICE uint32_t bytes_to_bf16x2(uint32_t word, int b0) {
  const float f0 = byte_to_f(word, b0), f1 = byte_to_f(word, b0 + 1);
  __nv_bfloat162 r = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<uint32_t*>(&r);
}

template <int BITS>
__global__ void __launch_bounds__(kSkThreads) skinny_kernel(SkArgs a) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const int row0 = blockIdx.x * 16;
  const int K = a.K, ngrp = K >> 7;
  const int ra = min(row0 + g, a.N - 1), rb = min(row0 + g + 8, a.N - 1);
  const size_t lo_stride = BITS == 8 ? (size_t)K : (size_t)K / 2;
  const size_t hi_stride = BITS == 6 ? (size_t)K / 4 : (size_t)K / 8;
  const bool tok_ok = g < a.M;
  const __nv_bfloat16* xrow = a.x + (size_t)(tok_ok ? g : 0) * a.x_stride;

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  // each warp handles a contiguous range of 128-groups
  const int gpw = (ngrp + kSkWarps - 1) / kSkWarps;
  const int g0 = warp * gpw, g1 = min(ngrp, g0 + gpw);
  for (int grp = g0; grp < g1; ++grp) {
    float part[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int bb = 0; bb < 2; ++bb) {
      const int blk = 2 * grp + bb;
      uint32_t la[4], lb[4];  // A-operand words for rows ra / rb
      if constexpr (BITS == 8) {
        const uint4 ua = ld_stream_v4(a.lo + ra * lo_stride + (size_t)blk * 64 + 16 * c);
        const uint4 ub = ld_stream_v4(a.lo + rb * lo_stride + (size_t)blk * 64 + 16 * c);
        la[0] = ua.x; la[1] = ua.y; la[2] = ua.z; la[3] = ua.w;
        lb[0] = ub.x; lb[1] = ub.y; lb[2] = ub.z; lb[3] = ub.w;
      } else {
        const uint2 ua = ld_stream_v2(a.lo + ra * lo_stride + (size_t)blk * 32 + 8 * c);
        const uint2 ub = ld_stream_v2(a.lo + rb * lo_stride + (size_t)blk * 32 + 8 * c);
        la[0] = ua.x; la[1] = ua.y;
        lb[0] = ub.x; lb[1] = ub.y;
        if constexpr (BITS == 6) {
          la[2] = *reinterpret_cast<const uint32_t*>(a.hi + ra * hi_stride + (size_t)blk * 16 + 4 * c);
          lb[2] = *reinterpret_cast<const uint32_t*>(a.hi + rb * hi_stride + (size_t)blk * 16 + 4 * c);
        } else if constexpr (BITS == 5) {
          la[2] = *reinterpret_cast<const uint32_t*>(a.hi + ra * hi_stride + (size_t)blk * 8 + 4 * (c >> 1));
          lb[2] = *reinterpret_cast<const uint32_t*>(a.hi + rb * hi_stride + (size_t)blk * 8 + 4 * (c >> 1));
        }
      }
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        uint32_t A[4];
        if constexpr (BITS == 8) {
          // thread c's 16 bytes: byte 8h + 2t + j  -> word 2h + t/2, byte 2(t%2) + j
          A[0] = bytes_to_bf16x2(la[t >> 1], 2 * (t & 1));
          A[1] = bytes_to_bf16x2(lb[t >> 1], 2 * (t & 1));
          A[2] = bytes_to_bf16x2(la[2 + (t >> 1)], 2 * (t & 1));
          A[3] = bytes_to_bf16x2(lb[2 + (t >> 1)], 2 * (t & 1));
        } else {
          A[0] = sub_zero<BITS>(decode_pair_bf16magic<BITS>(la[0], la[2], t, 0, c));
          A[1] = sub_zero<BITS>(decode_pair_bf16magic<BITS>(lb[0], lb[2], t, 0, c));
          A[2] = sub_zero<BITS>(decode_pair_bf16magic<BITS>(la[1], la[2], t, 1, c));
          A[3] = sub_zero<BITS>(decode_pair_bf16magic<BITS>(lb[1], lb[2], t, 1, c));
        }
        // B fragment: tokens n = g, k = blk*64 + 16t + {2c, 2c+1} and {2c+8, 2c+9}
        uint32_t B[2] = {0u, 0u};
        if (tok_ok) {
          const __nv_bfloat16* xp = xrow + (size_t)blk * 64 + 16 * t + 2 * c;
          B[0] = __ldg(reinterpret_cast<const unsigned int*>(xp));
          B[1] = __ldg(reinterpret_cast<const unsigned int*>(xp + 8));
        }
        mma_bf16(part, A, B);
      }
    }
    const float sa = __half2float(a.scales[(size_t)ra * ngrp + grp]);
    const float sb = __half2float(a.scales[(size_t)rb * ngrp + grp]);
    acc[0] = fmaf(sa, part[0], acc[0]);
    acc[1] = fmaf(sa, part[1], acc[1]);
    acc[2] = fmaf(sb, part[2], acc[2]);
    acc[3] = fmaf(sb, part[3], acc[3]);
  }

  // reduce the split-K partial tiles: C fragment (rows g / g+8, tokens 2c / 2c+1)
  __shared__ float red[kSkWarps][32][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) red[warp][lane][i] = acc[i];
  __syncthreads();
  if (warp != 0) return;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  for (int w = 0; w < kSkWarps; ++w)
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] += red[w][lane][i];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = row0 + g + (i >= 2 ? 8 : 0);
    const int tok = 2 * c + (i & 1);
    if (row >= a.N || tok >= a.M) continue;
    const size_t o = (size_t)tok * a.N + row;
    if (a.epi == EPI_BF16) reinterpret_cast<__nv_bfloat16*>(a.y)[o] = __float2bfloat16(v[i]);
    else if (a.epi == EPI_F32) reinterpret_cast<float*>(a.y)[o] = v[i];
    else reinterpret_cast<float*>(a.y)[o] += v[i];
  }
}

void qgemm_skinny(const QWeight& w, const void* x, int x_stride, int M, int epi, void* y, cudaStream_t stream) {
  SkArgs a{w.lo, w.hi, reinterpret_cast<const __half*>(w.scales), w.N, w.K,
           reinterpret_cast<const __nv_bfloat16*>(x), x_stride, M, y, epi};
  const int grid = (w.N + 15) / 16;
  switch (w.bits) {
    case 4: skinny_kernel<4><<<grid, kSkThreads, 0, stream>>>(a); break;
    case 5: skinny_kernel<5><<<grid, kSkThreads, 0, stream>>>(a); break;
    case 6: skinny_kernel<6><<<grid, kSkThreads, 0, stream>>>(a); break;
    default: skinny_kernel<8><<<grid, kSkThreads, 0, stream>>>(a); break;
  }
}

}  // namespace cck
