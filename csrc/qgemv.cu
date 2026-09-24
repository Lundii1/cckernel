// Dequant-fused GEMV for batch-1 decode (memory-bound: stream packed weights once at full BW).
//
// Mapping: 256 threads = 8 warps, each warp owns R consecutive rows; lane l walks the 64-weight
// blocks l, l+32, ... of those rows. Every block is 32 B (INT4) .. 64 B (INT8) contiguous per row,
// loaded with 128-bit streaming loads (L1::no_allocate). The activation vector is staged once per
// CTA in shared memory in the layout's e-order (bf16, 144 B stride per block -> conflict-free
// 16 B reads because 9 is odd), optionally normalised on the fly (prologue fusion).
// Integer -> float conversion uses the bf16 magic number 0x4300 (128.0) for 4/5/6 bit and the fp32
// 2^23 magic via prmt for 8 bit (Marlin / Kim et al. style: no I2F instructions).
#include "cck.h"
#include "common.cuh"

namespace cck {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kXStride = 72;  // bf16 elements per 64-block in shared memory (64 + 8 pad)

struct GemvKArgs {
  const uint8_t* lo;
  const uint8_t* hi;
  const __half* scales;
  int N, K;
  const void* x;
  const __nv_bfloat16* z;
  int head_dim;
  float eps;
  void* y;
};

// Store a bf16 pair (natural k0, k0+1; k0 even) into e-order shared memory.
CCK_DEVICE void store_pair_e(__nv_bfloat16* xs, int k0, __nv_bfloat162 v) {
  const int blk = k0 >> 6, e = perm64(k0 & 63);  // j=0 element; j=1 lands at e+1
  *reinterpret_cast<__nv_bfloat162*>(xs + blk * kXStride + e) = v;
}

template <int PRO>
CCK_DEVICE void prologue(const GemvKArgs& a, __nv_bfloat16* xs, float* red) {
  const int K = a.K, tid = threadIdx.x;
  if constexpr (PRO == PRO_BF16) {
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(a.x);
    for (int i = tid; i < K / 2; i += kThreads) store_pair_e(xs, 2 * i, x2[i]);
  } else if constexpr (PRO == PRO_RMSNORM) {
    const float2* x2 = reinterpret_cast<const float2*>(a.x);
    float ss = 0.f;
    for (int i = tid; i < K / 2; i += kThreads) {
      const float2 v = x2[i];
      ss += v.x * v.x + v.y * v.y;
    }
    ss = block_sum<kThreads>(ss, red);
    const float inv = rsqrtf(ss / K + a.eps);
    for (int i = tid; i < K / 2; i += kThreads) {
      const float2 v = x2[i];
      store_pair_e(xs, 2 * i, __floats2bfloat162_rn(v.x * inv, v.y * inv));
    }
  } else {  // PRO_GATED: per-head RMS of the GDN output, times silu(z)
    const __nv_bfloat16* o = reinterpret_cast<const __nv_bfloat16*>(a.x);
    const int hd = a.head_dim, per = hd / 32, warp = tid >> 5, lane = tid & 31;
    for (int h = warp; h < K / hd; h += kWarps) {
      float vals[8];
      float ss = 0.f;
      for (int i = 0; i < per; ++i) {
        vals[i] = __bfloat162float(o[h * hd + lane * per + i]);
        ss += vals[i] * vals[i];
      }
      ss = warp_sum(ss);
      const float inv = rsqrtf(ss / hd + a.eps);
      for (int i = 0; i < per; ++i) {
        const int k = h * hd + lane * per + i;
        const float zz = __bfloat162float(a.z[k]);
        const float v = bf16r(bf16r(vals[i] * inv) * silu(zz));
        xs[(k >> 6) * kXStride + perm64(k & 63)] = __float2bfloat16(v);
      }
    }
  }
}

template <int BITS, int PRO, int EPI, int R>
__global__ void __launch_bounds__(kThreads) qgemv_kernel(GemvKArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __shared__ float red[kWarps];
  prologue<PRO>(a, xs, red);
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * kWarps + warp) * R;
  if (row0 >= a.N) return;
  const int K = a.K, nblk = K >> 6, ngrp = K >> 7;
  constexpr int kLoBytes = BITS == 8 ? 64 : 32;
  constexpr int kHiBytes = BITS == 6 ? 16 : (BITS == 5 ? 8 : 0);
  const size_t lo_stride = BITS == 8 ? (size_t)K : (size_t)K / 2;
  const size_t hi_stride = BITS == 6 ? (size_t)K / 4 : (size_t)K / 8;

  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;

  for (int blk = lane; blk < nblk; blk += 32) {
    uint32_t lo[R][kLoBytes / 4];
    uint32_t hi[R][4];
    float sc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = min(row0 + r, a.N - 1);
      const uint8_t* lp = a.lo + row * lo_stride + (size_t)blk * kLoBytes;
#pragma unroll
      for (int v = 0; v < kLoBytes / 16; ++v) {
        const uint4 t = ld_stream_v4(lp + 16 * v);
        lo[r][4 * v] = t.x; lo[r][4 * v + 1] = t.y; lo[r][4 * v + 2] = t.z; lo[r][4 * v + 3] = t.w;
      }
      if constexpr (kHiBytes == 16) {
        const uint4 t = ld_stream_v4(a.hi + row * hi_stride + (size_t)blk * 16);
        hi[r][0] = t.x; hi[r][1] = t.y; hi[r][2] = t.z; hi[r][3] = t.w;
      } else if constexpr (kHiBytes == 8) {
        const uint2 t = ld_stream_v2(a.hi + row * hi_stride + (size_t)blk * 8);
        hi[r][0] = t.x; hi[r][1] = t.y;
      }
      sc[r] = __half2float(a.scales[(size_t)row * ngrp + (blk >> 1)]);
    }

    float p[R];
#pragma unroll
    for (int r = 0; r < R; ++r) p[r] = 0.f;
    const __nv_bfloat16* xb = xs + blk * kXStride;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      const uint4 xv = *reinterpret_cast<const uint4*>(xb + 8 * w);
      const float xf[8] = {bf16lo(xv.x), bf16hi(xv.x), bf16lo(xv.y), bf16hi(xv.y),
                           bf16lo(xv.z), bf16hi(xv.z), bf16lo(xv.w), bf16hi(xv.w)};
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float f[8];
        if constexpr (BITS == 8) {
#pragma unroll
          for (int i = 0; i < 8; ++i) f[i] = byte_to_f(lo[r][2 * w + (i >> 2)], i & 3);
        } else {
          decode8<BITS>(lo[r], hi[r], w, f);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) p[r] = fmaf(f[i], xf[i], p[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = fmaf(sc[r], p[r], acc[r]);
  }

#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = warp_sum(acc[r]);
  if (lane != 0) return;
  if constexpr (EPI == EPI_SWIGLU) {
#pragma unroll
    for (int r = 0; r + 1 < R; r += 2) {
      if (row0 + r + 1 < a.N) {
        const float g = bf16r(acc[r]), u = bf16r(acc[r + 1]);
        reinterpret_cast<__nv_bfloat16*>(a.y)[(row0 + r) >> 1] = __float2bfloat16(bf16r(silu(g)) * u);
      }
    }
  } else {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = row0 + r;
      if (row >= a.N) break;
      if constexpr (EPI == EPI_BF16) reinterpret_cast<__nv_bfloat16*>(a.y)[row] = __float2bfloat16(acc[r]);
      if constexpr (EPI == EPI_F32) reinterpret_cast<float*>(a.y)[row] = acc[r];
      if constexpr (EPI == EPI_ADD_F32) reinterpret_cast<float*>(a.y)[row] += acc[r];
    }
  }
}

// ---------------------------------------------------------------------------------------------
// dequantization to bf16 (natural k order): one thread per (row, 64-block)
// ---------------------------------------------------------------------------------------------
template <int BITS>
__global__ void dequant_kernel(const uint8_t* lo_p, const uint8_t* hi_p, const __half* scales, int N, int K,
                               __nv_bfloat16* out) {
  const int nblk = K >> 6;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)N * nblk) return;
  const int row = idx / nblk, blk = idx % nblk;
  constexpr int kLoBytes = BITS == 8 ? 64 : 32;
  uint32_t lo[kLoBytes / 4], hi[4] = {0, 0, 0, 0};
  const uint32_t* lp = reinterpret_cast<const uint32_t*>(lo_p + (size_t)row * (BITS == 8 ? K : K / 2) + (size_t)blk * kLoBytes);
  for (int i = 0; i < kLoBytes / 4; ++i) lo[i] = lp[i];
  if constexpr (BITS == 6) {
    const uint32_t* hp = reinterpret_cast<const uint32_t*>(hi_p + (size_t)row * (K / 4) + (size_t)blk * 16);
    for (int i = 0; i < 4; ++i) hi[i] = hp[i];
  } else if constexpr (BITS == 5) {
    const uint32_t* hp = reinterpret_cast<const uint32_t*>(hi_p + (size_t)row * (K / 8) + (size_t)blk * 8);
    for (int i = 0; i < 2; ++i) hi[i] = hp[i];
  }
  const float s = __half2float(scales[(size_t)row * (K >> 7) + (blk >> 1)]);
  __nv_bfloat16* o = out + (size_t)row * K + (size_t)blk * 64;
  for (int w = 0; w < 8; ++w) {
    float f[8];
    if constexpr (BITS == 8) {
      for (int i = 0; i < 8; ++i) f[i] = byte_to_f(lo[2 * w + (i >> 2)], i & 3);
    } else {
      decode8<BITS>(lo, hi, w, f);
    }
    for (int i = 0; i < 8; ++i) o[perm64(8 * w + i)] = __float2bfloat16(s * f[i]);
  }
}

// ---------------------------------------------------------------------------------------------
// host dispatch
// ---------------------------------------------------------------------------------------------
namespace {
constexpr int kR = 2;

template <int BITS, int PRO, int EPI>
void launch_gemv(const GemvKArgs& a, cudaStream_t st) {
  const int rows_per_cta = kWarps * kR;
  const int grid = (a.N + rows_per_cta - 1) / rows_per_cta;
  const size_t smem = (size_t)(a.K / 64) * kXStride * sizeof(__nv_bfloat16);
  auto kern = qgemv_kernel<BITS, PRO, EPI, kR>;
  if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<grid, kThreads, smem, st>>>(a);
}

template <int BITS, int PRO>
void dispatch_epi(int epi, const GemvKArgs& a, cudaStream_t st) {
  switch (epi) {
    case EPI_BF16: return launch_gemv<BITS, PRO, EPI_BF16>(a, st);
    case EPI_F32: return launch_gemv<BITS, PRO, EPI_F32>(a, st);
    case EPI_ADD_F32: return launch_gemv<BITS, PRO, EPI_ADD_F32>(a, st);
    default: return launch_gemv<BITS, PRO, EPI_SWIGLU>(a, st);
  }
}

template <int BITS>
void dispatch_pro(int pro, int epi, const GemvKArgs& a, cudaStream_t st) {
  switch (pro) {
    case PRO_BF16: return dispatch_epi<BITS, PRO_BF16>(epi, a, st);
    case PRO_RMSNORM: return dispatch_epi<BITS, PRO_RMSNORM>(epi, a, st);
    default: return dispatch_epi<BITS, PRO_GATED>(epi, a, st);
  }
}
}  // namespace

void qgemv(const QWeight& w, int pro, const void* x, const void* z, int head_dim, float eps, int epi, void* y,
           cudaStream_t stream) {
  GemvKArgs a{w.lo, w.hi, reinterpret_cast<const __half*>(w.scales), w.N, w.K, x,
              reinterpret_cast<const __nv_bfloat16*>(z), head_dim, eps, y};
  switch (w.bits) {
    case 4: return dispatch_pro<4>(pro, epi, a, stream);
    case 5: return dispatch_pro<5>(pro, epi, a, stream);
    case 6: return dispatch_pro<6>(pro, epi, a, stream);
    default: return dispatch_pro<8>(pro, epi, a, stream);
  }
}

void dequant_bf16(const QWeight& w, void* out, cudaStream_t stream) {
  const size_t n = (size_t)w.N * (w.K / 64);
  const int threads = 256;
  const int grid = (int)((n + threads - 1) / threads);
  const auto* s = reinterpret_cast<const __half*>(w.scales);
  auto* o = reinterpret_cast<__nv_bfloat16*>(out);
  switch (w.bits) {
    case 4: dequant_kernel<4><<<grid, threads, 0, stream>>>(w.lo, w.hi, s, w.N, w.K, o); break;
    case 5: dequant_kernel<5><<<grid, threads, 0, stream>>>(w.lo, w.hi, s, w.N, w.K, o); break;
    case 6: dequant_kernel<6><<<grid, threads, 0, stream>>>(w.lo, w.hi, s, w.N, w.K, o); break;
    default: dequant_kernel<8><<<grid, threads, 0, stream>>>(w.lo, w.hi, s, w.N, w.K, o); break;
  }
}

}  // namespace cck
