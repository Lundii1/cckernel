// Gated full attention for decode / verify (head_dim 256, GQA group 4) over a quantized KV cache.
//
// attn_prep: per (token, head) CTA of 256 threads: zero-centred RMSNorm (weights pre-shifted to
//   1 + w), partial rotate-half RoPE on the first `rot` dims (theta^-2i/rot frequencies computed on
//   the host in fp32 exactly like HF), bf16 rounding points mirroring the reference. Optionally a
//   randomized Hadamard rotation of the head (shared-memory FWHT, same butterfly order as
//   cckernel/hadamard.py), then the K / V append in the cache format (cckernel/kvq.py):
//     BF16  as is;  FP8  e4m3 (RNE, saturated);  FP4  E2M1 nibbles + one E4M3 scale per 16 channels,
//     the scale picked among the E4M3 code of absmax/6 and its neighbours by squared error (fp64).
//   A 16-channel group is a half-warp, so group reductions are shuffles. All rounding steps are
//   IEEE-exact (__fdiv_rn / __frcp_rn / __fmul_rn despite --use_fast_math), so the bytes match kvq.
// attn_decode: flash-decoding. Grid (Hkv, NS, M); a CTA owns one KV head and one split of the keys,
//   and processes the G = H/Hkv query heads of the group together so each K/V row is read once for
//   all of them (GQA packing); rows are dequantized in registers (FP4: 4 bytes + 1 scale byte per
//   lane and row, E2M1 decoded with a byte-permute LUT). Online softmax (Milakov & Gimelshein) per
//   warp, merged across warps in shared memory, then across splits by the last CTA to finish (atomic
//   ticket), which also un-rotates the output (inverse FWHT in shared memory) and applies the
//   head-specific sigmoid output gate (Gated Attention, arXiv 2505.06708).
#include <cuda_fp8.h>

#include "cck.h"
#include "common.cuh"

namespace cck {

constexpr int D = 256;
constexpr int G = 4;  // query heads per KV head
constexpr int GS = 16;  // FP4 scale group

// ---------------------------------------------------------------------------------------------
// number formats
// ---------------------------------------------------------------------------------------------
CCK_DEVICE uint8_t e4m3_enc(float x) { return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3); }
CCK_DEVICE float e4m3_dec(uint8_t c) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)c, __NV_E4M3)));
}
// E2M1 code (bit 3 = sign) with round-to-nearest-even thresholds, saturating at 6.
CCK_DEVICE uint32_t e2m1_enc(float y) {
  const float a = fabsf(y);
  const uint32_t m = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
  return m | (y < 0.0f ? 8u : 0u);
}
// E2M1 code -> float through the fp16 bit pattern (high bytes 00 38 3C 3E 40 42 44 46).
CCK_DEVICE float e2m1_dec(uint32_t c) {
  const uint32_t hb = __byte_perm(0x3E3C3800u, 0x46444240u, c & 7u);
  const unsigned short bits = (unsigned short)((hb << 8) | ((c & 8u) << 12));
  return __half2float(__ushort_as_half(bits));
}

// Orthonormal FWHT of the block's 256 values (thread d holds x) with a sign flip first:
// y = H diag(s) x / 16. `sh` is 256 floats of shared memory. Must be called by all 256 threads.
CCK_DEVICE float fwht256(float x, const float* signs, float* sh) {
  const int d = threadIdx.x;
  __syncthreads();
  sh[d] = x * signs[d];
#pragma unroll
  for (int h = 1; h < D; h <<= 1) {
    __syncthreads();
    const float a = sh[d], b = sh[d ^ h];
    const float r = (d & h) ? (b - a) : (a + b);
    __syncthreads();
    sh[d] = r;
  }
  __syncthreads();
  return sh[d] * 0.0625f;
}

// Store channel d of one 256-vector (value x, thread d) into a cache row in format F.
template <int F>
CCK_DEVICE void kv_store(float x, void* cache, uint8_t* scale, size_t row) {
  const int d = threadIdx.x;
  if constexpr (F == KV_BF16) {
    reinterpret_cast<__nv_bfloat16*>(cache)[row * D + d] = __float2bfloat16(x);
  } else if constexpr (F == KV_FP8) {
    reinterpret_cast<uint8_t*>(cache)[row * D + d] = e4m3_enc(x);
  } else {
    // half-warp = one 16-channel group
    float amax = fabsf(x);
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, 16));
    const int s0 = e4m3_enc(__fdiv_rn(amax, 6.0f));
    int best = 0;
    double best_err = 0.0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int dc = k == 0 ? 0 : (k == 1 ? 1 : (k == 2 ? -1 : -2));
      const int code = min(max(s0 + dc, 0), 0x7E);
      const float s = e4m3_dec((uint8_t)code);
      const float inv = s > 0.0f ? __frcp_rn(s) : 0.0f;
      const float q = __fmul_rn(e2m1_dec(e2m1_enc(__fmul_rn(x, inv))), s);
      const double e = (double)q - (double)x;
      double err = e * e;
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) err += __shfl_xor_sync(0xffffffffu, err, o, 16);
      if (k == 0 || err < best_err) {
        best_err = err;
        best = code;
      }
    }
    const float s = e4m3_dec((uint8_t)best);
    const float inv = s > 0.0f ? __frcp_rn(s) : 0.0f;
    const uint32_t c = e2m1_enc(__fmul_rn(x, inv));
    const uint32_t c_next = __shfl_down_sync(0xffffffffu, c, 1);
    if ((d & 1) == 0) reinterpret_cast<uint8_t*>(cache)[row * (D / 2) + d / 2] = (uint8_t)(c | (c_next << 4));
    if ((d & (GS - 1)) == 0) scale[row * (D / GS) + d / GS] = (uint8_t)best;
  }
}

// ---------------------------------------------------------------------------------------------
// attn_prep
// ---------------------------------------------------------------------------------------------
template <int KF, int VF>
__global__ void __launch_bounds__(D) attn_prep_kernel(AttnPrepArgs a) {
  const int m = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
  const int pos = *a.cur_len + m;
  const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(a.proj) + (size_t)m * a.proj_stride;
  const bool is_q = head < a.H;
  const int kvh = head - a.H;
  __shared__ float red[D / 32];
  __shared__ float xs[D];
  __shared__ float sh[D];
  const float x = __bfloat162float(is_q ? row[head * 2 * D + d] : row[a.H * 2 * D + kvh * D + d]);
  const float ss = block_sum<D>(x * x, red);
  const float xn = bf16r(x * rsqrtf(ss / D + a.eps) * (is_q ? a.q_norm[d] : a.k_norm[d]));
  xs[d] = xn;
  __syncthreads();
  float val = xn;
  if (d < a.rot) {
    const int half = a.rot / 2, i = d % half;
    float sn, cs;
    sincosf((float)pos * a.inv_freq[i], &sn, &cs);
    cs = bf16r(cs);
    sn = bf16r(sn);
    const float partner = d < half ? -xs[d + half] : xs[d - half];
    val = bf16r(bf16r(xn * cs) + bf16r(partner * sn));
  }
  if (a.rotate) val = fwht256(val, a.signs, sh);  // block-uniform branch
  if (is_q) {
    a.q_out[((size_t)m * a.H + head) * D + d] = val;
    return;
  }
  const size_t crow = (size_t)kvh * a.max_len + pos;
  kv_store<KF>(val, a.k_cache, a.k_scale, crow);
  float v = __bfloat162float(row[a.H * 2 * D + a.Hkv * D + kvh * D + d]);
  if (a.rotate) v = fwht256(v, a.signs, sh);
  kv_store<VF>(v, a.v_cache, a.v_scale, crow);
}

template <int KF, int VF>
static void launch_prep(const AttnPrepArgs& x, cudaStream_t stream) {
  dim3 grid(x.M, x.H + x.Hkv);
  attn_prep_kernel<KF, VF><<<grid, D, 0, stream>>>(x);
}

void attn_prep(const AttnPrepArgs& x, cudaStream_t stream) {
  if (x.kfmt == KV_BF16 && x.vfmt == KV_BF16) launch_prep<KV_BF16, KV_BF16>(x, stream);
  else if (x.kfmt == KV_FP8 && x.vfmt == KV_FP8) launch_prep<KV_FP8, KV_FP8>(x, stream);
  else if (x.kfmt == KV_FP8 && x.vfmt == KV_FP4) launch_prep<KV_FP8, KV_FP4>(x, stream);
  else if (x.kfmt == KV_FP4 && x.vfmt == KV_FP4) launch_prep<KV_FP4, KV_FP4>(x, stream);
}

// ---------------------------------------------------------------------------------------------
// attn_decode
// ---------------------------------------------------------------------------------------------
// The 8 channels [8*lane, 8*lane+8) of cache row t, dequantized to fp32.
template <int F>
CCK_DEVICE void kv_load8(const void* cache, const uint8_t* scale, size_t row, int lane, float* f) {
  if constexpr (F == KV_BF16) {
    const uint4 r = *reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(cache) + row * D + lane * 8);
    f[0] = bf16lo(r.x); f[1] = bf16hi(r.x); f[2] = bf16lo(r.y); f[3] = bf16hi(r.y);
    f[4] = bf16lo(r.z); f[5] = bf16hi(r.z); f[6] = bf16lo(r.w); f[7] = bf16hi(r.w);
  } else if constexpr (F == KV_FP8) {
    const uint2 r = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint8_t*>(cache) + row * D + lane * 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[i] = e4m3_dec((uint8_t)(r.x >> (8 * i)));
      f[4 + i] = e4m3_dec((uint8_t)(r.y >> (8 * i)));
    }
  } else {
    const uint32_t w = *reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(cache) + row * (D / 2) + lane * 4);
    const float s = e4m3_dec(scale[row * (D / GS) + (lane >> 1)]);
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = e2m1_dec((w >> (4 * i)) & 0xFu) * s;
  }
}

constexpr int kDecThreads = 128;
constexpr int kDecWarps = kDecThreads / 32;

template <int KF, int VF>
__global__ void __launch_bounds__(kDecThreads) attn_decode_kernel(AttnDecodeArgs a) {
  const int kvh = blockIdx.x, split = blockIdx.y, m = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int L = *a.cur_len + m + 1;  // keys visible to token m (causal)
  const int chunk = max(64, (L + a.NS - 1) / a.NS);
  const int start = split * chunk, end = min(L, start + chunk);
  const float scale = rsqrtf((float)D);

  // q for the G heads of this group; lane owns dims [8*lane, 8*lane+8)
  float qv[G][8];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const float* qp = a.q + ((size_t)m * a.H + kvh * G + g) * D + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) qv[g][i] = qp[i] * scale;
  }
  float mx[G], ls[G], acc[G][8];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    mx[g] = -INFINITY;
    ls[g] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[g][i] = 0.f;
  }
  const size_t base = (size_t)kvh * a.max_len;
  for (int t = start + warp; t < end; t += kDecWarps) {
    float kf[8], vf[8];
    kv_load8<KF>(a.k_cache, a.k_scale, base + t, lane, kf);
    kv_load8<VF>(a.v_cache, a.v_scale, base + t, lane, vf);
    float sc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float d0 = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) d0 = fmaf(qv[g][i], kf[i], d0);
      sc[g] = warp_sum(d0);
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float mn = fmaxf(mx[g], sc[g]);
      const float corr = __expf(mx[g] - mn);  // exp(-inf) = 0 on the first key
      const float p = __expf(sc[g] - mn);
      ls[g] = ls[g] * corr + p;
      mx[g] = mn;
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[g][i] = fmaf(acc[g][i], corr, p * vf[i]);
    }
  }

  // ---- merge the 4 warps
  __shared__ float s_acc[kDecWarps][G][D];
  __shared__ float s_ml[kDecWarps][G][2];
#pragma unroll
  for (int g = 0; g < G; ++g) {
#pragma unroll
    for (int i = 0; i < 8; ++i) s_acc[warp][g][lane * 8 + i] = acc[g][i];
    if (lane == 0) {
      s_ml[warp][g][0] = mx[g];
      s_ml[warp][g][1] = ls[g];
    }
  }
  __syncthreads();
  const size_t pbase = ((size_t)m * a.NS + split) * a.H + kvh * G;
  for (int idx = threadIdx.x; idx < G * D; idx += kDecThreads) {
    const int g = idx / D, d = idx % D;
    float M_ = -INFINITY;
    for (int w = 0; w < kDecWarps; ++w) M_ = fmaxf(M_, s_ml[w][g][0]);
    float num = 0.f, den = 0.f;
    if (M_ != -INFINITY) {
      for (int w = 0; w < kDecWarps; ++w) {
        const float f = __expf(s_ml[w][g][0] - M_);
        num += f * s_acc[w][g][d];
        den += f * s_ml[w][g][1];
      }
    }
    a.part_acc[(pbase + g) * D + d] = num;
    if (d == 0) {
      a.part_ml[(pbase + g) * 2] = M_;
      a.part_ml[(pbase + g) * 2 + 1] = den;
    }
  }

  // ---- last CTA of this (token, kv head) combines the splits
  __shared__ int s_last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int prev = atomicAdd(&a.counters[m * a.Hkv + kvh], 1);
    s_last = (prev == a.NS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  float(*s_o)[D] = s_acc[0];  // reuse: [G][D] combined (rotated) outputs
  for (int idx = threadIdx.x; idx < G * D; idx += kDecThreads) {
    const int g = idx / D, d = idx % D, head = kvh * G + g;
    float M_ = -INFINITY;
    for (int s = 0; s < a.NS; ++s) {
      const volatile float* ml = a.part_ml + (((size_t)m * a.NS + s) * a.H + head) * 2;
      M_ = fmaxf(M_, ml[0]);
    }
    float num = 0.f, den = 0.f;
    for (int s = 0; s < a.NS; ++s) {
      const size_t b = ((size_t)m * a.NS + s) * a.H + head;
      const volatile float* ml = a.part_ml + b * 2;
      const float mm = ml[0];
      if (mm == -INFINITY) continue;
      const float f = __expf(mm - M_);
      num += f * ((const volatile float*)a.part_acc)[b * D + d];
      den += f * ml[1];
    }
    s_o[g][d] = num / den;
  }
  if (a.rotate) {  // o = Q^T o_rot = diag(s) H o_rot / 16, butterflies in the order of hadamard.fwht
    for (int h = 1; h < D; h <<= 1) {
      __syncthreads();
      for (int p = threadIdx.x; p < G * D / 2; p += kDecThreads) {
        const int g = p / (D / 2), r = p % (D / 2);
        const int i = (r / h) * 2 * h + (r % h);  // index with bit h clear
        const float x0 = s_o[g][i], x1 = s_o[g][i + h];
        s_o[g][i] = x0 + x1;
        s_o[g][i + h] = x0 - x1;
      }
    }
  }
  __syncthreads();
  const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(a.proj) + (size_t)m * a.proj_stride;
  for (int idx = threadIdx.x; idx < G * D; idx += kDecThreads) {
    const int g = idx / D, d = idx % D, head = kvh * G + g;
    const float o = bf16r(a.rotate ? s_o[g][d] * 0.0625f * a.signs[d] : s_o[g][d]);
    const float gt = bf16r(sigmoidf_(__bfloat162float(row[head * 2 * D + D + d])));
    reinterpret_cast<__nv_bfloat16*>(a.out)[(size_t)m * a.H * D + head * D + d] = __float2bfloat16(o * gt);
  }
  if (threadIdx.x == 0) a.counters[m * a.Hkv + kvh] = 0;  // re-arm for the next call / graph replay
}

template <int KF, int VF>
static void launch_decode(const AttnDecodeArgs& x, cudaStream_t stream) {
  dim3 grid(x.Hkv, x.NS, x.M);
  attn_decode_kernel<KF, VF><<<grid, kDecThreads, 0, stream>>>(x);
}

void attn_decode(const AttnDecodeArgs& x, cudaStream_t stream) {
  if (x.kfmt == KV_BF16 && x.vfmt == KV_BF16) launch_decode<KV_BF16, KV_BF16>(x, stream);
  else if (x.kfmt == KV_FP8 && x.vfmt == KV_FP8) launch_decode<KV_FP8, KV_FP8>(x, stream);
  else if (x.kfmt == KV_FP8 && x.vfmt == KV_FP4) launch_decode<KV_FP8, KV_FP4>(x, stream);
  else if (x.kfmt == KV_FP4 && x.vfmt == KV_FP4) launch_decode<KV_FP4, KV_FP4>(x, stream);
}

}  // namespace cck
