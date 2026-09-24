// Gated full attention for decode / verify (head_dim 256, GQA group 4).
//
// attn_prep: per (token, head) CTA of 256 threads: zero-centred RMSNorm (weights pre-shifted to
//   1 + w), partial rotate-half RoPE on the first `rot` dims (theta^-2i/rot frequencies computed on
//   the host in fp32 exactly like HF), bf16 rounding points mirroring the reference, K/V append.
// attn_decode: flash-decoding. Grid (Hkv, NS, M); a CTA owns one KV head and one split of the keys,
//   and processes the G = H/Hkv query heads of the group together so each K/V row is read once for
//   all of them (GQA packing). Online softmax (Milakov & Gimelshein) per warp, merged across warps
//   in shared memory, then across splits by the last CTA to finish (atomic ticket), which also
//   applies the head-specific sigmoid output gate (Gated Attention, arXiv 2505.06708).
#include "cck.h"
#include "common.cuh"

namespace cck {

constexpr int D = 256;
constexpr int G = 4;  // query heads per KV head

struct PrepKArgs {
  const __nv_bfloat16* proj;
  int proj_stride;
  const float* q_norm;
  const float* k_norm;
  const float* inv_freq;
  float* q_out;
  __nv_bfloat16* k_cache;
  __nv_bfloat16* v_cache;
  const int* cur_len;
  int max_len, M, H, Hkv, rot;
  float eps;
};

__global__ void __launch_bounds__(D) attn_prep_kernel(PrepKArgs a) {
  const int m = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
  const int pos = *a.cur_len + m;
  const __nv_bfloat16* row = a.proj + (size_t)m * a.proj_stride;
  const bool is_q = head < a.H;
  const int kvh = head - a.H;
  __shared__ float red[D / 32];
  __shared__ float xs[D];
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
  if (is_q) {
    a.q_out[((size_t)m * a.H + head) * D + d] = val;
  } else {
    const size_t off = ((size_t)kvh * a.max_len + pos) * D + d;
    a.k_cache[off] = __float2bfloat16(val);
    a.v_cache[off] = row[a.H * 2 * D + a.Hkv * D + kvh * D + d];
  }
}

void attn_prep(const AttnPrepArgs& x, cudaStream_t stream) {
  PrepKArgs a{reinterpret_cast<const __nv_bfloat16*>(x.proj), x.proj_stride, x.q_norm, x.k_norm, x.inv_freq, x.q_out,
              reinterpret_cast<__nv_bfloat16*>(x.k_cache), reinterpret_cast<__nv_bfloat16*>(x.v_cache), x.cur_len,
              x.max_len, x.M, x.H, x.Hkv, x.rot, x.eps};
  dim3 grid(x.M, x.H + x.Hkv);
  attn_prep_kernel<<<grid, D, 0, stream>>>(a);
}

// ---------------------------------------------------------------------------------------------

struct DecKArgs {
  const float* q;
  const __nv_bfloat16* k_cache;
  const __nv_bfloat16* v_cache;
  const __nv_bfloat16* proj;
  int proj_stride;
  float* part_acc;
  float* part_ml;
  int* counters;
  __nv_bfloat16* out;
  const int* cur_len;
  int max_len, M, H, Hkv, NS;
};

constexpr int kDecThreads = 128;
constexpr int kDecWarps = kDecThreads / 32;

__global__ void __launch_bounds__(kDecThreads) attn_decode_kernel(DecKArgs a) {
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
  const __nv_bfloat16* Kb = a.k_cache + (size_t)kvh * a.max_len * D + lane * 8;
  const __nv_bfloat16* Vb = a.v_cache + (size_t)kvh * a.max_len * D + lane * 8;
  for (int t = start + warp; t < end; t += kDecWarps) {
    const uint4 kr = *reinterpret_cast<const uint4*>(Kb + (size_t)t * D);
    const uint4 vr = *reinterpret_cast<const uint4*>(Vb + (size_t)t * D);
    const float kf[8] = {bf16lo(kr.x), bf16hi(kr.x), bf16lo(kr.y), bf16hi(kr.y),
                         bf16lo(kr.z), bf16hi(kr.z), bf16lo(kr.w), bf16hi(kr.w)};
    const float vf[8] = {bf16lo(vr.x), bf16hi(vr.x), bf16lo(vr.y), bf16hi(vr.y),
                         bf16lo(vr.z), bf16hi(vr.z), bf16lo(vr.w), bf16hi(vr.w)};
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
  const __nv_bfloat16* row = a.proj + (size_t)m * a.proj_stride;
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
    const float o = bf16r(num / den);
    const float gt = bf16r(sigmoidf_(__bfloat162float(row[head * 2 * D + D + d])));
    a.out[(size_t)m * a.H * D + head * D + d] = __float2bfloat16(o * gt);
  }
  if (threadIdx.x == 0) a.counters[m * a.Hkv + kvh] = 0;  // re-arm for the next call / graph replay
}

void attn_decode(const AttnDecodeArgs& x, cudaStream_t stream) {
  DecKArgs a{x.q, reinterpret_cast<const __nv_bfloat16*>(x.k_cache), reinterpret_cast<const __nv_bfloat16*>(x.v_cache),
             reinterpret_cast<const __nv_bfloat16*>(x.proj), x.proj_stride, x.part_acc, x.part_ml, x.counters,
             reinterpret_cast<__nv_bfloat16*>(x.out), x.cur_len, x.max_len, x.M, x.H, x.Hkv, x.NS};
  dim3 grid(x.Hkv, x.NS, x.M);
  attn_decode_kernel<<<grid, kDecThreads, 0, stream>>>(a);
}

}  // namespace cck
