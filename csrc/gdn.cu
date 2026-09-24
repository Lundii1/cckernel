// Gated DeltaNet decode / speculative-verify kernel (M <= 8 tokens per call).
//
// Math (HF layout, state S in R^{dk x dv} per value head; arXiv 2412.06464):
//   S_t = a_t S_{t-1} + k_t u_t^T,   u_t = b_t (v_t - a_t S_{t-1}^T k_t),   o_t = S_t^T q_t
// One pass over S per token: with r = S^T k and p = S^T q of the old state,
//   u = b (v - a r),   o = a p + u (k.q),   S' = a S + k u^T.
// Column j of every quantity depends only on column j of S, so each CTA owns a 128 x 32 slice
// (grid = Hv x 4 = 128 CTAs), kept in registers (thread = one column x 32 rows).
//
// Deferred commit (TreeWY arXiv 2608.20961 / ReplaySSM): the tokens of a call are *pending*; the
// kernel stores their pseudo-values u_i and log-decays g_i. The next call commits the first
// n_commit of them by replaying S <- a_i S + k_i u_i^T (no per-token state snapshots, S is still
// read once and written once per call). k_i is recomputed from the conv-input ring buffer.
// Conv inputs live in a 32-slot ring indexed by absolute position, so rollback is free: slots at
// positions >= cur_len are simply overwritten later. New tokens read their own inputs from `proj`,
// so no CTA ever reads a ring slot written in the same call.
#include "cck.h"
#include "common.cuh"

namespace cck {

constexpr int DK = 128, DV = 128, SL = 32, NSL = DV / SL, KW = 4, RING = 32;
constexpr int kGdnThreads = 128;

struct GdnKArgs {
  const __nv_bfloat16* proj;
  int proj_stride;
  __nv_bfloat16* ring;
  const float* conv_w;
  const float* A_log;
  const float* dt_bias;
  float* S;
  float* pend_u;
  float* pend_g;
  __nv_bfloat16* out;
  const int* cur_len;
  const int* n_commit;
  int M, Hk, Hv, C;
};

// Conv input of channel ch at absolute position P (P < L -> ring, else current proj row P - L).
CCK_DEVICE float conv_input(const GdnKArgs& a, int ch, int P, int L) {
  if (P < 0) return 0.f;
  if (P < L) return __bfloat162float(a.ring[(size_t)ch * RING + (P & (RING - 1))]);
  return __bfloat162float(a.proj[(size_t)(P - L) * a.proj_stride + ch]);
}

// silu(conv) output at position P for channel ch, rounded like the bf16 reference.
CCK_DEVICE float conv_out(const GdnKArgs& a, int ch, int P, int L) {
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < KW; ++j) acc = fmaf(a.conv_w[ch * KW + j], conv_input(a, ch, P - (KW - 1) + j, L), acc);
  return bf16r(silu(bf16r(acc)));
}

__global__ void __launch_bounds__(kGdnThreads) gdn_decode_kernel(GdnKArgs a) {
  const int h = blockIdx.x, sl = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rep = a.Hv / a.Hk, kh = h / rep;
  const int key_dim = a.Hk * DK;
  const int L = *a.cur_len, ncommit = *a.n_commit, M = a.M;
  const int qch = kh * DK + tid, kch = key_dim + kh * DK + tid;
  const int vch = 2 * key_dim + h * DV + sl * SL + lane;
  const int col = sl * SL + lane;  // value column owned by this thread

  __shared__ float qs[DK], ks[DK];
  __shared__ float red[4], part[4][SL][2];
  __shared__ float gate_g, gate_b;

  // ---- 1. load the committed state slice: rows warp*32 .. +31, column `col`
  float* Sh = a.S + (size_t)h * DK * DV;
  float s[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) s[i] = Sh[(size_t)(warp * 32 + i) * DV + col];

  // ---- 2. commit the accepted tokens of the previous call
  for (int i = 0; i < ncommit; ++i) {
    const int P = L - ncommit + i;
    const float kv = conv_out(a, kch, P, L);
    const float ss = block_sum<kGdnThreads>(kv * kv, red);
    ks[tid] = kv * rsqrtf(ss + 1e-6f);
    __syncthreads();
    const float u = a.pend_u[((size_t)i * a.Hv + h) * DV + col];
    const float alpha = __expf(a.pend_g[((size_t)i * a.Hv + h) * NSL + sl]);
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = fmaf(alpha, s[r], ks[warp * 32 + r] * u);
    __syncthreads();
  }
  if (ncommit > 0) {
#pragma unroll
    for (int i = 0; i < 32; ++i) Sh[(size_t)(warp * 32 + i) * DV + col] = s[i];
  }

  // ---- 3. record new conv inputs in the ring (owners only; slots >= L are never read this call)
  const bool qk_owner = (h % rep == 0) && sl == 0;
  for (int m = 0; m < M; ++m) {
    const int slot = (L + m) & (RING - 1);
    const __nv_bfloat16* row = a.proj + (size_t)m * a.proj_stride;
    if (qk_owner) {
      a.ring[(size_t)qch * RING + slot] = row[qch];
      a.ring[(size_t)kch * RING + slot] = row[kch];
    }
    if (warp == 0) a.ring[(size_t)vch * RING + slot] = row[vch];
  }

  // ---- 4. process the new tokens (pending until the host reports acceptance)
  const float A = __expf(a.A_log[h]);
  for (int m = 0; m < M; ++m) {
    const int P = L + m;
    const float qv = conv_out(a, qch, P, L);
    const float kv = conv_out(a, kch, P, L);
    const float vv = conv_out(a, vch, P, L);
    const float sq = block_sum<kGdnThreads>(qv * qv, red);
    const float sk = block_sum<kGdnThreads>(kv * kv, red);
    const float qn = qv * rsqrtf(sq + 1e-6f) * rsqrtf((float)DK);
    const float kn = kv * rsqrtf(sk + 1e-6f);
    qs[tid] = qn;
    ks[tid] = kn;
    if (tid == 0) {
      const __nv_bfloat16* row = a.proj + (size_t)m * a.proj_stride;
      const int boff = a.C + a.Hv * DV;
      const float bb = __bfloat162float(row[boff + h]);
      const float aa = __bfloat162float(row[boff + a.Hv + h]);
      gate_b = sigmoidf_(bb);
      gate_g = -A * softplusf_(aa + a.dt_bias[h]);
    }
    const float kq = block_sum<kGdnThreads>(qn * kn, red);  // also publishes qs/ks/gates
    const float g = gate_g, beta = gate_b, alpha = __expf(g);
    // partial r = S^T k and p = S^T q over this thread's 32 rows
    float pr = 0.f, pp = 0.f;
#pragma unroll
    for (int r = 0; r < 32; ++r) {
      pr = fmaf(s[r], ks[warp * 32 + r], pr);
      pp = fmaf(s[r], qs[warp * 32 + r], pp);
    }
    part[warp][lane][0] = pr;
    part[warp][lane][1] = pp;
    __syncthreads();
    const float rr = part[0][lane][0] + part[1][lane][0] + part[2][lane][0] + part[3][lane][0];
    const float pq = part[0][lane][1] + part[1][lane][1] + part[2][lane][1] + part[3][lane][1];
    // v for this column: every warp computes it redundantly, cheap
    const float u = beta * (vv - alpha * rr);
    const float o = alpha * pq + u * kq;
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = fmaf(alpha, s[r], ks[warp * 32 + r] * u);
    if (warp == 0) {
      a.out[(size_t)m * a.Hv * DV + h * DV + col] = __float2bfloat16(o);
      a.pend_u[((size_t)m * a.Hv + h) * DV + col] = u;
      if (lane == 0) a.pend_g[((size_t)m * a.Hv + h) * NSL + sl] = g;
    }
    __syncthreads();
  }
}

void gdn_decode(const GdnDecodeArgs& x, cudaStream_t stream) {
  GdnKArgs a{reinterpret_cast<const __nv_bfloat16*>(x.proj), x.proj_stride, reinterpret_cast<__nv_bfloat16*>(x.ring),
             x.conv_w, x.A_log, x.dt_bias, x.S, x.pend_u, x.pend_g, reinterpret_cast<__nv_bfloat16*>(x.out),
             x.cur_len, x.n_commit, x.M, x.Hk, x.Hv, x.C};
  dim3 grid(x.Hv, NSL);
  gdn_decode_kernel<<<grid, kGdnThreads, 0, stream>>>(a);
}

}  // namespace cck
