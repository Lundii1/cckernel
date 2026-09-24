// Host-side launcher API for cckernel (plain C++, no torch). Implemented in the .cu files.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace cck {

// Packed quantized weight (see cckernel/quant.py). lo: [N, K/2] (4/5/6 bit) or [N, K] (8 bit);
// hi: [N, K/4] (6 bit), [N, K/8] (5 bit), else null; scales: fp16 [N, K/128].
struct QWeight {
  const uint8_t* lo;
  const uint8_t* hi;
  const void* scales;
  int bits, N, K;
};

enum GemvPro { PRO_BF16 = 0, PRO_RMSNORM = 1, PRO_GATED = 2 };
enum GemvEpi { EPI_BF16 = 0, EPI_F32 = 1, EPI_ADD_F32 = 2, EPI_SWIGLU = 3 };

// y = W x for a single token (decode). Prologues: PRO_BF16 x = bf16[K]; PRO_RMSNORM x = fp32
// residual [K], normalised in-kernel (weights carry the folded norm gain); PRO_GATED x = bf16 GDN
// output [K] per-head RMS-normalised and multiplied by silu(z) (z = bf16[K], head size head_dim).
// Epilogues: bf16 store, fp32 store, fp32 residual add (y += Wx), SwiGLU on interleaved rows.
void qgemv(const QWeight& w, int pro, const void* x, const void* z, int head_dim, float eps, int epi, void* y,
           cudaStream_t stream);

// Tensor-core GEMM for 2..8 tokens: y[M, N] (row stride N) = x[M, K] (row stride x_stride) W^T.
// epi: EPI_BF16 / EPI_F32 / EPI_ADD_F32.
void qgemm_skinny(const QWeight& w, const void* x, int x_stride, int M, int epi, void* y, cudaStream_t stream);

// Dequantize to bf16 [N, K] in natural k order (prefill fallback / debugging).
void dequant_bf16(const QWeight& w, void* out, cudaStream_t stream);

// Gated DeltaNet decode / verify step for M <= 8 tokens (see gdn.cu for the full contract).
struct GdnDecodeArgs {
  const void* proj;       // bf16 [M, proj_stride]: [qkv (C) | z (Hv*DV) | b (Hv) | a (Hv)]
  int proj_stride;
  void* ring;             // bf16 [C, 32] ring buffer of conv inputs, slot = position & 31
  const float* conv_w;    // [C, 4]
  const float* A_log;     // [Hv]
  const float* dt_bias;   // [Hv]
  float* S;               // [Hv, DK, DV] committed recurrent state
  float* pend_u;          // [8, Hv, DV] pseudo-values of the last step's tokens
  float* pend_g;          // [8, Hv, DV/32] log-decays of the last step's tokens
  void* out;              // bf16 [M, Hv*DV] (pre-norm core output)
  const int* cur_len;     // committed length (position of the first new token), device scalar
  const int* n_commit;    // how many of the previous step's tokens were accepted, device scalar
  int M, Hk, Hv, C;
};
void gdn_decode(const GdnDecodeArgs& a, cudaStream_t stream);

// KV cache formats (cckernel/kvq.py): BF16 [Hkv, max_len, D] bf16; FP8 [Hkv, max_len, D] e4m3 codes;
// FP4 [Hkv, max_len, D/2] E2M1 nibbles (low nibble = even channel) + [Hkv, max_len, D/16] E4M3 scales.
enum KvFormat { KV_BF16 = 0, KV_FP8 = 1, KV_FP4 = 2 };

// Attention: qk-norm + partial RoPE (+ per-head Hadamard rotation) + quantized KV-cache append.
struct AttnPrepArgs {
  const void* proj;       // bf16 [M, proj_stride]: [q|gate per head (H*2D) | k (Hkv*D) | v (Hkv*D)]
  int proj_stride;
  const float* q_norm;    // [D] (1 + w)
  const float* k_norm;    // [D] (1 + w)
  const float* inv_freq;  // [rot/2]
  const float* signs;     // [D] Hadamard signs (used when rotate)
  float* q_out;           // [M, H, D] (rotated when rotate)
  void* k_cache;          // K data (format kfmt)
  uint8_t* k_scale;       // FP4 scales or null
  void* v_cache;
  uint8_t* v_scale;
  const int* cur_len;
  int max_len, M, H, Hkv, rot, kfmt, vfmt, rotate;
  float eps;
};
void attn_prep(const AttnPrepArgs& a, cudaStream_t stream);

// Split-KV flash-decoding with GQA packing, in-kernel dequantization of the KV cache, split combine,
// inverse rotation and the sigmoid output gate.
struct AttnDecodeArgs {
  const float* q;         // [M, H, D]
  const void* k_cache;
  const uint8_t* k_scale;
  const void* v_cache;
  const uint8_t* v_scale;
  const float* signs;     // [D]
  const void* proj;       // bf16 [M, proj_stride] (gate read from the q_proj part)
  int proj_stride;
  float* part_acc;        // [M, NS, H, D]
  float* part_ml;         // [M, NS, H, 2]
  int* counters;          // [M, Hkv] zero-initialised
  void* out;              // bf16 [M, H*D]
  const int* cur_len;
  int max_len, M, H, Hkv, NS, kfmt, vfmt, rotate;
};
void attn_decode(const AttnDecodeArgs& a, cudaStream_t stream);

}  // namespace cck
