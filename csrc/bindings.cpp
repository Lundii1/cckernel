// PyTorch bindings: thin wrappers that pass raw pointers + the current CUDA stream to the launchers
// (so every op is CUDA-graph capturable).
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cck.h"

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

template <typename T = void>
T* ptr(const torch::Tensor& t) {
  return t.defined() && t.numel() ? reinterpret_cast<T*>(t.data_ptr()) : nullptr;
}

cck::QWeight qweight(const torch::Tensor& lo, const torch::Tensor& hi, const torch::Tensor& scales, int64_t bits,
                     int64_t N, int64_t K) {
  CHECK_CUDA(lo);
  CHECK_CONTIG(lo);
  CHECK_CUDA(scales);
  TORCH_CHECK(scales.scalar_type() == torch::kHalf, "scales must be fp16");
  TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
  TORCH_CHECK(bits == 4 || bits == 5 || bits == 6 || bits == 8, "bits must be 4, 5, 6 or 8");
  if (bits == 5 || bits == 6) {
    CHECK_CUDA(hi);
    CHECK_CONTIG(hi);
  }
  return cck::QWeight{ptr<const uint8_t>(lo), (bits == 5 || bits == 6) ? ptr<const uint8_t>(hi) : nullptr,
                      ptr<const void>(scales), (int)bits, (int)N, (int)K};
}

void qgemv(torch::Tensor lo, torch::Tensor hi, torch::Tensor scales, int64_t bits, int64_t N, int64_t K, int64_t pro,
           torch::Tensor x, torch::Tensor z, int64_t head_dim, double eps, int64_t epi, torch::Tensor y) {
  CHECK_CUDA(x);
  CHECK_CUDA(y);
  const bool want_f32_x = pro == cck::PRO_RMSNORM;
  TORCH_CHECK(x.scalar_type() == (want_f32_x ? torch::kFloat : torch::kBFloat16), "unexpected x dtype for prologue");
  const bool want_f32_y = epi == cck::EPI_F32 || epi == cck::EPI_ADD_F32;
  TORCH_CHECK(y.scalar_type() == (want_f32_y ? torch::kFloat : torch::kBFloat16), "unexpected y dtype for epilogue");
  cck::qgemv(qweight(lo, hi, scales, bits, N, K), (int)pro, ptr<const void>(x), ptr<const void>(z), (int)head_dim,
             (float)eps, (int)epi, ptr(y), stream());
}

void qgemm_skinny(torch::Tensor lo, torch::Tensor hi, torch::Tensor scales, int64_t bits, int64_t N, int64_t K,
                  torch::Tensor x, int64_t M, int64_t epi, torch::Tensor y) {
  CHECK_CUDA(x);
  CHECK_CUDA(y);
  TORCH_CHECK(M >= 1 && M <= 8, "qgemm_skinny supports 1..8 tokens");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.stride(-1) == 1, "x must be bf16 with unit inner stride");
  TORCH_CHECK(y.is_contiguous() && y.size(-1) == N, "y must be contiguous [*, N]");
  TORCH_CHECK(y.scalar_type() == (epi == cck::EPI_BF16 ? torch::kBFloat16 : torch::kFloat),
              "unexpected y dtype for epilogue");
  cck::qgemm_skinny(qweight(lo, hi, scales, bits, N, K), ptr<const void>(x), (int)x.stride(0), (int)M, (int)epi,
                    ptr(y), stream());
}

void dequant(torch::Tensor lo, torch::Tensor hi, torch::Tensor scales, int64_t bits, int64_t N, int64_t K,
             torch::Tensor out) {
  CHECK_CUDA(out);
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.numel() >= N * K, "out must be bf16 with N*K elements");
  cck::dequant_bf16(qweight(lo, hi, scales, bits, N, K), ptr(out), stream());
}

void gdn_decode(torch::Tensor proj, torch::Tensor ring, torch::Tensor conv_w, torch::Tensor A_log, torch::Tensor dt_bias,
                torch::Tensor S, torch::Tensor pend_u, torch::Tensor pend_g, torch::Tensor out, torch::Tensor cur_len,
                torch::Tensor n_commit, int64_t M, int64_t Hk, int64_t Hv, int64_t C) {
  TORCH_CHECK(M >= 1 && M <= 8, "gdn_decode supports 1..8 tokens");
  TORCH_CHECK(S.size(-1) == 128 && S.size(-2) == 128, "gdn_decode is compiled for dk = dv = 128");
  cck::GdnDecodeArgs a{ptr<const void>(proj), (int)proj.stride(0), ptr(ring), ptr<const float>(conv_w),
                       ptr<const float>(A_log), ptr<const float>(dt_bias), ptr<float>(S), ptr<float>(pend_u),
                       ptr<float>(pend_g), ptr(out), ptr<const int>(cur_len), ptr<const int>(n_commit),
                       (int)M, (int)Hk, (int)Hv, (int)C};
  cck::gdn_decode(a, stream());
}

void check_kv(const torch::Tensor& data, const torch::Tensor& scale, int64_t fmt, const char* what) {
  CHECK_CUDA(data);
  CHECK_CONTIG(data);
  TORCH_CHECK(fmt >= 0 && fmt <= 2, what, ": unknown KV format");
  const int64_t d = data.size(-1) * (fmt == cck::KV_FP4 ? 2 : 1);
  TORCH_CHECK(d == 256, what, ": attention kernels are compiled for head_dim 256");
  TORCH_CHECK(data.scalar_type() == (fmt == cck::KV_BF16 ? torch::kBFloat16 : torch::kByte), what, ": dtype");
  if (fmt == cck::KV_FP4) {
    CHECK_CUDA(scale);
    CHECK_CONTIG(scale);
    TORCH_CHECK(scale.size(-1) == 16 && scale.size(1) == data.size(1), what, ": FP4 scale shape");
  }
}

void check_formats(int64_t kfmt, int64_t vfmt) {
  TORCH_CHECK((kfmt == 0 && vfmt == 0) || (kfmt == 1 && vfmt == 1) || (kfmt == 1 && vfmt == 2) ||
                  (kfmt == 2 && vfmt == 2),
              "supported KV formats: bf16/bf16, fp8/fp8, fp8/fp4, fp4/fp4");
}

void attn_prep(torch::Tensor proj, torch::Tensor q_norm, torch::Tensor k_norm, torch::Tensor inv_freq, torch::Tensor q_out,
               torch::Tensor k_cache, torch::Tensor k_scale, torch::Tensor v_cache, torch::Tensor v_scale,
               torch::Tensor signs, torch::Tensor cur_len, int64_t M, int64_t H, int64_t Hkv, double eps, int64_t kfmt,
               int64_t vfmt, bool rotate) {
  check_formats(kfmt, vfmt);
  check_kv(k_cache, k_scale, kfmt, "k_cache");
  check_kv(v_cache, v_scale, vfmt, "v_cache");
  TORCH_CHECK(signs.numel() == 256 && signs.scalar_type() == torch::kFloat, "signs must be fp32 [256]");
  cck::AttnPrepArgs a{ptr<const void>(proj), (int)proj.stride(0), ptr<const float>(q_norm), ptr<const float>(k_norm),
                      ptr<const float>(inv_freq), ptr<const float>(signs), ptr<float>(q_out), ptr(k_cache),
                      ptr<uint8_t>(k_scale), ptr(v_cache), ptr<uint8_t>(v_scale), ptr<const int>(cur_len),
                      (int)k_cache.size(1), (int)M, (int)H, (int)Hkv, (int)inv_freq.numel() * 2, (int)kfmt,
                      (int)vfmt, (int)rotate, (float)eps};
  cck::attn_prep(a, stream());
}

void attn_decode(torch::Tensor q, torch::Tensor k_cache, torch::Tensor k_scale, torch::Tensor v_cache,
                 torch::Tensor v_scale, torch::Tensor signs, torch::Tensor proj, torch::Tensor part_acc,
                 torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out, torch::Tensor cur_len, int64_t M,
                 int64_t H, int64_t Hkv, int64_t NS, int64_t kfmt, int64_t vfmt, bool rotate) {
  TORCH_CHECK(H == 4 * Hkv, "attn_decode is compiled for GQA group size 4");
  check_formats(kfmt, vfmt);
  check_kv(k_cache, k_scale, kfmt, "k_cache");
  check_kv(v_cache, v_scale, vfmt, "v_cache");
  cck::AttnDecodeArgs a{ptr<const float>(q), ptr<const void>(k_cache), ptr<const uint8_t>(k_scale),
                        ptr<const void>(v_cache), ptr<const uint8_t>(v_scale), ptr<const float>(signs),
                        ptr<const void>(proj), (int)proj.stride(0), ptr<float>(part_acc), ptr<float>(part_ml),
                        ptr<int>(counters), ptr(out), ptr<const int>(cur_len), (int)k_cache.size(1), (int)M, (int)H,
                        (int)Hkv, (int)NS, (int)kfmt, (int)vfmt, (int)rotate};
  cck::attn_decode(a, stream());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "cckernel CUDA ops (sm_89)";
  m.def("qgemv", &qgemv, "dequant-fused GEMV with prologue/epilogue fusion");
  m.def("qgemm_skinny", &qgemm_skinny, "tensor-core GEMM for 2..8 tokens (speculative verify)");
  m.def("dequant", &dequant, "dequantize packed weight to bf16");
  m.def("gdn_decode", &gdn_decode, "Gated DeltaNet decode/verify step with deferred commit");
  m.def("attn_prep", &attn_prep, "qk-norm + partial RoPE + Hadamard rotation + quantized KV append");
  m.def("attn_decode", &attn_decode, "split-KV GQA flash-decoding over a quantized KV cache, gated output");
}
