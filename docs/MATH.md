# The math behind the kernels

Target: `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`. It is an SFT of Qwen3.5-9B with 32 layers:
- 24 Gated DeltaNet (GDN) layers;
- 8 gated full-attention layers (GQA 16/4, head_dim 256, partial RoPE on 64 dims);
- a SwiGLU MLP (4096 → 12288);
- a 248,320-token vocabulary.

The GPU is an RTX 4060 Ti 16 GB: sm_89, 288 GB/s, 34 SMs, 32 MB L2.

Each section names the paper (found through alphaXiv) and the kernel or tool it became.

## 1. Where the time goes (roofline)

Batch-1 decode reads every weight once per token:

| Component | Params |
|---|---|
| MLP | 4.83 B |
| GDN projections | 1.62 B |
| Attention projections | 0.47 B |
| lm_head | 1.02 B |
| **Total** | **7.94 B** |

Other per-token traffic:
- GDN recurrent state: 24 × 32 heads × 128 × 128 × 4 B = 48 MB, read and written;
- KV cache: 32 KB per context token (only 8 layers have a KV cache).

So decode is weight-bandwidth-bound:

| Weights | Bytes/token | Ceiling on a 4060 Ti |
|---|---|---|
| INT8 | 8.06 GB | 35.7 tok/s |
| INT6 (lm_head INT8) | 6.33 GB | 45.5 tok/s |
| INT5 (lm_head INT6) | 5.21 GB | 55.3 tok/s |

The levers, in order:
1. fewer bits at equal quality (§4, §5);
2. more tokens per weight pass (speculation, §7);
3. fewer launches and passes (fusion, CUDA graphs).

## 2. Gated delta rule: three equivalent algorithms

*Gated Delta Networks*, arXiv 2412.06464; *Parallelizing Linear Transformers with the Delta Rule*, arXiv 2406.06484.

Per value head, in HF layout (S ∈ R^{dk×dv}):

    S_t = α_t S_{t-1} + k_t ũ_tᵀ,   ũ_t = β_t (v_t − α_t S_{t-1}ᵀ k_t),   o_t = S_tᵀ q_t
    α_t = exp(g_t),  g_t = −exp(A_log)·softplus(a_t + dt_bias),  β_t = σ(b_t)

q and k are L2-normalized, and q is scaled by 1/√128.

**(a) One-pass decode (`csrc/gdn.cu`).** With r = Sᵀk and p = Sᵀq of the *old* state:

    ũ = β (v − α r),   o = α p + ũ (k·q),   S' = α S + k ũᵀ

- S is read once and written once per token, which is the minimum possible traffic.
- Column j of every quantity depends only on column j of S.
- So a CTA owns a 128×32 slice (Hv × 4 = 128 CTAs), kept in registers.
- The cross-thread work per token is two block reductions (the ‖q‖, ‖k‖ norms and k·q) plus one 4-way row reduction.

**(b) Chunkwise WY/UT prefill (`torch_ops.gdn_chunk`, `reference.gdn_chunked`).**
- Within a chunk of C tokens starting from state S₀, with γ_r = exp(G_r) and G = cumsum(g):

        (I + L) Ũ = diag(β)(V − diag(γ) K S₀),   L_ri = β_r (γ_r/γ_i) k_r·k_i  (i < r)
        U = T V,  W = T diag(γ) K,  T = (I+L)⁻¹ diag(β)        ⇒  Ũ = U − W S₀
        O      = diag(γ) Q S₀ + (QKᵀ ⊙ Γ ⊙ M) Ũ,               Γ_ri = γ_r/γ_i
        S_next = γ_C S₀ + (diag(γ_C/γ) K)ᵀ Ũ

- Every exponent is a difference G_r − G_i with r ≥ i, so it is ≤ 0 and cannot overflow; `test_chunked_strong_decay_is_stable` checks this.
- The triangular system is one batched `solve_triangular` per layer; only the state scan is sequential.

**(c) Deferred commit for speculation.**
- Source: TreeWY, arXiv 2608.20961 (and ReplaySSM, which it cites).
- Unrolling (a):

        S_a = γ_a S₀ + Σ_{i≤a} (γ_a/γ_i) k_i ũ_iᵀ

- So S is never snapshotted per draft token. The kernel stores only the pseudo-values ũ_i and log-decays g_i of a step.
- The next call replays the first `n_commit` of them onto the committed state.
- k_i is recomputed from the conv-input ring buffer.
- Per call, S is still read once and written once.

The conv state is a 32-slot ring indexed by absolute position:
- Rolling back is free: slots at positions ≥ `cur_len` are simply overwritten later.
- New tokens read their own inputs from the projection output, so no CTA reads a slot written in the same launch.

All three algorithms agree to 1e-9 in fp64 (`tests/cpu/test_gdn_math.py`).

*Why GDN survives 4-bit* (arXiv 2609.04098) shows two things:
- the softplus/exp gate parameterization compresses quantization error;
- the delta rule erases state error instead of accumulating it.

So all GDN projections, including `in_proj_a` and `in_proj_b`, are quantized like any other matrix. Only conv, norms, A_log and dt_bias stay in fp32.

## 3. Gated attention decode

*Gated Attention for LLMs*, arXiv 2505.06708; online softmax (Milakov & Gimelshein); flash-decoding.

**`attn_prep`:**
- zero-centred per-head RMSNorm, with the (1+w) weights pre-shifted;
- rotate-half RoPE on dims [0, 64), with θ = 10⁷ and inverse frequencies computed exactly like HF in fp32;
- K/V append to a bf16 cache;
- bf16 rounding points that follow HF's dataflow.

**`attn_decode`:**
- Grid is (KV head, split, token). Keys are read once per KV head for all 4 query heads of the group (GQA packing).
- Each warp keeps an online softmax per head. The 4 warps are merged in shared memory.
- The last CTA to finish (atomic ticket, re-armed for graph replay) merges the splits with log-sum-exp.
- That CTA also applies the head-specific output gate σ(gate).

## 4. Rotations and folding (free at runtime)

*QuIP#*, arXiv 2402.04396, Lemma 3.1; QuaRot / SpinQuant R1.

Q = H₄₀₉₆ diag(s)/64 is orthogonal and rms(Qx) = rms(x), so rotating the residual stream is exact:
- The embedding rows and every residual writer become `Q W`: `o_proj`, `out_proj`, `down`.
- Every residual reader becomes `W diag(1+γ) Qᵀ`: all input projections, `gate/up`, `lm_head`.
- Each zero-centred norm gain (1+γ) is folded into its consumers.
- The GDN gated-norm weight is folded into `out_proj`.

The runtime norms become weightless. Incoherence processing makes each weight entry roughly Gaussian with no outliers: `max |W_ij| ≤ μ‖W‖_F/√(mn)` with μ = 2 log(4mn/δ). `test_fold_with_hadamard` shows the folded network reproduces the original logits to 1e-6 in fp64.

## 5. Quantization and bit allocation

*Pushing the Limits of LLM Quantization via the Linearity Theorem* (HIGGS), arXiv 2411.17525.

- **Linearity theorem.** E[PPL] ≈ PPL* + Σ_l α_l t_l², where t_l² = ‖W_l − Ŵ_l‖²/‖W_l‖² and α_l does not depend on the quantizer.
- **Quantizer.** Symmetric INT-b (b ∈ {4, 5, 6, 8}) with groups of 128 along K and fp16 scales. The clip ratio is chosen per group to minimise MSE. After rotation the weights are near-Gaussian, so this is the Gaussian-optimal uniform grid.
- **Allocator (`cckernel/alloc.py`).** Solves exactly the multiple-choice knapsack min Σ α_l t_l²(b_l) subject to Σ bytes ≤ budget, by DP.
  - t_l²(b) is measured on the real weights.
  - α_l comes from `tools/calibrate_alpha.py`: Gaussian noise injection plus KL, fitted through the origin, as in HIGGS Alg. 3. Without calibration it falls back to a documented prior.

For a 16 GB card at minimum quality loss, the `quality` preset (all INT8 after rotation) is the right default:
- INT8 RTN with MSE clipping gives t² ≈ 1e-4 on Gaussian groups, which is effectively lossless;
- the total is 7.5 GiB of weights plus a 1.9 GiB bf16 embedding plus the KV cache;
- that fits 16 GB with a 128K-token context.

The `balanced` and `fast` presets trade quality for speed through the knapsack.

## 6. Packed layout and dequantization

*MARLIN*, arXiv 2408.11743 (Kim et al. dequant trick); *QTIP*, arXiv 2406.11235 (bitshift / magic-number decoding).

**Block coordinates.** Within each 64-weight block, a weight has coordinates (c, h, t, j):
- the natural index is k = 16t + 8h + 2c + j;
- the storage order is e = 16c + 8h + 2t + j.

The map between them swaps two 2-bit fields, so it is an involution.

**What each kernel gets:**
- **GEMV.** It streams each row sequentially: 32–64 B per lane with 128-bit `ld.global.nc.L1::no_allocate` loads. It stages x in e-order in shared memory (144 B stride, so 16 B reads are bank-conflict-free).
- **Skinny GEMM.** Lane (g, c) finds exactly the pairs (k, k+1) that `mma.m16n8k16` needs in low words (2c, 2c+1) and high word c.

**Integer-to-float conversion (no I2F):**
- 4, 5 and 6 bit: `lop3(x >> 4t, 0x000F000F, 0x43004300)` gives the bf16 pair (128+u, 128+u′). The high plane is merged with one more `lop3`.
- 8 bit: `prmt` into the fp32 2²³ magic, followed by `fsub`.

## 7. Speculative decoding without an MTP head

The checkpoint ships no `mtp.*` weights. Drafts come from the token history instead:
- a prompt-lookup n-gram drafter (n = 4..1) with an adaptive draft length;
- this suits agentic and coding outputs, which copy their context.

Verification is exact:
- **Temperature 0:** accept the longest matching prefix, then emit the target's own token.
- **Temperature > 0:** the draft distribution is a point mass q = δ_d. Speculative sampling (Leviathan et al.; Chen et al.) then reduces to: accept d with probability p(d), otherwise sample from p with d removed. `test_sampling_accept_is_exact` checks this empirically.

The verify pass for M ≤ 8 tokens uses:
- **`qgemm_skinny`** (FlashDecoding++, arXiv 2311.01282, flat GEMM; QuIP#-style operand swap). The weights are the A operand and the tokens are the N = 8 columns, so verifying 8 tokens streams the same bytes as one decode step.
- **The GDN kernel** processes the M tokens sequentially, with S held in registers and the deferred commit (§2c).
- **The attention kernel** handles M causal queries.

## 8. Not implemented yet

The literature points to these next steps:
- **Tensor-core chunked GDN prefill.** Prefill currently uses cuBLAS batched `trsm` plus matmuls.
- **Fused dequant-GEMM for prefill.** Prefill currently dequantizes each matrix once per prompt (layer-major), then uses cuBLAS.
- **FP8-E4M3 KV cache with calibrated scales.** 2609.04098 reports this as performance-free.
- **Tree drafts via the tree WY solve** (TreeWY).
- **An MTP draft head with an FR-Spec vocabulary subset.** FR-Spec is arXiv 2502.14856.
- **Sub-4-bit QTIP trellis codes.**
