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
- **Tree drafts via the tree WY solve** (TreeWY).
- **An MTP draft head with an FR-Spec vocabulary subset.** FR-Spec is arXiv 2502.14856.
- **Sub-4-bit QTIP trellis codes.**

## 9. Quantized KV cache (DeepSeek-V4.1-Flash)

*DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*, arXiv 2609.19969, sec. 2.4.4.

**What the paper does.** It stores its main KV cache as FP4:
- E2M1 values with one E4M3 scale per 16 channels, which is NVFP4 without the second-level scale;
- K is quantized after RoPE and dequantized in the attention kernel before use;
- quantization-aware training makes it accurate.

**What we can transfer.** Only the format carries over, because we cannot retrain. Our 8 attention layers cost 32 KiB of bf16 KV per context token:

| Format | Bytes per token | Size vs bf16 |
|---|---|---|
| fp8 | 16 KiB | ½ |
| k8v4 (K fp8, V fp4) | 12.5 KiB | 0.39 |
| fp4 | 9 KiB | 0.28 |

At 262,144 tokens, the model's full context, the fp4 cache takes 2.4 GB, where bf16 takes 8.6 GB. So the full context fits a 16 GB card next to the 9.4 GB of weights.

**Replacing QAT with an MSE scale search.** For a group x ∈ R¹⁶:
- the absmax scale s₀ = E4M3(max|x|/6) is not MSE-optimal on the non-uniform E2M1 grid {0, ½, 1, 1½, 2, 3, 4, 6};
- we try the E4M3 codes s₀−2 … s₀+1 and keep the smallest ‖x − s·E2M1(x/s)‖²;
- on Gaussian data this lowers the error below absmax scaling, to about 0.74% relative MSE (`test_fp4_scale_search_beats_absmax_and_error_level`).

**Rotation.** An optional randomized Hadamard rotation Q = H diag(s)/16 is applied per head:
- q and k are rotated after RoPE, which is exact because qᵀk = (Qq)ᵀ(Qk);
- v is rotated too, and the output is un-rotated before the gate: o = Σ p_i v_i = Qᵀ Σ p_i (Q v_i);
- the un-rotation comes before the gate because the sigmoid output gate is elementwise and does not commute with Q.

With 16-channel groups plus the MSE search, plain fp4 already handles synthetic outlier channels well: outliers only affect their own group. So whether the rotation helps is measured on the real model (`tools/eval_quality.py`, variants `fp4` and `fp4-norot`).

**Kernels (`csrc/attn.cu`):**
- `attn_prep` runs the shared-memory FWHT (the same butterfly order as `hadamard.fwht`, so bit-identical), group absmax over a half-warp, the fp64 error comparison and the nibble pack.
- `attn_decode` dequantizes the 8 channels of a lane in registers. For FP4 that is 4 bytes of nibbles plus 1 scale byte per row. E2M1 is decoded through a `byte_perm` lookup of the fp16 high bytes. The last CTA un-rotates the combined output with an in-place shared-memory FWHT.
- Every rounding step is IEEE-exact (`__fdiv_rn`, `__frcp_rn`, `__fmul_rn`) despite `--use_fast_math`, so the cache bytes match `cckernel/kvq.py`.

**Prefill.** It quantizes into the cache first, then attends over the dequantized cache. Prefill, decode and verify therefore read identical values, and speculative verification stays exact (`test_fp4_speculative_verify_equals_greedy`).

**Measured on the real model** (`tools/eval_quality.py`, 8,704 positions against the fp32 reference, [`eval_kv.json`](eval_kv.json)).

KL on stable positions:

| KV cache | KL | Increase over bf16 KV |
|---|---|---|
| bf16 | 1.6e-3 | — |
| fp8 | 2.7e-3 | +1.1e-3 |
| k8v4 | 4.3e-3 | +2.7e-3 |
| fp4 | 6.8e-3 | +5.2e-3 |
| fp4 without rotation | 7.1e-3 | +5.5e-3 |

- **Perplexity** is within ±0.1 % throughout, and top-1 agreement falls from 98.3 % to 96.7 %.
- **Rotation:** it helps a little.
- **Range:** with rotation, no written value exceeds 30, far inside E4M3's 448.
- **Why the default is automatic:** post-training FP4 is not free, and KV reads are a small part of a decode step at short context. So `kvq.auto_format` keeps bf16 while the full cache would be at most 1/8 of the weight bytes, then steps down fp8 → k8v4 → fp4. For the 9B model the switch points are about 31K, 61K and 79K tokens.

**What is deliberately not quantized.** The 24 GDN states (48 MiB fp32) are the analogue of the paper's local SWA state, which it keeps at higher precision because it is "sensitive to quantization". They are read and written every token and are small next to the weights, so they stay fp32.

## 10. Confidence-scheduled verification (DSpark)

*DSpark: Confidence-Scheduled Speculative Decoding*, arXiv 2607.05147 (Alg. 1), as used by DeepSeek-V4.1-Flash.

**The objective.** A verify step of M = 1 + ℓ tokens costs T(M, ctx). It emits 1 + Σ_{j≤ℓ} a_j tokens in expectation, where:
- a_j = Π_{i≤j} c_i is the prefix-survival probability;
- c_i is the probability that draft token i is accepted, given tokens < i were.

The scheduler picks ℓ* = argmax (1 + Σ_{j≤ℓ} a_j) / T(1+ℓ, ctx).

**The step-cost curve T.**
- It is profiled by `Engine.profile_costs` as a + b·M + c·M·ctx.
- The c term exists because `attn_decode` reads the KV cache once per verified token. A smaller KV format therefore also makes verification cheaper at long context.
- Without a profile, the scheduler uses the 4060 Ti bandwidth model `StepCost.roofline`.

**Confidence without a trained head.**
- DSpark trains a confidence head and calibrates it with sequential temperature scaling. Our drafts come from n-gram lookup, so c_j is estimated online instead.
- It uses Beta-smoothed counts in a back-off hierarchy of buckets: (regime, match order, agreeing earlier occurrences, position) → … → position → global.
- Counts are censored after the first rejection.
- Empirical frequencies are calibrated by construction; the ECE is reported per run.

**Hindsight learning.** The counts are updated from *every* proposed draft, not only the verified part. Once its positions have been emitted, draft token j counts as accepted iff it and all earlier draft tokens match the emitted tokens.
- Under greedy decoding this is exactly the verification outcome.
- Under sampling it has the same probability p(d).
- It fixes a lock-in failure seen on the real model: after a run of misses, a scheduler that learns only from verified tokens stops verifying, and then never observes anything again.

**Exactness.** DSpark must stop admission early because its draft tokens are sampled, so a later confidence depends on an earlier sample. An n-gram draft and the confidence state are deterministic functions of the history. Any length rule computed from them before verification therefore leaves the output distribution unchanged, so the global argmax is allowed. `test_scheduled_sampling_is_exact` checks this: TV distance to the exact sequence distribution under temperature 1.

**Exact replay.** Under greedy decoding the output does not depend on the policy. `spec.replay` therefore derives, from one recorded output, the exact steps and verified drafts of every policy under any cost curve. It shares the same `SpecPolicy` code, and `test_replay_matches_policy_decisions` checks it.

## 11. Persistent prefix cache

This follows DeepSeek-V4.1-Flash sec. 3.2.1. The paper keeps its global KV for prefix reuse and snapshots the local state only at the end of the prompt and the end of the output; those are the points that regeneration and multi-turn requests hit.

Our hybrid has the same split:
- **Attention layers:** an append-only KV cache, stored in the engine's KV format, so fp4 snapshots are 3.6× smaller.
- **GDN layers:** a fixed 48 MiB recurrent state that can be snapshotted but not truncated.

**Implementation.**
- `Engine.session_state()` / `load_session_state()` save and restore the KV prefix, S, the conv ring and the deferred-commit buffers. Restore is done in place, so CUDA graphs stay valid.
- `PrefixCache` restores the longest snapshot whose tokens are a prefix of the request and prefills only the rest.
- An exact hit needs no prefill at all, because the snapshot stores the last-token logits.

## 12. CPU backend

`cckernel/cpu.py` runs the real 9B model on an AVX-512/AMX CPU, so every test and measurement in this repository could run without a GPU.

**Decode linears.** INT8 weights are kept group-major, int8 [K/128, N, 128]. The engine calls `aten._weight_int8pack_mm` on each 128-wide group with unit scales. The fp16 group scales are applied in fp32 as the groups are accumulated.

**Consequence for speculative decoding.** Every row is computed independently in a fixed order, so the logits of a verified token do not depend on M, bit for bit. Greedy speculative decoding therefore reproduces plain decoding exactly on the real model.

**Prefill.** Matrices are dequantized to bf16 and multiplied with AMX GEMMs. Short prompts go through the int8 decode path instead.

**Attention.** An fp32 mirror of the dequantized KV cache is kept incrementally, so a decode step does not re-decode the whole cache.
