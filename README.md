# cckernel

CUDA kernels and a lean runtime for **[XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)**. The model is Qwen3.5-9B with hybrid Gated DeltaNet and gated attention layers.

- **Hardware:** an **RTX 4060 Ti 16 GB** (Ada, sm_89).
- **Quantization:** the model is quantized to fit in 16 GB with minimal quality loss.
- **Papers:** every kernel is derived from a paper; see [`docs/MATH.md`](docs/MATH.md).
- **DeepSeek-V4.1-Flash:** the parts of arXiv 2609.19969 that work without retraining are integrated, and their accuracy and speed are measured (see [below](#deepseek-v41-flash-integration-arxiv-260919969)).

## What's inside

| Piece | Where | Idea (paper) |
|---|---|---|
| Norm + residual-Hadamard folding | `cckernel/quant.py`, `hadamard.py` | Incoherence processing is exact and free at runtime (QuIP# 2402.04396, QuaRot) |
| INT4/5/6/8 group-128 quantizer with MSE clipping | `quant.py` | Gaussian-optimal grids after rotation (HIGGS 2411.17525) |
| Per-matrix bit allocation under a VRAM budget | `alloc.py`, `tools/calibrate_alpha.py` | Linearity theorem plus an exact knapsack (HIGGS) |
| Dequant-fused GEMV with a fused RMSNorm / gated-norm prologue and residual / SwiGLU epilogue | `csrc/qgemv.cu` | Streaming loads, magic-number dequant (Marlin 2408.11743) |
| Gated DeltaNet decode, reading and writing S once per token | `csrc/gdn.cu` | One-pass delta rule, column-split state (Gated DeltaNet 2412.06464) |
| Deferred GDN commit for speculative rollback, with no state snapshots | `csrc/gdn.cu` | TreeWY 2608.20961 / ReplaySSM |
| GQA split-KV decode attention with in-kernel dequant, combine and σ-gate | `csrc/attn.cu` | Flash-decoding, gated attention 2505.06708 |
| **FP4 / FP8 / K8V4 KV cache** with an MSE scale search and an optional per-head Hadamard rotation | `cckernel/kvq.py`, `csrc/attn.cu` | **DeepSeek-V4.1-Flash 2609.19969 §2.4.4** (FP4 = E2M1 + E4M3 per 16 channels) |
| **Confidence-scheduled speculative verification** | `cckernel/spec.py` | **DSpark 2607.05147**, used by DeepSeek-V4.1-Flash |
| **Persistent prefix cache** (KV prefix + recurrent-state snapshots) | `cckernel/prefix_cache.py` | **DeepSeek-V4.1-Flash §3.2.1** |
| Tensor-core skinny GEMM (2–8 tokens) for verification | `csrc/skinny.cu` | Flat GEMM (FlashDecoding++ 2311.01282) |
| Chunked WY/UT prefill | `cckernel/torch_ops.py` | DeltaNet chunkwise algorithm (2406.06484) |
| N-gram speculative decoding with exact acceptance | `cckernel/spec.py` | Speculative sampling with a point-mass draft |
| CUDA-graph decode engine, layer-major prefill | `cckernel/engine.py` | |
| **CPU backend** that runs the 9B model (int8 GEMV, AMX prefill) | `cckernel/cpu.py` | Used for every measurement in this README |

## Quick start (Linux or Windows, RTX 40xx)

```bash
pip install torch safetensors transformers    # a CUDA toolkit matching your torch build is required to compile
pip install -e .                              # builds cckernel._C for sm_89

# 1. quantize once: INT8 "quality" recipe with a built-in accuracy report (KL / top-1 / perplexity vs bf16).
#    Streams tensors straight from the Hub with HTTP range requests, so the 18.8 GB shards are never stored.
python tools/quantize_stream.py --hf-repo XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B --out /models/mimo-cck-q8

# 2. chat / generate
#    --spec            n-gram speculative decoding with confidence-scheduled verification
#    --kv auto         KV cache format: bf16 for short contexts, fp8 / k8v4 / fp4 as --max-len grows
#    --prefix-cache    reuse snapshots of earlier prompts / answers instead of prefilling them again
python -m cckernel.generate --model /models/mimo-cck-q8 --chat --spec --kv auto --max-len 131072 \
    --prefix-cache /tmp/cck_prefix --prompt "Write a C function that reverses a linked list."

# 3. check and measure on your GPU
pytest tests                                  # CPU math + emulated engine; tests/gpu run the real kernels
python bench/kernels.py                       # per-kernel GB/s vs 288 GB/s
python bench/decode.py --model /models/mimo-cck-q8 --ctx 512 8192 32768 --kv bf16 fp4 --profile
python tools/eval_generate.py --model /models/mimo-cck-q8 --device cuda --out eval_gpu.json   # coherence + speed
```

With no GPU, add `--device cpu` to `generate` or the tools. This uses the CPU backend, which needs about 11 GB of RAM and a CPU with AVX-512 / AMX. On the 4-core Xeon used here it decodes at about 1.5 tok/s.

## Fitting 16 GB with minimal quality loss

| Preset | Bits | Weights | + embedding (bf16) | Decode ceiling at 288 GB/s |
|---|---|---|---|---|
| `quality` (default) | INT8 everywhere, Hadamard-rotated | 7.5 GiB | 9.4 GiB | 35.7 tok/s |
| `balanced` | knapsack at 6.5 bpw, lm_head ≥ 8 | ≈6.2 GiB | ≈8.1 GiB | ≈43 tok/s |
| `fast` | knapsack at 5.25 bpw, lm_head ≥ 6 | ≈5.0 GiB | ≈6.9 GiB | ≈54 tok/s |

On top of the weights come:
- the KV cache, which depends on the format (below);
- the GDN state, 48 MiB;
- about 0.8 GiB of scratch space and CUDA context.

`quality_report.json`, written next to the checkpoint, records the measured weight-quantization loss.

## DeepSeek-V4.1-Flash integration (arXiv 2609.19969)

**What transfers.** Most of the paper needs pre-training or QAT: the causal encoder-decoder, CSA2 cross-layer KV reuse, Engram, the trained DSpark drafter and QAT itself. Three ideas apply to an already-trained model:

1. **Quantized KV cache (§2.4.4).**
   - The paper's FP4 format is E2M1 values with one E4M3 scale per 16 channels, quantized after RoPE.
   - FP8 (E4M3) and K8V4 (K fp8, V fp4) are provided as well.
   - Two changes replace the missing QAT: an MSE search over the E4M3 scale codes, and an optional per-head Hadamard rotation. The rotation is exact for qᵀk; the output is un-rotated before the gate.
   - The CUDA kernels encode and decode bit-identically to `cckernel/kvq.py`.
   - The GDN recurrent state stays fp32. It plays the role of the paper's sensitive local state, which the paper keeps at higher precision.
2. **DSpark confidence-scheduled verification (§2.4.3).** For each step, the verification length ℓ maximises (1 + Σ a_j) / T(1+ℓ, ctx), where:
   - a_j are prefix-survival probabilities;
   - T is the step time, profiled per machine or taken from the 4060 Ti bandwidth model.

   DSpark's trained confidence head is replaced by online counts for our n-gram drafts. They learn in hindsight from every draft, and they stay exact because they only read the past.
3. **Persistent prefix cache (§3.2.1).** Snapshots are taken at the end of the prompt and the end of the answer. A later request restores the longest cached prefix and prefills only the rest.

All numbers below were measured in this repository's container on CPU with the real engine: 4-core Xeon with AMX, **no GPU**. GPU speed is **modeled** (bandwidth roofline) wherever it says so. [`docs/eval_kv.json`](docs/eval_kv.json), [`docs/eval_generate.json`](docs/eval_generate.json) and [`docs/samples/`](docs/samples) hold the raw data.

### Accuracy vs the original bf16 model

The reference is fp32 with HF semantics, run on the original bf16 weights streamed from the Hub. The eval set is 8,704 positions: 2 × 256 tokens plus 2 × 4096 tokens of WikiText-2 and code.

**11 positions are left out of the stable columns.** They sit in a repetitive list in the long prose sequence, where copy heads are on a knife-edge. There, the reference *and* every quantized variant each predict confidently wrong tokens (e.g. " members" after "bass ("), at different positions. Those positions dominate the plain mean KL.

| KV cache | Bytes/token | KL vs original (stable positions) | KL vs bf16-KV engine | Top-1 agreement | Perplexity (original 5.226) |
|---|---|---|---|---|---|
| bf16 (before) | 32,768 | 1.6e-3 | 0 | 98.3 % | 5.216 |
| fp8 | 16,384 | 2.7e-3 | 1.3e-3 | 97.9 % | 5.211 |
| k8v4 | 12,800 | 4.3e-3 | 3.7e-3 | 97.4 % | 5.219 |
| **fp4 (paper format)** | **9,216** | 6.8e-3 | 1.0e-2 | 96.7 % | 5.232 |
| fp4, no rotation | 9,216 | 7.1e-3 | 1.1e-2 | 96.6 % | 5.233 |

**Findings:**
- **Perplexity does not change measurably**, at most +0.1 %.
- **The token-level distribution does change.** Without the paper's QAT, FP4 moves about 1.6 % of top-1 predictions.
- **The rotation helps a little.**
- **Rule fixed before measuring:** KL at most +5e-4 and top-1 at most −0.5 points over bf16 KV. No quantized format passes it (fp8: +1.1e-3 KL, −0.4 points top-1).

That, together with the speed table, is why the default is **`--kv auto`**:
- bf16 up to about 31K tokens, where KV reads are under 12.5 % of a step's traffic, so compressing them would buy about 1–9 % speed at an accuracy cost;
- fp8 to about 61K tokens, k8v4 to about 79K, and fp4 beyond, where it pays off.

### Speed and memory

**Modeled RTX 4060 Ti decode ceiling** (288 GB/s at 80 % efficiency, `tools/kv_roofline.py`):

| KV cache | Longest context that fits in 15.5 GiB | 4K ctx | 32K ctx | 128K ctx | 256K ctx |
|---|---|---|---|---|---|
| bf16 | 172K | 27.8 tok/s | 25.0 | 18.5 | out of memory |
| fp8 | 262K (full) | 28.0 | 26.5 | 22.4 | 18.5 |
| fp4 | 262K (full) | 28.1 | 27.2 (+9 %) | 24.6 (+33 %) | 21.8 |

**Speculative decoding** is measured as the exact replay of each policy on the recorded greedy outputs of the 15-prompt suite. Greedy output does not depend on the policy; on this backend a verified token's logits match plain decoding bit for bit.

| Step-cost model | Old EWMA drafter | Always verify all | **Confidence-scheduled** |
|---|---|---|---|
| This CPU (measured T(M), verifying 8 tokens = 2.9 × one step) | 1.08× | 0.77× | **1.14×** |
| 4060 Ti model, short context | 1.50× | 1.51× | **1.53×** |
| 4060 Ti model, +32K context, bf16 KV | 1.27× | 1.04× | **1.28×** |
| 4060 Ti model, +32K context, fp4 KV | 1.42× | 1.33× | **1.43×** |

(Speedup over no speculation, bf16-KV outputs. On the fp4-KV outputs, the scheduler is at 1.19×, 1.61×, 1.32× and 1.48×.)

**What this shows:**
- The scheduler matches the old heuristic where verification is almost free, and avoids its losses where verification is expensive (this CPU; long contexts).
- FP4 KV makes verification cheaper at long context. Each verified token reads the KV cache, so FP4 lifts speculation from 1.28× to 1.43×.

**Measured on this CPU with the real engine** (hindsight-learning scheduler, [`docs/eval_validate.json`](docs/eval_validate.json)):
- Speculative output was token-for-token identical to the no-speculation run on every prompt.
- The replayed step count matched the real run exactly (8/8).

| Prompt | Tokens / step | Decode speed | vs no speculation |
|---|---|---|---|
| Code edit (add type hints: copies the input) | 2.75 (fp4) / 2.42 (bf16) | 2.78 / 2.62 tok/s | **1.84× / 1.73×** |
| Arithmetic word problem | 1.49 / 1.53 | 1.76 / 1.95 tok/s | 1.15× / 1.27× |
| One-sentence summary | 1.52 / 1.50 | 1.68 / 1.75 tok/s | 1.12× / 1.16× |
| FizzBuzz (little to copy) | 1.16 / 1.18 | 1.54 / 1.58 tok/s | 1.02× / 1.05× |


**Prefix cache.** In the two-turn chat test, turn 2 restored 46 of its 70 prompt tokens from the snapshot and prefilled only the remaining 24. A cached 6K-token document would save about 100 s of CPU prefill per request here, or about 1–2 s on a 4060 Ti.

### Coherence and output quality

This is the 15-prompt suite in `tools/eval_generate.py`: chat template, greedy decoding, thinking off except for 2 prompts. Every check is automatic:
- math answers;
- generated Python run against unit tests;
- JSON validity;
- facts;
- a two-turn memory question through the prefix cache;
- reasoning prompts;
- degeneration metrics.

| Configuration | Passed | Repeated 4-grams | Distinct-2 |
|---|---|---|---|
| bf16 KV, no speculation (before) | 14 / 15 | 0.186 | 0.831 |
| fp4 KV + scheduled speculation (new) | 14 / 15 | 0.188 | 0.832 |

**How the two runs compare:**
- 12 of the 15 outputs are word-for-word identical between the two configurations. The other 3 differ in wording and still pass. For example, the code edit uses built-in generics instead of `typing`.
- **The one failure is the same in both:**
  - The JSON prompt returned `{"name": "Ada Lovelace", "age": 34, "languages": ["Python", "C++", "Go"]` without the closing brace.
  - At the last list item, the next-token probabilities were a three-way tie: `"]` 0.36, `",` 0.32 and `"]}` 0.30. Greedy decoding took `"]`.
  - After that, the closing brace was spread over `}</`, `` }` ``, `}"` and `}` (about 0.40 together), so end-of-message won at 0.41.
  - This is a greedy-decoding quirk that INT8 noise can tip either way, not a KV-cache effect.

**Needle in a haystack.** A passphrase was inserted at 10 %, 50 % and 90 % depth of a 6,043-token WikiText-2 document. It was retrieved 3/3 with bf16 KV and 3/3 with fp4 KV.

Full transcripts: [`docs/samples/`](docs/samples).

## Compared with llama.cpp

The baseline is running the same model with llama.cpp, using the official GGUF `ggml-org/MiMo-V2.6-Distill-Qwen-9B-GGUF`. Q8_0 is the same quality class as our INT8. Script: [`bench/vs_llamacpp.py`](bench/vs_llamacpp.py).

**Measured on this machine's CPU** (4-core Xeon with AMX, llama.cpp build 97a418b, [`docs/vs_llamacpp_cpu.json`](docs/vs_llamacpp_cpu.json)):

| | llama.cpp Q8_0 | cckernel INT8 |
|---|---|---|
| Raw decode, 0 / 4K context | 4.6 / 4.5 tok/s (f16 KV); 4.8 / 4.7 (q8_0 KV) | 1.5 / 1.4 tok/s (bf16 KV); 1.5 / 1.5 (fp4 KV) |
| 15-prompt suite, no speculation | 5.2 tok/s, 15/15 pass | 1.5 tok/s, 14/15 pass |
| 15-prompt suite with n-gram speculation | 6.0 tok/s (`--spec-type ngram-simple`, 1.16×) | 1.8 tok/s (confidence-scheduled, 1.23×) |
| Tokens per step with speculation (hardware independent) | 1.21 (it drafts only on the code-edit prompt) | ≈1.45 (it also drafts on math and prose) |

- **On CPU, llama.cpp is about 3× faster.** Its hand-tuned AVX-512/AMX int8 kernels beat this repository's CPU backend, which exists to test and measure the model without a GPU. For CPU inference, use llama.cpp.
- **Both speculators are exact.** On this suite ours gets more tokens per step, but the gain is content-dependent.
- **Quality:** llama.cpp Q8_0 also passes the JSON prompt, where our INT8 build hits a three-way near-tie (see above).

**Modeled on an RTX 4060 Ti** (bandwidth roofline, `tools/kv_roofline.py` method). Both engines are assumed to reach 80 % of 288 GB/s. This is **not measured**: cckernel's CUDA kernels have not run on a GPU yet, while llama.cpp's are mature, so treat the cckernel columns as an upper bound.

| Decode tok/s, one sequence | 4K context | 32K context | 128K context |
|---|---|---|---|
| llama.cpp BF16 GGUF (the unquantized model) | 14.3 (15.6 GiB before KV, so it does not really fit a 16 GB card) | 13.5 | does not fit |
| llama.cpp Q8_0, f16 KV (default) | 26.5 | 23.9 | 17.9 |
| llama.cpp Q8_0, q8_0 / q4_0 KV (`-ctk/-ctv`) | 26.7 / 26.8 | 25.3 / 26.0 | 21.3 / 23.6 |
| **cckernel INT8, `--kv auto`** | **27.8** (bf16 KV) | **26.5** (fp8 KV) | **24.6** (fp4 KV) |

With speculation on the 15-prompt mix (replayed with a 4060 Ti cost model):
- **cckernel:** 1.53× at short context and about 1.4× at 32K.
- **llama.cpp `ngram-simple`:** about 1.2× (1.21 tokens/step with nearly free verification).

So the expected end-to-end gain for this prompt mix:

| cckernel INT8 + speculation vs … | Short context | 32K context |
|---|---|---|
| llama.cpp BF16 | ≈3× | ≈3×, if BF16 fits at all |
| llama.cpp Q8_0 | ≈1.6× | ≈1.55× (f16 KV) |
| llama.cpp Q8_0 + its n-gram speculation | ≈1.3× | ≈1.3× |

Raw decode alone is only 1.05× faster than Q8_0 at 4K. That comes from 8.1 instead of 8.5 bits per weight, and is within kernel-efficiency noise. The real GPU numbers come from:

```bash
python bench/vs_llamacpp.py --gguf MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf --llama-bin ~/llama.cpp/build/bin \
    --model /models/mimo-cck-q8 --ctx 0 8192 32768 --out vs_llamacpp_gpu.json
```

<!--MMLU-->

## Validation status

- **70 CPU tests pass.** New since the DeepSeek integration:
  - KV formats against brute force;
  - scale search and rotation;
  - the engine with a quantized cache against the oracle;
  - exact speculative verification with the fp4 cache;
  - CPU backend parity and M-invariance;
  - scheduler argmax, censoring and sampling exactness;
  - replay against live runs;
  - session and prefix-cache round trips.

  Earlier suites: the gated-delta-rule algorithms, folding, packing, quantizer and knapsack, and the emulated end-to-end engine.
- **The CUDA sources compile for sm_89** with nvcc 13.0 and no register spills. All 8 attention kernel variants and the extension link.
- **The real 9B model runs end to end on the CPU backend.** That covers every accuracy, coherence, retrieval and speed number above.
- **Not yet run on a GPU.** `tests/gpu` checks every kernel, including each KV format byte for byte, against its emulation. The benchmarks and `tools/eval_generate.py --device cuda` produce the real GPU numbers.
- **Weight quantization** (`quality`, INT8, rotated, MSE clip): [`docs/quality_report_q8.json`](docs/quality_report_q8.json), measured on 512 tokens against the fp32 reference:
  - top-1 agreement 98.05 %;
  - KL 8.4e-4;
  - perplexity 8.803 → 8.811.

## Layout

```
cckernel/    config, loader, reference (fp32 oracle), quant + packing, alloc, folded model,
             torch_ops (prefill), kvq (KV formats), emu (CUDA-op emulation), cpu (CPU backend),
             engine, spec (drafter + scheduler + replay), prefix_cache, generate
csrc/        common.cuh, qgemv.cu, gdn.cu, attn.cu, skinny.cu, bindings.cpp
tools/       quantize.py, quantize_stream.py, calibrate_alpha.py, eval_quality.py, eval_generate.py, eval_mmlu_pro.py,
             kv_roofline.py
tests/cpu    math + emulated end-to-end tests      tests/gpu   kernel/engine/HF parity
bench/       kernels.py, decode.py, vs_llamacpp.py
docs/        MATH.md, eval_kv.json, eval_generate.json, kv_roofline.json, samples/
```
