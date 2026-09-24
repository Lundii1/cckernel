# cckernel

These are CUDA kernels and a lean runtime for **[XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)**, which is Qwen3.5-9B with hybrid Gated DeltaNet and gated attention layers. They target an **RTX 4060 Ti 16 GB** (Ada, sm_89). The model is quantized so that it fits in 16 GB with minimal quality loss. Every kernel is derived from a paper; see [`docs/MATH.md`](docs/MATH.md).

## What's inside

| Piece | Where | Idea (paper) |
|---|---|---|
| Norm + residual-Hadamard folding | `cckernel/quant.py`, `hadamard.py` | Incoherence processing is exact and free at runtime (QuIP# 2402.04396, QuaRot) |
| INT4/5/6/8 group-128 quantizer with MSE clipping | `quant.py` | Gaussian-optimal grids after rotation (HIGGS 2411.17525) |
| Per-matrix bit allocation under a VRAM budget | `alloc.py`, `tools/calibrate_alpha.py` | Linearity theorem plus an exact knapsack (HIGGS) |
| Dequant-fused GEMV with a fused RMSNorm / gated-norm prologue and residual / SwiGLU epilogue | `csrc/qgemv.cu` | Streaming loads, magic-number dequant (Marlin 2408.11743) |
| Gated DeltaNet decode, reading and writing S once per token | `csrc/gdn.cu` | One-pass delta rule, column-split state (Gated DeltaNet 2412.06464) |
| Deferred GDN commit for speculative rollback, no state snapshots | `csrc/gdn.cu` | TreeWY 2608.20961 / ReplaySSM |
| GQA split-KV decode attention with in-kernel combine and σ-gate | `csrc/attn.cu` | Flash-decoding, gated attention 2505.06708 |
| Tensor-core skinny GEMM (2..8 tokens) for verification | `csrc/skinny.cu` | Flat GEMM (FlashDecoding++ 2311.01282) |
| Chunked WY/UT prefill | `cckernel/torch_ops.py` | DeltaNet chunkwise algorithm (2406.06484) |
| N-gram speculative decoding with exact acceptance | `cckernel/spec.py` | Speculative sampling with a point-mass draft |
| CUDA-graph decode engine, layer-major prefill | `cckernel/engine.py` | |

## Quick start (Linux or Windows, RTX 40xx)

```bash
pip install torch safetensors transformers    # a CUDA toolkit matching your torch build is required to compile
pip install -e .                              # builds cckernel._C for sm_89

# 1. quantize once: INT8 "quality" recipe with a built-in accuracy report (KL / top-1 / perplexity vs bf16).
#    Streams tensors straight from the Hub with HTTP range requests, so the 18.8 GB shards are never stored;
#    peak RAM is a few GB. Use --model DIR instead of --hf-repo for a local copy, --device cuda to speed it up.
python tools/quantize_stream.py --hf-repo XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B --out /models/mimo-cck-q8

# 2. chat / generate (add --spec for n-gram speculative decoding)
python -m cckernel.generate --model /models/mimo-cck-q8 --chat --spec --prompt "Write a C function that reverses a linked list."

# 3. check and measure on your GPU
pytest tests                                  # CPU math + emulated engine; tests/gpu run the real kernels
python bench/kernels.py                       # per-kernel GB/s vs 288 GB/s
python bench/decode.py --model /models/mimo-cck-q8 --ctx 512 8192 32768
```

## Fitting 16 GB with minimal quality loss

| Preset | Bits | Weights | + embedding (bf16) | Decode ceiling at 288 GB/s |
|---|---|---|---|---|
| `quality` (default) | INT8 everywhere, Hadamard-rotated | 7.5 GiB | 9.4 GiB | 35.7 tok/s |
| `balanced` | knapsack at 6.5 bpw, lm_head ≥ 8 | ≈6.2 GiB | ≈8.1 GiB | ≈43 tok/s |
| `fast` | knapsack at 5.25 bpw, lm_head ≥ 6 | ≈5.0 GiB | ≈6.9 GiB | ≈54 tok/s |

On top of the weights come the KV cache (1 GiB per 32K tokens, since only 8 layers carry one), the GDN state (48 MiB) and about 0.8 GiB of scratch and CUDA context. `quality` therefore runs a 128K context inside 16 GB.

`quality_report.json`, written next to the checkpoint, records the measured loss. Both sides of the comparison are fp32, so it isolates the weight-quantization error. It contains:
- per-matrix t²;
- the relative residual-stream error after every layer;
- mean and max KL(p_bf16 ‖ p_quant);
- top-1 agreement;
- reference vs quantized perplexity on WikiText-2 and code.

The `balanced` and `fast` presets use `tools/quantize.py`. For a measured, rather than prior-based, allocation, calibrate α on the INT8 model first:

```bash
python tools/calibrate_alpha.py --model /models/mimo-cck-q8 --text some_corpus.txt --out alphas.json
python tools/quantize.py --model /models/MiMo-V2.6-Distill-Qwen-9B --out /models/mimo-cck-b --preset balanced --alphas alphas.json
```

## Validation status

- **In the development container (no GPU): 37 CPU tests pass.**
  - They cover the three gated-delta-rule algorithms and the deferred commit, to 1e-9.
  - Folding the norms and the Hadamard rotation leaves the model's function unchanged, to 1e-6.
  - The packing layout and the emulated GEMV and mma-fragment decode are exact.
  - The quantizer error follows theory, and the knapsack matches brute force.
  - N-gram acceptance is exact.
  - The **full engine** runs through a torch emulation of every CUDA op. That covers prefill, decode, speculative verify/rollback (identical to greedy) and multi-turn prefill after decode.
- **The CUDA sources compile for sm_89** with nvcc 13.0, with no register spills, and the extension links against torch.
- **Not yet run on a GPU.** `tests/gpu` checks every kernel against its emulation, plus the GPU engine against the CPU engine and optional HF parity (`CCK_HF_MODEL=... CCK_MODEL=... pytest tests/gpu/test_hf_parity.py -s`). The benchmarks also still need a GPU run.
- **The real model has been quantized** with `quality` (INT8, rotated, MSE clip), streamed from the Hub. The run took 27.5 min on 4 CPU cores and fetched 16.7 GiB. The full report is [`docs/quality_report_q8.json`](docs/quality_report_q8.json). On 512 eval tokens (256 WikiText-2 + 256 code), measured against the fp32 reference of the original bf16 weights:

  | Metric | Value |
  |---|---|
  | Weights / embedding | 7.51 GiB INT8 / 1.89 GiB bf16 |
  | Per-matrix t² (mean / max) | 4.3e-5 / 5.7e-5 |
  | KL(p_bf16 ‖ p_int8), mean / max | 8.4e-4 / 4.6e-2 nats |
  | Top-1 next-token agreement | 98.05 % (10 / 512 tokens differ) |
  | Perplexity, WikiText-2 | 16.14 → 16.11 |
  | Perplexity, code | 4.80 → 4.82 |
  | Perplexity, combined | 8.803 → 8.811 (+0.09 %) |
  | Residual-stream error after the last layer | 2.2 % (prose), 3.6 % (code) |

## Layout

```
cckernel/    config, loader, reference (fp32 oracle), quant + packing, alloc, folded model,
             torch_ops (prefill), emu (CUDA-op emulation), engine, spec, generate
csrc/        common.cuh, qgemv.cu, gdn.cu, attn.cu, skinny.cu, bindings.cpp
tools/       quantize.py, calibrate_alpha.py
tests/cpu    math + emulated end-to-end tests      tests/gpu   kernel/engine/HF parity
bench/       kernels.py, decode.py
docs/        MATH.md
```
