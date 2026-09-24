#!/usr/bin/env python
"""Quantize MiMo-V2.6-Distill-Qwen-9B (or any Qwen3.5-style checkpoint) into the cck format.

Fits a 16 GB GPU while minimising quality loss:
  * norm folding + residual-stream random Hadamard rotation (exact, free at runtime),
  * group-128 symmetric INT-b with MSE-optimal clipping on near-Gaussian (rotated) weights,
  * per-matrix bitwidths from the linearity-theorem knapsack under a byte budget.

Presets (weights only; embedding kept in bf16 = 2.0 GB, KV cache/state come on top):
  quality   every matrix INT8                        ~8.1 GB + 2.0 GB   (near-lossless, default)
  balanced  knapsack, avg ~6.5 bit, lm_head >= 8     ~6.6 GB + 2.0 GB
  fast      knapsack, avg ~5.25 bit, lm_head >= 6    ~5.4 GB + 2.0 GB

Example:
  python tools/quantize.py --model /models/MiMo-V2.6-Distill-Qwen-9B --out /models/mimo-cck-q8 --device cuda
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cckernel import quant  # noqa: E402
from cckernel.alloc import CHOICES, allocate, predicted_loss, prior_alphas  # noqa: E402
from cckernel.config import TextConfig  # noqa: E402
from cckernel.hadamard import random_signs  # noqa: E402
from cckernel.loader import Checkpoint  # noqa: E402
from cckernel.quant_io import write_cck  # noqa: E402

PRESETS = {
    "quality": dict(avg_bits=None, uniform=8, lm_head_min=8),
    "balanced": dict(avg_bits=6.5, uniform=None, lm_head_min=8),
    "fast": dict(avg_bits=5.25, uniform=None, lm_head_min=6),
}


def layer_mats(cfg, i, get, signs, device):
    return quant.fold_layer(cfg, i, lambda n: get(n).to(device), signs.to(device) if signs is not None else None)


def measure_t2(cfg, ckpt, signs, device, clip_grid) -> dict[str, dict[int, float]]:
    """Relative quantization MSE t^2(b) of every matrix at every candidate bitwidth."""
    t2 = {}
    for i in range(cfg.num_hidden_layers):
        mats = layer_mats(cfg, i, ckpt.get, signs, device)
        for k, w in mats.items():
            if k not in quant.LINEAR_NAMES:
                continue
            name = f"layers.{i}.{k}"
            t2[name] = {b: quant.rel_mse(w, quant.dequant_rtn(*quant.quantize_rtn(w, b, clip_grid=clip_grid), b))
                        for b in CHOICES}
        print(f"  t2 layer {i:2d}: " + ", ".join(f"{k}={t2[f'layers.{i}.{k}'][8]:.2e}@8b" for k in mats
                                                  if k in quant.LINEAR_NAMES), flush=True)
    g = quant.fold_globals(cfg, lambda n: ckpt.get(n).to(device), signs.to(device) if signs is not None else None)
    lm = g["lm_head"]
    t2["lm_head"] = {b: quant.rel_mse(lm, quant.dequant_rtn(*quant.quantize_rtn(lm, b, clip_grid=clip_grid), b))
                     for b in CHOICES}
    return t2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF checkpoint directory (config.json + safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="quality")
    ap.add_argument("--avg-bits", type=float, default=None, help="override the knapsack target (bits/weight)")
    ap.add_argument("--bits-json", default=None, help="explicit {matrix: bits} assignment")
    ap.add_argument("--alphas", default=None, help="JSON of calibrated alpha_l (tools/calibrate_alpha.py)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="Hadamard sign seed")
    ap.add_argument("--no-rotate", action="store_true", help="disable the residual Hadamard rotation")
    ap.add_argument("--clip-grid", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    src, out = Path(args.model), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = TextConfig.from_hf(src)
    ckpt = Checkpoint(src)
    signs = None if args.no_rotate else random_signs(cfg.hidden_size, args.seed)
    preset = PRESETS[args.preset]
    mats = quant.matrix_list(cfg)

    # ---------------------------------------------------------------- bit assignment
    t2 = None
    if args.bits_json:
        bits = json.loads(Path(args.bits_json).read_text())
    elif preset["uniform"] and args.avg_bits is None:
        bits = {name: preset["uniform"] for name, _, _ in mats}
    else:
        print("measuring t^2(b) for the knapsack ...", flush=True)
        with torch.no_grad():
            t2 = measure_t2(cfg, ckpt, signs, args.device, args.clip_grid)
        alphas = json.loads(Path(args.alphas).read_text()) if args.alphas else prior_alphas(cfg)
        numel = {name: N * K for name, N, K in mats}
        target = args.avg_bits or preset["avg_bits"]
        budget = sum(numel.values()) * quant.bits_per_weight(target) / 8
        items = [(name, numel[name], t2[name], alphas[name]) for name, _, _ in mats]
        bits = allocate(items, budget, min_bits={"lm_head": preset["lm_head_min"]})
        print(f"knapsack: target {target} bpw, predicted sum(alpha t^2) = {predicted_loss(items, bits):.4e} "
              f"(uniform-8: {predicted_loss(items, {n: 8 for n in bits}):.4e})")

    # ---------------------------------------------------------------- quantize + write
    extra = {"preset": args.preset, "source": str(src)}
    if t2 is not None:
        extra["t2_table"] = t2
    manifest = write_cck(cfg, ckpt.get, out, bits, seed=None if args.no_rotate else args.seed, device=args.device,
                         clip_grid=args.clip_grid, extra=extra, log=lambda m: print(m, flush=True))
    total_bytes, stats = manifest["quantized_weight_bytes"], manifest["stats"]
    for f in src.iterdir():  # tokenizer / chat template / generation config
        if f.suffix in (".json", ".jinja", ".txt", ".model") and "safetensors" not in f.name:
            if f.name not in ("config.json",):
                shutil.copy(f, out / f.name)
    shutil.copy(src / "config.json", out / "hf_config.json")
    avg = sum(s["bits"] * s["N"] * s["K"] for s in stats.values()) / sum(s["N"] * s["K"] for s in stats.values())
    print(f"done in {time.time() - t0:.0f}s: quantized weights {total_bytes / 2**30:.2f} GiB "
          f"(avg {avg:.2f} bits + scales), embedding {manifest['embed_bytes'] / 2**30:.2f} GiB -> {out}")


if __name__ == "__main__":
    main()
