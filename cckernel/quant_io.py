"""Write a folded + quantized checkpoint in the cck-v1 format (used by tools/quantize.py and tests).

Layout of the output directory:
  cck_manifest.json          config, bit map, Hadamard seed, per-matrix t^2
  layer-XX.safetensors       packed planes + scales of the layer's fused matrices, small fp32 tensors
  globals.safetensors        lm_head (packed) and the rotated bf16 embedding
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch

from . import quant
from .config import TextConfig
from .hadamard import random_signs


def write_cck(cfg: TextConfig, get: Callable[[str], torch.Tensor], out: str | Path, bits: dict[str, int],
              seed: int | None = 0, device="cpu", clip_grid: int = 20, extra: dict | None = None,
              log: Callable[[str], None] | None = None) -> dict:
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signs = random_signs(cfg.hidden_size, seed).to(device) if seed is not None else None
    dget = lambda n: get(n).to(device)  # noqa: E731
    stats, total = {}, 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(cfg.num_hidden_layers):
            folded = quant.fold_layer(cfg, i, dget, signs)
            sd = {}
            for k, w in folded.items():
                name = f"layers.{i}.{k}"
                if k in quant.LINEAR_NAMES:
                    ql = quant.QLinear.from_weight(w, bits[name], clip_grid=clip_grid)
                    stats[name] = {"bits": bits[name], "t2": quant.rel_mse(w, ql.dequant()), "N": ql.N, "K": ql.K}
                    sd.update({kk: vv.cpu().contiguous() for kk, vv in ql.state(name).items()})
                    total += ql.nbytes()
                else:
                    sd[name] = w.float().cpu().contiguous()
            save_file(sd, str(out / f"layer-{i:02d}.safetensors"))
            if log:
                log(f"layer {i:2d} {cfg.layer_types[i]:17s} " + " ".join(
                    f"{k}:{bits[f'layers.{i}.{k}']}b" for k in folded if k in quant.LINEAR_NAMES)
                    + f"  [{time.time() - t0:6.0f}s]")
        g = quant.fold_globals(cfg, dget, signs)
        ql = quant.QLinear.from_weight(g["lm_head"], bits["lm_head"], clip_grid=clip_grid)
        stats["lm_head"] = {"bits": bits["lm_head"], "t2": quant.rel_mse(g["lm_head"], ql.dequant()), "N": ql.N, "K": ql.K}
        total += ql.nbytes()
        sd = {k: v.cpu().contiguous() for k, v in ql.state("lm_head").items()}
        sd["embed"] = g["embed"].to(torch.bfloat16).cpu().contiguous()
        save_file(sd, str(out / "globals.safetensors"))
    manifest = {
        "format": "cck-v1",
        "config": cfg.to_dict(),
        "bits": bits,
        "group_size": quant.GROUP,
        "hadamard_seed": seed,
        "stats": stats,
        "quantized_weight_bytes": total,
        "embed_bytes": cfg.vocab_size * cfg.hidden_size * 2,
    }
    manifest.update(extra or {})
    (out / "cck_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
