"""Write a folded + quantized checkpoint in the cck-v1 format (used by the tools and tests).

Layout of the output directory:
  cck_manifest.json          config, bit map, Hadamard seed, per-matrix t^2
  layer-XX.safetensors       packed planes + scales of the layer's fused matrices, small fp32 tensors
  globals.safetensors        lm_head (packed) and the rotated bf16 embedding

Everything is streamed: one decoder layer at a time, and the 248320-row embedding / lm_head in row
chunks, so peak host memory stays a few GB for the 9B model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch

from . import kvq, quant
from .config import TextConfig
from .hadamard import random_signs

ROW_CHUNK = 16384


def quantize_layer(cfg: TextConfig, i: int, get: Callable[[str], torch.Tensor], signs, bits: dict[str, int],
                   device="cpu", clip_grid: int = 20) -> dict:
    """Fold + quantize decoder layer i. Returns {short name: QLinear (on CPU) | small fp32 tensor}."""
    folded = quant.fold_layer(cfg, i, lambda n: get(n).to(device), signs)
    out = {}
    for k, w in folded.items():
        if k in quant.LINEAR_NAMES:
            out[k] = quant.QLinear.from_weight(w, bits[f"layers.{i}.{k}"], clip_grid=clip_grid).to("cpu")
        else:
            out[k] = w.float().cpu().contiguous()
    return out


def layer_state(i: int, parts: dict) -> dict[str, torch.Tensor]:
    sd = {}
    for k, v in parts.items():
        name = f"layers.{i}.{k}"
        if isinstance(v, quant.QLinear):
            sd.update({kk: vv.contiguous() for kk, vv in v.state(name).items()})
        else:
            sd[name] = v
    return sd


def rotate_embedding(cfg: TextConfig, get_rows: Callable[[str, int, int], torch.Tensor], signs, device="cpu",
                     row_chunk: int = ROW_CHUNK, on_chunk: Callable | None = None) -> torch.Tensor:
    """Row-chunked embedding rotation -> bf16 [V, d] on CPU. ``on_chunk(r0, r1, raw_rows)``."""
    V, d = cfg.vocab_size, cfg.hidden_size
    emb = torch.empty(V, d, dtype=torch.bfloat16)
    for r0 in range(0, V, row_chunk):
        r1 = min(V, r0 + row_chunk)
        raw = get_rows("embed_tokens.weight", r0, r1).to(device)
        if on_chunk is not None:
            on_chunk(r0, r1, raw)
        emb[r0:r1] = quant.fold_embed_rows(raw, signs).to(torch.bfloat16).cpu()
    return emb


def quantize_lm_head(cfg: TextConfig, get_rows: Callable[[str, int, int], torch.Tensor], final_norm: torch.Tensor,
                     signs, bits: int, device="cpu", clip_grid: int = 20, row_chunk: int = ROW_CHUNK,
                     on_chunk: Callable | None = None) -> quant.QLinear:
    """Row-chunked lm_head fold + quantization. ``on_chunk(r0, r1, raw_rows, ql_chunk)``."""
    parts = []
    for r0 in range(0, cfg.vocab_size, row_chunk):
        r1 = min(cfg.vocab_size, r0 + row_chunk)
        raw = get_rows("lm_head.weight", r0, r1).to(device)
        ql = quant.QLinear.from_weight(quant.fold_lm_head_rows(raw, final_norm.to(device), signs), bits,
                                       clip_grid=clip_grid).to("cpu")
        if on_chunk is not None:
            on_chunk(r0, r1, raw, ql)
        parts.append(ql)
        del raw
    return quant.QLinear.concat(parts)


def quantize_globals(cfg: TextConfig, get_rows: Callable[[str, int, int], torch.Tensor], final_norm: torch.Tensor,
                     signs, lm_bits: int, device="cpu", clip_grid: int = 20, row_chunk: int = ROW_CHUNK,
                     on_chunk: Callable | None = None):
    """(rotated bf16 embedding, packed lm_head), both streamed in row chunks."""
    emb = rotate_embedding(cfg, get_rows, signs, device, row_chunk)
    return emb, quantize_lm_head(cfg, get_rows, final_norm, signs, lm_bits, device, clip_grid, row_chunk, on_chunk)


def manifest_dict(cfg: TextConfig, bits, seed, stats, total, extra=None) -> dict:
    m = {
        "format": "cck-v1",
        "config": cfg.to_dict(),
        "bits": bits,
        "group_size": quant.GROUP,
        "hadamard_seed": seed,
        "stats": stats,
        "quantized_weight_bytes": total,
        "embed_bytes": cfg.vocab_size * cfg.hidden_size * 2,
        # runtime defaults read by the engine (overridable per run: Engine(kv_format=...), generate --kv)
        "runtime": {"kv_cache": kvq.DEFAULT_KV, "kv_rotate_seed": kvq.KV_SEED, "spec_policy": "sched"},
    }
    m.update(extra or {})
    return m


def write_cck(cfg: TextConfig, get: Callable[[str], torch.Tensor], out: str | Path, bits: dict[str, int],
              seed: int | None = 0, device="cpu", clip_grid: int = 20, extra: dict | None = None,
              log: Callable[[str], None] | None = None,
              get_rows: Callable[[str, int, int], torch.Tensor] | None = None) -> dict:
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signs = random_signs(cfg.hidden_size, seed).to(device) if seed is not None else None
    if get_rows is None:  # small models: slice full tensors
        get_rows = lambda n, a, b: get(n)[a:b]  # noqa: E731
    stats, total = {}, 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(cfg.num_hidden_layers):
            parts = quantize_layer(cfg, i, get, signs, bits, device, clip_grid)
            for k, v in parts.items():
                if isinstance(v, quant.QLinear):
                    stats[f"layers.{i}.{k}"] = {"bits": v.bits, "t2": v.t2, "N": v.N, "K": v.K}
                    total += v.nbytes()
            save_file(layer_state(i, parts), str(out / f"layer-{i:02d}.safetensors"))
            if log:
                log(f"layer {i:2d} {cfg.layer_types[i]:17s} " + " ".join(
                    f"{k}:{v.bits}b(t2={v.t2:.1e})" for k, v in parts.items() if isinstance(v, quant.QLinear))
                    + f"  [{time.time() - t0:6.0f}s]")
            del parts
        emb, lm = quantize_globals(cfg, get_rows, get("norm.weight"), signs, bits["lm_head"], device, clip_grid)
        stats["lm_head"] = {"bits": lm.bits, "t2": lm.t2, "N": lm.N, "K": lm.K}
        total += lm.nbytes()
        sd = {k: v.contiguous() for k, v in lm.state("lm_head").items()}
        sd["embed"] = emb
        save_file(sd, str(out / "globals.safetensors"))
    manifest = manifest_dict(cfg, bits, seed, stats, total, extra)
    (out / "cck_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
