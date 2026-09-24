#!/usr/bin/env python
"""Bandwidth model of decode on a 16 GB card per KV-cache format: bytes per token, tok/s ceiling and
whether the context fits in VRAM. Modeled, not measured (bench/decode.py measures on a GPU).

  python tools/kv_roofline.py --model /models/mimo-cck-q8 [--bandwidth 288e9 --vram-gib 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cckernel import kvq  # noqa: E402
from cckernel.config import TextConfig  # noqa: E402


def table(man: dict, bandwidth: float, efficiency: float, vram_gib: float, ctxs, formats) -> dict:
    cfg = TextConfig.from_dict(man["config"])
    w = man["quantized_weight_bytes"]
    emb = man["embed_bytes"]
    state = 24 * 32 * 128 * 128 * 4  # GDN state, read + written each token
    overhead = 0.8 * 2**30  # CUDA context, scratch, step buffers
    out = {}
    for fmt in formats:
        bpt = kvq.kv_bytes_per_token(fmt, len(cfg.attn_layers), cfg.num_key_value_heads, cfg.head_dim)
        rows = {}
        for ctx in ctxs:
            step_bytes = w + 2 * state + ctx * bpt
            vram = w + emb + state + ctx * bpt + overhead
            rows[ctx] = {"bytes_per_step": step_bytes, "tok_s": bandwidth * efficiency / step_bytes,
                         "vram_gib": vram / 2**30, "fits": vram / 2**30 <= vram_gib}
        max_ctx = int((vram_gib * 2**30 - (w + emb + state + overhead)) / bpt)
        out[fmt] = {"kv_bytes_per_token": bpt, "rows": rows, "max_context": min(max_ctx, 262144)}  # max_position_embeddings
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bandwidth", type=float, default=288e9)
    ap.add_argument("--efficiency", type=float, default=0.8, help="achieved fraction of peak bandwidth")
    ap.add_argument("--vram-gib", type=float, default=15.5, help="usable VRAM (16 GB card minus driver reserve)")
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 32768, 131072, 262144])
    ap.add_argument("--formats", nargs="+", default=["bf16", "fp8", "k8v4", "fp4"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    man = json.loads((Path(args.model) / "cck_manifest.json").read_text())
    t = table(man, args.bandwidth, args.efficiency, args.vram_gib, args.ctx, args.formats)
    print(f"| KV format | bytes/token | max context in {args.vram_gib:g} GiB | "
          + " | ".join(f"{c // 1024}K ctx tok/s" for c in args.ctx) + " |")
    print("|---|---|---|" + "---|" * len(args.ctx))
    for fmt, r in t.items():
        cells = [f"{r['rows'][c]['tok_s']:.1f}" + ("" if r["rows"][c]["fits"] else " (OOM)") for c in args.ctx]
        print(f"| {fmt} | {r['kv_bytes_per_token']:,} | {r['max_context']:,} | " + " | ".join(cells) + " |")
    if args.json:
        Path(args.json).write_text(json.dumps({"bandwidth": args.bandwidth, "efficiency": args.efficiency,
                                               "vram_gib": args.vram_gib, "table": t}, indent=1))


if __name__ == "__main__":
    main()
