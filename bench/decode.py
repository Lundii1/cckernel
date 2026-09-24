#!/usr/bin/env python
"""End-to-end decode and prefill throughput of a quantized model.

  python bench/decode.py --model /models/mimo-cck-q8 --ctx 512 8192 32768 --steps 128
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cckernel.engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[512, 8192, 32768])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--verify-m", type=int, nargs="+", default=[4, 8], help="also time M-token verify steps")
    args = ap.parse_args()
    eng = Engine(args.model, max_len=max(args.ctx) + args.steps + 16)
    print(eng.vram_report())
    g = torch.Generator().manual_seed(0)
    for ctx in args.ctx:
        eng.reset()
        ids = torch.randint(0, eng.cfg.vocab_size, (ctx,), generator=g).tolist()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tok = int(eng.prefill(ids).argmax())
        torch.cuda.synchronize()
        tp = time.perf_counter() - t0
        eng.step([tok]); eng.commit(1)  # capture graph outside the timed region
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            tok = int(eng.step([tok])[0].argmax())
            eng.commit(1)
        torch.cuda.synchronize()
        td = (time.perf_counter() - t0) / args.steps
        line = f"ctx {ctx:6d}: prefill {ctx / tp:8.0f} tok/s ({tp:6.2f} s) | decode {1 / td:6.1f} tok/s ({td * 1e3:6.2f} ms/tok)"
        for M in args.verify_m:
            eng.step([tok] * M)  # warm/capture
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(16):
                eng.step([tok] * M)
            torch.cuda.synchronize()
            line += f" | verify M={M}: {(time.perf_counter() - t0) / 16 * 1e3:6.2f} ms"
        print(line, flush=True)


if __name__ == "__main__":
    main()
