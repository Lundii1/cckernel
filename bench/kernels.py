#!/usr/bin/env python
"""Per-kernel timing and effective DRAM bandwidth at the model's real shapes (run on the GPU).

  python bench/kernels.py            # all bit widths
  python bench/kernels.py --bits 8 6
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cckernel import _C, quant  # noqa: E402

SHAPES = {  # (N, K) of the fused matrices of MiMo-V2.6-Distill-Qwen-9B
    "gdn in_proj": (12352, 4096), "gdn out_proj": (4096, 4096), "attn qkv": (10240, 4096), "attn o_proj": (4096, 4096),
    "mlp gate_up": (24576, 4096), "mlp down": (4096, 12288), "lm_head": (248320, 4096),
}


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 5, 6, 8])
    ap.add_argument("--peak-gbs", type=float, default=288.0, help="RTX 4060 Ti: 288 GB/s")
    args = ap.parse_args()
    dev = torch.device("cuda")
    print(f"{'matrix':14s} {'bits':>4s} {'M':>2s} {'us':>9s} {'GB/s':>7s} {'%peak':>6s}")
    for bits in args.bits:
        for name, (N, K) in SHAPES.items():
            w = torch.randn(N, K, device=dev) * 0.02
            q = quant.QLinear.from_weight(w, bits, clip_grid=1)
            del w
            lo = (q.planes["q8"] if bits == 8 else q.planes["lo"]).to(dev)
            hi = q.planes["hi"].to(dev) if bits in (5, 6) else torch.empty(0, dtype=torch.uint8, device=dev)
            sc = q.scales.to(dev)
            nbytes = lo.numel() + hi.numel() + sc.numel() * 2
            x = torch.randn(K, device=dev)
            y = torch.empty(N, device=dev)
            t = timeit(lambda: _C.qgemv(lo, hi, sc, bits, N, K, 1, x, x, 128, 1e-6, 1, y))
            print(f"{name:14s} {bits:4d} {1:2d} {t * 1e6:9.1f} {nbytes / t / 1e9:7.1f} {nbytes / t / 1e9 / args.peak_gbs * 100:5.1f}%")
            xb = torch.randn(8, K, device=dev).to(torch.bfloat16)
            yb = torch.empty(8, N, device=dev)
            for M in (4, 8):
                t = timeit(lambda: _C.qgemm_skinny(lo, hi, sc, bits, N, K, xb, M, 1, yb))
                print(f"{name:14s} {bits:4d} {M:2d} {t * 1e6:9.1f} {nbytes / t / 1e9:7.1f} {nbytes / t / 1e9 / args.peak_gbs * 100:5.1f}%")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
