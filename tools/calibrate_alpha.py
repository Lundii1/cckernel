#!/usr/bin/env python
"""Measure the linearity-theorem coefficients alpha_l (HIGGS, arXiv 2411.17525, Alg. 3).

For every quantized matrix W_l, inject Gaussian noise of relative size t (||dW|| = t ||W||),
measure the mean KL(p_ref || p_noisy) over calibration tokens, and fit KL ~= alpha_l * t^2.
Run it on a near-lossless INT8 (``--preset quality``) model on the GPU; feed the JSON to
``tools/quantize.py --alphas``. With --random-tokens it is data-free (as in HIGGS).

Example:
  python tools/calibrate_alpha.py --model /models/mimo-cck-q8 --text calib.txt --out alphas.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cckernel import quant  # noqa: E402
from cckernel.engine import Engine  # noqa: E402


def calib_batches(args, eng: Engine):
    g = torch.Generator().manual_seed(0)
    if args.text:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        ids = tok(Path(args.text).read_text())["input_ids"]
        n = args.seq_len
        return [ids[i:i + n] for i in range(0, max(1, len(ids) - n), n)][: args.num_seqs]
    return [torch.randint(0, eng.cfg.vocab_size, (args.seq_len,), generator=g).tolist() for _ in range(args.num_seqs)]


def swap(eng: Engine, name: str, w: torch.Tensor, bits: int):
    q = quant.QLinear.from_weight(w, bits, clip_grid=8)
    old = eng.lin[name]
    sd = q.state(name)
    eng.lin[name] = eng._qlin(sd, name, bits, (q.N, q.K))
    return old


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="cck model directory (preferably the INT8 'quality' preset)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=None, help="calibration text file (default: random tokens)")
    ap.add_argument("--num-seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.03, 0.06])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    eng = Engine(args.model, device=args.device, max_len=args.seq_len + 8, use_graphs=False)
    batches = calib_batches(args, eng)
    refs = [eng.score(b) for b in batches]
    alphas = {}
    t0 = time.time()
    names = [n for n, _, _ in quant.matrix_list(eng.cfg)]
    gen = torch.Generator(device=eng.dev).manual_seed(1)
    for j, name in enumerate(names):
        W = eng.lin[name].dequant(torch.empty(eng.lin[name].N * eng.lin[name].K, dtype=torch.bfloat16,
                                              device=eng.dev)).float()
        wn = W.norm()
        xs, ys = [], []
        for t in args.levels:
            noise = torch.randn(W.shape, generator=gen, device=eng.dev)
            Wn = W + noise * (t * wn / noise.norm())
            old = swap(eng, name, Wn, 8)
            kl = 0.0
            for b, ref in zip(batches, refs):
                lp = eng.score(b)
                kl += float((ref.exp() * (ref - lp)).sum(-1).mean())
            eng.lin[name] = old
            xs.append(t * t)
            ys.append(kl / len(batches))
        x, y = torch.tensor(xs), torch.tensor(ys)
        alphas[name] = max(float((x * y).sum() / (x * x).sum()), 1e-6)  # least squares through the origin
        print(f"[{j + 1:3d}/{len(names)}] {name:24s} alpha={alphas[name]:.4g}  KL@t={args.levels[-1]}: {ys[-1]:.3e}"
              f"  [{time.time() - t0:5.0f}s]", flush=True)
        del W
    Path(args.out).write_text(json.dumps(alphas, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
