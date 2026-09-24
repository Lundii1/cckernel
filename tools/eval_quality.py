#!/usr/bin/env python
"""Accuracy of the quantized model (and of each KV-cache format) against the original bf16 model.

Two steps, so the expensive reference is computed once:

  # 1. reference: stream the original bf16 weights from the Hub (HTTP ranges, nothing stored),
  #    run the fp32 HF-semantics forward layer-major over the eval sequences and keep compact
  #    statistics per position: top-128 logits + indices, logsumexp, NLL of the true next token.
  python tools/eval_quality.py reference --hf-repo XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B \\
      --cck-model /models/mimo-cck-q8 --out ref.pt

  # 2. score the cckernel engine (the real prefill path: dequantized bf16 GEMMs, quantized KV
  #    cache written then read back) for each KV variant against the reference.
  python tools/eval_quality.py score --cck-model /models/mimo-cck-q8 --ref ref.pt \\
      --variants bf16 fp8 k8v4 fp4 fp4-norot --out docs/eval_kv.json

Eval sequences: 2 x 256 tokens (WikiText-2 test + code; the set of the original quality report) and
2 x 4096 tokens of the same sources (long context, where KV quantization errors accumulate).

KL(p_ref || p_q) is computed over the reference top-128 tokens plus one bucket for the remaining
mass (a lower bound of the full KL by the log-sum inequality; the top-128 hold > 99% of the mass
almost everywhere, and the bucket mass is reported).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cckernel import kvq  # noqa: E402
from cckernel.config import TextConfig  # noqa: E402
from cckernel.reference import RefCache, RefModel, rmsnorm_zc  # noqa: E402

TOPK = 128


def log(msg):
    print(msg, flush=True)


def build_sequences(cck_model: Path, n_short: int, n_long: int, vocab: int) -> dict[str, list[list[int]]]:
    from quantize_stream import eval_sequences

    tj = cck_model / "tokenizer.json"
    tok_json = tj.read_bytes() if tj.exists() else None
    return {"short": eval_sequences(tok_json, n_short, None, vocab), "long": eval_sequences(tok_json, n_long, None, vocab)}


# ------------------------------------------------------------------------------------------ reference
def cmd_reference(args):
    t0 = time.time()
    cck = Path(args.cck_model)
    cfg = TextConfig.from_dict(json.loads((cck / "cck_manifest.json").read_text())["config"])
    if args.model:
        from cckernel.loader import Checkpoint

        ck = Checkpoint(Path(args.model))
    else:
        from cckernel.remote import RemoteCheckpoint

        ck = RemoteCheckpoint(args.hf_repo, args.revision)
    sets = build_sequences(cck, args.n_short, args.n_long, cfg.vocab_size)
    seqs = [s for name in ("short", "long") for s in sets[name]]
    log(f"sequences: {[len(s) for s in seqs]} tokens")
    V, eps = cfg.vocab_size, cfg.rms_norm_eps

    need = sorted({t for s in seqs for t in s})
    rows: dict[int, torch.Tensor] = {}
    ROW = 16384
    for r0 in range(0, V, ROW):
        r1 = min(V, r0 + ROW)
        want = [t for t in need if r0 <= t < r1]
        if want:
            chunk = ck.get_rows("embed_tokens.weight", r0, r1, dtype=None)
            for t in want:
                rows[t] = chunk[t - r0].float()
    h = [torch.stack([rows[t] for t in s]) for s in seqs]
    del rows
    log(f"embedding rows gathered [{time.time() - t0:.0f}s]")

    keys_of = lambda i: [k for k in ck.keys() if k.startswith(f"layers.{i}.")]  # noqa: E731
    fetch = lambda i: {k: ck.get(k, dtype=None) for k in keys_of(i)}  # noqa: E731
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(fetch, 0)
    for i in range(cfg.num_hidden_layers):
        tl = time.time()
        wl = nxt.result()
        if i + 1 < cfg.num_hidden_layers:
            nxt = pool.submit(fetch, i + 1)
        p = f"layers.{i}."
        with torch.no_grad():
            ref = RefModel(cfg, {k: v.float() for k, v in wl.items()}, gdn_algo="chunked")
            for j in range(len(h)):
                pos = torch.arange(h[j].shape[0])
                x = rmsnorm_zc(h[j], ref.W(p + "input_layernorm.weight"), eps)
                if cfg.layer_types[i] == "linear_attention":
                    h[j] = h[j] + ref.gdn(i, x, RefCache(cfg))
                else:
                    h[j] = h[j] + ref.attn(i, x, RefCache(cfg), pos)
                x = rmsnorm_zc(h[j], ref.W(p + "post_attention_layernorm.weight"), eps)
                h[j] = h[j] + ref.mlp(i, x)
        wl = ref = None
        log(f"layer {i:2d} {cfg.layer_types[i]:17s} [{time.time() - tl:.0f}s / {time.time() - t0:.0f}s]")
    pool.shutdown()

    # lm_head in row chunks with running logsumexp / top-k / true-token logits
    xr = torch.cat([rmsnorm_zc(x, ck.get("norm.weight"), eps) for x in h])  # [T, d]
    offs = [0]
    for s in seqs:
        offs.append(offs[-1] + len(s))
    Tt = xr.shape[0]
    nxt_tok = torch.full((Tt,), -1, dtype=torch.long)
    for j, s in enumerate(seqs):
        nxt_tok[offs[j]:offs[j + 1] - 1] = torch.tensor(s[1:])
    lse = torch.full((Tt,), float("-inf"))
    topv = torch.full((Tt, TOPK), float("-inf"))
    topi = torch.zeros((Tt, TOPK), dtype=torch.long)
    true_logit = torch.zeros(Tt)
    for r0 in range(0, V, ROW):
        r1 = min(V, r0 + ROW)
        W = ck.get_rows("lm_head.weight", r0, r1)  # fp32
        lg = xr @ W.t()  # [T, rows] fp32
        lse = torch.logaddexp(lse, torch.logsumexp(lg, -1))
        cv, ci = lg.topk(min(TOPK, r1 - r0), dim=-1)
        allv, alli = torch.cat([topv, cv], 1), torch.cat([topi, ci + r0], 1)
        topv, sel = allv.topk(TOPK, dim=-1)
        topi = alli.gather(1, sel)
        m = (nxt_tok >= r0) & (nxt_tok < r1)
        true_logit[m] = lg[m, nxt_tok[m] - r0]
        del lg, W
    log(f"lm_head done [{time.time() - t0:.0f}s]")
    torch.save({"seqs": seqs, "sets": {"short": [0, 1], "long": [2, 3]}, "offs": offs, "lse": lse,
                "topv": topv, "topi": topi.to(torch.int32), "true_logit": true_logit, "next": nxt_tok,
                "source": args.hf_repo or args.model}, args.out)
    log(f"reference saved to {args.out} [{time.time() - t0:.0f}s]")


# ------------------------------------------------------------------------------------------ scoring
def metrics_for(ref, j: int, lp: torch.Tensor, start: int) -> dict:
    """Per-position metrics for positions [start, start+len(lp)) of sequence j (lp: log-probs)."""
    o = ref["offs"][j] + start
    n = lp.shape[0]
    lse, topv, topi = ref["lse"][o:o + n], ref["topv"][o:o + n], ref["topi"][o:o + n].long()
    lr = topv - lse[:, None]  # reference log-probs of its top-128
    pr = lr.exp()
    lq = lp.gather(1, topi)
    pr_tail = (1 - pr.sum(-1)).clamp_min(1e-12)
    pq_tail = (1 - lq.exp().sum(-1)).clamp_min(1e-12)
    kl = (pr * (lr - lq)).sum(-1) + pr_tail * (pr_tail.log() - pq_tail.log())
    agree = (lp.argmax(-1) == topi[:, 0]).float()
    nx = ref["next"][o:o + n]
    valid = nx >= 0
    nll_q = -lp[valid].gather(1, nx[valid][:, None])[:, 0]
    nll_r = -(ref["true_logit"][o:o + n][valid] - lse[valid])
    return {"kl": kl, "agree": agree, "nll_q": nll_q, "nll_r": nll_r, "tail": pr_tail}


def summarize(parts: list[dict]) -> dict:
    cat = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    return {"tokens": int(cat["kl"].numel()), "kl_mean": float(cat["kl"].mean()), "kl_p99": float(cat["kl"].quantile(0.99)),
            "kl_max": float(cat["kl"].max()), "top1_agreement": float(cat["agree"].mean()),
            "ppl_ref": float(cat["nll_r"].mean().exp()), "ppl_quant": float(cat["nll_q"].mean().exp()),
            "ref_tail_mass_mean": float(cat["tail"].mean())}


VARIANTS = {"bf16": ("bf16", False), "fp8": ("fp8", True), "k8v4": ("k8v4", True), "fp4": ("fp4", True),
            "fp4-norot": ("fp4", False), "fp8-norot": ("fp8", False)}


def cmd_score(args):
    from cckernel.engine import Engine

    ref = torch.load(args.ref, weights_only=False)
    seqs = ref["seqs"]
    eng = Engine(args.cck_model, device=args.device, max_len=max(len(s) for s in seqs) + 16,
                 prefill_chunk=args.prefill_chunk, kv_format="bf16")
    log(eng.vram_report())
    out = {"reference": ref.get("source"), "sequences": {k: [len(seqs[j]) for j in v] for k, v in ref["sets"].items()},
           "variants": {}}
    if Path(args.out).exists():
        out = json.loads(Path(args.out).read_text())
    for name in args.variants:
        if name in out["variants"] and not args.force:
            log(f"{name}: already scored, skipping")
            continue
        fmt, rot = VARIANTS[name]
        eng.set_kv_format(fmt, rot)
        kvq.STATS = {}
        t0 = time.time()
        per_seq = []
        for j, s in enumerate(seqs):
            parts = []
            eng.score(s, fn=lambda st, lp, j=j, parts=parts: parts.append(metrics_for(ref, j, lp, st)))
            per_seq.append(parts)
        res = {}
        for set_name, idx in ref["sets"].items():
            res[set_name] = summarize([p for j in idx for p in per_seq[j]])
        res["all"] = summarize([p for ps in per_seq for p in ps])
        res["kv_bytes_per_token"] = eng.kv_bytes_per_token()
        res["max_abs_written"] = {("bf16", "fp8", "fp4")[k]: v for k, v in kvq.STATS.items()}
        res["seconds"] = time.time() - t0
        kvq.STATS = None
        out["variants"][name] = res
        log(f"{name:10s} " + " | ".join(
            f"{k}: KL {v['kl_mean']:.2e} top1 {v['top1_agreement'] * 100:.2f}% ppl {v['ppl_ref']:.3f}->{v['ppl_quant']:.3f}"
            for k, v in res.items() if isinstance(v, dict) and "kl_mean" in v) + f"  [{res['seconds']:.0f}s]")
        Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"written {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--hf-repo", default="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B")
    r.add_argument("--revision", default="main")
    r.add_argument("--model", default=None, help="local HF checkpoint dir instead of streaming")
    r.add_argument("--cck-model", required=True, help="cck checkpoint (config + tokenizer)")
    r.add_argument("--n-short", type=int, default=256)
    r.add_argument("--n-long", type=int, default=4096)
    r.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--cck-model", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--variants", nargs="+", default=["bf16", "fp8", "k8v4", "fp4", "fp4-norot"],
                   choices=sorted(VARIANTS))
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.add_argument("--prefill-chunk", type=int, default=2048)
    s.add_argument("--force", action="store_true")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    (cmd_reference if args.cmd == "reference" else cmd_score)(args)


if __name__ == "__main__":
    main()
