#!/usr/bin/env python
"""Streamed, memory-bounded quantization of the real checkpoint with a built-in accuracy report.

One layer-major pass over the checkpoint (local directory or the Hugging Face Hub via HTTP range
requests: the 18.8 GB of shards are never stored). For every decoder layer it
  (a) fetches the bf16 tensors,
  (b) advances an fp32 reference forward (HF semantics) on the evaluation tokens,
  (c) folds norms + the residual Hadamard rotation and quantizes (default: INT8, group 128, MSE clip),
      writing layer-XX.safetensors,
  (d) advances the same tokens through the folded model on the *dequantized* weights,
  (e) records the relative residual error ||Q^T h_q - h_ref|| / ||h_ref||.
The embedding is rotated first (its eval rows seed both forwards) and lm_head is quantized last in row
chunks while computing both sets of logits: KL(p_ref || p_q), top-1 agreement and perplexity are
reported. Both forwards run in fp32, so the report isolates the weight-quantization loss.

  python tools/quantize_stream.py --hf-repo XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B --out /models/mimo-cck-q8
  python tools/quantize_stream.py --model /models/MiMo-V2.6-Distill-Qwen-9B --out /models/mimo-cck-q8 --device cuda
Peak host memory is a few GB (one layer + the bf16 embedding + two [T, V] logit tables).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cckernel import quant  # noqa: E402
from cckernel.config import TextConfig  # noqa: E402
from cckernel.folded import FoldedCache, FoldedModel, rms  # noqa: E402
from cckernel.hadamard import fwht, random_signs  # noqa: E402
from cckernel.quant_io import layer_state, manifest_dict, quantize_layer, quantize_lm_head, rotate_embedding  # noqa: E402
from cckernel.reference import RefCache, RefModel, rmsnorm_zc  # noqa: E402

WIKITEXT = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
SMALL_EXT = (".json", ".jinja", ".txt", ".model")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ source
class Source:
    """Uniform access to a local checkpoint directory or a Hub repository."""

    def __init__(self, model: str | None, hf_repo: str | None, revision: str, endpoint: str = "https://huggingface.co"):
        if hf_repo:
            from cckernel.remote import RemoteCheckpoint

            self.ck = RemoteCheckpoint(hf_repo, revision=revision, endpoint=endpoint)
            self.remote = True
            self.config_raw = json.loads(self.ck.fetch("config.json"))
        else:
            from cckernel.loader import Checkpoint

            self.ck = Checkpoint(model)
            self.remote = False
            self.dir = Path(model)
            self.config_raw = json.loads((self.dir / "config.json").read_text())

    def small_files(self) -> dict[str, bytes]:
        if self.remote:
            names = [f for f in self.ck.list_files()
                     if f.endswith(SMALL_EXT) and "/" not in f and not f.startswith("model.safetensors")]
            out = {}
            for f in names:
                try:
                    out[f] = self.ck.fetch(f)
                except Exception as e:  # noqa: BLE001
                    log(f"  (skip {f}: {e})")
            return out
        return {f.name: f.read_bytes() for f in self.dir.iterdir()
                if f.suffix in SMALL_EXT and not f.name.startswith("model.safetensors")}


def config_from_raw(raw: dict, tmp: Path) -> TextConfig:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "config.json").write_text(json.dumps(raw))
    return TextConfig.from_hf(tmp / "config.json")


# ------------------------------------------------------------------------------------------------ eval set
def eval_sequences(tokenizer_json: bytes | None, n_tok: int, text_file: str | None, vocab: int) -> list[list[int]]:
    """Two sequences of n_tok tokens: natural text (WikiText-2 test, or a file) and code (this repo)."""
    if tokenizer_json is None:  # tests with fake checkpoints: deterministic pseudo-tokens
        g = torch.Generator().manual_seed(0)
        return [torch.randint(0, vocab, (n_tok,), generator=g).tolist() for _ in range(2)]
    root = Path(__file__).resolve().parents[1]
    code = "\n\n".join(p.read_text() for p in sorted((root / "cckernel").glob("*.py")))
    if text_file:
        prose = Path(text_file).read_text()
    else:
        prose = None
        try:
            import io

            import pyarrow.parquet as pq

            with urllib.request.urlopen(urllib.request.Request(WIKITEXT, headers={"User-Agent": "cckernel"}),
                                        timeout=60) as r:
                tbl = pq.read_table(io.BytesIO(r.read()))
            prose = "".join(tbl.column("text").to_pylist()[:2000])
            log("eval text: WikiText-2 test")
        except Exception as e:  # noqa: BLE001
            log(f"eval text: WikiText-2 unavailable ({type(e).__name__}); using docs/MATH.md")
        if not prose:
            prose = (root / "docs" / "MATH.md").read_text()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(tokenizer_json.decode())
    seqs = []
    for text in (prose, code):
        ids = tok.encode(text).ids
        start = min(len(ids) // 4, max(0, len(ids) - n_tok))
        seqs.append(ids[start:start + n_tok])
    return seqs


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src_g = ap.add_mutually_exclusive_group(required=True)
    src_g.add_argument("--model", help="local HF checkpoint directory")
    src_g.add_argument("--hf-repo", help="Hub repository id, streamed with range requests")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--endpoint", default="https://huggingface.co", help=argparse.SUPPRESS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=8, choices=quant.SUPPORTED_BITS, help="uniform bitwidth (quality=8)")
    ap.add_argument("--bits-json", default=None, help="per-matrix bit map (overrides --bits)")
    ap.add_argument("--seed", type=int, default=0, help="Hadamard sign seed")
    ap.add_argument("--clip-grid", type=int, default=20)
    ap.add_argument("--eval-tokens", type=int, default=256, help="tokens per eval sequence (2 sequences)")
    ap.add_argument("--eval-text", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--upload", default=os.environ.get("CCK_UPLOAD_REPO"), help="Hub repo to upload the result to")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Source(args.model, args.hf_repo, args.revision, args.endpoint)
    cfg = config_from_raw(src.config_raw, out / ".tmp")
    ck = src.ck
    dev = torch.device(args.device)
    mats = quant.matrix_list(cfg)
    bits = json.loads(Path(args.bits_json).read_text()) if args.bits_json else {n: args.bits for n, _, _ in mats}
    signs = random_signs(cfg.hidden_size, args.seed)
    signs_d = signs.to(dev)
    small = src.small_files()
    for name, data in small.items():
        (out / ("hf_config.json" if name == "config.json" else name)).write_bytes(data)
    seqs = eval_sequences(small.get("tokenizer.json"), args.eval_tokens, args.eval_text, cfg.vocab_size)
    log(f"{cfg.num_hidden_layers} layers, vocab {cfg.vocab_size}; eval: {[len(s) for s in seqs]} tokens; device {dev}")

    # ---- embedding (rotated, streamed) + eval rows for both forwards
    need = sorted({t for s in seqs for t in s})
    rows_raw: dict[int, torch.Tensor] = {}

    def grab(r0, r1, raw):
        for t in need:
            if r0 <= t < r1:
                rows_raw[t] = raw[t - r0].float().cpu()

    with torch.no_grad():
        emb = rotate_embedding(cfg, lambda n, a, b: ck.get_rows(n, a, b, dtype=None), signs_d, dev, on_chunk=grab)
    h_ref = [torch.stack([rows_raw[t] for t in s]) for s in seqs]
    h_q = [emb[torch.tensor(s)].float() for s in seqs]  # the engine uses the bf16 rotated embedding
    log(f"embedding rotated ({emb.numel() * 2 / 2**30:.2f} GiB bf16) [{time.time() - t0:.0f}s]")

    # ---- layer-major pass with prefetch of the next layer
    eps = cfg.rms_norm_eps
    keys_of = lambda i: [k for k in ck.keys() if k.startswith(f"layers.{i}.")]  # noqa: E731
    fetch = lambda i: {k: ck.get(k, dtype=None) for k in keys_of(i)}  # noqa: E731
    stats, total, layer_err = {}, 0, []
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(fetch, 0)
    for i in range(cfg.num_hidden_layers):
        tl = time.time()
        wl = nxt.result()
        if i + 1 < cfg.num_hidden_layers:
            nxt = pool.submit(fetch, i + 1)
        with torch.no_grad():
            # (c) quantize + write
            parts = quantize_layer(cfg, i, lambda n: wl[n], signs_d, bits, dev, args.clip_grid)
            for k, v in parts.items():
                if isinstance(v, quant.QLinear):
                    stats[f"layers.{i}.{k}"] = {"bits": v.bits, "t2": v.t2, "N": v.N, "K": v.K}
                    total += v.nbytes()
            from safetensors.torch import save_file

            save_file(layer_state(i, parts), str(out / f"layer-{i:02d}.safetensors"))
            # (b) reference forward (fp32, HF semantics)
            p = f"layers.{i}."
            ref = RefModel(cfg, {k: v.float() for k, v in wl.items()}, gdn_algo="chunked")
            # (d) folded forward on dequantized weights
            ft = {p + k: (v.dequant() if isinstance(v, quant.QLinear) else v) for k, v in parts.items()}
            fm = FoldedModel(cfg, ft, gdn_algo="chunked")
            errs = []
            for j in range(len(seqs)):
                T = h_ref[j].shape[0]
                pos = torch.arange(T)
                x = rmsnorm_zc(h_ref[j], ref.W(p + "input_layernorm.weight"), eps)
                if cfg.layer_types[i] == "linear_attention":
                    h_ref[j] = h_ref[j] + ref.gdn(i, x, RefCache(cfg))
                else:
                    h_ref[j] = h_ref[j] + ref.attn(i, x, RefCache(cfg), pos)
                x = rmsnorm_zc(h_ref[j], ref.W(p + "post_attention_layernorm.weight"), eps)
                h_ref[j] = h_ref[j] + ref.mlp(i, x)
                xq = rms(h_q[j], eps)
                if cfg.layer_types[i] == "linear_attention":
                    h_q[j] = h_q[j] + fm.gdn(i, xq, FoldedCache())
                else:
                    h_q[j] = h_q[j] + fm.attn(i, xq, FoldedCache(), pos)
                h_q[j] = h_q[j] + fm.mlp(i, rms(h_q[j], eps))
                back = fwht(h_q[j]) * signs  # Q^T h_q
                errs.append(float((back - h_ref[j]).norm() / h_ref[j].norm()))
            layer_err.append(errs)
        wl = ref = fm = ft = parts = None  # free the layer before the next one arrives
        log(f"layer {i:2d} {cfg.layer_types[i]:17s} t2="
            + ",".join(f"{stats[n]['t2']:.1e}" for n in stats if n.startswith(p))
            + f"  resid err {', '.join(f'{e:.2e}' for e in errs)}  [{time.time() - tl:.0f}s / {time.time() - t0:.0f}s]")
    pool.shutdown()

    # ---- lm_head: quantize in row chunks and compute both logit tables
    final_norm = ck.get("norm.weight")
    xr = [rmsnorm_zc(h, final_norm, eps) for h in h_ref]
    xq = [rms(h, eps) for h in h_q]
    V = cfg.vocab_size
    lr = [torch.empty(h.shape[0], V) for h in h_ref]
    lq = [torch.empty(h.shape[0], V) for h in h_q]

    def on_chunk(r0, r1, raw, ql):
        w_ref = raw.float().cpu()
        w_q = ql.dequant()
        for j in range(len(seqs)):
            lr[j][:, r0:r1] = xr[j] @ w_ref.t()
            lq[j][:, r0:r1] = xq[j] @ w_q.t()

    with torch.no_grad():
        lm = quantize_lm_head(cfg, lambda n, a, b: ck.get_rows(n, a, b, dtype=None), final_norm, signs_d,
                              bits["lm_head"], dev, args.clip_grid, on_chunk=on_chunk)
    stats["lm_head"] = {"bits": lm.bits, "t2": lm.t2, "N": lm.N, "K": lm.K}
    total += lm.nbytes()
    from safetensors.torch import save_file

    sd = {k: v.contiguous() for k, v in lm.state("lm_head").items()}
    sd["embed"] = emb
    save_file(sd, str(out / "globals.safetensors"))
    del sd, lm

    # ---- report
    report = {"per_sequence": [], "per_layer_resid_err": layer_err}
    kls, agree, nll_r, nll_q, n = [], 0, 0.0, 0.0, 0
    for j, s in enumerate(seqs):
        a, b = torch.log_softmax(lr[j], -1), torch.log_softmax(lq[j], -1)
        kl = (a.exp() * (a - b)).sum(-1)
        tgt = torch.tensor(s[1:])
        nr = -a[:-1].gather(1, tgt[:, None]).squeeze(1)
        nq = -b[:-1].gather(1, tgt[:, None]).squeeze(1)
        ag = int((a.argmax(-1) == b.argmax(-1)).sum())
        report["per_sequence"].append({"tokens": len(s), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                                       "top1_agreement": ag / len(s), "ppl_ref": float(nr.mean().exp()),
                                       "ppl_quant": float(nq.mean().exp())})
        kls.append(kl)
        agree += ag
        nll_r += float(nr.sum())
        nll_q += float(nq.sum())
        n += len(s) - 1
    kl = torch.cat(kls)
    t2s = [v["t2"] for v in stats.values()]
    report["summary"] = {
        "bits": sorted(set(bits.values())), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
        "top1_agreement": agree / sum(len(s) for s in seqs),
        "ppl_ref": float(torch.tensor(nll_r / n).exp()), "ppl_quant": float(torch.tensor(nll_q / n).exp()),
        "t2_max": max(t2s), "t2_mean": sum(t2s) / len(t2s),
        "final_resid_err": layer_err[-1],
        "weights_gib": total / 2**30, "embed_gib": emb.numel() * 2 / 2**30,
        "fetched_gib": getattr(ck, "bytes_fetched", 0) / 2**30, "seconds": time.time() - t0,
    }
    (out / "quality_report.json").write_text(json.dumps(report, indent=1))
    manifest = manifest_dict(cfg, bits, args.seed, stats, total,
                             {"source": args.hf_repo or args.model, "preset": f"uniform-{args.bits}" if not args.bits_json else "custom",
                              "quality": report["summary"]})
    (out / "cck_manifest.json").write_text(json.dumps(manifest, indent=1))
    shutil.rmtree(out / ".tmp", ignore_errors=True)
    sm = report["summary"]
    log(f"done: weights {sm['weights_gib']:.2f} GiB + embedding {sm['embed_gib']:.2f} GiB | KL mean {sm['kl_mean']:.2e} "
        f"max {sm['kl_max']:.2e} | top-1 agreement {sm['top1_agreement'] * 100:.2f}% | ppl {sm['ppl_ref']:.3f} -> "
        f"{sm['ppl_quant']:.3f} | t2 max {sm['t2_max']:.1e} | {sm['seconds'] / 60:.1f} min -> {out}")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.upload, commit_message="cckernel INT8 (quality) checkpoint")
        log(f"uploaded to https://huggingface.co/{args.upload}")


if __name__ == "__main__":
    main()
