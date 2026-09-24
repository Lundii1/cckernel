#!/usr/bin/env python
"""MMLU-Pro accuracy of the original model, the cckernel INT8 model (per KV format) and llama.cpp.

  python tools/eval_mmlu_pro.py --backend cck --model /models/mimo-cck-q8 --kv bf16 fp4 --out mmlu.json
  python tools/eval_mmlu_pro.py --backend ref --model /models/mimo-cck-q8 --out mmlu.json        # original bf16, streamed
  python tools/eval_mmlu_pro.py --backend llamacpp --model /models/mimo-cck-q8 --gguf X.gguf --llama-bin DIR --out mmlu.json
  python tools/eval_mmlu_pro.py --summary --out mmlu.json

Protocol (the same for every backend, so they are directly comparable):
  * TIGER-Lab/MMLU-Pro test split; ``--per-category N`` questions from each of the 14 categories (fixed seed),
    or ``--per-category 0`` for all 12,032.
  * Zero-shot, no chain of thought: the model's chat template with thinking disabled, the question and its
    lettered options, then the assistant turn prefilled with "The answer is (". The prediction is the option
    letter with the highest next-token logit (restricted to the valid letters).
  This is a likelihood-style variant: much cheaper than the official 5-shot CoT protocol, so absolute scores
  are lower than published CoT numbers, but every backend is scored on exactly the same prompts.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
LETTERS = "ABCDEFGHIJ"
PREFIX = "The answer is ("


def log(m):
    print(m, flush=True)


def load_questions(per_category: int, seed: int = 0) -> list[dict]:
    import pyarrow.parquet as pq

    with urllib.request.urlopen(urllib.request.Request(DATA, headers={"User-Agent": "cckernel"}), timeout=120) as r:
        rows = pq.read_table(io.BytesIO(r.read())).to_pylist()
    by_cat: dict[str, list[dict]] = {}
    for q in rows:
        by_cat.setdefault(q["category"], []).append(q)
    g = torch.Generator().manual_seed(seed)
    out = []
    for cat in sorted(by_cat):
        qs = by_cat[cat]
        idx = torch.randperm(len(qs), generator=g).tolist()
        out += [qs[i] for i in (idx[:per_category] if per_category else idx)]
    return out


def prompt_text(tok, q: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    user = (f"The following is a multiple choice question about {q['category']}. Choose the correct option.\n\n"
            f"Question: {q['question']}\n\nOptions:\n{opts}\n\nAnswer with the letter of the correct option.")
    chat = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    return chat + PREFIX


def letter_ids(tok) -> list[int]:
    ids = []
    for L in LETTERS:
        t = tok(L, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, (L, t)
        ids.append(t[0])
    return ids


# ------------------------------------------------------------------------------------------ backends
def run_cck(args, qs, tok) -> dict:
    """Layer-major batched scoring on the cckernel engine: every weight matrix is dequantized once for all
    questions; each question is its own sequence (fresh GDN state and KV cache) through the engine's real
    prefill code (quantized KV written then read back)."""
    import torch.nn.functional as F

    from cckernel import torch_ops as T
    from cckernel.engine import Engine

    seqs = [tok(prompt_text(tok, q), add_special_tokens=False)["input_ids"] for q in qs]
    lids = letter_ids(tok)
    eng = Engine(args.model, device=args.device, max_len=max(map(len, seqs)) + 8, kv_format=args.kv[0])
    c, eps, dev = eng.cfg, eng.cfg.rms_norm_eps, eng.dev
    rms = lambda t: (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)  # noqa: E731
    res = {}
    for kv in args.kv:
        eng.set_kv_format(kv)
        t0 = time.time()
        H = [eng.embed[torch.tensor(s, device=dev)].float() for s in seqs]
        groups, cur, n = [], [], 0  # batch the projections over groups of ~args.group_tokens tokens
        for j, s in enumerate(seqs):
            cur.append(j)
            n += len(s)
            if n >= args.group_tokens:
                groups.append(cur)
                cur, n = [], 0
        if cur:
            groups.append(cur)
        for i, lt in enumerate(c.layer_types):
            p = f"layers.{i}."
            names = ("in_proj", "out_proj") if lt == "linear_attention" else ("qkv_proj", "o_proj")
            W_in = eng.lin[p + names[0]].dequant(eng.scratch[0])
            W_out = eng.lin[p + names[1]].dequant(eng.scratch[1])
            for grp in groups:
                proj = torch.cat([rms(H[j]) for j in grp]) @ W_in.t()
                mixes, o = [], 0
                for j in grp:
                    L = len(seqs[j])
                    if lt == "linear_attention":
                        eng.S[i].zero_()
                        mixes.append(eng._gdn_prefill(i, proj[o:o + L], 0))
                    else:
                        mixes.append(eng._attn_prefill(i, proj[o:o + L], 0))
                    o += L
                out = (torch.cat(mixes) @ W_out.t()).float()
                o = 0
                for j in grp:
                    L = len(seqs[j])
                    H[j] += out[o:o + L]
                    o += L
            W_gu = eng.lin[p + "gate_up"].dequant(eng.scratch[0])
            W_dn = eng.lin[p + "down"].dequant(eng.scratch[1])
            for grp in groups:
                gu = torch.cat([rms(H[j]) for j in grp]) @ W_gu.t()
                hm = (T.bf16r(F.silu(gu[:, 0::2].float())) * gu[:, 1::2].float()).to(torch.bfloat16)
                del gu
                out = (hm @ W_dn.t()).float()
                o = 0
                for j in grp:
                    L = len(seqs[j])
                    H[j] += out[o:o + L]
                    o += L
            if i % 4 == 3:
                log(f"  [{kv}] layer {i + 1}/{c.num_hidden_layers} [{time.time() - t0:.0f}s]")
        last = rms(torch.stack([h[-1] for h in H])).float()
        W = torch.stack([eng.lin["lm_head"].dequant_rows(r, r + 1, eng.scratch[0])[0].float().clone() for r in lids])
        logits = last @ W.t()  # [n_questions, 10]
        res[f"cckernel INT8, KV {kv}"] = finalize(qs, logits, time.time() - t0)
        del H
    return res


def run_ref(args, qs, tok) -> dict:
    """The original bf16 weights streamed from the Hub (HTTP ranges), HF semantics with bf16 matmuls
    (as the model is normally run), layer-major over all questions."""
    from cckernel.config import TextConfig
    from cckernel.reference import RefCache, RefModel, rmsnorm_zc
    from cckernel.remote import RemoteCheckpoint

    from concurrent.futures import ThreadPoolExecutor

    cfg = TextConfig.from_dict(json.loads((Path(args.model) / "cck_manifest.json").read_text())["config"])
    ck = RemoteCheckpoint(args.hf_repo)
    seqs = [tok(prompt_text(tok, q), add_special_tokens=False)["input_ids"] for q in qs]
    lids = letter_ids(tok)
    t0 = time.time()
    need = sorted({t for s in seqs for t in s})
    rows = {}
    for r0 in range(0, cfg.vocab_size, 16384):
        want = [t for t in need if r0 <= t < r0 + 16384]
        if want:
            chunk = ck.get_rows("embed_tokens.weight", r0, min(cfg.vocab_size, r0 + 16384), dtype=None)
            for t in want:
                rows[t] = chunk[t - r0]
    H = [torch.stack([rows[t] for t in s]).float() for s in seqs]
    del rows
    keys_of = lambda i: [k for k in ck.keys() if k.startswith(f"layers.{i}.")]  # noqa: E731
    fetch = lambda i: {k: ck.get(k, dtype=None) for k in keys_of(i)}  # noqa: E731
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(fetch, 0)
    eps = cfg.rms_norm_eps
    for i in range(cfg.num_hidden_layers):
        wl = nxt.result()
        if i + 1 < cfg.num_hidden_layers:
            nxt = pool.submit(fetch, i + 1)
        ref = RefModel(cfg, wl, gdn_algo="chunked", dtype=torch.bfloat16)
        p = f"layers.{i}."
        for j in range(len(H)):
            pos = torch.arange(H[j].shape[0])
            x = rmsnorm_zc(H[j], ref.W(p + "input_layernorm.weight"), eps).to(torch.bfloat16)
            mix = ref.gdn(i, x, RefCache(cfg)) if cfg.layer_types[i] == "linear_attention" else ref.attn(i, x, RefCache(cfg), pos)
            H[j] = H[j] + mix.float()
            x = rmsnorm_zc(H[j], ref.W(p + "post_attention_layernorm.weight"), eps).to(torch.bfloat16)
            H[j] = H[j] + ref.mlp(i, x).float()
        log(f"  [ref] layer {i + 1}/{cfg.num_hidden_layers} [{time.time() - t0:.0f}s]")
    pool.shutdown()
    last = rmsnorm_zc(torch.stack([h[-1] for h in H]), ck.get("norm.weight"), eps).to(torch.bfloat16).float()
    W = torch.stack([ck.get_rows("lm_head.weight", r, r + 1)[0].to(torch.bfloat16).float() for r in lids])
    return {"original bf16": finalize(qs, last @ W.t(), time.time() - t0)}


def run_llamacpp(args, qs, tok) -> dict:
    import subprocess

    cmd = [str(Path(args.llama_bin) / "llama-server"), "-m", args.gguf, "-ngl", "99" if args.device == "cuda" else "0",
           "-c", "4096", "--port", str(args.port), "--parallel", "1", "--no-webui", "-fa", "on"]
    if args.threads:
        cmd += ["-t", str(args.threads)]
    log("  $ " + " ".join(cmd))
    base = f"http://127.0.0.1:{args.port}"

    def start():
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return p
            except Exception:  # noqa: BLE001
                if p.poll() is not None:
                    break
                time.sleep(1)
        raise RuntimeError("llama-server did not start")

    t0 = time.time()
    proc = start()
    try:
        preds = []
        for k, q in enumerate(qs):
            valid = LETTERS[: len(q["options"])]
            body = {"prompt": prompt_text(tok, q), "n_predict": 1, "temperature": 0.0, "cache_prompt": False,
                    "grammar": "root ::= [" + valid + "]"}
            for attempt in range(3):  # restart the server if it went away (e.g. killed under memory pressure)
                try:
                    req = urllib.request.Request(base + "/completion", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=600) as r:
                        preds.append(json.loads(r.read())["content"].strip()[:1])
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"  [llama.cpp] question {k}: {type(e).__name__}; restarting the server")
                    proc.kill()
                    proc = start()
            else:
                raise RuntimeError(f"question {k} failed 3 times")
            if k % 20 == 19:
                log(f"  [llama.cpp] {k + 1}/{len(qs)} [{time.time() - t0:.0f}s]")
    finally:
        proc.terminate()
        proc.wait(30)
    logits = torch.full((len(qs), 10), -1e9)
    for k, pch in enumerate(preds):
        if pch in LETTERS:
            logits[k, LETTERS.index(pch)] = 0.0
    return {f"llama.cpp {Path(args.gguf).stem.split('-')[-1]}": finalize(qs, logits, time.time() - t0)}


# ------------------------------------------------------------------------------------------ scoring
def finalize(qs, logits: torch.Tensor, seconds: float) -> dict:
    preds = []
    for k, q in enumerate(qs):
        n = len(q["options"])
        preds.append(LETTERS[int(logits[k, :n].argmax())])
    correct = [p == q["answer"] for p, q in zip(preds, qs)]
    return {"preds": preds, "correct": correct, "accuracy": sum(correct) / len(qs), "seconds": seconds}


def summary(result: dict):
    qs_cat = result["categories"]
    names = list(result["runs"])
    log(f"\nMMLU-Pro subset: {len(qs_cat)} questions, zero-shot, answer-letter scoring (no CoT)")
    for n in names:
        r = result["runs"][n]
        acc = sum(r["correct"]) / len(r["correct"])
        se = (acc * (1 - acc) / len(r["correct"])) ** 0.5
        log(f"  {n:28s} {acc * 100:5.1f} % ± {se * 100:.1f}")
    if len(names) > 1:
        log("  agreement of predicted letters:")
        for a in names:
            pa = result["runs"][a]["preds"]
            log("   " + a[:28].ljust(28) + " " + " ".join(
                f"{sum(x == y for x, y in zip(pa, result['runs'][b]['preds'])) / len(pa) * 100:5.1f}" for b in names))
    cats = sorted(set(qs_cat))
    log("  per category (" + ", ".join(names) + "):")
    for cat in cats:
        idx = [k for k, cc in enumerate(qs_cat) if cc == cat]
        log(f"   {cat:18s} " + "  ".join(
            f"{sum(result['runs'][n]['correct'][k] for k in idx) / len(idx) * 100:5.1f}" for n in names))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cck", "ref", "llamacpp"])
    ap.add_argument("--model", required=True, help="cck checkpoint (tokenizer / chat template / config)")
    ap.add_argument("--kv", nargs="+", default=["bf16", "fp4"])
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--group-tokens", type=int, default=12000)
    ap.add_argument("--hf-repo", default="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--llama-bin", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    outp = Path(args.out)
    result = json.loads(outp.read_text()) if outp.exists() else {"runs": {}}
    if args.summary:
        summary(result)
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    qs = load_questions(args.per_category)
    ids = [q["question_id"] for q in qs]
    if "question_ids" in result:
        assert result["question_ids"] == ids, "a different question subset is already stored in --out"
    result.update({"question_ids": ids, "categories": [q["category"] for q in qs],
                   "answers": [q["answer"] for q in qs], "protocol": "zero-shot, chat template (thinking off), "
                   "assistant prefix 'The answer is (', argmax over valid option letters"})
    log(f"{len(qs)} questions, mean prompt {sum(len(tok(prompt_text(tok, q))['input_ids']) for q in qs[:50]) / min(50, len(qs)):.0f} tokens")
    run = {"cck": run_cck, "ref": run_ref, "llamacpp": run_llamacpp}[args.backend]
    for name, r in run(args, qs, tok).items():
        result["runs"][name] = r
        log(f"{name}: {r['accuracy'] * 100:.1f} % [{r['seconds']:.0f}s]")
        outp.write_text(json.dumps(result, indent=1))
    summary(result)


if __name__ == "__main__":
    main()
