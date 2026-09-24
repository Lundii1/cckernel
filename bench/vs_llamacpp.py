#!/usr/bin/env python
"""Head-to-head decode speed and output quality: llama.cpp (GGUF) vs cckernel, same machine.

  # RTX 4060 Ti (build llama.cpp with -DGGML_CUDA=ON; the GGUF: ggml-org/MiMo-V2.6-Distill-Qwen-9B-GGUF, Q8_0)
  python bench/vs_llamacpp.py --gguf MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf --llama-bin ~/llama.cpp/build/bin \\
      --model /models/mimo-cck-q8 --ctx 0 8192 32768 --out vs_llamacpp_gpu.json
  # CPU: add --device cpu --threads <cores>

Two parts:
  raw    Pure decode speed at each context depth, no speculation (content-independent):
         llama-bench tg at depth d (for each --llama-kv cache type) vs cckernel single-token steps at depth d
         (for each --kv format). Same quantity on both sides: tokens/s of one sequence.
  suite  The 15 prompts of tools/eval_generate.py, greedy, chat template: llama-server without and with its
         own n-gram speculation (--spec-type) vs cckernel without and with confidence-scheduled speculation.
         Reports decode tok/s (generated tokens / decode time) and the automatic pass/fail checks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------------------------------ raw decode
def llama_bench(bin_dir: Path, gguf: str, depths, ngl: int, threads: int | None, kv: str, n: int, reps: int) -> dict:
    cmd = [str(bin_dir / "llama-bench"), "-m", gguf, "-p", "0", "-n", str(n), "-d", ",".join(map(str, depths)),
           "-ngl", str(ngl), "-fa", "on", "-ctk", kv, "-ctv", kv, "-r", str(reps), "-o", "json"]
    if threads:
        cmd += ["-t", str(threads)]
    log("  $ " + " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    rows = json.loads(out[out.index("["):])
    return {int(r["n_depth"]): float(r["avg_ts"]) for r in rows}


def cck_decode_tps(eng, ctx: int, steps: int) -> float:
    """Tokens/s of single-token decode steps with ``ctx`` tokens already in the cache (content does not
    matter for the timing, so no prefill is needed: the same method as Engine.profile_costs)."""
    eng.reset()
    eng.len_host = ctx
    eng.cur_len.fill_(ctx)
    eng._kv_changed()
    eng.step([0])  # warm up / capture the graph
    eng._sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        eng.step([0])
    eng._sync()
    dt = (time.perf_counter() - t0) / steps
    eng.reset()
    return 1.0 / dt


# ------------------------------------------------------------------------------------------ suite
def _post(url: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def llama_server_suite(bin_dir: Path, gguf: str, ngl: int, threads: int | None, spec: str | None, port: int,
                       prompts) -> list[dict]:
    from eval_generate import check

    cmd = [str(bin_dir / "llama-server"), "-m", gguf, "-ngl", str(ngl), "-c", "8192", "--port", str(port),
           "--jinja", "--reasoning-format", "none", "-fa", "on", "--parallel", "1", "--no-webui"]
    if threads:
        cmd += ["-t", str(threads)]
    if spec:
        cmd += ["--spec-type", spec]
    log("  $ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        res = []
        for p in prompts:
            turns = p.get("turns") or [p["prompt"]]
            msgs, text, stats = [], "", []
            for t in turns:
                msgs.append({"role": "user", "content": t})
                r = _post(base + "/v1/chat/completions", {
                    "messages": msgs, "temperature": 0.0, "max_tokens": p["max_new"], "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": bool(p.get("think", False))}})
                content = r["choices"][0]["message"]["content"]
                msgs.append({"role": "assistant", "content": content.split("</think>")[-1].strip()})
                text += ("\n\n[turn 2]\n" if text else "") + content
                stats.append(r.get("timings", {}))
            ok, why = check(p, text)
            tim = stats[-1]
            row = {"id": p["id"], "ok": ok, "why": why, "text": text,
                   "predicted_n": sum(s.get("predicted_n", 0) for s in stats),
                   "predicted_ms": sum(s.get("predicted_ms", 0.0) for s in stats),
                   "draft_n": sum(s.get("draft_n", 0) for s in stats),
                   "draft_n_accepted": sum(s.get("draft_n_accepted", 0) for s in stats),
                   "prompt_per_second": tim.get("prompt_per_second")}
            log(f"    {p['id']:17s} {'PASS' if ok else 'FAIL'} {row['predicted_n']:4d} tok "
                f"{row['predicted_n'] / max(row['predicted_ms'], 1e-9) * 1e3:6.2f} tok/s  {why[:50]!r}")
            res.append(row)
        return res
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()


def cck_suite(eng, tok, prompts, policy: str) -> list[dict]:
    from eval_generate import run_prompts

    import tempfile

    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    eos.discard(None)
    cost = None
    if policy == "sched":
        from cckernel.spec import StepCost

        cost = StepCost.from_profile(eng.profile_costs(ctxs=(256, 2048)))
    res = run_prompts(eng, tok, eos, policy, cost, prompts, Path(tempfile.mkdtemp(prefix="cck_pc_")))
    return [{"id": r["id"], "ok": r["ok"], "why": r["why"], "text": r["text"],
             "predicted_n": len(r["out_ids"]) - 1, "predicted_ms": r["stats"]["decode_s"] * 1e3,
             "tokens_per_step": r["stats"]["tokens_per_step"]} for r in res]


def summarize_suite(res: list[dict]) -> dict:
    n = sum(r["predicted_n"] for r in res)
    ms = sum(r["predicted_ms"] for r in res)
    return {"pass": sum(r["ok"] for r in res), "prompts": len(res), "tokens": n, "decode_tok_s": n / max(ms, 1e-9) * 1e3}


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-bin", required=True, help="directory with llama-bench and llama-server")
    ap.add_argument("--model", required=True, help="cckernel checkpoint directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--ctx", type=int, nargs="+", default=[0, 8192, 32768])
    ap.add_argument("--steps", type=int, default=64, help="decode steps per cckernel measurement")
    ap.add_argument("--kv", nargs="+", default=["bf16", "fp4"], help="cckernel KV formats for the raw part")
    ap.add_argument("--llama-kv", nargs="+", default=["f16", "q8_0"], help="llama.cpp KV cache types")
    ap.add_argument("--llama-spec", default="ngram-simple", help="llama-server --spec-type for the suite ('' = skip)")
    ap.add_argument("--parts", nargs="+", default=["raw", "suite"], choices=["raw", "suite"])
    ap.add_argument("--suite-kv", default="auto", help="cckernel KV format for the prompt suite")
    ap.add_argument("--only", nargs="*", default=None, help="suite prompt ids")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--force", action="store_true", help="re-run parts already present in --out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    bin_dir = Path(args.llama_bin).expanduser()
    ngl = 99 if args.device == "cuda" else 0
    if args.threads and args.device == "cpu":
        torch.set_num_threads(args.threads)
    outp = Path(args.out)
    result = json.loads(outp.read_text()) if outp.exists() else {}
    result.update({"device": args.device, "gguf": args.gguf, "model": args.model})

    # llama.cpp first, then cckernel: each needs ~10 GB, so they must not be resident at the same time
    raw = result.get("raw", {"llama.cpp": {}, "cckernel": {}})
    suite = result.get("suite", {})
    if "raw" in args.parts:
        for kv in args.llama_kv:
            if kv in raw["llama.cpp"] and not args.force:
                continue
            log(f"== llama-bench, KV {kv}")
            raw["llama.cpp"][kv] = llama_bench(bin_dir, args.gguf, args.ctx, ngl, args.threads, kv, 64 if ngl else 16, 2)
            log(f"  {raw['llama.cpp'][kv]}")
        result["raw"] = raw
        outp.write_text(json.dumps(result, indent=1))
    prompts = None
    if "suite" in args.parts:
        from eval_generate import PROMPTS

        prompts = [p for p in PROMPTS if not args.only or p["id"] in args.only]
        runs = [("llama.cpp", None)] + ([("llama.cpp+" + args.llama_spec, args.llama_spec)] if args.llama_spec else [])
        for name, spec in runs:
            if name in suite and not args.force:
                continue
            log(f"== suite: {name}")
            suite[name] = llama_server_suite(bin_dir, args.gguf, ngl, args.threads, spec, args.port, prompts)
            result["suite"] = suite
            outp.write_text(json.dumps(result, indent=1))

    from cckernel.engine import Engine

    eng = Engine(args.model, device=args.device, max_len=max(args.ctx) + 1024, kv_format=args.kv[0])
    if "raw" in args.parts:
        for kv in args.kv:
            if kv in raw["cckernel"] and not args.force:
                continue
            eng.set_kv_format(kv)
            log(f"== cckernel, KV {kv}")
            raw["cckernel"][kv] = {c: cck_decode_tps(eng, c, args.steps) for c in args.ctx}
            log(f"  {raw['cckernel'][kv]}")
        result["raw"] = raw
        outp.write_text(json.dumps(result, indent=1))
    if "suite" in args.parts:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        eng.set_kv_format(args.suite_kv)
        for policy in ("none", "sched"):
            name = f"cckernel{'+sched' if policy == 'sched' else ''}"
            if name in suite and not args.force:
                continue
            log(f"== suite: {name} (KV {eng.kv_format})")
            suite[name] = cck_suite(eng, tok, prompts, policy)
            result["suite"] = suite
            outp.write_text(json.dumps(result, indent=1))
        result["suite_summary"] = {k: summarize_suite(v) for k, v in suite.items()}

    # ---- report
    if "raw" in result:
        r = result["raw"]
        log("\nRaw decode, one sequence, no speculation (tok/s):")
        log("| context | " + " | ".join(f"llama.cpp KV {k}" for k in r["llama.cpp"]) + " | "
            + " | ".join(f"cckernel KV {k}" for k in r["cckernel"]) + " |")
        for c in args.ctx:
            cells = [f"{r['llama.cpp'][k].get(c, r['llama.cpp'][k].get(str(c), float('nan'))):.2f}" for k in r["llama.cpp"]]
            cells += [f"{r['cckernel'][k].get(c, r['cckernel'][k].get(str(c), float('nan'))):.2f}" for k in r["cckernel"]]
            log(f"| {c} | " + " | ".join(cells) + " |")
    if "suite_summary" in result:
        log("\nPrompt suite (greedy, chat template):")
        for k, v in result["suite_summary"].items():
            log(f"  {k:28s} {v['pass']}/{v['prompts']} pass  {v['tokens']:5d} tokens  {v['decode_tok_s']:.2f} tok/s")
    outp.write_text(json.dumps(result, indent=1))
    log(f"written {outp}")


if __name__ == "__main__":
    main()
