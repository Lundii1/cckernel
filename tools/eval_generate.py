#!/usr/bin/env python
"""Coherence, long-context retrieval and speed tests of the quantized model with the real engine.

  python tools/eval_generate.py --model /models/mimo-cck-q8 --out docs/eval_generate.json \\
      --configs bf16:none fp4:sched --samples docs/samples

Each config is KV_FORMAT:SPEC_POLICY. Every prompt is generated greedily (the model's chat template,
thinking off unless the prompt says otherwise) and checked automatically:
  math      the final number is right
  code      the generated Python passes unit tests (run in a subprocess)
  json      the output parses and has the requested keys
  fact      the answer contains the expected word
  multiturn a follow-up question about the first turn is answered (second turn via the prefix cache)
  think     reasoning-mode prompts (capped) reach the right final answer
plus degeneration metrics (repeated 4-grams, distinct-2) on every output.

Needle-in-a-haystack: a passphrase inserted at 10/50/90% depth of ~6K tokens of WikiText-2 prose.

Speed: prefill time, decode s/token, tokens per step, and the exact greedy replay (spec.replay) of
every speculative policy on the recorded outputs under the measured step-cost curve of this machine
and under the RTX 4060 Ti bandwidth model (short and long context).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cckernel.engine import Engine  # noqa: E402
from cckernel.generate import encode_chat, generate  # noqa: E402
from cckernel.prefix_cache import PrefixCache  # noqa: E402
from cckernel.spec import StepCost, replay  # noqa: E402

CODE_EDIT_SRC = '''def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda iv: iv[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [tuple(iv) for iv in merged]


def moving_average(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        if i >= window - 1:
            out.append(total / window)
    return out
'''

PROMPTS = [
    {"id": "math_train", "kind": "math", "max_new": 160, "answer": "270",
     "prompt": "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. How many kilometers does it "
               "travel in total? Show the calculation briefly, then give the final number on the last line."},
    {"id": "math_discount", "kind": "math", "max_new": 160, "answer": "22",
     "prompt": "A shirt costs $25. It is discounted by 20%, and then a 10% tax is added to the discounted price. What "
               "is the final price in dollars? Show the steps briefly, then give the final number on the last line."},
    {"id": "math_mul", "kind": "math", "max_new": 64, "answer": "391",
     "prompt": "What is 17 multiplied by 23? Reply with just the number."},
    {"id": "code_prime", "kind": "code", "max_new": 256,
     "prompt": "Write a Python function `is_prime(n)` that returns True if n is a prime number and False otherwise. "
               "Reply with only the code in a ```python block.",
     "tests": "assert is_prime(2) and is_prime(97) and is_prime(7919)\n"
              "assert not is_prime(1) and not is_prime(0) and not is_prime(100) and not is_prime(-7)\n"},
    {"id": "code_fizzbuzz", "kind": "code", "max_new": 256,
     "prompt": "Write a Python function `fizzbuzz(n)` that returns a list of strings for the numbers 1..n using the "
               "usual rules (Fizz for multiples of 3, Buzz for 5, FizzBuzz for both, otherwise the number). Reply with "
               "only the code in a ```python block.",
     "tests": "r = fizzbuzz(15)\nassert r[0] == '1' and r[2] == 'Fizz' and r[4] == 'Buzz' and r[14] == 'FizzBuzz'\n"
              "assert len(r) == 15\n"},
    {"id": "code_words", "kind": "code", "max_new": 200,
     "prompt": "Write a Python function `reverse_words(s)` that returns the words of the string s in reverse order, "
               "separated by single spaces. Reply with only the code in a ```python block.",
     "tests": "assert reverse_words('the quick brown fox') == 'fox brown quick the'\n"
              "assert reverse_words('hello') == 'hello'\n"},
    {"id": "json_profile", "kind": "json", "max_new": 160, "keys": ["name", "age", "languages"],
     "prompt": "Return a JSON object with the keys \"name\", \"age\" and \"languages\" (a list) describing a "
               "fictional software engineer. Reply with only the JSON."},
    {"id": "fact_capital", "kind": "fact", "max_new": 32, "answer": "Canberra",
     "prompt": "What is the capital city of Australia? Answer in one word."},
    {"id": "fact_author", "kind": "fact", "max_new": 32, "answer": "Austen",
     "prompt": "Who wrote the novel 'Pride and Prejudice'? Answer with the author's name only."},
    {"id": "fact_gold", "kind": "fact", "max_new": 32, "answer": "Au",
     "prompt": "What is the chemical symbol of gold? Answer with the symbol only."},
    {"id": "summary", "kind": "summary", "max_new": 96, "answer": "photosynthesis",
     "prompt": "Summarize the following paragraph in one sentence.\n\nPhotosynthesis is the process by which green "
               "plants, algae and some bacteria convert light energy into chemical energy. Using sunlight, they take "
               "in carbon dioxide and water and produce glucose and oxygen. The process takes place mainly in the "
               "chloroplasts, where the pigment chlorophyll absorbs light. Photosynthesis is the foundation of most "
               "food chains and is responsible for the oxygen in Earth's atmosphere."},
    {"id": "code_edit", "kind": "edit", "max_new": 400,
     "prompt": "Add type hints and a one-line docstring to each function below. Keep the logic exactly the same. "
               "Reply with only the updated code in a ```python block.\n\n```python\n" + CODE_EDIT_SRC + "```",
     "tests": "assert merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]\n"
              "assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]\n"},
    {"id": "multiturn", "kind": "multiturn", "max_new": 48, "answer": "Lyon",
     "turns": ["Hi! My name is Priya and I live in Lyon. Please just say hello back in one sentence.",
               "Which city do I live in? Answer with the city name only."]},
    {"id": "think_strawberry", "kind": "think", "max_new": 512, "answer": "3", "think": True,
     "prompt": "How many times does the letter r appear in the word 'strawberry'? Give the final answer as a number."},
    {"id": "think_batball", "kind": "think", "max_new": 512, "answer": "5", "think": True,
     "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the "
               "ball cost? Give the final answer as a number of cents."},
]

NEEDLE = "The secret passphrase is blue-harbor-4172."
NEEDLE_Q = "What is the secret passphrase mentioned in the text above? Answer with the passphrase only."


def log(m):
    print(m, flush=True)


def final_answer(text: str) -> str:
    return text.split("</think>")[-1].replace("<|im_end|>", "").strip()


def code_block(text: str) -> str:
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[0] if m else text


def run_python(code: str, tests: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + tests)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=20)
        return r.returncode == 0, (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def degeneration(ids: list[int]) -> dict:
    n4 = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    n2 = [tuple(ids[i:i + 2]) for i in range(len(ids) - 1)]
    return {"repeat4": 1 - len(set(n4)) / max(len(n4), 1), "distinct2": len(set(n2)) / max(len(n2), 1)}


def check(p: dict, text: str) -> tuple[bool, str]:
    ans = final_answer(text)
    k = p["kind"]
    if k in ("math", "think"):
        nums = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
        ok = bool(nums) and float(nums[-1]) == float(p["answer"])
        return ok, f"last number {nums[-1] if nums else None}"
    if k in ("code", "edit"):
        code = code_block(ans)
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"syntax error {e}"
        ok, err = run_python(code, p["tests"])
        if ok and k == "edit":
            ok = "->" in code and ('"""' in code or "'''" in code)
            return ok, "tests pass" + ("" if ok else ", but no type hints/docstrings")
        return ok, "tests pass" if ok else err
    if k == "json":
        body = code_block(ans) if "```" in ans else ans
        try:
            obj = json.loads(body[body.index("{"): body.rindex("}") + 1])
            ok = all(key in obj for key in p["keys"]) and isinstance(obj.get("languages"), list)
            return ok, f"keys {sorted(obj)}"
        except (ValueError, json.JSONDecodeError) as e:
            return False, f"invalid json: {e}"
    return p["answer"].lower() in ans.lower(), ans[:80]


# ------------------------------------------------------------------------------------------ runs
def run_prompts(eng, tok, eos, policy, cost, prompts, pc_dir):
    res = []
    for p in prompts:
        eng.reset()
        t0 = time.perf_counter()
        if p["kind"] == "multiturn":
            pc = PrefixCache(pc_dir / f"{eng.kv_format}_{policy}")
            msgs = [{"role": "user", "content": p["turns"][0]}]
            ids1 = encode_chat(tok, msgs)
            out1, st1 = generate(eng, ids1, p["max_new"], eos, spec=policy != "none", policy=policy, cost=cost)
            pc.save(eng, ids1 + out1[:-1])
            text1 = tok.decode(out1, skip_special_tokens=True)
            msgs += [{"role": "assistant", "content": final_answer(text1)}, {"role": "user", "content": p["turns"][1]}]
            ids = encode_chat(tok, msgs)
            # turn 2: restore the longest cached prefix, prefill only the new tokens
            fresh = eng.len_host
            t1 = time.perf_counter()
            eng.reset()
            n, lg = pc.restore(eng, ids)
            first = lg if n == len(ids) else eng.prefill(ids[n:])
            t_restore = time.perf_counter() - t1
            out, st = generate(eng, ids, p["max_new"], eos, spec=policy != "none", policy=policy, cost=cost,
                               prefilled_logits=first)
            st["prefix_reused"], st["prefix_total"], st["turn2_prefill_s"] = n, len(ids), t_restore
            st["turn1_len"] = fresh
            text = text1 + "\n\n[turn 2]\n" + tok.decode(out, skip_special_tokens=False)
        else:
            ids = encode_chat(tok, [{"role": "user", "content": p["prompt"]}], think=p.get("think", False))
            out, st = generate(eng, ids, p["max_new"], eos, spec=policy != "none", policy=policy, cost=cost)
            text = tok.decode(out, skip_special_tokens=False)
        ok, why = check(p, text)
        r = {"id": p["id"], "kind": p["kind"], "ok": ok, "why": why, "text": text, "prompt_ids": ids, "out_ids": out,
             "stats": st, "wall_s": time.perf_counter() - t0, **degeneration(out)}
        log(f"  {p['id']:17s} {'PASS' if ok else 'FAIL'}  {len(out):4d} tok  {st['decode_tok_s']:.2f} tok/s  "
            f"{st['tokens_per_step']:.2f} tok/step  {why[:60]!r}")
        res.append(r)
    return res


def needle(eng, tok, eos, depths, n_tokens, prose):
    ids_prose = tok(prose, add_special_tokens=False)["input_ids"][: n_tokens]
    out = []
    for dpt in depths:
        cut = int(len(ids_prose) * dpt)
        text = tok.decode(ids_prose[:cut]) + "\n" + NEEDLE + "\n" + tok.decode(ids_prose[cut:])
        ids = encode_chat(tok, [{"role": "user", "content": text + "\n\n" + NEEDLE_Q}])
        eng.reset()
        t0 = time.perf_counter()
        gen, st = generate(eng, ids, 24, eos)
        ans = final_answer(tok.decode(gen, skip_special_tokens=True))
        ok = "blue-harbor-4172" in ans
        out.append({"depth": dpt, "prompt_tokens": len(ids), "ok": ok, "answer": ans, "prefill_s": st["prefill_s"],
                    "wall_s": time.perf_counter() - t0})
        log(f"  needle depth {dpt:.0%} ({len(ids)} tok): {'PASS' if ok else 'FAIL'} {ans[:50]!r} "
            f"[prefill {st['prefill_s']:.0f}s]")
    return out


def replay_table(runs: dict, costs: dict) -> dict:
    """Exact replay of every policy on each run's recorded outputs under each cost model."""
    table = {}
    for rname, res in runs.items():
        for cname, (cost, ctx_off) in costs.items():
            tot = {}
            for pol in ("none", "ewma", "fixed", "sched"):
                agg = {"steps": 0, "tokens": 0, "time": 0.0, "verified_drafts": 0, "rejected_drafts": 0}
                for r in res:
                    x = replay(r["prompt_ids"], r["out_ids"], pol, cost, ctx0=ctx_off)
                    for k in agg:
                        agg[k] += x[k]
                agg["tokens_per_step"] = agg["tokens"] / max(agg["steps"], 1)
                agg["tok_per_s"] = agg["tokens"] / max(agg["time"], 1e-12)
                tot[pol] = agg
            table[f"{rname} | {cname}"] = tot
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="+", default=["bf16:none", "fp4:sched"])
    ap.add_argument("--only", nargs="*", default=None, help="prompt ids to run")
    ap.add_argument("--needle", nargs="*", default=["bf16", "fp4"], help="KV formats for the needle test")
    ap.add_argument("--needle-tokens", type=int, default=6000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", default=None, help="directory for markdown transcripts")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    eng = Engine(args.model, device=args.device, max_len=args.needle_tokens + 1024, kv_format="bf16")
    log(eng.vram_report())
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    for e in (eng.cfg.eos_token_id if isinstance(eng.cfg.eos_token_id, list) else [eng.cfg.eos_token_id]):
        if e is not None:
            eos.add(e)
    eos.discard(None)
    outp = Path(args.out)
    result = json.loads(outp.read_text()) if outp.exists() else {"runs": {}, "needle": {}, "profiles": {}}
    prompts = [p for p in PROMPTS if not args.only or p["id"] in args.only]
    pc_dir = Path(tempfile.mkdtemp(prefix="cck_pc_"))

    for cfgname in args.configs:
        kv, policy = cfgname.split(":")
        eng.set_kv_format(kv)
        prof = eng.profile_costs(ctxs=(256, 2048))
        result["profiles"][f"{args.device}_{kv}"] = prof
        cost = StepCost.from_profile(prof)
        log(f"step cost ({kv}): {prof['a']:.3f}s + {prof['b']:.4f}s*M + {prof['c'] * 1e6:.3f}us*M*ctx")
        log(f"== {cfgname}")
        res = run_prompts(eng, tok, eos, policy, cost, prompts, pc_dir)
        prev = {r["id"]: r for r in result["runs"].get(cfgname, [])}
        prev.update({r["id"]: r for r in res})
        result["runs"][cfgname] = list(prev.values())
        outp.write_text(json.dumps(result, indent=1))

    if args.needle:
        import io
        import urllib.request

        import pyarrow.parquet as pq

        from quantize_stream import WIKITEXT

        with urllib.request.urlopen(urllib.request.Request(WIKITEXT, headers={"User-Agent": "cckernel"}), timeout=60) as r:
            prose = "".join(pq.read_table(io.BytesIO(r.read())).column("text").to_pylist()[:1500])
        for kv in args.needle:
            eng.set_kv_format(kv)
            log(f"== needle {kv}")
            result["needle"][kv] = needle(eng, tok, eos, [0.1, 0.5, 0.9], args.needle_tokens, prose)
            outp.write_text(json.dumps(result, indent=1))

    # exact replay of all policies on the recorded outputs
    w = sum(q.nbytes() for q in eng.lin.values())
    costs = {}
    for key, prof in result["profiles"].items():
        costs[f"measured {key}"] = (StepCost.from_profile(prof), 0)
    for kv in ("bf16", "fp4"):
        from cckernel import kvq

        kvb = kvq.kv_bytes_per_token(kv, len(eng.cfg.attn_layers), eng.cfg.num_key_value_heads, eng.cfg.head_dim)
        rc = StepCost.roofline(w, kvb)
        costs[f"4060Ti model {kv} short ctx"] = (rc, 0)
        costs[f"4060Ti model {kv} +32K ctx"] = (rc, 32768)
    result["replay"] = replay_table({k: v for k, v in result["runs"].items()}, costs)
    outp.write_text(json.dumps(result, indent=1))

    if args.samples:
        sd = Path(args.samples)
        sd.mkdir(parents=True, exist_ok=True)
        for cfgname, res in result["runs"].items():
            lines = [f"# Samples: {cfgname}\n", "Greedy decoding, chat template, thinking off unless noted.\n"]
            for r in res:
                p = next(x for x in PROMPTS if x["id"] == r["id"])
                q = p.get("prompt") or "\n\n".join(p["turns"])
                lines += [f"## {r['id']} — {'PASS' if r['ok'] else 'FAIL'} ({r['why'][:80]})\n",
                          "**Prompt:**\n", "```text\n" + q.strip() + "\n```\n", "**Output:**\n",
                          "```text\n" + r["text"].strip() + "\n```\n"]
            (sd / f"{cfgname.replace(':', '_')}.md").write_text("\n".join(lines))
    log(f"written {outp}")


if __name__ == "__main__":
    main()
