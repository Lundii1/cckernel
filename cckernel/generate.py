"""Command-line generation with the cckernel engine.

  python -m cckernel.generate --model /models/mimo-cck-q8 --chat --prompt "Write a quicksort in C."
  python -m cckernel.generate --model /models/mimo-cck-q8 --prompt-file task.txt --spec --max-new 1024
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from .engine import MAXM, Engine
from .spec import SpecPolicy, StepCost, _filter_logits, accept


def encode_chat(tok, messages: list[dict], think: bool = False, add_generation_prompt: bool = True) -> list[int]:
    """Token ids of a chat (the model's own template; robust to tokenizer API differences)."""
    text = tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False,
                                   enable_thinking=think)
    return tok(text, add_special_tokens=False)["input_ids"]


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, gen: torch.Generator | None) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = _filter_logits(logits.float() / temperature, top_k, top_p).softmax(-1)
    return int(torch.multinomial(probs, 1, generator=gen))


def step_cost(eng: Engine, profile: bool = True) -> StepCost:
    """The engine's measured T(M, ctx) (profiled once, cached next to the model), or the RTX 4060 Ti
    roofline model when profiling is disabled."""
    if profile:
        return StepCost.from_profile(eng.profile_costs())
    w = sum(q.nbytes() for q in eng.lin.values())
    return StepCost.roofline(w, eng.kv_bytes_per_token())


def generate(eng: Engine, prompt_ids: list[int], max_new: int, eos: set[int], temperature=0.0, top_k=0, top_p=1.0,
             spec=False, max_draft=MAXM - 1, seed=0, on_tokens=None, policy: str = "sched", cost: StepCost | None = None,
             prefilled_logits: torch.Tensor | None = None):
    """Generate up to max_new tokens. With ``spec`` the n-gram drafter proposes tokens and ``policy``
    decides how many to verify ("sched": confidence-scheduled, "ewma", "fixed"). ``prefilled_logits``:
    the prompt is already in the engine (prefix-cache hit) and these are its last-token logits."""
    gen = torch.Generator(device=eng.dev).manual_seed(seed)
    t0 = time.perf_counter()
    lg = prefilled_logits if prefilled_logits is not None else eng.prefill(prompt_ids)
    tok = sample(lg, temperature, top_k, top_p, gen)
    if eng.dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    out = [tok]
    pol = SpecPolicy(policy if spec else "none", cost=cost if cost is not None else (step_cost(eng) if spec and
                     policy == "sched" else None), max_draft=max_draft)
    pol.reset(list(prompt_ids) + [tok])
    if on_tokens:
        on_tokens([tok])
    while len(out) < max_new and tok not in eos and eng.len_host + MAXM < eng.max_len:
        draft = pol.plan(eng.len_host)
        lg = eng.step([tok] + draft)
        em = accept(lg, draft, temperature, top_k, top_p, gen) if draft else [sample(lg[0], temperature, top_k, top_p, gen)]
        if eos & set(em):  # stop at the first EOS
            em = em[: min(em.index(e) for e in eos if e in em) + 1]
        eng.commit(len(em))
        pol.observe(draft, em)
        out += em
        tok = out[-1]
        if on_tokens:
            on_tokens(em)
    if eng.dev.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    st = pol.stats
    stats = {
        "prompt_tokens": len(prompt_ids),
        "prefill_s": t1 - t0,
        "prefill_tok_s": len(prompt_ids) / max(t1 - t0, 1e-9),
        "new_tokens": len(out),
        "decode_s": t2 - t1,
        "decode_tok_s": (len(out) - 1) / max(t2 - t1, 1e-9),
        "steps": st["steps"],
        "tokens_per_step": (len(out) - 1) / max(st["steps"], 1),
        "draft_acceptance": st["accepted"] / max(st["drafted_verified"], 1),
        "verified_drafts": st["drafted_verified"],
        "rejected_drafts": st["drafted_verified"] - st["accepted"],
        "policy": pol.kind,
        "ece": pol.ece(),
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--chat", action="store_true", help="apply the model's chat template")
    ap.add_argument("--think", action="store_true", help="enable_thinking in the chat template")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--spec", action="store_true", help="n-gram speculative decoding")
    ap.add_argument("--spec-policy", choices=["sched", "ewma", "fixed"], default="sched",
                    help="verification length: confidence-scheduled (default), EWMA heuristic, or always max")
    ap.add_argument("--kv", choices=["auto", "bf16", "fp8", "k8v4", "fp4"], default=None,
                    help="KV cache format (default: the manifest's runtime.kv_cache; auto = bf16 for short contexts, "
                         "fp8/k8v4/fp4 as --max-len grows)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prefix-cache", default=None, help="directory of persistent prompt/output snapshots")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    text = args.prompt if args.prompt is not None else open(args.prompt_file).read()
    if args.chat:
        ids = encode_chat(tok, [{"role": "user", "content": text}], think=args.think)
    else:
        ids = tok(text)["input_ids"]
    eng = Engine(args.model, device=args.device, max_len=args.max_len, use_graphs=not args.no_graphs, kv_format=args.kv)
    print(eng.vram_report(), file=sys.stderr)
    eos = set()
    for e in (eng.cfg.eos_token_id, tok.eos_token_id):
        eos |= set(e) if isinstance(e, (list, tuple)) else ({e} if e is not None else set())
    state = {"ids": [], "text": ""}

    def emit(new):  # incremental detokenisation (robust to multi-token characters)
        state["ids"] += new
        text = tok.decode(state["ids"], skip_special_tokens=False)
        sys.stdout.write(text[len(state["text"]):])
        sys.stdout.flush()
        state["text"] = text

    pc, first = None, None
    if args.prefix_cache:
        from .prefix_cache import PrefixCache

        pc = PrefixCache(args.prefix_cache)
        t = time.perf_counter()
        n, lg = pc.restore(eng, ids)
        if n:
            first = lg if n == len(ids) else eng.prefill(ids[n:])
            print(f"prefix cache: reused {n}/{len(ids)} prompt tokens ({time.perf_counter() - t:.2f}s)", file=sys.stderr)
        else:
            first = eng.prefill(ids)
        pc.save(eng, ids, first)
    out, st = generate(eng, ids, args.max_new, eos, args.temperature, args.top_k, args.top_p, args.spec,
                       seed=args.seed, on_tokens=emit, policy=args.spec_policy, prefilled_logits=first)
    if pc is not None:
        pc.save(eng, ids + out[:-1])
    print(file=sys.stderr)
    print(" | ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items()), file=sys.stderr)


if __name__ == "__main__":
    main()
