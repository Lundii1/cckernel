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
from .spec import NGramDrafter, _filter_logits, accept


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, gen: torch.Generator | None) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = _filter_logits(logits.float() / temperature, top_k, top_p).softmax(-1)
    return int(torch.multinomial(probs, 1, generator=gen))


def generate(eng: Engine, prompt_ids: list[int], max_new: int, eos: set[int], temperature=0.0, top_k=0, top_p=1.0,
             spec=False, max_draft=MAXM - 1, seed=0, on_tokens=None):
    gen = torch.Generator(device=eng.dev).manual_seed(seed)
    t0 = time.perf_counter()
    lg = eng.prefill(prompt_ids)
    tok = sample(lg, temperature, top_k, top_p, gen)
    if eng.dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    out = [tok]
    steps = drafted = accepted = 0
    drafter = NGramDrafter(max_draft=max_draft) if spec else None
    if drafter:
        drafter.reset(prompt_ids + [tok])
    if on_tokens:
        on_tokens([tok])
    while len(out) < max_new and tok not in eos and eng.len_host + MAXM < eng.max_len:
        draft = drafter.propose()[:max_draft] if drafter else []
        lg = eng.step([tok] + draft)
        em = accept(lg, draft, temperature, top_k, top_p, gen) if draft else [sample(lg[0], temperature, top_k, top_p, gen)]
        if eos & set(em):  # stop at the first EOS
            em = em[: min(em.index(e) for e in eos if e in em) + 1]
        eng.commit(len(em))
        steps += 1
        drafted += len(draft)
        accepted += len(em) - 1
        if drafter:
            drafter.update_stats(len(draft), len(em) - 1)
            drafter.extend(em)
        out += em
        tok = out[-1]
        if on_tokens:
            on_tokens(em)
    if eng.dev.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    stats = {
        "prompt_tokens": len(prompt_ids),
        "prefill_s": t1 - t0,
        "prefill_tok_s": len(prompt_ids) / max(t1 - t0, 1e-9),
        "new_tokens": len(out),
        "decode_s": t2 - t1,
        "decode_tok_s": (len(out) - 1) / max(t2 - t1, 1e-9),
        "steps": steps,
        "tokens_per_step": (len(out) - 1) / max(steps, 1),
        "draft_acceptance": accepted / max(drafted, 1),
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
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    text = args.prompt if args.prompt is not None else open(args.prompt_file).read()
    if args.chat:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True,
                                      enable_thinking=args.think)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
    else:
        ids = tok(text)["input_ids"]
    eng = Engine(args.model, max_len=args.max_len, use_graphs=not args.no_graphs)
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

    _, st = generate(eng, ids, args.max_new, eos, args.temperature, args.top_k, args.top_p, args.spec, seed=args.seed,
                     on_tokens=emit)
    print(file=sys.stderr)
    print(" | ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items()), file=sys.stderr)


if __name__ == "__main__":
    main()
