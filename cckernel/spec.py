"""Speculative decoding: training-free n-gram (prompt-lookup) drafter + exact acceptance.

The checkpoint ships no MTP head, so drafts come from the token history itself: the longest
suffix (n = max_n .. 1) that occurred earlier proposes the tokens that followed it. Agentic/coding
workloads copy spans of their context, which is exactly what this exploits.

Acceptance is exact (output distribution = the target model's):
  * temperature 0: accept the longest prefix where the target argmax equals the draft, then emit
    the target token at the first mismatch (the "bonus" token).
  * temperature T > 0: the draft distribution is a point mass q = delta_d, so speculative sampling
    (Leviathan et al. 2023; Chen et al. 2023) reduces to: accept d with probability p(d); on
    rejection sample from p with d removed (the residual max(0, p - q) normalised).
"""

from __future__ import annotations

import math

import torch


class NGramDrafter:
    def __init__(self, max_n: int = 4, min_n: int = 1, max_draft: int = 7):
        self.max_n, self.min_n, self.max_draft = max_n, min_n, max_draft
        self.tokens: list[int] = []
        self.index: list[dict[tuple, int]] = [dict() for _ in range(max_n + 1)]
        self.ewma = 2.0  # running estimate of accepted drafts per step

    def reset(self, tokens: list[int]):
        self.tokens = []
        self.index = [dict() for _ in range(self.max_n + 1)]
        self.extend(tokens)

    def extend(self, new_tokens: list[int]):
        for tok in new_tokens:
            # register every n-gram that ends at the current last token, pointing at the next slot
            self.tokens.append(int(tok))
            L = len(self.tokens)
            for n in range(1, self.max_n + 1):
                if L - 1 >= n:
                    # n-gram ending at position L-2 is followed by position L-1: register once the
                    # continuation exists (so a lookup never returns an empty continuation)
                    key = tuple(self.tokens[L - 1 - n: L - 1])
                    self.index[n][key] = L - 1

    def draft_len(self) -> int:
        return max(1, min(self.max_draft, int(math.ceil(self.ewma + 1.0))))

    def propose(self) -> list[int]:
        k = self.draft_len()
        L = len(self.tokens)
        for n in range(min(self.max_n, L), self.min_n - 1, -1):
            pos = self.index[n].get(tuple(self.tokens[L - n:]))
            if pos is not None:
                return self.tokens[pos: pos + k]
        return []

    def update_stats(self, n_drafted: int, n_accepted: int):
        if n_drafted:
            self.ewma = 0.7 * self.ewma + 0.3 * n_accepted


def _filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        srt, idx = torch.sort(logits, dim=-1, descending=True)
        probs = srt.softmax(-1)
        drop = probs.cumsum(-1) - probs > top_p
        srt = srt.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, srt)
    return logits


def accept(logits: torch.Tensor, draft: list[int], temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
           generator: torch.Generator | None = None) -> list[int]:
    """logits: [len(draft)+1, V] target logits at the verified positions (row i predicts the token
    after draft[:i]). Returns the emitted tokens (accepted drafts + one target token)."""
    M = len(draft) + 1
    assert logits.shape[0] == M
    if temperature <= 0.0:
        tgt = logits.argmax(-1).tolist()
        out = []
        for i, d in enumerate(draft):
            if tgt[i] != d:
                out.append(tgt[i])
                return out
            out.append(d)
        out.append(tgt[-1])
        return out
    probs = _filter_logits(logits.float() / temperature, top_k, top_p).softmax(-1)
    out = []
    for i, d in enumerate(draft):
        p = probs[i]
        r = torch.rand((), generator=generator, device=p.device)
        if r < p[d]:
            out.append(d)
            continue
        res = p.clone()
        res[d] = 0.0
        if float(res.sum()) <= 0:  # p is a point mass on d (cannot really be rejected)
            out.append(d)
            continue
        out.append(int(torch.multinomial(res / res.sum(), 1, generator=generator)))
        return out
    out.append(int(torch.multinomial(probs[-1], 1, generator=generator)))
    return out
