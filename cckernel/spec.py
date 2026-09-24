"""Speculative decoding: training-free n-gram (prompt-lookup) drafter, confidence-scheduled
verification, exact acceptance.

The checkpoint ships no MTP head, so drafts come from the token history itself: the longest
suffix (n = max_n .. 1) that occurred earlier proposes the tokens that followed it. Agentic/coding
workloads copy spans of their context, which is exactly what this exploits.

Acceptance is exact (output distribution = the target model's):
  * temperature 0: accept the longest prefix where the target argmax equals the draft, then emit
    the target token at the first mismatch (the "bonus" token).
  * temperature T > 0: the draft distribution is a point mass q = delta_d, so speculative sampling
    (Leviathan et al. 2023; Chen et al. 2023) reduces to: accept d with probability p(d); on
    rejection sample from p with d removed (the residual max(0, p - q) normalised).

Verification length (DSpark, arXiv 2607.05147, as used by DeepSeek-V4.1-Flash): verifying a draft
token costs step time, and it only pays off if the token survives. With c_j the conditional
probability that draft token j is accepted given tokens < j were, the prefix-survival probability is
a_j = prod_{i<=j} c_i, a step verifying l drafts emits 1 + sum_{j<=l} a_j tokens in expectation, and
we pick l maximising that divided by the profiled step time T(1 + l, ctx) (tokens per second).
DSpark predicts c_j with a trained head; we estimate it online from counts of how n-gram drafts with
the same features fared (match order, how many earlier occurrences agree, position, recent regime).
Empirical frequencies are calibrated by construction, which is what the throughput objective needs.

Exactness: DSpark must stop its greedy admission early because its draft tokens are *sampled* and
the next confidence depends on them. An n-gram draft is a deterministic function of the history, and
so is the confidence model's state, so any length rule computed from them before verification leaves
the accepted distribution unchanged; the global argmax over l is allowed.
"""

from __future__ import annotations

import math

import torch


class NGramDrafter:
    def __init__(self, max_n: int = 4, min_n: int = 1, max_draft: int = 7, occ: int = 4):
        self.max_n, self.min_n, self.max_draft, self.occ = max_n, min_n, max_draft, occ
        self.tokens: list[int] = []
        self.index: list[dict[tuple, list[int]]] = [dict() for _ in range(max_n + 1)]
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
                    occ = self.index[n].setdefault(key, [])
                    occ.append(L - 1)
                    if len(occ) > self.occ:
                        del occ[0]

    def draft_len(self) -> int:
        return max(1, min(self.max_draft, int(math.ceil(self.ewma + 1.0))))

    def propose_full(self) -> tuple[list[int], list[tuple[int, int]]]:
        """Longest-suffix match, continuation of its most recent occurrence (up to max_draft tokens),
        and per draft position the features (match order n, number of stored occurrences whose
        continuation agrees at that position)."""
        L = len(self.tokens)
        for n in range(min(self.max_n, L), self.min_n - 1, -1):
            occ = self.index[n].get(tuple(self.tokens[L - n:]))
            if occ:
                pos = occ[-1]
                draft = self.tokens[pos: pos + self.max_draft]
                feats = []
                for j, d in enumerate(draft):
                    agree = sum(1 for p in occ if p + j < L and self.tokens[p + j] == d)
                    feats.append((n, agree))
                return draft, feats
        return [], []

    def propose(self) -> list[int]:
        """Draft with the EWMA length heuristic (the original policy)."""
        return self.propose_full()[0][: self.draft_len()]

    def update_stats(self, n_drafted: int, n_accepted: int):
        if n_drafted:
            self.ewma = 0.7 * self.ewma + 0.3 * n_accepted


# --------------------------------------------------------------------------------------------------
# confidence-scheduled verification
# --------------------------------------------------------------------------------------------------
class StepCost:
    """Step time model T(M, ctx) = a + b*M + c*M*ctx (seconds), M = verified tokens incl. the input."""

    def __init__(self, a: float, b: float = 0.0, c: float = 0.0):
        self.a, self.b, self.c = a, b, c

    def __call__(self, M: int, ctx: int) -> float:
        return self.a + self.b * M + self.c * M * ctx

    @classmethod
    def from_profile(cls, prof: dict) -> "StepCost":
        return cls(max(prof["a"], 1e-6), max(prof["b"], 0.0), max(prof["c"], 0.0))

    @classmethod
    def roofline(cls, weight_bytes: float, kv_bytes_per_token: float, state_bytes: float = 1e8,
                 bandwidth: float = 288e9, efficiency: float = 0.8, per_token_frac: float = 0.03) -> "StepCost":
        """Bandwidth model of one decode/verify step on a GPU (default RTX 4060 Ti, 288 GB/s): all weights
        and the recurrent state are streamed once per step; each verified token reads the KV cache once
        (attn_decode runs one CTA group per token); a small per-token compute term."""
        bw = bandwidth * efficiency
        a = (weight_bytes + state_bytes) / bw
        return cls(a * (1 - per_token_frac), a * per_token_frac, kv_bytes_per_token / bw)

    def to_dict(self):
        return {"a": self.a, "b": self.b, "c": self.c}


class ConfidenceModel:
    """Online estimate of c_j = P(draft token j accepted | tokens < j accepted).

    Beta-smoothed acceptance counts in a back-off hierarchy of feature buckets
    (regime, n, agree, j) -> (n, agree, j) -> (agree, j) -> (j,) -> (), each level shrunk towards
    its parent with ``k`` pseudo-counts. Only positions up to the first rejection are observed."""

    def __init__(self, k: float = 4.0, prior: float = 0.5):
        self.k, self.prior = k, prior
        self.counts: dict[tuple, list[float]] = {}
        self.regime_ewma = 0.5

    def regime(self) -> int:
        return 0 if self.regime_ewma < 0.3 else (1 if self.regime_ewma < 0.7 else 2)

    def _keys(self, feat: tuple[int, int], j: int) -> list[tuple]:
        n, agree = min(feat[0], 4), min(feat[1], 3)
        jj = min(j, 7)
        return [(), (jj,), (agree, jj), (n, agree, jj), (self.regime(), n, agree, jj)]

    def conf(self, feats: list[tuple[int, int]]) -> list[float]:
        out = []
        for j, f in enumerate(feats):
            p = self.prior
            for key in self._keys(f, j):
                acc, tries = self.counts.get(key, (0.0, 0.0))
                p = (acc + self.k * p) / (tries + self.k)
            out.append(p)
        return out

    def update(self, feats: list[tuple[int, int]], n_verified: int, n_accepted: int):
        for j in range(min(n_verified, n_accepted + 1)):  # censored after the first rejection
            ok = 1.0 if j < n_accepted else 0.0
            for key in self._keys(feats[j], j):
                c = self.counts.setdefault(key, [0.0, 0.0])
                c[0] += ok
                c[1] += 1.0
        if n_verified:
            self.regime_ewma = 0.7 * self.regime_ewma + 0.3 * (n_accepted / n_verified)


def schedule(conf: list[float], cost: StepCost, ctx: int) -> tuple[int, list[float]]:
    """argmax_l (1 + sum_{j<l} a_j) / T(1 + l, ctx) with a_j = prod_{i<=j} c_i. Returns (l, a)."""
    a, run = [], 1.0
    for c in conf:
        run *= c
        a.append(run)
    best_l, best, tau = 0, 1.0 / cost(1, ctx), 1.0
    for l_, aj in enumerate(a, start=1):
        tau += aj
        th = tau / cost(1 + l_, ctx)
        if th > best:
            best_l, best = l_, th
    return best_l, a


class SpecPolicy:
    """Per-step drafting decision shared by generation and the exact replay simulator.

    kind: "sched" (confidence-scheduled, default), "ewma" (original heuristic), "fixed" (always the
    longest draft), "none" (no speculation).

    The confidence model learns in hindsight from *every* proposed draft, verified or not: once the
    tokens at the draft's positions have been emitted, draft token j counts as accepted iff it and all
    earlier draft tokens equal what was emitted. That is exactly the verification outcome under greedy
    decoding (and has the acceptance probability p(d) under sampling), so the scheduler cannot lock
    itself out after a run of misses, and it only ever reads the past (exactness is preserved)."""

    def __init__(self, kind: str = "sched", cost: StepCost | None = None, max_draft: int = 7, max_n: int = 4):
        self.kind = kind
        self.cost = cost or StepCost(1.0, 0.03, 0.0)
        self.drafter = NGramDrafter(max_n=max_n, max_draft=max_draft)
        self.conf = ConfidenceModel()
        self._pending: list[tuple[int, list[int], list[tuple[int, int]], list[float]]] = []
        self.calib: list[tuple[float, float]] = []  # (predicted survival, observed) for ECE
        self.stats = {"steps": 0, "drafted_verified": 0, "accepted": 0, "proposed": 0}

    def reset(self, tokens: list[int]):
        self.drafter.reset(tokens)
        self._pending = []

    def plan(self, ctx: int) -> list[int]:
        if self.kind == "none":
            return []
        draft, feats = self.drafter.propose_full()
        if not draft:
            return []
        self.stats["proposed"] += len(draft)
        conf = self.conf.conf(feats)
        if self.kind == "ewma":
            l_ = min(len(draft), self.drafter.draft_len())
        elif self.kind == "fixed":
            l_ = len(draft)
        else:
            l_ = schedule(conf, self.cost, ctx)[0]
        a, run = [], 1.0
        for c in conf:
            run *= c
            a.append(run)
        self._pending.append((len(self.drafter.tokens), draft, feats, a))
        return draft[:l_]

    def observe(self, draft: list[int], emitted: list[int]):
        n_acc = len(emitted) - 1
        self.stats["steps"] += 1
        self.stats["drafted_verified"] += len(draft)
        self.stats["accepted"] += n_acc
        if draft:
            self.drafter.update_stats(len(draft), n_acc)
        self.drafter.extend(emitted)
        self._resolve()

    def _resolve(self):
        toks, keep = self.drafter.tokens, []
        L = len(toks)
        for p0, d, f, a in self._pending:
            avail = min(L - p0, len(d))
            n = 0
            while n < avail and d[n] == toks[p0 + n]:
                n += 1
            mismatch = n < avail
            if not mismatch and avail < len(d):
                keep.append((p0, d, f, a))  # not all of its positions emitted yet
                continue
            evaluated = n + 1 if mismatch else n
            self.conf.update(f, evaluated, n)
            self.calib.extend((a[j], 1.0 if j < n else 0.0) for j in range(evaluated))
        self._pending = keep

    def ece(self, bins: int = 10) -> float:
        if not self.calib:
            return float("nan")
        tot, err = len(self.calib), 0.0
        for b in range(bins):
            sel = [(p, o) for p, o in self.calib if b / bins <= p < (b + 1) / bins or (b == bins - 1 and p == 1.0)]
            if sel:
                err += len(sel) / tot * abs(sum(p for p, _ in sel) / len(sel) - sum(o for _, o in sel) / len(sel))
        return err


def replay(prompt: list[int], out: list[int], kind: str, cost: StepCost, max_draft: int = 7, ctx0: int = 0) -> dict:
    """Exact replay of greedy speculative generation. Under greedy decoding the output does not depend
    on the drafting policy, so from one recorded (prompt, output) pair the steps, verified drafts and
    modeled time of any policy under any step-cost curve follow exactly: a draft token is accepted iff
    it equals the recorded next token. Mirrors ``generate`` step for step (same SpecPolicy code).
    ``ctx0`` adds context in front of the prompt for the cost model only (a long-running session)."""
    pol = SpecPolicy(kind, cost, max_draft)
    pol.reset(list(prompt) + out[:1])
    i, ctx, t = 1, len(prompt) + ctx0, 0.0
    while i < len(out):
        draft = pol.plan(ctx)
        n = 0
        while n < len(draft) and i + n < len(out) and draft[n] == out[i + n]:
            n += 1
        emitted = out[i: i + n + 1]
        t += cost(1 + len(draft), ctx)
        pol.observe(draft, emitted)
        ctx += len(emitted)
        i += len(emitted)
    st = pol.stats
    return {"policy": kind, "steps": st["steps"], "tokens": len(out) - 1, "tokens_per_step": (len(out) - 1) / max(st["steps"], 1),
            "verified_drafts": st["drafted_verified"], "rejected_drafts": st["drafted_verified"] - st["accepted"],
            "time": t, "tok_per_s": (len(out) - 1) / max(t, 1e-12), "ece": pol.ece()}


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
