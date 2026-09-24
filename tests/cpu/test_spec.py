"""N-gram drafter and exactness of the acceptance rule."""

import torch

from cckernel.spec import NGramDrafter, accept


def test_drafter_finds_previous_continuation():
    d = NGramDrafter(max_n=3, max_draft=4)
    d.reset([5, 6, 7, 8, 9, 1, 5, 6, 7])
    d.ewma = 10
    assert d.propose() == [8, 9, 1, 5]


def test_drafter_no_match():
    d = NGramDrafter()
    d.reset([1, 2, 3])
    assert d.propose() == []


def test_greedy_accept():
    V = 10
    logits = torch.full((4, V), -10.0)
    for i, t in enumerate([3, 4, 7, 2]):
        logits[i, t] = 10.0
    assert accept(logits, [3, 4, 5]) == [3, 4, 7]
    assert accept(logits, [3, 4, 7]) == [3, 4, 7, 2]
    assert accept(logits, [9, 4, 7]) == [3]


def test_sampling_accept_is_exact():
    """First emitted token must be distributed exactly as softmax(logits[0]/T) for any draft."""
    torch.manual_seed(0)
    V = 5
    logits = torch.randn(2, V)
    p = (logits[0] / 0.7).softmax(-1)
    g = torch.Generator().manual_seed(1)
    n = 40000
    counts = torch.zeros(V)
    for _ in range(n):
        counts[accept(logits, [2], temperature=0.7, generator=g)[0]] += 1
    torch.testing.assert_close(counts / n, p, atol=0.01, rtol=0)


# ------------------------------------------------------------------------------------------ scheduler
from cckernel.spec import ConfidenceModel, SpecPolicy, StepCost, replay, schedule  # noqa: E402


def test_schedule_is_argmax_of_expected_throughput():
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        conf = torch.rand(int(torch.randint(0, 8, (1,), generator=g)), generator=g).tolist()
        cost = StepCost(*(torch.rand(3, generator=g) * torch.tensor([1.0, 0.3, 1e-4])).tolist())
        ctx = int(torch.randint(0, 5000, (1,), generator=g))
        l_, a = schedule(conf, cost, ctx)
        th = []
        for ll in range(len(conf) + 1):
            tau = 1 + sum(float(torch.tensor(conf[:j + 1]).prod()) for j in range(ll))
            th.append(tau / cost(1 + ll, ctx))
        assert abs(th[l_] - max(th)) < 1e-9 * max(th)


def test_schedule_extremes():
    flat = StepCost(1.0, 0.0, 0.0)
    assert schedule([0.9, 0.8, 0.1], flat, 0)[0] == 3  # verification is free: verify everything
    steep = StepCost(1.0, 1.0, 0.0)
    assert schedule([0.2, 0.2], steep, 0)[0] == 0  # expensive and unlikely: do not speculate


def test_confidence_counts_are_censored():
    cm = ConfidenceModel()
    feats = [(3, 1)] * 5
    cm.update(feats, n_verified=5, n_accepted=2)
    assert cm.counts[(0,)] == [1.0, 1.0] and cm.counts[(1,)] == [1.0, 1.0]
    assert cm.counts[(2,)] == [0.0, 1.0] and (3,) not in cm.counts and (4,) not in cm.counts
    c = cm.conf(feats)
    assert c[0] > 0.5 and c[2] < 0.5


class _MarkovEngine:
    """Engine stand-in whose next-token distribution depends only on the previous token."""

    def __init__(self, logits):
        self.L = logits
        self.len_host, self.max_len, self.dev = 0, 10 ** 6, torch.device("cpu")

    def prefill(self, ids):
        self.len_host += len(ids)
        return self.L[ids[-1]]

    def step(self, toks):
        return self.L[torch.tensor(toks)]

    def commit(self, n):
        self.len_host += n


def test_scheduled_sampling_is_exact():
    """With temperature > 0, the distribution of generated sequences is the target's for every policy."""
    from cckernel.generate import generate

    torch.manual_seed(0)
    V, n_new, N = 3, 4, 6000
    L = torch.randn(V, V) * 1.5
    P = L.softmax(-1)
    prompt = [0, 1, 2, 0, 1, 2, 0, 1]
    exact = {}
    for seq in torch.cartesian_prod(*[torch.arange(V)] * n_new).tolist():
        p, prev = 1.0, prompt[-1]
        for t in seq:
            p *= float(P[prev, t])
            prev = t
        exact[tuple(seq)] = p
    for policy in ("sched", "fixed"):
        counts = {}
        for s in range(N):
            out, _ = generate(_MarkovEngine(L), prompt, n_new, set(), temperature=1.0, spec=True, seed=s,
                              policy=policy, cost=StepCost(1.0, 0.05, 0.0))
            key = tuple(out[:n_new])
            counts[key] = counts.get(key, 0) + 1
        tv = 0.5 * sum(abs(counts.get(k, 0) / N - p) for k, p in exact.items())
        assert tv < 0.06, (policy, tv)


def test_replay_matches_policy_decisions():
    """replay() re-derives the same steps as a live run from the recorded output."""
    from cckernel.generate import generate

    L = torch.randn(7, 7) * 3
    prompt = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    cost = StepCost(1.0, 0.05, 0.0)
    for policy in ("sched", "ewma", "fixed"):
        out, st = generate(_MarkovEngine(L), prompt, 40, set(), spec=True, policy=policy, cost=cost)
        r = replay(prompt, out, policy, cost)
        assert r["steps"] == st["steps"] and r["verified_drafts"] == st["verified_drafts"]
    assert replay(prompt, out, "none", cost)["steps"] == len(out) - 1
    assert SpecPolicy("none").plan(0) == []
