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
