"""Equivalence of the three gated-delta-rule algorithms and the deferred commit (fp64)."""

import torch

from cckernel.reference import gdn_chunked, gdn_commit, gdn_recurrent, gdn_step_onepass, l2norm

torch.manual_seed(0)
DT = torch.float64


def _inputs(T=37, H=3, dk=16, dv=8):
    q = l2norm(torch.randn(T, H, dk, dtype=DT)) * dk ** -0.5
    k = l2norm(torch.randn(T, H, dk, dtype=DT))
    v = torch.randn(T, H, dv, dtype=DT)
    g = -torch.rand(T, H, dtype=DT) * 2.0  # log decays <= 0
    beta = torch.rand(T, H, dtype=DT)
    S0 = torch.randn(H, dk, dv, dtype=DT)
    return q, k, v, g, beta, S0


def test_onepass_matches_recurrent():
    q, k, v, g, beta, S0 = _inputs()
    o_ref, S_ref = gdn_recurrent(q, k, v, g, beta, S0)
    S = S0.clone()
    outs = []
    for t in range(q.shape[0]):
        o, _, S = gdn_step_onepass(S, q[t], k[t], v[t], g[t], beta[t])
        outs.append(o)
    torch.testing.assert_close(torch.stack(outs), o_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(S, S_ref, rtol=1e-10, atol=1e-10)


def test_onepass_column_independence():
    """Each value column of S evolves independently -> dv can be split across CTAs."""
    q, k, v, g, beta, S0 = _inputs(T=1)
    o, _, S1 = gdn_step_onepass(S0, q[0], k[0], v[0], g[0], beta[0])
    for sl in (slice(0, 3), slice(3, 8)):
        o_s, _, S_s = gdn_step_onepass(S0[..., sl], q[0], k[0], v[0][..., sl], g[0], beta[0])
        torch.testing.assert_close(o_s, o[..., sl])
        torch.testing.assert_close(S_s, S1[..., sl])


def test_chunked_matches_recurrent():
    for T, C in ((37, 8), (64, 16), (5, 16), (130, 64)):
        q, k, v, g, beta, S0 = _inputs(T=T)
        o_ref, S_ref = gdn_recurrent(q, k, v, g, beta, S0)
        o, S = gdn_chunked(q, k, v, g, beta, S0, chunk=C)
        torch.testing.assert_close(o, o_ref, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(S, S_ref, rtol=1e-9, atol=1e-9)


def test_chunked_strong_decay_is_stable():
    """Very negative log-decays: all exponents are differences G_r - G_i <= 0 (no overflow)."""
    q, k, v, g, beta, S0 = _inputs(T=64)
    g = g * 40.0
    o_ref, S_ref = gdn_recurrent(q, k, v, g, beta, S0)
    o, S = gdn_chunked(q, k, v, g, beta, S0, chunk=64)
    assert torch.isfinite(o).all()
    torch.testing.assert_close(o, o_ref, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(S, S_ref, rtol=1e-8, atol=1e-8)


def test_deferred_commit_reconstructs_every_prefix():
    q, k, v, g, beta, S0 = _inputs(T=8)
    S = S0.clone()
    states, us = [], []
    for t in range(8):
        _, u, S = gdn_step_onepass(S, q[t], k[t], v[t], g[t], beta[t])
        states.append(S)
        us.append(u)
    us = torch.stack(us)
    for a in range(0, 9):
        Sa = gdn_commit(S0, k[:a], us[:a], g[:a])
        torch.testing.assert_close(Sa, S0 if a == 0 else states[a - 1], rtol=1e-10, atol=1e-10)
