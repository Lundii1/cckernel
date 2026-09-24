"""Vectorised torch implementations used by the engine's prefill path (and as GPU test oracles).

Rounding points mirror the CUDA kernels (which mirror HF's bf16 dataflow) so prefill and decode
produce consistent caches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bf16r(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def conv_silu(x_hist: torch.Tensor, x_new: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x_hist: [K-1, C] previous inputs, x_new: [T, C], w: [C, K] -> bf16-rounded silu(conv) [T, C]."""
    K = w.shape[1]
    full = torch.cat([x_hist, x_new], dim=0).float()
    win = full.unfold(0, K, 1)  # [T, C, K]
    acc = (win * w[None]).sum(-1)
    return bf16r(F.silu(bf16r(acc)))


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gdn_chunk(q, k, v, g, beta, S0, chunk: int = 64):
    """Chunkwise gated delta rule (WY/UT), vectorised over chunks; only the state scan is serial.

    q, k: [T, H, dk], v: [T, H, dv], g, beta: [T, H], S0: [H, dk, dv] (fp32). See docs/MATH.md.
    """
    T, H, dk = k.shape
    dv = v.shape[-1]
    pad = (-T) % chunk
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, 0, 0, pad)) for x in (g, beta))
    n = q.shape[0] // chunk
    rs = lambda x: x.reshape(n, chunk, H, -1).permute(2, 0, 1, 3)  # noqa: E731  [H, n, C, d]
    q, k, v = rs(q), rs(k), rs(v)
    g = g.reshape(n, chunk, H).permute(2, 0, 1)
    beta = beta.reshape(n, chunk, H).permute(2, 0, 1)
    G = torch.cumsum(g, dim=-1)
    idx = torch.arange(chunk, device=q.device)
    incl = idx[:, None] >= idx[None, :]
    diff = (G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf"))
    Gam = diff.exp()
    KK = k @ k.transpose(-1, -2)
    L = (beta[..., :, None] * Gam * KK).masked_fill(~(idx[:, None] > idx[None, :]), 0.0)
    A = L + torch.eye(chunk, device=q.device, dtype=q.dtype)
    rhs = torch.cat([beta[..., None] * v, (beta * G.exp())[..., None] * k], dim=-1)
    X = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
    U, W = X[..., :dv], X[..., dv:]
    QK = (q @ k.transpose(-1, -2)) * Gam
    qg = G.exp()[..., None] * q
    kt = (G[..., -1:] - G).exp()[..., None] * k
    decay = G[..., -1].exp()
    S = S0.clone()
    outs = []
    for c in range(n):
        Ut = U[:, c] - W[:, c] @ S
        outs.append(qg[:, c] @ S + QK[:, c] @ Ut)
        S = decay[:, c, None, None] * S + kt[:, c].transpose(-1, -2) @ Ut
    o = torch.stack(outs, dim=1).permute(1, 2, 0, 3).reshape(n * chunk, H, dv)
    return o[:T], S


def rope_bf16(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Partial rotate-half RoPE on x[T, H, D] (first 2*len(inv_freq) dims) with the kernel's rounding."""
    rot = inv_freq.numel() * 2
    ang = positions.float()[:, None] * inv_freq[None, :]
    cs, sn = bf16r(torch.cos(ang))[:, None, :], bf16r(torch.sin(ang))[:, None, :]
    xr = x[..., :rot]
    x1, x2 = xr[..., : rot // 2], xr[..., rot // 2:]
    cs2, sn2 = torch.cat([cs, cs], -1), torch.cat([sn, sn], -1)
    partner = torch.cat([-x2, x1], dim=-1)
    out = bf16r(bf16r(xr * cs2) + bf16r(partner * sn2))
    return torch.cat([out, x[..., rot:]], dim=-1)


def rms_norm_w(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return bf16r(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w)


def gated_head_norm(o: torch.Tensor, z: torch.Tensor, head_dim: int, eps: float) -> torch.Tensor:
    """bf16(bf16(rms_head(o)) * silu(z)) for o, z: [T, H*head_dim] (gated-norm weight folded away)."""
    T = o.shape[0]
    of = o.float().reshape(T, -1, head_dim)
    t = bf16r(of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps))
    return (t * F.silu(z.float().reshape(T, -1, head_dim))).reshape(T, -1).to(torch.bfloat16)


def inv_freq_hf(rotary_dim: int, theta: float, device=None) -> torch.Tensor:
    """Exactly HF's default RoPE inverse frequencies (computed in fp32)."""
    return (1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                               / rotary_dim)))
