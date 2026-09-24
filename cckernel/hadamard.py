"""Randomized Hadamard transforms (QuIP#, arXiv 2402.04396; QuaRot/SpinQuant R1 rotation).

Q = H_n diag(s) / sqrt(n) with H_n the Sylvester Walsh-Hadamard matrix and s a random sign vector.
Q is orthogonal, so rms(Q x) = rms(x): the rotation commutes with a weightless RMSNorm and can be
folded into the weights of every residual-stream reader and writer at zero runtime cost.

All transforms are implemented with the O(n log n) fast Walsh-Hadamard transform; the n x n
matrix is never materialized.
"""

from __future__ import annotations

import torch


def fwht(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard transform along ``dim`` (size must be a power of two)."""
    x = x.transpose(dim, -1) if dim not in (-1, x.ndim - 1) else x
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"fwht needs a power-of-two size, got {n}")
    shape = x.shape
    y = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        h *= 2
    y = y.reshape(shape) * (n ** -0.5)
    return y.transpose(dim, -1) if dim not in (-1, x.ndim - 1) else y


def random_signs(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float32)


def rotate_vec(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Q x for x[..., n] (applied along the last dim)."""
    return fwht(x * signs.to(x.dtype), dim=-1)


def rotate_reader(w: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """W Q^T for a residual reader W[out, n]. (Q^T = diag(s) H / sqrt(n), H symmetric.)"""
    return fwht(w * signs.to(w.dtype)[None, :], dim=-1)


def rotate_writer(w: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Q W for a residual writer W[n, in]."""
    return fwht(w * signs.to(w.dtype)[:, None], dim=0)
