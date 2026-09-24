"""Per-matrix bit allocation from the linearity theorem (HIGGS, arXiv 2411.17525).

For small enough relative errors t_l^2 = ||W_l - W^_l||^2 / ||W_l||^2 the perplexity increase is
additive: E[PPL] - PPL* ~= sum_l alpha_l t_l^2(b_l), with layer constants alpha_l independent of the
quantizer. Given measured t_l^2(b) for every candidate bitwidth and alpha_l (calibrated by noise
injection, or a prior), the optimal assignment under a memory budget is a multiple-choice knapsack,
solved exactly here by dynamic programming over a discretised budget.
"""

from __future__ import annotations

import math

import numpy as np

from .config import TextConfig
from .quant import bits_per_weight, matrix_list

CHOICES = (4, 5, 6, 8)


def prior_alphas(cfg: TextConfig) -> dict[str, float]:
    """Data-free prior for alpha_l when no calibration is available.

    Heuristic, documented as such: the logit layer and the residual *writers* (down / out / o
    projections) are the most sensitive per unit of relative error, and so are the first and last
    blocks. Replace with ``tools/calibrate_alpha.py`` output for a measured allocation.
    """
    L = cfg.num_hidden_layers
    out = {}
    for name, N, K in matrix_list(cfg):
        if name == "lm_head":
            out[name] = 8.0
            continue
        i = int(name.split(".")[1])
        depth = 1.6 if i < 2 or i >= L - 2 else 1.0
        kind = name.split(".")[-1]
        typ = {"down": 1.5, "out_proj": 1.3, "o_proj": 1.3, "in_proj": 1.0, "qkv_proj": 1.1, "gate_up": 1.0}[kind]
        out[name] = depth * typ
    return out


def allocate(items: list[tuple[str, int, dict[int, float], float]], budget_bytes: float,
             unit_bytes: float | None = None, min_bits: dict[str, int] | None = None) -> dict[str, int]:
    """items: (name, numel, {bits: t2}, alpha). Minimise sum alpha*t2 s.t. sum bytes <= budget.

    Costs are rounded up to ``unit_bytes`` (default: budget / 20000, i.e. a 20k-cell DP), so the
    returned assignment always satisfies the real budget.
    """
    min_bits = min_bits or {}
    unit_bytes = unit_bytes or max(1.0, budget_bytes / 20000)
    B = int(budget_bytes // unit_bytes)
    INF = float("inf")
    dp = np.full(B + 1, INF)
    dp[0] = 0.0
    choice = []
    for name, numel, t2, alpha in items:
        opts = [(b, math.ceil(numel * bits_per_weight(b) / 8 / unit_bytes), alpha * t2[b])
                for b in sorted(t2) if b >= min_bits.get(name, 0)]
        new = np.full(B + 1, INF)
        arg = np.full(B + 1, -1, dtype=np.int8)
        for oi, (b, c, loss) in enumerate(opts):
            if c > B:
                continue
            cand = np.full(B + 1, INF)
            cand[c:] = dp[: B + 1 - c] + loss
            better = cand < new
            new[better] = cand[better]
            arg[better] = oi
        dp = new
        choice.append((opts, arg))
    if not np.isfinite(dp).any():
        raise ValueError("budget too small for the allowed bitwidths")
    j = int(np.argmin(dp))
    out = {}
    for (name, *_), (opts, arg) in zip(reversed(items), reversed(choice)):
        oi = int(arg[j])
        b, c, _ = opts[oi]
        out[name] = b
        j -= c
    return out


def predicted_loss(items, bits: dict[str, int]) -> float:
    return sum(alpha * t2[bits[name]] for name, _, t2, alpha in items)
