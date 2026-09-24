"""Torch implementation of the decoder on *folded* weights (fused matrices, weightless norms,
rotated residual stream). It defines the exact semantics the CUDA engine implements and serves as
the CPU/GPU fallback path. Linear layers are callables so packed (quantized) layers can be used.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .config import TextConfig
from .reference import _f, apply_partial_rope, causal_conv1d, gdn_chunked, gdn_gates, gdn_recurrent, l2norm, rope_cos_sin


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = _f(x)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class FoldedCache:
    def __init__(self):
        self.conv, self.rec, self.k, self.v = {}, {}, {}, {}
        self.pos = 0


class FoldedModel:
    """``lin(name)`` returns a function x[T, K] -> y[T, N]; ``t(name)`` returns a small fp32 tensor."""

    def __init__(self, cfg: TextConfig, tensors: dict[str, torch.Tensor], linears: dict[str, Callable] | None = None,
                 gdn_algo: str = "recurrent", dtype=torch.float32):
        self.cfg = cfg
        self.tensors = tensors
        self.dtype = dtype
        self.gdn_algo = gdn_algo
        self.linears = linears or {}

    def lin(self, name: str, x: torch.Tensor) -> torch.Tensor:
        f = self.linears.get(name)
        if f is not None:
            return f(x)
        return x.to(self.dtype) @ self.tensors[name].to(self.dtype).t()

    def t(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def gdn(self, i, x, cache: FoldedCache):
        c, p = self.cfg, f"layers.{i}."
        T = x.shape[0]
        Hk, Hv, dk, dv = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
        proj = self.lin(p + "in_proj", x)
        C = c.gdn_conv_dim
        mixed, z, b, a = torch.split(proj, [C, c.gdn_value_dim, Hv, Hv], dim=-1)
        mixed, cache.conv[i] = causal_conv1d(mixed, self.t(p + "conv_w"), cache.conv.get(i))
        q, k, v = torch.split(mixed, [c.gdn_key_dim, c.gdn_key_dim, c.gdn_value_dim], dim=-1)
        q = l2norm(_f(q.reshape(T, Hk, dk).repeat_interleave(Hv // Hk, dim=1))) * dk ** -0.5
        k = l2norm(_f(k.reshape(T, Hk, dk).repeat_interleave(Hv // Hk, dim=1)))
        v = _f(v.reshape(T, Hv, dv))
        g, beta = gdn_gates(a, b, self.t(p + "A_log"), self.t(p + "dt_bias"))
        S0 = cache.rec.get(i)
        if S0 is None:
            S0 = torch.zeros(Hv, dk, dv, dtype=q.dtype, device=x.device)
        algo = gdn_chunked if self.gdn_algo == "chunked" else gdn_recurrent
        o, cache.rec[i] = algo(q, k, v, g, beta, S0.to(q.dtype))
        o = rms(o, c.rms_norm_eps) * F.silu(_f(z.reshape(T, Hv, dv)))  # gated-norm weight folded in out_proj
        return self.lin(p + "out_proj", o.reshape(T, -1).to(self.dtype))

    def attn(self, i, x, cache: FoldedCache, positions):
        c, p = self.cfg, f"layers.{i}."
        T, H, Hkv, D = x.shape[0], c.num_attention_heads, c.num_key_value_heads, c.head_dim
        proj = self.lin(p + "qkv_proj", x)
        qg, k, v = torch.split(proj, [c.attn_q_dim, c.attn_kv_dim, c.attn_kv_dim], dim=-1)
        qg = qg.reshape(T, H, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(T, H * D)
        q = rms(q, c.rms_norm_eps) * self.t(p + "q_norm")
        k = rms(k.reshape(T, Hkv, D), c.rms_norm_eps) * self.t(p + "k_norm")
        v = _f(v.reshape(T, Hkv, D))
        cos, sin = rope_cos_sin(positions.cpu(), c.rotary_dim, c.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
        K = torch.cat([cache.k[i], k]) if i in cache.k else k
        V = torch.cat([cache.v[i], v]) if i in cache.v else v
        cache.k[i], cache.v[i] = K, V
        rep = H // Hkv
        scores = torch.einsum("thd,shd->hts", q, K.repeat_interleave(rep, dim=1)) * D ** -0.5
        kpos = torch.arange(K.shape[0], device=x.device)[None, :]
        scores = scores.masked_fill((kpos > positions[:, None])[None], float("-inf"))
        o = torch.einsum("hts,shd->thd", scores.softmax(-1), V.repeat_interleave(rep, dim=1)).reshape(T, H * D)
        o = o * torch.sigmoid(_f(gate))
        return self.lin(p + "o_proj", o.to(self.dtype))

    def mlp(self, i, x):
        gu = self.lin(f"layers.{i}.gate_up", x)
        h = F.silu(_f(gu[:, 0::2])) * _f(gu[:, 1::2])
        return self.lin(f"layers.{i}.down", h.to(self.dtype))

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache: FoldedCache) -> torch.Tensor:
        c = self.cfg
        T = ids.shape[0]
        positions = torch.arange(cache.pos, cache.pos + T, device=ids.device)
        h = _f(self.t("embed")[ids])
        eps = c.rms_norm_eps
        for i, lt in enumerate(c.layer_types):
            x = rms(h, eps).to(self.dtype)
            h = h + _f(self.gdn(i, x, cache) if lt == "linear_attention" else self.attn(i, x, cache, positions))
            x = rms(h, eps).to(self.dtype)
            h = h + _f(self.mlp(i, x))
        cache.pos += T
        return _f(self.lin("lm_head", rms(h, eps).to(self.dtype)))


def fold_all(cfg: TextConfig, w: dict[str, torch.Tensor], signs, dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Fold a full in-memory weight dict (tests / small models)."""
    from .quant import fold_globals, fold_layer

    get = w.__getitem__
    out = {}
    for i in range(cfg.num_hidden_layers):
        for k, v in fold_layer(cfg, i, get, signs, dtype).items():
            out[f"layers.{i}.{k}"] = v
    out.update(fold_globals(cfg, get, signs, dtype))
    return out
