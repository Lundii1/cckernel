"""Pure-PyTorch fp32 reference of the Qwen3.5 text decoder (HF ``modeling_qwen3_5`` semantics).

This is the oracle for every CUDA kernel and for the offline weight folding. It also contains the
*algorithmic* models of the kernels (one-pass delta-rule step, chunked WY/UT prefill, deferred
commit for speculative decoding), so the math can be verified on a CPU.

Notation, HF layout: per value head the state is S in R^{dk x dv} and
    S_t = a_t S_{t-1} + k_t u_t^T,   u_t = b_t (v_t - a_t S_{t-1}^T k_t),   o_t = S_t^T q_t
with a_t = exp(g_t), g_t = -exp(A_log) softplus(a + dt_bias) and b_t = sigmoid(b)
(Gated DeltaNet, arXiv 2412.06464, eq. 10, transposed).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import TextConfig

# ------------------------------------------------------------------------------------------------
# elementwise building blocks
# ------------------------------------------------------------------------------------------------


def _f(x: torch.Tensor) -> torch.Tensor:
    """Upcast to fp32 unless already fp64 (tests run the reference in fp64)."""
    return x if x.dtype == torch.float64 else x.float()


def rmsnorm_zc(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Zero-centered RMSNorm used by Qwen3.5: x / rms(x) * (1 + w)."""
    x = _f(x)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + _f(w))


def rmsnorm_gated(x: torch.Tensor, w: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
    """GDN output norm: w * x / rms(x) * silu(z) (plain, not zero-centered, weight)."""
    x = _f(x)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return _f(w) * x * F.silu(_f(z))


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gdn_gates(a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
    """Returns (g, beta): log-decay g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)."""
    g = -_f(A_log).exp() * F.softplus(_f(a) + _f(dt_bias))
    return g, torch.sigmoid(_f(b))


def causal_conv1d(x: torch.Tensor, w: torch.Tensor, state: torch.Tensor | None):
    """Depthwise causal conv + SiLU.

    x: [T, C] new inputs, w: [C, K], state: [C, K-1] previous inputs (oldest first) or None.
    Returns (y [T, C], new_state [C, K-1]).
    """
    T, C = x.shape
    K = w.shape[1]
    if state is None:
        state = x.new_zeros(C, K - 1)
    full = torch.cat([state.t().to(x.dtype), x], dim=0)  # [K-1+T, C]
    windows = full.unfold(0, K, 1)  # [T, C, K]
    y = F.silu((_f(windows) * _f(w)[None]).sum(-1))
    return y, full[-(K - 1):].t().contiguous()


def rope_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    freqs = positions.double()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, H, D]; rotate_half RoPE on the first ``cos.shape[-1]`` dims."""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    x1, x2 = xr[..., : rd // 2], xr[..., rd // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    c, s = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)
    return torch.cat([xr * c + rot * s, xp], dim=-1)


# ------------------------------------------------------------------------------------------------
# gated delta rule: three equivalent algorithms
# ------------------------------------------------------------------------------------------------


def gdn_recurrent(q, k, v, g, beta, S0):
    """Token-by-token gated delta rule, exactly as HF ``torch_recurrent_gated_delta_rule``.

    q, k: [T, H, dk] (already L2-normalized, q already scaled by dk^-0.5), v: [T, H, dv],
    g, beta: [T, H], S0: [H, dk, dv]. Returns (o [T, H, dv], S [H, dk, dv]).
    """
    S = S0.clone()
    out = []
    for t in range(q.shape[0]):
        S = S * g[t].exp()[:, None, None]
        kv_mem = torch.einsum("hkv,hk->hv", S, k[t])
        delta = (v[t] - kv_mem) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        out.append(torch.einsum("hkv,hk->hv", S, q[t]))
    return torch.stack(out), S


def gdn_step_onepass(S, q, k, v, g, beta):
    """One decode step written the way the CUDA kernel computes it (single pass over S).

    Using r = S^T k and p = S^T q of the *old* state (one read of S):
        u = beta (v - alpha r),  o = alpha p + u (k.q),  S' = alpha S + k u^T  (one write of S).
    Every quantity for value column j only needs column j of S -> columns split across CTAs.
    Returns (o [H, dv], u [H, dv], S').
    """
    alpha = g.exp()[:, None]
    r = torch.einsum("hkv,hk->hv", S, k)
    p = torch.einsum("hkv,hk->hv", S, q)
    u = beta[:, None] * (v - alpha * r)
    kq = (k * q).sum(-1, keepdim=True)
    o = alpha * p + u * kq
    S_new = alpha[:, :, None] * S + k[:, :, None] * u[:, None, :]
    return o, u, S_new


def gdn_commit(S0, ks, us, gs):
    """Deferred commit (TreeWY arXiv 2608.20961 / ReplaySSM): rebuild S_a from S_0 and pending
    pseudo-values without having stored S_1..S_a:
        S_a = gamma_a S_0 + sum_{i<=a} (gamma_a / gamma_i) k_i u_i^T,  gamma_i = exp(sum_{j<=i} g_j).
    ks: [a, H, dk], us: [a, H, dv], gs: [a, H].
    """
    if ks.shape[0] == 0:
        return S0.clone()
    G = torch.cumsum(gs, dim=0)  # [a, H]
    w = (G[-1][None] - G).exp()  # gamma_a / gamma_i, <= 1
    return G[-1].exp()[:, None, None] * S0 + torch.einsum("ah,ahk,ahv->hkv", w, ks, us)


def solve_unit_lower(L: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve (I + L) X = B for strictly lower-triangular L by forward substitution."""
    n = L.shape[-1]
    X = B.clone()
    for r in range(n):
        if r:
            X[..., r, :] = B[..., r, :] - (L[..., r, :r, None] * X[..., :r, :]).sum(-2)
    return X


def gdn_chunked(q, k, v, g, beta, S0, chunk: int = 64):
    """Chunkwise-parallel gated delta rule (WY representation + UT transform).

    Derivation (HF layout), within a chunk starting at state S_0 with gamma_r = exp(G_r),
    G = cumsum(g):
        (I + L) U~ = diag(beta) (V - diag(gamma) K S_0),   L_ri = beta_r (gamma_r/gamma_i) k_r.k_i, i<r
        U = T V, W = T diag(gamma) K with T = (I+L)^{-1} diag(beta)       ->  U~ = U - W S_0
        O      = diag(gamma) Q S_0 + (Q K^T . Gamma . M) U~,   Gamma_ri = gamma_r / gamma_i
        S_next = gamma_C S_0 + (diag(gamma_C / gamma) K)^T U~
    Every exponent is a difference G_r - G_i with r >= i, hence <= 0: no overflow.
    """
    T, H, dk = k.shape
    dv = v.shape[-1]
    pad = (-T) % chunk
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, 0, 0, pad)) for x in (g, beta))
    n = q.shape[0] // chunk
    S = S0.clone()
    outs = []
    tri_strict = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool), -1)
    tri_incl = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool), 0)
    for c in range(n):
        sl = slice(c * chunk, (c + 1) * chunk)
        qc, kc, vc = (x[sl].transpose(0, 1) for x in (q, k, v))  # [H, C, d]
        gc, bc = g[sl].t(), beta[sl].t()  # [H, C]
        G = torch.cumsum(gc, dim=-1)
        Gam = (G[:, :, None] - G[:, None, :]).masked_fill(~tri_incl, float("-inf")).exp()
        KK = kc @ kc.transpose(-1, -2)
        L = (bc[:, :, None] * Gam * KK).masked_fill(~tri_strict, 0.0)
        rhs = torch.cat([bc[:, :, None] * vc, bc[:, :, None] * G.exp()[:, :, None] * kc], dim=-1)
        X = solve_unit_lower(L, rhs)
        U, W = X[..., :dv], X[..., dv:]
        Ut = U - W @ S  # [H, C, dv]
        QK = (qc @ kc.transpose(-1, -2)) * Gam  # Gam already zero above the diagonal
        o = (G.exp()[:, :, None] * qc) @ S + QK @ Ut
        Kt = (G[:, -1:] - G).exp()[:, :, None] * kc
        S = G[:, -1].exp()[:, None, None] * S + Kt.transpose(-1, -2) @ Ut
        outs.append(o.transpose(0, 1))
    return torch.cat(outs)[:T], S


# ------------------------------------------------------------------------------------------------
# full model
# ------------------------------------------------------------------------------------------------


class RefCache:
    def __init__(self, cfg: TextConfig):
        self.conv = {}  # layer -> [C, K-1]
        self.rec = {}  # layer -> [Hv, dk, dv]
        self.k = {}  # layer -> [T, Hkv, D]
        self.v = {}
        self.pos = 0


class RefModel:
    """fp32 reference. ``w`` maps canonical names (see loader.canonical_name) to tensors."""

    def __init__(self, cfg: TextConfig, w: dict[str, torch.Tensor], gdn_algo: str = "recurrent", dtype=torch.float32):
        self.cfg = cfg
        self.w = {k: v.to(dtype) for k, v in w.items()}
        self.gdn_algo = gdn_algo
        self.dtype = dtype

    def W(self, name):
        return self.w[name]

    # -- mixers ----------------------------------------------------------------------------------
    def gdn(self, i: int, x: torch.Tensor, cache: RefCache) -> torch.Tensor:
        c, p = self.cfg, f"layers.{i}.linear_attn."
        T = x.shape[0]
        Hk, Hv, dk, dv = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
        mixed = x @ self.W(p + "in_proj_qkv.weight").t()
        z = x @ self.W(p + "in_proj_z.weight").t()
        b = x @ self.W(p + "in_proj_b.weight").t()
        a = x @ self.W(p + "in_proj_a.weight").t()
        convw = self.W(p + "conv1d.weight").reshape(c.gdn_conv_dim, -1)
        mixed, cache.conv[i] = causal_conv1d(mixed, convw, cache.conv.get(i))
        q, k, v = torch.split(mixed, [c.gdn_key_dim, c.gdn_key_dim, c.gdn_value_dim], dim=-1)
        q = q.reshape(T, Hk, dk).repeat_interleave(Hv // Hk, dim=1)
        k = k.reshape(T, Hk, dk).repeat_interleave(Hv // Hk, dim=1)
        v = v.reshape(T, Hv, dv)
        g, beta = gdn_gates(a, b, self.W(p + "A_log"), self.W(p + "dt_bias"))
        q = l2norm(q) * dk ** -0.5
        k = l2norm(k)
        S0 = cache.rec.get(i)
        if S0 is None:
            S0 = torch.zeros(Hv, dk, dv, dtype=q.dtype)
        if self.gdn_algo == "chunked":
            o, S = gdn_chunked(q, k, v, g, beta, S0)
        else:
            o, S = gdn_recurrent(q, k, v, g, beta, S0)
        cache.rec[i] = S
        o = rmsnorm_gated(o, self.W(p + "norm.weight"), z.reshape(T, Hv, dv), c.rms_norm_eps).reshape(T, -1)
        return o @ self.W(p + "out_proj.weight").t()

    def attn(self, i: int, x: torch.Tensor, cache: RefCache, positions: torch.Tensor) -> torch.Tensor:
        c, p = self.cfg, f"layers.{i}.self_attn."
        T, H, Hkv, D = x.shape[0], c.num_attention_heads, c.num_key_value_heads, c.head_dim
        qg = (x @ self.W(p + "q_proj.weight").t()).reshape(T, H, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(T, H * D)
        k = (x @ self.W(p + "k_proj.weight").t()).reshape(T, Hkv, D)
        v = (x @ self.W(p + "v_proj.weight").t()).reshape(T, Hkv, D)
        q = rmsnorm_zc(q, self.W(p + "q_norm.weight"), c.rms_norm_eps)
        k = rmsnorm_zc(k, self.W(p + "k_norm.weight"), c.rms_norm_eps)
        cos, sin = rope_cos_sin(positions, c.rotary_dim, c.rope_theta)
        q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
        K = torch.cat([cache.k[i], k]) if i in cache.k else k
        V = torch.cat([cache.v[i], v]) if i in cache.v else v
        cache.k[i], cache.v[i] = K, V
        rep = H // Hkv
        Kr, Vr = K.repeat_interleave(rep, dim=1), V.repeat_interleave(rep, dim=1)
        scores = torch.einsum("thd,shd->hts", q, Kr) * D ** -0.5
        S_len = K.shape[0]
        qpos = positions[:, None]
        kpos = torch.arange(S_len)[None, :]
        scores = scores.masked_fill((kpos > qpos)[None], float("-inf"))
        o = torch.einsum("hts,shd->thd", scores.softmax(-1), Vr).reshape(T, H * D)
        o = o * torch.sigmoid(gate)
        return o @ self.W(p + "o_proj.weight").t()

    def mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        p = f"layers.{i}.mlp."
        return (F.silu(x @ self.W(p + "gate_proj.weight").t()) * (x @ self.W(p + "up_proj.weight").t())) @ self.W(
            p + "down_proj.weight"
        ).t()

    # -- forward ---------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache: RefCache, return_hidden: bool = False) -> torch.Tensor:
        c = self.cfg
        T = ids.shape[0]
        positions = torch.arange(cache.pos, cache.pos + T)
        h = self.W("embed_tokens.weight")[ids].to(self.dtype)
        for i, lt in enumerate(c.layer_types):
            p = f"layers.{i}."
            x = rmsnorm_zc(h, self.W(p + "input_layernorm.weight"), c.rms_norm_eps).to(self.dtype)
            if lt == "linear_attention":
                h = h + self.gdn(i, x, cache)
            else:
                h = h + self.attn(i, x, cache, positions)
            x = rmsnorm_zc(h, self.W(p + "post_attention_layernorm.weight"), c.rms_norm_eps).to(self.dtype)
            h = h + self.mlp(i, x)
        cache.pos += T
        hn = rmsnorm_zc(h, self.W("norm.weight"), c.rms_norm_eps).to(self.dtype)
        logits = hn @ self.W("lm_head.weight").t()
        return (logits, h) if return_hidden else logits


def random_weights(cfg: TextConfig, seed: int = 0, scale: float = 0.05) -> dict[str, torch.Tensor]:
    """Random weights with the checkpoint's names and shapes (for tests)."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, s=scale):
        return torch.randn(*shape, generator=g) * s

    d, I = cfg.hidden_size, cfg.intermediate_size
    w = {"embed_tokens.weight": r(cfg.vocab_size, d, s=1.0), "lm_head.weight": r(cfg.vocab_size, d), "norm.weight": r(d, s=0.1)}
    for i, lt in enumerate(cfg.layer_types):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = r(d, s=0.1)
        w[p + "post_attention_layernorm.weight"] = r(d, s=0.1)
        w[p + "mlp.gate_proj.weight"] = r(I, d)
        w[p + "mlp.up_proj.weight"] = r(I, d)
        w[p + "mlp.down_proj.weight"] = r(d, I)
        if lt == "linear_attention":
            q = p + "linear_attn."
            w[q + "in_proj_qkv.weight"] = r(cfg.gdn_conv_dim, d)
            w[q + "in_proj_z.weight"] = r(cfg.gdn_value_dim, d)
            w[q + "in_proj_b.weight"] = r(cfg.linear_num_value_heads, d)
            w[q + "in_proj_a.weight"] = r(cfg.linear_num_value_heads, d)
            w[q + "conv1d.weight"] = r(cfg.gdn_conv_dim, 1, cfg.linear_conv_kernel_dim, s=0.3)
            w[q + "A_log"] = torch.empty(cfg.linear_num_value_heads).uniform_(math.log(0.5), math.log(8), generator=g)
            w[q + "dt_bias"] = r(cfg.linear_num_value_heads, s=0.5)
            w[q + "norm.weight"] = 1.0 + r(cfg.linear_value_head_dim, s=0.1)
            w[q + "out_proj.weight"] = r(d, cfg.gdn_value_dim)
        else:
            q = p + "self_attn."
            w[q + "q_proj.weight"] = r(cfg.attn_q_dim, d)
            w[q + "k_proj.weight"] = r(cfg.attn_kv_dim, d)
            w[q + "v_proj.weight"] = r(cfg.attn_kv_dim, d)
            w[q + "o_proj.weight"] = r(d, cfg.num_attention_heads * cfg.head_dim)
            w[q + "q_norm.weight"] = r(cfg.head_dim, s=0.1)
            w[q + "k_norm.weight"] = r(cfg.head_dim, s=0.1)
    return w
