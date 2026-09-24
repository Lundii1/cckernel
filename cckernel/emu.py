"""Torch emulation of every CUDA op in ``cckernel._C`` with identical signatures and in-place
semantics. Used (1) as the backend when the engine runs on CPU (so the full engine, including
ring buffers, deferred commit and speculative verification, is testable without a GPU) and (2) as
the oracle in tests/gpu that checks each kernel against it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import torch_ops as T
from .quant import GROUP, dequant_rtn, unpack

RING = 32
_WCACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _w(lo, hi, scales, bits, N, K) -> torch.Tensor:
    # keyed by object id; holding a reference to `lo` guarantees the id is not reused
    hit = _WCACHE.get(id(lo))
    if hit is not None and hit[0] is lo:
        return hit[1]
    planes = {"q8": lo} if bits == 8 else ({"lo": lo, "hi": hi} if bits in (5, 6) else {"lo": lo})
    planes = {k: v.cpu() for k, v in planes.items()}
    w = dequant_rtn(unpack(planes, bits, K), scales.cpu(), bits).to(lo.device)
    _WCACHE[id(lo)] = (lo, w)
    return w


def clear_cache():
    _WCACHE.clear()


def _epilogue(yv: torch.Tensor, y: torch.Tensor, epi: int):
    if epi == 0:
        y.copy_(yv.to(torch.bfloat16))
    elif epi == 1:
        y.copy_(yv)
    elif epi == 2:
        y.add_(yv)
    else:
        g, u = T.bf16r(yv[..., 0::2]), T.bf16r(yv[..., 1::2])
        y.copy_((T.bf16r(F.silu(g)) * u).to(torch.bfloat16))


def qgemv(lo, hi, scales, bits, N, K, pro, x, z, head_dim, eps, epi, y):
    W = _w(lo, hi, scales, bits, N, K)
    if pro == 1:
        xf = x.float()
        xin = T.bf16r(xf * torch.rsqrt(xf.pow(2).mean() + eps))
    elif pro == 0:
        xin = x.float()
    else:
        o = x.float().reshape(-1, head_dim)
        t = T.bf16r(o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps))
        xin = T.bf16r(t * F.silu(z.float().reshape(-1, head_dim))).reshape(-1)
    _epilogue(W @ xin, y, epi)


def qgemm_skinny(lo, hi, scales, bits, N, K, x, M, epi, y):
    W = _w(lo, hi, scales, bits, N, K)
    _epilogue(x[:M].float() @ W.t(), y[:M], epi)


def dequant(lo, hi, scales, bits, N, K, out):
    out[: N * K].view(N, K).copy_(_w(lo, hi, scales, bits, N, K).to(torch.bfloat16))


def gdn_decode(proj, ring, conv_w, A_log, dt_bias, S, pend_u, pend_g, out, cur_len, n_commit, M, Hk, Hv, C):
    L, a = int(cur_len.item()), int(n_commit.item())
    rep, kd, Vd = Hv // Hk, Hk * 128, Hv * 128
    dev = proj.device

    def conv_out(P):  # all C channels at absolute position P
        vals = []
        for Q in range(P - 3, P + 1):
            if Q < 0:
                vals.append(torch.zeros(C, device=dev))
            elif Q < L:
                vals.append(ring[:, Q & (RING - 1)].float())
            else:
                vals.append(proj[Q - L, :C].float())
        acc = (torch.stack(vals, dim=-1) * conv_w).sum(-1)
        return T.bf16r(F.silu(T.bf16r(acc)))

    for i in range(a):
        kv = conv_out(L - a + i)[kd:2 * kd].reshape(Hk, 128)
        kv = T.l2norm(kv).repeat_interleave(rep, dim=0)
        S.mul_(pend_g[i, :, 0].exp()[:, None, None]).add_(kv[:, :, None] * pend_u[i][:, None, :])
    for m in range(M):
        ring[:, (L + m) & (RING - 1)] = proj[m, :C]
    Sw = S.clone()
    boff = C + Vd
    for m in range(M):
        y = conv_out(L + m)
        q = (T.l2norm(y[:kd].reshape(Hk, 128)) * 128 ** -0.5).repeat_interleave(rep, dim=0)
        k = T.l2norm(y[kd:2 * kd].reshape(Hk, 128)).repeat_interleave(rep, dim=0)
        v = y[2 * kd:].reshape(Hv, 128)
        bb, aa = proj[m, boff:boff + Hv].float(), proj[m, boff + Hv:boff + 2 * Hv].float()
        beta = torch.sigmoid(bb)
        g = -A_log.exp() * F.softplus(aa + dt_bias)
        alpha = g.exp()[:, None]
        r = torch.einsum("hkv,hk->hv", Sw, k)
        p = torch.einsum("hkv,hk->hv", Sw, q)
        u = beta[:, None] * (v - alpha * r)
        o = alpha * p + u * (k * q).sum(-1, keepdim=True)
        Sw = alpha[:, :, None] * Sw + k[:, :, None] * u[:, None, :]
        out[m].copy_(o.reshape(-1).to(torch.bfloat16))
        pend_u[m].copy_(u)
        pend_g[m].copy_(g[:, None].expand(-1, pend_g.shape[-1]))


def attn_prep(proj, q_norm, k_norm, inv_freq, q_out, k_cache, v_cache, cur_len, M, H, Hkv, eps):
    L = int(cur_len.item())
    D = k_cache.shape[-1]
    pos = torch.arange(L, L + M, device=proj.device)
    qg = proj[:M, : H * 2 * D].float().reshape(M, H, 2 * D)
    k = proj[:M, H * 2 * D: H * 2 * D + Hkv * D].float().reshape(M, Hkv, D)
    v = proj[:M, H * 2 * D + Hkv * D: H * 2 * D + 2 * Hkv * D].reshape(M, Hkv, D)
    q = T.rope_bf16(T.rms_norm_w(qg[..., :D], q_norm, eps), pos, inv_freq)
    k = T.rope_bf16(T.rms_norm_w(k, k_norm, eps), pos, inv_freq)
    q_out[:M].copy_(q)
    k_cache[:, L:L + M] = k.transpose(0, 1).to(torch.bfloat16)
    v_cache[:, L:L + M] = v.transpose(0, 1)


def attn_decode(q, k_cache, v_cache, proj, part_acc, part_ml, counters, out, cur_len, M, H, Hkv, NS):
    L = int(cur_len.item())
    D = k_cache.shape[-1]
    rep = H // Hkv
    for m in range(M):
        Lm = L + m + 1
        K = k_cache[:, :Lm].float().repeat_interleave(rep, dim=0)  # [H, Lm, D]
        V = v_cache[:, :Lm].float().repeat_interleave(rep, dim=0)
        s = torch.einsum("hd,hld->hl", q[m].float(), K) * D ** -0.5
        o = torch.einsum("hl,hld->hd", s.softmax(-1), V)
        gate = proj[m, : H * 2 * D].float().reshape(H, 2 * D)[:, D:]
        out[m].copy_((T.bf16r(o) * T.bf16r(torch.sigmoid(gate))).reshape(-1).to(torch.bfloat16))
