"""Fast CPU backend for the engine (same op API as ``cckernel._C`` / ``cckernel.emu``).

``emu`` is the exact oracle for the CUDA kernels, but it caches every weight as a dense fp32 matrix,
which is 32 GB for the 9B model. This backend keeps INT8 weights as int8 and is fast enough to run
the real model on a many-core x86 CPU with AMX / AVX-512:

  * decode / verify linears (M <= 8 tokens): the weights are stored group-major, int8
    [K/128, N, 128]. Each 128-wide group goes through ``aten._weight_int8pack_mm`` (int8 x bf16,
    unit scales), and the fp16 group scales are applied in fp32 while accumulating. Every row is
    computed independently in a fixed order, so a token's result does not depend on how many tokens
    are verified with it. That keeps greedy speculative decoding identical to plain decoding.
  * prefill: matrices are dequantized to bf16 (the same values as the CUDA ``dequant``) and
    multiplied with oneDNN / AMX bf16 GEMMs by the engine's torch prefill path.
  * Gated DeltaNet and attention reuse the ``emu`` math. For attention an fp32 mirror of the
    dequantized KV cache is maintained incrementally (rows are decoded once, when written), so a
    decode step does not re-decode the whole quantized cache. The engine calls ``kv_changed()``
    whenever it rewrites the cache outside ``attn_prep`` (prefill, reset, session restore).

Weights with 4/5/6 bits fall back to per-call dequantization (correct, but slow).
"""

from __future__ import annotations

import torch

from . import emu, kvq
from . import torch_ops as T
from .emu import _epilogue
from .quant import GROUP, PERM64, dequant_rtn, unpack

gdn_decode = emu.gdn_decode  # re-exported op
_S32: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # id(lo) -> (lo, fp32 scales [G, N])
_ONES: dict[int, torch.Tensor] = {}
ROW_CHUNK = 8192
small_prefill = 96  # prompts up to this many tokens are prefilled through the int8 decode path


def _prepared(lo: torch.Tensor) -> bool:
    return lo.dtype == torch.int8 and lo.dim() == 3


def prepare(q) -> None:
    """Convert an engine ``QLin`` in place: packed uint8 [N, K] (PERM64 order, zero point 128)
    -> int8 [K/128, N, 128] in natural k order. Converted in row chunks (bounded temporaries)."""
    if q.bits != 8 or _prepared(q.lo):
        return
    N, K = q.N, q.K
    G = K // GROUP
    w8 = torch.empty(G, N, GROUP, dtype=torch.int8)
    perm = PERM64.to(torch.long)
    for r0 in range(0, N, ROW_CHUNK):
        r1 = min(N, r0 + ROW_CHUNK)
        u = q.lo[r0:r1].reshape(r1 - r0, K // 64, 64)[..., perm]  # storage (e) order -> natural k order
        w8[:, r0:r1] = (u.to(torch.int16) - 128).to(torch.int8).reshape(r1 - r0, G, GROUP).transpose(0, 1)
    q.lo = w8
    _S32[id(w8)] = (w8, q.scales.float().t().contiguous())


def _scales32(lo, scales):
    hit = _S32.get(id(lo))
    if hit is not None and hit[0] is lo:
        return hit[1]
    s = scales.float().t().contiguous()
    _S32[id(lo)] = (lo, s)
    return s


def _ones(n: int) -> torch.Tensor:
    t = _ONES.get(n)
    if t is None:
        t = _ONES[n] = torch.ones(n, dtype=torch.bfloat16)
    return t


def _dense_rows(lo, hi, scales, bits, N, K, r0, r1) -> torch.Tensor:
    """fp32 rows [r0, r1) of the dequantized matrix."""
    if _prepared(lo):
        w = lo[:, r0:r1].transpose(0, 1).float()  # [rows, G, 128]
        return (w * scales[r0:r1].float()[..., None]).reshape(r1 - r0, K)
    planes = {"q8": lo[r0:r1]} if bits == 8 else (
        {"lo": lo[r0:r1], "hi": hi[r0:r1]} if bits in (5, 6) else {"lo": lo[r0:r1]})
    return dequant_rtn(unpack(planes, bits, K), scales[r0:r1], bits)


def matmul(lo, hi, scales, bits, N, K, x: torch.Tensor) -> torch.Tensor:
    """fp32 [M, N] = bf16 x[M, K] @ W^T."""
    x = x.to(torch.bfloat16)
    M = x.shape[0]
    if not _prepared(lo):
        return torch.cat([x.float() @ _dense_rows(lo, hi, scales, bits, N, K, r0, min(N, r0 + ROW_CHUNK)).t()
                          for r0 in range(0, N, ROW_CHUNK)], dim=1)
    G = K // GROUP
    s32 = _scales32(lo, scales)
    xg = x.reshape(M, G, GROUP).transpose(0, 1).contiguous()
    one = _ones(N)
    y = torch.zeros(M, N, dtype=torch.float32)
    for g in range(G):
        y.addcmul_(torch.ops.aten._weight_int8pack_mm(xg[g], lo[g], one), s32[g])
    return y


def qgemv(lo, hi, scales, bits, N, K, pro, x, z, head_dim, eps, epi, y):
    # prologues computed exactly like the engine's M > 1 path (same torch ops, row-wise)
    if pro == 1:
        r = x.float()[None]
        xin = (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
    elif pro == 0:
        xin = x[None].to(torch.bfloat16)
    else:
        xin = T.gated_head_norm(x[None], z[None], head_dim, eps)
    _epilogue(matmul(lo, hi, scales, bits, N, K, xin)[0], y, epi)


def qgemm_skinny(lo, hi, scales, bits, N, K, x, M, epi, y):
    _epilogue(matmul(lo, hi, scales, bits, N, K, x[:M]), y[:M], epi)


def dequant_rows(lo, hi, scales, bits, N, K, r0, r1, out):
    o = out[: (r1 - r0) * K].view(r1 - r0, K)
    for a in range(r0, r1, ROW_CHUNK):
        b = min(r1, a + ROW_CHUNK)
        o[a - r0:b - r0] = _dense_rows(lo, hi, scales, bits, N, K, a, b).to(torch.bfloat16)


def dequant(lo, hi, scales, bits, N, K, out):
    dequant_rows(lo, hi, scales, bits, N, K, 0, N, out)


# ------------------------------------------------------------------------------------------ attention
_MIRROR: dict[int, list] = {}  # id(cache data) -> [data, fp32 mirror [Hkv, max_len, D], valid rows]


def kv_changed():
    for ent in _MIRROR.values():
        ent[2] = 0


def _mirror(data, scale, f, upto):
    ent = _MIRROR.get(id(data))
    if ent is None or ent[0] is not data:
        D = data.shape[2] * (2 if f == kvq.FP4 else 1)
        ent = _MIRROR[id(data)] = [data, torch.zeros(data.shape[0], data.shape[1], D), 0]
    if ent[2] < upto:
        v = ent[2]
        ent[1][:, v:upto] = kvq.decode(f, data[:, v:upto], scale[:, v:upto] if f == kvq.FP4 else None)
        ent[2] = upto
    return ent


def attn_prep(proj, q_norm, k_norm, inv_freq, q_out, k_cache, k_scale, v_cache, v_scale, signs, cur_len, M, H, Hkv,
              eps, kfmt=0, vfmt=0, rotate=False):
    emu.attn_prep(proj, q_norm, k_norm, inv_freq, q_out, k_cache, k_scale, v_cache, v_scale, signs, cur_len, M, H,
                  Hkv, eps, kfmt, vfmt, rotate)
    L = int(cur_len.item())
    for f, data, scale in ((kfmt, k_cache, k_scale), (vfmt, v_cache, v_scale)):
        ent = _mirror(data, scale, f, L)  # rows before L valid; rows >= L are rewritten now
        ent[1][:, L:L + M] = kvq.decode(f, data[:, L:L + M], scale[:, L:L + M] if f == kvq.FP4 else None)
        ent[2] = L + M


def attn_decode(q, k_cache, k_scale, v_cache, v_scale, signs, proj, part_acc, part_ml, counters, out, cur_len, M, H,
                Hkv, NS, kfmt=0, vfmt=0, rotate=False):
    L = int(cur_len.item())
    K = _mirror(k_cache, k_scale, kfmt, L + M)[1]
    V = _mirror(v_cache, v_scale, vfmt, L + M)[1]
    emu.attn_rows(q, K, V, proj, out, L, M, H, Hkv, rotate, signs)
