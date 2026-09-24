"""Quantized KV cache formats (DeepSeek-V4.1-Flash, arXiv 2609.19969, sec. 2.4.4).

The paper stores its main KV cache as FP4: E2M1 values with one E4M3 scale per 16 channels (NVFP4
without the second-level global scale), quantized *after* RoPE and dequantized before attention,
and makes it accurate with quantization-aware training. We cannot retrain, so two training-free
substitutes keep the post-training error small:

  * a randomized Hadamard rotation of every 256-dim head (q and k after RoPE, and v). q.k is
    invariant under an orthogonal Q, and o = sum_i p_i Q v_i = Q (sum_i p_i v_i), so the attention
    output is un-rotated (o = Q^T o_rot) before the elementwise sigmoid gate. The rotation spreads
    the RoPE / outlier channels of K over all 16-channel groups (QuaRot-style incoherence).
  * an MSE scale search per group: the E4M3 code of absmax/6 and its neighbours (-2 .. +1 codes)
    are tried and the one with the smallest squared error wins (the E2M1 grid is non-uniform, so
    absmax scaling is not optimal).

Formats (bytes per token per 256-dim head: bf16 512, fp8 256, fp4 144):
  bf16  the unquantized cache (no rotation)
  fp8   E4M3 per element, saturated to +-448
  fp4   packed E2M1 nibbles (low nibble = even channel) + one E4M3 scale per 16 channels
A layer's K and V formats are chosen independently (``KV_FORMATS``); ``k8v4`` keeps K in fp8.

Rounding is spelled out (round-to-nearest-even thresholds for E2M1, IEEE fp32 reciprocal, errors in
fp64) so the CUDA kernels reproduce these bytes exactly; there is no native FP4 conversion on sm_89.
"""

from __future__ import annotations

import torch

from .hadamard import fwht, random_signs

BF16, FP8, FP4 = 0, 1, 2
KV_FORMATS = {"bf16": (BF16, BF16), "fp8": (FP8, FP8), "k8v4": (FP8, FP4), "fp4": (FP4, FP4)}
KV_SEED = 0x4B56  # Hadamard sign seed of the KV rotation (stored in the manifest runtime block)
GROUP = 16
E4M3_MAX_CODE = 0x7E  # 448; 0x7F is NaN in e4m3fn
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def bytes_per_value(f: int) -> float:
    return {BF16: 2.0, FP8: 1.0, FP4: 0.5 + 1.0 / GROUP}[f]


def kv_bytes_per_token(fmt: str, n_layers: int, hkv: int, d: int) -> int:
    kf, vf = KV_FORMATS[fmt]
    return int(n_layers * hkv * d * (bytes_per_value(kf) + bytes_per_value(vf)))


# ------------------------------------------------------------------------------------------ rotation
def kv_signs(seed: int = KV_SEED, d: int = 256) -> torch.Tensor:
    return random_signs(d, seed)


def rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Q x along the last dim (Q = H diag(s) / sqrt(d)), fp32."""
    return fwht(x.float() * signs.to(x.device), dim=-1)


def unrotate(y: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Q^T y along the last dim (Q^T = diag(s) H / sqrt(d)), fp32."""
    return fwht(y.float(), dim=-1) * signs.to(y.device)


# ------------------------------------------------------------------------------------------ scalars
def e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> e4m3fn codes (uint8), round-to-nearest-even, saturated to +-448."""
    return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_decode(c: torch.Tensor) -> torch.Tensor:
    return c.view(torch.float8_e4m3fn).float()


def e2m1_encode(y: torch.Tensor) -> torch.Tensor:
    """fp32 -> E2M1 codes 0..15 (bit 3 = sign), round-to-nearest-even, saturating at 6."""
    a = y.abs()
    m = ((a > 0.25).to(torch.uint8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0))
    return m.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)


def e2m1_decode(c: torch.Tensor) -> torch.Tensor:
    return E2M1_VALUES.to(c.device)[c.long()]


# ------------------------------------------------------------------------------------------ fp4 groups
def fp4_encode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x[..., D] fp32 -> (packed uint8 [..., D/2], scale codes uint8 [..., D/16])."""
    x = x.float()
    shp = x.shape
    g = x.reshape(*shp[:-1], shp[-1] // GROUP, GROUP)
    amax = g.abs().amax(-1)
    s0 = e4m3_encode(amax / 6.0).to(torch.int16)
    best_err = best_code = best_q = None
    for dc in (0, 1, -1, -2):  # evaluation order = tie-break order (first minimum wins)
        code = (s0 + dc).clamp(0, E4M3_MAX_CODE).to(torch.uint8)
        s = e4m3_decode(code)
        inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
        q = e2m1_encode(g * inv[..., None])
        err = ((e2m1_decode(q) * s[..., None]).double() - g.double()).pow(2).sum(-1)
        if best_err is None:
            best_err, best_code, best_q = err, code, q
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_code = torch.where(better, code, best_code)
            best_q = torch.where(better[..., None], q, best_q)
    q = best_q.reshape(*shp[:-1], shp[-1])
    packed = q[..., 0::2] | (q[..., 1::2] << 4)
    return packed.contiguous(), best_code.contiguous()


def fp4_decode(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    q = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    v = e2m1_decode(q)
    s = e4m3_decode(scales)
    return (v.reshape(*s.shape, GROUP) * s[..., None]).reshape(v.shape)


# ------------------------------------------------------------------------------------------ cache API
STATS: dict | None = None  # set to {} to record the largest |value| written per format (range checks)


def encode(f: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if STATS is not None:
        STATS[f] = max(STATS.get(f, 0.0), float(x.abs().max()))
    if f == BF16:
        return x.to(torch.bfloat16), None
    if f == FP8:
        return e4m3_encode(x), None
    return fp4_encode(x)


def decode(f: int, data: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    if f == BF16:
        return data.float()
    if f == FP8:
        return e4m3_decode(data)
    return fp4_decode(data, scale)


def alloc(f: int, hkv: int, max_len: int, d: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache storage for one of K / V of one layer: (data, scale). Unused scale is empty."""
    empty = torch.empty(0, dtype=torch.uint8, device=device)
    if f == BF16:
        return torch.zeros(hkv, max_len, d, dtype=torch.bfloat16, device=device), empty
    if f == FP8:
        return torch.zeros(hkv, max_len, d, dtype=torch.uint8, device=device), empty
    return (torch.zeros(hkv, max_len, d // 2, dtype=torch.uint8, device=device),
            torch.zeros(hkv, max_len, d // GROUP, dtype=torch.uint8, device=device))


def fake_quant(f: int, x: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize (used by the folded reference model and the evaluation tools)."""
    return decode(f, *encode(f, x)) if f != BF16 else x.to(torch.bfloat16).float()
