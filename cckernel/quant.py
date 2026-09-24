"""Weight folding, quantization and the packed "cck" weight layout.

Pipeline (see docs/MATH.md):
  1. fold zero-centered RMSNorm weights (1+w) into the consumer linears; fold the GDN gated-norm
     weight into out_proj; runtime norms become weightless.
  2. fold a random-sign Hadamard rotation Q of the residual stream into every residual reader (W Q^T)
     and writer (Q W) plus the embedding (QuaRot R1 / QuIP# incoherence). Exact: rms(Qx) = rms(x).
  3. symmetric round-to-nearest INT-b (b in {4,5,6,8}) with groups of 128 along K and per-group
     fp16 scales whose clip ratio minimises the group MSE (after the rotation the weights are close
     to Gaussian, HIGGS arXiv 2411.17525).
  4. pack into the 64-weight block layout consumed by both the CUDA-core GEMV and the tensor-core
     skinny GEMM (mma.m16n8k16 A-fragment order).

Block layout (64 consecutive k of one row). Every weight has coordinates (c, h, t, j) with
    k_local = 16 t + 8 h + 2 c + j,      c in 0..3 (thread-in-quad), h in 0..1, t in 0..3, j in 0..1.
The storage/decode order is e = 16 c + 8 h + 2 t + j, i.e. the 2-bit fields c and t are swapped;
this permutation is an involution (``PERM64``). The GEMV stages x in e-order so it can stream the
weights sequentially; the mma kernel gives lane (g, c) exactly the pairs (k, k+1) it needs.
  * low plane (b = 4, 5, 6): 8 uint32 words per block, word w = 2c + h, nibble of (t, j) at bit 4t + 16j
    -> one ``lop3(x >> 4t, 0x000F000F, 0x43004300)`` yields the bf16x2 pair (128 + u_k, 128 + u_{k+1}).
  * high plane b = 6: 4 words per block, word c, 2-bit field of (h, t, j) at bit 2(4h + t) + 16j.
  * high plane b = 5: 2 words per block, word c >> 1, bit of (h, t, j) at 16j + 8(c & 1) + 4h + t.
  * b = 8: 64 bytes per block, byte 16c + 8h + 2t + j (= e).
Stored value u = q + 2^(b-1) (unsigned); w = scale * (u - 2^(b-1)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import hadamard
from .config import TextConfig

GROUP = 128
BLOCK = 64
SUPPORTED_BITS = (4, 5, 6, 8)


def _perm64() -> torch.Tensor:
    e = torch.arange(64)
    c, h, t, j = (e >> 4) & 3, (e >> 3) & 1, (e >> 1) & 3, e & 1
    return 16 * t + 8 * h + 2 * c + j


PERM64 = _perm64()  # e -> k_local (involution)


def permute_x(x: torch.Tensor) -> torch.Tensor:
    """x[..., K] in natural order -> e-order used by the packed layout."""
    K = x.shape[-1]
    return x.reshape(*x.shape[:-1], K // 64, 64)[..., PERM64.to(x.device)].reshape(x.shape)


# ------------------------------------------------------------------------------------------------
# quantization
# ------------------------------------------------------------------------------------------------


def quantize_rtn(w: torch.Tensor, bits: int, group: int = GROUP, clip_grid: int = 20, row_chunk: int = 4096):
    """Symmetric per-group RTN with MSE-optimal clipping.

    Returns (u uint8 [N, K], scales fp16 [N, K/group]).
    """
    assert bits in SUPPORTED_BITS
    N, K = w.shape
    assert K % group == 0, (K, group)
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    alphas = torch.linspace(1.0, 0.55, clip_grid) if clip_grid > 1 else torch.ones(1)
    us, ss = [], []
    for r0 in range(0, N, row_chunk):
        g = w[r0:r0 + row_chunk].float().reshape(-1, K // group, group)
        amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        best_err = None
        best_s = None
        for a in alphas:
            s = (a * amax / qmax).half().float().clamp_min(1e-10)  # score with the stored precision
            q = torch.clamp(torch.round(g / s), qmin, qmax)
            err = ((q * s - g) ** 2).sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_s = err, s
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_s = torch.where(better, s, best_s)
        q = torch.clamp(torch.round(g / best_s), qmin, qmax)
        us.append((q - qmin).to(torch.uint8).reshape(-1, K))
        ss.append(best_s.reshape(-1, K // group).half())
    return torch.cat(us), torch.cat(ss)


def dequant_rtn(u: torch.Tensor, scales: torch.Tensor, bits: int, group: int = GROUP) -> torch.Tensor:
    N, K = u.shape
    z = 2 ** (bits - 1)
    return ((u.float() - z).reshape(N, K // group, group) * scales.float()[..., None]).reshape(N, K)


def rel_mse(w: torch.Tensor, w_hat: torch.Tensor) -> float:
    """t^2 of the linearity theorem: ||W - W_hat||_F^2 / ||W||_F^2."""
    return float(((w.float() - w_hat.float()) ** 2).sum() / (w.float() ** 2).sum().clamp_min(1e-30))


# ------------------------------------------------------------------------------------------------
# packing
# ------------------------------------------------------------------------------------------------


def _words_to_bytes(words: torch.Tensor) -> torch.Tensor:
    """int64 [..., n] holding uint32 values -> uint8 [..., 4n] little endian."""
    b = torch.stack([(words >> (8 * i)) & 0xFF for i in range(4)], dim=-1)
    return b.reshape(*words.shape[:-1], -1).to(torch.uint8)


def _bytes_to_words(b: torch.Tensor) -> torch.Tensor:
    b = b.to(torch.int64).reshape(*b.shape[:-1], -1, 4)
    return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)


def _coords():
    """Yield (c, h, t, j, k_local) for the 64 weights of a block."""
    for c in range(4):
        for h in range(2):
            for t in range(4):
                for j in range(2):
                    yield c, h, t, j, 16 * t + 8 * h + 2 * c + j


def pack(u: torch.Tensor, bits: int) -> dict[str, torch.Tensor]:
    """u uint8 [N, K] -> packed planes (uint8, row-major)."""
    N, K = u.shape
    assert K % BLOCK == 0
    ub = u.to(torch.int64).reshape(N, K // BLOCK, BLOCK)
    if bits == 8:
        return {"q8": ub[..., PERM64.to(u.device)].reshape(N, K).to(torch.uint8).contiguous()}
    lo = torch.zeros(N, K // BLOCK, 8, dtype=torch.int64, device=u.device)
    hi_words = {6: 4, 5: 2, 4: 0}[bits]
    hi = torch.zeros(N, K // BLOCK, max(hi_words, 1), dtype=torch.int64, device=u.device)
    for c, h, t, j, k in _coords():
        v = ub[..., k]
        lo[..., 2 * c + h] |= (v & 15) << (4 * t + 16 * j)
        if bits == 6:
            hi[..., c] |= ((v >> 4) & 3) << (2 * (4 * h + t) + 16 * j)
        elif bits == 5:
            hi[..., c >> 1] |= ((v >> 4) & 1) << (16 * j + 8 * (c & 1) + 4 * h + t)
    out = {"lo": _words_to_bytes(lo).reshape(N, K // 2).contiguous()}
    if bits in (5, 6):
        out["hi"] = _words_to_bytes(hi).reshape(N, -1).contiguous()
    return out


def unpack(p: dict[str, torch.Tensor], bits: int, K: int) -> torch.Tensor:
    """Inverse of ``pack`` (oracle for the kernels)."""
    if bits == 8:
        q = p["q8"].to(torch.int64)
        N = q.shape[0]
        return q.reshape(N, K // BLOCK, BLOCK)[..., PERM64.to(q.device)].reshape(N, K).to(torch.uint8)
    lo = _bytes_to_words(p["lo"]).reshape(p["lo"].shape[0], K // BLOCK, 8)
    N = lo.shape[0]
    hi = _bytes_to_words(p["hi"]).reshape(N, K // BLOCK, -1) if bits in (5, 6) else None
    u = torch.zeros(N, K // BLOCK, BLOCK, dtype=torch.int64, device=lo.device)
    for c, h, t, j, k in _coords():
        v = (lo[..., 2 * c + h] >> (4 * t + 16 * j)) & 15
        if bits == 6:
            v |= ((hi[..., c] >> (2 * (4 * h + t) + 16 * j)) & 3) << 4
        elif bits == 5:
            v |= ((hi[..., c >> 1] >> (16 * j + 8 * (c & 1) + 4 * h + t)) & 1) << 4
        u[..., k] = v
    return u.reshape(N, K).to(torch.uint8)


@dataclass
class QLinear:
    """A packed linear layer: y = W x with W [N, K]."""

    bits: int
    N: int
    K: int
    planes: dict[str, torch.Tensor]
    scales: torch.Tensor  # fp16 [N, K/128]

    def dequant(self) -> torch.Tensor:
        return dequant_rtn(unpack(self.planes, self.bits, self.K), self.scales, self.bits)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.planes.values()) + self.scales.numel() * 2

    @classmethod
    def from_weight(cls, w: torch.Tensor, bits: int, clip_grid: int = 20) -> "QLinear":
        u, s = quantize_rtn(w, bits, clip_grid=clip_grid)
        return cls(bits, w.shape[0], w.shape[1], pack(u, bits), s)

    def state(self, prefix: str) -> dict[str, torch.Tensor]:
        d = {f"{prefix}.{k}": v for k, v in self.planes.items()}
        d[f"{prefix}.scales"] = self.scales
        return d

    @classmethod
    def from_state(cls, sd: dict[str, torch.Tensor], prefix: str, bits: int, N: int, K: int) -> "QLinear":
        planes = {k: sd[f"{prefix}.{k}"] for k in ("lo", "hi", "q8") if f"{prefix}.{k}" in sd}
        return cls(bits, N, K, planes, sd[f"{prefix}.scales"])


def bits_per_weight(bits: int) -> float:
    return bits + 16.0 / GROUP


# ------------------------------------------------------------------------------------------------
# emulation of the CUDA decode paths (for CPU tests of the index math)
# ------------------------------------------------------------------------------------------------


def emulate_gemv(q: QLinear, x: torch.Tensor) -> torch.Tensor:
    """Mirror of qgemv's lane decode: stream blocks in storage order against e-ordered x."""
    xe = permute_x(x.float()).reshape(q.K // BLOCK, BLOCK)
    z = 2 ** (q.bits - 1)
    y = torch.zeros(q.N)
    if q.bits == 8:
        wq = q.planes["q8"].to(torch.int64).reshape(q.N, q.K // BLOCK, BLOCK).float() - z
    else:
        lo = _bytes_to_words(q.planes["lo"]).reshape(q.N, q.K // BLOCK, 8)
        hi = _bytes_to_words(q.planes["hi"]).reshape(q.N, q.K // BLOCK, -1) if q.bits in (5, 6) else None
        vals = []
        for w in range(8):  # storage order == e order
            c, h = w >> 1, w & 1
            for t in range(4):
                for j in range(2):
                    v = (lo[..., w] >> (4 * t + 16 * j)) & 15
                    if q.bits == 6:
                        v |= ((hi[..., c] >> (2 * (4 * h + t) + 16 * j)) & 3) << 4
                    elif q.bits == 5:
                        v |= ((hi[..., c >> 1] >> (16 * j + 8 * (c & 1) + 4 * h + t)) & 1) << 4
                    vals.append(v)
        wq = torch.stack(vals, dim=-1).float() - z
    part = (wq * xe[None]).sum(-1)  # [N, blocks]
    s = q.scales.float().repeat_interleave(GROUP // BLOCK, dim=1)
    y = (part * s).sum(-1)
    return y


def emulate_mma_a_fragment(q: QLinear, row0: int, kblk: int, kstep: int, lane: int) -> list[tuple[float, float]]:
    """The four bf16x2 A-registers lane `lane` builds for rows row0..row0+15 and the 16-wide k-step
    `kstep` (0..3) of 64-block `kblk`, as integer values (u - z). Fragment convention of
    mma.m16n8k16.row: a0=(g, 2c..2c+1) a1=(g+8, 2c..) a2=(g, 2c+8..) a3=(g+8, 2c+8..)."""
    g, c = lane >> 2, lane & 3
    z = 2 ** (q.bits - 1)
    t = kstep
    regs = []
    for rr, h in ((0, 0), (8, 0), (0, 1), (8, 1)):
        row = row0 + g + rr
        pair = []
        for j in range(2):
            if q.bits == 8:
                byte = q.planes["q8"][row, kblk * 64 + 16 * c + 8 * h + 2 * t + j].item()
                pair.append(byte - z)
            else:
                lo = _bytes_to_words(q.planes["lo"][row].reshape(1, -1)).reshape(-1, 8)[kblk]
                v = (lo[2 * c + h].item() >> (4 * t + 16 * j)) & 15
                if q.bits in (5, 6):
                    hw = _bytes_to_words(q.planes["hi"][row].reshape(1, -1)).reshape(q.K // BLOCK, -1)[kblk]
                    if q.bits == 6:
                        v |= ((hw[c].item() >> (2 * (4 * h + t) + 16 * j)) & 3) << 4
                    else:
                        v |= ((hw[c >> 1].item() >> (16 * j + 8 * (c & 1) + 4 * h + t)) & 1) << 4
                pair.append(v - z)
        regs.append(tuple(pair))
    return regs


# ------------------------------------------------------------------------------------------------
# folding (norm weights + residual Hadamard) into fused matrices
# ------------------------------------------------------------------------------------------------


def fold_layer(cfg: TextConfig, i: int, get, signs: torch.Tensor | None, dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Return the folded, fused fp tensors of decoder layer i.

    ``get(name)`` returns checkpoint tensors by canonical name. ``signs`` = Hadamard sign vector or
    None (no rotation). Output names are relative to ``layers.{i}.``.
    """
    p = f"layers.{i}."
    lt = cfg.layer_types[i]
    ln_in = 1.0 + get(p + "input_layernorm.weight").to(dtype)
    ln_post = 1.0 + get(p + "post_attention_layernorm.weight").to(dtype)

    def reader(w, ln):
        w = w.to(dtype) * ln[None, :]
        return hadamard.rotate_reader(w, signs) if signs is not None else w

    def writer(w):
        w = w.to(dtype)
        return hadamard.rotate_writer(w, signs) if signs is not None else w

    out: dict[str, torch.Tensor] = {}
    if lt == "linear_attention":
        a = p + "linear_attn."
        fused = torch.cat([get(a + "in_proj_qkv.weight"), get(a + "in_proj_z.weight"), get(a + "in_proj_b.weight"),
                           get(a + "in_proj_a.weight")], dim=0)
        out["in_proj"] = reader(fused, ln_in)
        nw = get(a + "norm.weight").to(dtype).repeat(cfg.linear_num_value_heads)
        out["out_proj"] = writer(get(a + "out_proj.weight").to(dtype) * nw[None, :])
        out["conv_w"] = get(a + "conv1d.weight").to(torch.float32).reshape(cfg.gdn_conv_dim, -1).contiguous()
        out["A_log"] = get(a + "A_log").to(torch.float32)
        out["dt_bias"] = get(a + "dt_bias").to(torch.float32)
    else:
        a = p + "self_attn."
        fused = torch.cat([get(a + "q_proj.weight"), get(a + "k_proj.weight"), get(a + "v_proj.weight")], dim=0)
        out["qkv_proj"] = reader(fused, ln_in)
        out["o_proj"] = writer(get(a + "o_proj.weight"))
        out["q_norm"] = (1.0 + get(a + "q_norm.weight").to(torch.float32)).contiguous()
        out["k_norm"] = (1.0 + get(a + "k_norm.weight").to(torch.float32)).contiguous()
    g = reader(get(p + "mlp.gate_proj.weight"), ln_post)
    u = reader(get(p + "mlp.up_proj.weight"), ln_post)
    out["gate_up"] = torch.stack([g, u], dim=1).reshape(2 * g.shape[0], g.shape[1])  # rows g0,u0,g1,u1,...
    out["down"] = writer(get(p + "mlp.down_proj.weight"))
    return out


def fold_globals(cfg: TextConfig, get, signs: torch.Tensor | None, dtype=torch.float32) -> dict[str, torch.Tensor]:
    emb = get("embed_tokens.weight").to(dtype)
    lm = get("lm_head.weight").to(dtype) * (1.0 + get("norm.weight").to(dtype))[None, :]
    if signs is not None:
        emb = hadamard.rotate_reader(emb, signs)  # rows e -> Q e  (same op as a reader: E Q^T)
        lm = hadamard.rotate_reader(lm, signs)
    return {"embed": emb, "lm_head": lm}


LINEAR_NAMES = ("in_proj", "out_proj", "qkv_proj", "o_proj", "gate_up", "down")


def matrix_list(cfg: TextConfig) -> list[tuple[str, int, int]]:
    """(name, N, K) of every quantized matrix, in model order."""
    d, I = cfg.hidden_size, cfg.intermediate_size
    out = []
    for i, lt in enumerate(cfg.layer_types):
        p = f"layers.{i}."
        if lt == "linear_attention":
            out += [(p + "in_proj", cfg.gdn_in_dim, d), (p + "out_proj", d, cfg.gdn_value_dim)]
        else:
            out += [(p + "qkv_proj", cfg.attn_in_dim, d), (p + "o_proj", d, cfg.num_attention_heads * cfg.head_dim)]
        out += [(p + "gate_up", 2 * I, d), (p + "down", d, I)]
    out.append(("lm_head", cfg.vocab_size, d))
    return out


def model_bytes(cfg: TextConfig, bits: dict[str, int], embed_bytes_per: int = 2) -> int:
    """Approximate VRAM of the weights for a bit assignment."""
    tot = 0
    for name, N, K in matrix_list(cfg):
        tot += math.ceil(N * K * bits_per_weight(bits[name]) / 8)
    tot += cfg.vocab_size * cfg.hidden_size * embed_bytes_per
    return tot
