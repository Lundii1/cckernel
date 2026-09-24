"""cckernel inference engine for the quantized (cck-v1) model on one sm_89 GPU.

Decode / verify steps (M = 1..8 tokens) run entirely on the custom CUDA kernels and are captured
into CUDA graphs (one per M). Prefill uses torch (dequantized bf16 GEMMs through cuBLAS, SDPA and a
vectorised chunked Gated DeltaNet) and writes the same caches the kernels use.

Cache layout shared by all paths:
  GDN layer:  ring  bf16 [C, 32]   conv inputs by absolute position (slot = pos & 31)
              S     fp32 [Hv, 128, 128]  committed recurrent state
              pend_u / pend_g   pseudo-values / log-decays of the last step (deferred commit)
  attention:  k/v cache bf16 [Hkv, max_len, 256] (K after qk-norm + RoPE)
  scalars:    cur_len = committed tokens, n_commit = tokens of the last step accepted by the host
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from . import torch_ops as T
from .config import TextConfig
from .quant import GROUP

MAXM = 8
RING = 32
PRO_BF16, PRO_RMSNORM, PRO_GATED = 0, 1, 2
EPI_BF16, EPI_F32, EPI_ADD_F32, EPI_SWIGLU = 0, 1, 2, 3


_BACKEND = {"mod": None}


def _ext():
    """The CUDA extension, or the torch emulation when running on CPU (set by Engine)."""
    if _BACKEND["mod"] is None:
        from . import _C  # noqa: WPS433  (built by setup.py)

        _BACKEND["mod"] = _C
    return _BACKEND["mod"]


class QLin:
    """A device-resident packed linear layer."""

    def __init__(self, sd: dict, prefix: str, bits: int, N: int, K: int, device):
        self.bits, self.N, self.K = bits, N, K
        key = "q8" if bits == 8 else "lo"
        self.lo = sd[f"{prefix}.{key}"].to(device)
        self.hi = sd[f"{prefix}.hi"].to(device) if bits in (5, 6) else torch.empty(0, dtype=torch.uint8, device=device)
        self.scales = sd[f"{prefix}.scales"].to(device)
        assert self.scales.shape == (N, K // GROUP), (prefix, self.scales.shape)

    def nbytes(self):
        return self.lo.numel() + self.hi.numel() + self.scales.numel() * 2

    def gemv(self, pro, x, y, epi, z=None, head_dim=0, eps=1e-6):
        z = z if z is not None else torch.empty(0, device=x.device, dtype=torch.bfloat16)
        _ext().qgemv(self.lo, self.hi, self.scales, self.bits, self.N, self.K, pro, x, z, head_dim, eps, epi, y)

    def skinny(self, x, M, y, epi):
        _ext().qgemm_skinny(self.lo, self.hi, self.scales, self.bits, self.N, self.K, x, M, epi, y)

    def dequant(self, scratch):
        w = scratch[: self.N * self.K].view(self.N, self.K)
        _ext().dequant(self.lo, self.hi, self.scales, self.bits, self.N, self.K, w)
        return w


class Engine:
    def __init__(self, model_dir: str | Path, device: str = "cuda", max_len: int = 32768, attn_splits: int = 32,
                 prefill_chunk: int = 2048, use_graphs: bool = True):
        self.dir = Path(model_dir)
        man = json.loads((self.dir / "cck_manifest.json").read_text())
        assert man["format"] == "cck-v1", man["format"]
        self.manifest = man
        self.cfg = cfg = TextConfig.from_dict(man["config"])
        self.dev = torch.device(device)
        if self.dev.type != "cuda" or os.environ.get("CCK_EMULATE") == "1":
            from . import emu

            _BACKEND["mod"] = emu
            use_graphs = False
        self.max_len, self.NS, self.prefill_chunk, self.use_graphs = max_len, attn_splits, prefill_chunk, use_graphs
        assert cfg.linear_key_head_dim == 128 and cfg.linear_value_head_dim == 128, "kernels compiled for dk=dv=128"
        assert cfg.head_dim == 256 and cfg.num_attention_heads == 4 * cfg.num_key_value_heads

        from safetensors.torch import load_file

        bits = man["bits"]
        self.lin: dict[str, QLin] = {}
        self.small: dict[str, torch.Tensor] = {}
        from .quant import matrix_list

        shapes = {n: (N, K) for n, N, K in matrix_list(cfg)}
        for i in range(cfg.num_hidden_layers):
            sd = load_file(str(self.dir / f"layer-{i:02d}.safetensors"))
            for name in [n for n in shapes if n.startswith(f"layers.{i}.")]:
                self.lin[name] = QLin(sd, name, bits[name], *shapes[name], self.dev)
            for k, v in sd.items():
                if k.split(".")[-1] in ("conv_w", "A_log", "dt_bias", "q_norm", "k_norm"):
                    self.small[k] = v.to(self.dev, torch.float32).contiguous()
        g = load_file(str(self.dir / "globals.safetensors"))
        self.lin["lm_head"] = QLin(g, "lm_head", bits["lm_head"], *shapes["lm_head"], self.dev)
        self.embed = g["embed"].to(self.dev)
        del g

        self._alloc_state()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

    # ------------------------------------------------------------------------------------------
    def _alloc_state(self):
        c, d = self.cfg, self.dev
        bf, f32 = torch.bfloat16, torch.float32
        C, Hv, Hk = c.gdn_conv_dim, c.linear_num_value_heads, c.linear_num_key_heads
        self.ring = {i: torch.zeros(C, RING, dtype=bf, device=d) for i in c.gdn_layers}
        self.S = {i: torch.zeros(Hv, 128, 128, dtype=f32, device=d) for i in c.gdn_layers}
        self.pend_u = {i: torch.zeros(MAXM, Hv, 128, dtype=f32, device=d) for i in c.gdn_layers}
        self.pend_g = {i: torch.zeros(MAXM, Hv, 4, dtype=f32, device=d) for i in c.gdn_layers}
        Hkv, D = c.num_key_value_heads, c.head_dim
        self.kc = {i: torch.zeros(Hkv, self.max_len, D, dtype=bf, device=d) for i in c.attn_layers}
        self.vc = {i: torch.zeros(Hkv, self.max_len, D, dtype=bf, device=d) for i in c.attn_layers}
        self.cur_len = torch.zeros(1, dtype=torch.int32, device=d)
        self.n_commit = torch.zeros(1, dtype=torch.int32, device=d)
        self.len_host = 0
        # step buffers (static addresses for graph capture)
        H = c.num_attention_heads
        self.tok = torch.zeros(MAXM, dtype=torch.int64, device=d)
        self.resid = torch.zeros(MAXM, c.hidden_size, dtype=f32, device=d)
        self.xn = torch.zeros(MAXM, c.hidden_size, dtype=bf, device=d)
        self.proj_g = torch.zeros(MAXM, c.gdn_in_dim, dtype=bf, device=d)
        self.proj_a = torch.zeros(MAXM, c.attn_in_dim, dtype=bf, device=d)
        self.o_gdn = torch.zeros(MAXM, c.gdn_value_dim, dtype=bf, device=d)
        self.xg = torch.zeros(MAXM, c.gdn_value_dim, dtype=bf, device=d)
        self.attn_out = torch.zeros(MAXM, H * D, dtype=bf, device=d)
        self.gu = torch.zeros(MAXM, 2 * c.intermediate_size, dtype=bf, device=d)
        self.h_mlp = torch.zeros(MAXM, c.intermediate_size, dtype=bf, device=d)
        self.qbuf = torch.zeros(MAXM, H, D, dtype=f32, device=d)
        self.part_acc = torch.zeros(MAXM, self.NS, H, D, dtype=f32, device=d)
        self.part_ml = torch.zeros(MAXM, self.NS, H, 2, dtype=f32, device=d)
        self.counters = torch.zeros(MAXM, Hkv, dtype=torch.int32, device=d)
        self.logits = torch.zeros(MAXM, c.vocab_size, dtype=f32, device=d)
        self.inv_freq = T.inv_freq_hf(c.rotary_dim, c.rope_theta, d)
        biggest = max(q.N * q.K for n, q in self.lin.items() if n != "lm_head")
        self.scratch = torch.empty(biggest, dtype=bf, device=d)

    def reset(self):
        for t in list(self.ring.values()) + list(self.S.values()):
            t.zero_()
        self.cur_len.zero_()
        self.n_commit.zero_()
        self.len_host = 0

    def vram_report(self) -> str:
        w = sum(q.nbytes() for q in self.lin.values())
        e = self.embed.numel() * 2
        kv = sum(t.numel() * 2 for t in list(self.kc.values()) + list(self.vc.values()))
        st = sum(t.numel() * 4 for t in self.S.values())
        return (f"weights {w / 2**30:.2f} GiB | embed {e / 2**30:.2f} GiB | KV cache ({self.max_len} tok) "
                f"{kv / 2**30:.2f} GiB | GDN state {st / 2**20:.0f} MiB"
                + (f" | allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB" if self.dev.type == "cuda" else ""))

    # ------------------------------------------------------------------------------------------
    # decode / verify step (kernels only)
    # ------------------------------------------------------------------------------------------
    def _linear_M(self, name, x_bf16, M, y, epi):
        """M-token linear with the tensor-core skinny kernel (M >= 2)."""
        self.lin[name].skinny(x_bf16, M, y, epi)

    def _rms_bf16(self, M):
        r = self.resid[:M]
        self.xn[:M].copy_((r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)).to(torch.bfloat16))

    def _step_body(self, M: int):
        c, ext, eps = self.cfg, _ext(), self.cfg.rms_norm_eps
        self.resid[:M].copy_(self.embed.index_select(0, self.tok[:M]).float())
        H, Hkv, Hv, Hk, C, Vd = (c.num_attention_heads, c.num_key_value_heads, c.linear_num_value_heads,
                                 c.linear_num_key_heads, c.gdn_conv_dim, c.gdn_value_dim)
        for i, lt in enumerate(c.layer_types):
            p = f"layers.{i}."
            if lt == "linear_attention":
                if M == 1:
                    self.lin[p + "in_proj"].gemv(PRO_RMSNORM, self.resid[0], self.proj_g[0], EPI_BF16, eps=eps)
                else:
                    self._rms_bf16(M)
                    self._linear_M(p + "in_proj", self.xn, M, self.proj_g, EPI_BF16)
                ext.gdn_decode(self.proj_g, self.ring[i], self.small[p + "conv_w"], self.small[p + "A_log"],
                               self.small[p + "dt_bias"], self.S[i], self.pend_u[i], self.pend_g[i], self.o_gdn,
                               self.cur_len, self.n_commit, M, Hk, Hv, C)
                if M == 1:
                    self.lin[p + "out_proj"].gemv(PRO_GATED, self.o_gdn[0], self.resid[0], EPI_ADD_F32,
                                                  z=self.proj_g[0, C:C + Vd], head_dim=128, eps=eps)
                else:
                    self.xg[:M].copy_(T.gated_head_norm(self.o_gdn[:M], self.proj_g[:M, C:C + Vd], 128, eps))
                    self._linear_M(p + "out_proj", self.xg, M, self.resid, EPI_ADD_F32)
            else:
                if M == 1:
                    self.lin[p + "qkv_proj"].gemv(PRO_RMSNORM, self.resid[0], self.proj_a[0], EPI_BF16, eps=eps)
                else:
                    self._rms_bf16(M)
                    self._linear_M(p + "qkv_proj", self.xn, M, self.proj_a, EPI_BF16)
                ext.attn_prep(self.proj_a, self.small[p + "q_norm"], self.small[p + "k_norm"], self.inv_freq, self.qbuf,
                              self.kc[i], self.vc[i], self.cur_len, M, H, Hkv, eps)
                ext.attn_decode(self.qbuf, self.kc[i], self.vc[i], self.proj_a, self.part_acc, self.part_ml,
                                self.counters, self.attn_out, self.cur_len, M, H, Hkv, self.NS)
                if M == 1:
                    self.lin[p + "o_proj"].gemv(PRO_BF16, self.attn_out[0], self.resid[0], EPI_ADD_F32)
                else:
                    self._linear_M(p + "o_proj", self.attn_out, M, self.resid, EPI_ADD_F32)
            if M == 1:
                self.lin[p + "gate_up"].gemv(PRO_RMSNORM, self.resid[0], self.h_mlp[0], EPI_SWIGLU, eps=eps)
                self.lin[p + "down"].gemv(PRO_BF16, self.h_mlp[0], self.resid[0], EPI_ADD_F32)
            else:
                self._rms_bf16(M)
                self._linear_M(p + "gate_up", self.xn, M, self.gu, EPI_BF16)
                g, u = self.gu[:M, 0::2].float(), self.gu[:M, 1::2].float()
                self.h_mlp[:M].copy_((T.bf16r(F.silu(g)) * u).to(torch.bfloat16))
                self._linear_M(p + "down", self.h_mlp, M, self.resid, EPI_ADD_F32)
        if M == 1:
            self.lin["lm_head"].gemv(PRO_RMSNORM, self.resid[0], self.logits[0], EPI_F32, eps=eps)
        else:
            self._rms_bf16(M)
            self._linear_M("lm_head", self.xn, M, self.logits, EPI_F32)

    def _capture(self, M: int):
        # warm up (allocates cuBLAS/allocator state outside the graph); restore state afterwards
        snap = self._snapshot()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self._step_body(M)
        torch.cuda.current_stream().wait_stream(s)
        self._restore(snap)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step_body(M)
        self._restore(snap)
        self.graphs[M] = g

    def _snapshot(self):
        return {k: [t.clone() for t in d.values()] for k, d in
                (("S", self.S), ("pu", self.pend_u), ("pg", self.pend_g))}

    def _restore(self, snap):
        for t, v in zip(self.S.values(), snap["S"]):
            t.copy_(v)
        for t, v in zip(self.pend_u.values(), snap["pu"]):
            t.copy_(v)
        for t, v in zip(self.pend_g.values(), snap["pg"]):
            t.copy_(v)

    @torch.no_grad()
    def step(self, tokens: list[int]) -> torch.Tensor:
        """Run M = len(tokens) tokens at positions cur_len.. (not committed). Returns logits [M, V]."""
        M = len(tokens)
        assert 1 <= M <= MAXM and self.len_host + M <= self.max_len
        self.tok[:M].copy_(torch.tensor(tokens, dtype=torch.int64))
        if self.use_graphs:
            if M not in self.graphs:
                self._capture(M)
            self.graphs[M].replay()
        else:
            self._step_body(M)
        return self.logits[:M]

    def commit(self, n: int):
        """The first n tokens of the last step are part of the sequence."""
        self.len_host += n
        self.cur_len.fill_(self.len_host)
        self.n_commit.fill_(n)

    # ------------------------------------------------------------------------------------------
    # prefill (torch + cuBLAS on dequantized weights)
    # ------------------------------------------------------------------------------------------
    def _lin_prefill(self, name, x):
        return x @ self.lin[name].dequant(self.scratch).t()

    def _commit_torch(self):
        """Apply pending GDN tokens of the last step to S (same math as the kernel)."""
        a = int(self.n_commit.item())
        if a == 0:
            return
        c = self.cfg
        L = self.len_host
        Hk, Hv = c.linear_num_key_heads, c.linear_num_value_heads
        for i in c.gdn_layers:
            ring, w = self.ring[i], self.small[f"layers.{i}.conv_w"]
            kd = c.gdn_key_dim
            for j in range(a):
                P = L - a + j
                pos = torch.arange(P - 3, P + 1, device=self.dev)
                x = ring[kd:2 * kd, (pos & (RING - 1))].float() * (pos >= 0).float()[None]  # [kd, 4]
                acc = (x * w[kd:2 * kd]).sum(-1)
                kv = T.bf16r(F.silu(T.bf16r(acc))).reshape(Hk, 128)
                kv = T.l2norm(kv).repeat_interleave(Hv // Hk, dim=0)  # [Hv, 128]
                u = self.pend_u[i][j]
                alpha = self.pend_g[i][j, :, 0].exp()
                self.S[i].mul_(alpha[:, None, None]).add_(kv[:, :, None] * u[:, None, :])
        self.n_commit.zero_()

    @torch.no_grad()
    def prefill(self, ids: list[int]) -> torch.Tensor:
        """Process prompt tokens (committed). Returns fp32 logits [V] of the last token."""
        self._commit_torch()
        for s in range(0, len(ids), self.prefill_chunk):
            self._prefill_chunk(ids[s:s + self.prefill_chunk])
        return self.logits[0]

    def _prefill_chunk(self, ids: list[int]):
        c, eps, dev = self.cfg, self.cfg.rms_norm_eps, self.dev
        n = len(ids)
        P0 = self.len_host
        assert P0 + n <= self.max_len
        pos = torch.arange(P0, P0 + n, device=dev)
        h = self.embed[torch.tensor(ids, device=dev)].float()
        H, Hkv, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        Hk, Hv, C, Vd, kd = c.linear_num_key_heads, c.linear_num_value_heads, c.gdn_conv_dim, c.gdn_value_dim, c.gdn_key_dim
        for i, lt in enumerate(c.layer_types):
            p = f"layers.{i}."
            x = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
            if lt == "linear_attention":
                proj = self._lin_prefill(p + "in_proj", x)
                mixed_in, z = proj[:, :C], proj[:, C:C + Vd]
                b, a = proj[:, C + Vd:C + Vd + Hv].float(), proj[:, C + Vd + Hv:].float()
                hp = torch.arange(P0 - 3, P0, device=dev)
                hist = self.ring[i][:, hp & (RING - 1)].t().float() * (hp >= 0).float()[:, None]
                mixed = T.conv_silu(hist, mixed_in.float(), self.small[p + "conv_w"])
                keep = min(3, n)
                slots = torch.arange(P0 + n - keep, P0 + n, device=dev) & (RING - 1)
                self.ring[i][:, slots] = mixed_in[n - keep:].t()
                q = mixed[:, :kd].reshape(n, Hk, 128)
                k = mixed[:, kd:2 * kd].reshape(n, Hk, 128)
                v = mixed[:, 2 * kd:].reshape(n, Hv, 128)
                q = (T.l2norm(q) * 128 ** -0.5).repeat_interleave(Hv // Hk, dim=1)
                k = T.l2norm(k).repeat_interleave(Hv // Hk, dim=1)
                g = -self.small[p + "A_log"].exp() * F.softplus(a + self.small[p + "dt_bias"])
                beta = torch.sigmoid(b)
                o, S_new = T.gdn_chunk(q, k, v, g, beta, self.S[i])
                self.S[i].copy_(S_new)  # in place: graphs hold the buffer address
                xo = T.gated_head_norm(o.reshape(n, Vd).to(torch.bfloat16), z, 128, eps)
                h = h + self._lin_prefill(p + "out_proj", xo).float()
            else:
                proj = self._lin_prefill(p + "qkv_proj", x)
                qg = proj[:, :c.attn_q_dim].reshape(n, H, 2 * D)
                q, gate = qg[..., :D], qg[..., D:].reshape(n, H * D)
                k = proj[:, c.attn_q_dim:c.attn_q_dim + Hkv * D].reshape(n, Hkv, D)
                v = proj[:, c.attn_q_dim + Hkv * D:].reshape(n, Hkv, D)
                q = T.rope_bf16(T.rms_norm_w(q, self.small[p + "q_norm"], eps), pos, self.inv_freq)
                k = T.rope_bf16(T.rms_norm_w(k, self.small[p + "k_norm"], eps), pos, self.inv_freq)
                self.kc[i][:, P0:P0 + n] = k.transpose(0, 1).to(torch.bfloat16)
                self.vc[i][:, P0:P0 + n] = v.transpose(0, 1)
                Ls = P0 + n
                mask = torch.arange(Ls, device=dev)[None, :] <= pos[:, None]
                o = F.scaled_dot_product_attention(q.transpose(0, 1)[None].to(torch.bfloat16), self.kc[i][None, :, :Ls],
                                                   self.vc[i][None, :, :Ls], attn_mask=mask, enable_gqa=True)
                o = o[0].transpose(0, 1).reshape(n, H * D).float()
                o = (T.bf16r(o) * T.bf16r(torch.sigmoid(gate.float()))).to(torch.bfloat16)
                h = h + self._lin_prefill(p + "o_proj", o).float()
            x = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
            gu = self._lin_prefill(p + "gate_up", x)
            hm = (T.bf16r(F.silu(gu[:, 0::2].float())) * gu[:, 1::2].float()).to(torch.bfloat16)
            h = h + self._lin_prefill(p + "down", hm).float()
        self.resid[0].copy_(h[-1])
        self.lin["lm_head"].gemv(PRO_RMSNORM, self.resid[0], self.logits[0], EPI_F32, eps=eps)
        self.len_host += n
        self.cur_len.fill_(self.len_host)
        self.n_commit.zero_()
