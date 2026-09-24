"""cckernel inference engine for the quantized (cck-v1) model on one sm_89 GPU.

Decode / verify steps (M = 1..8 tokens) run entirely on the custom CUDA kernels and are captured
into CUDA graphs (one per M). Prefill uses torch (dequantized bf16 GEMMs through cuBLAS, SDPA and a
vectorised chunked Gated DeltaNet) and writes the same caches the kernels use.

Cache layout shared by all paths:
  GDN layer:  ring  bf16 [C, 32]   conv inputs by absolute position (slot = pos & 31)
              S     fp32 [Hv, 128, 128]  committed recurrent state
              pend_u / pend_g   pseudo-values / log-decays of the last step (deferred commit)
  attention:  K / V caches in the chosen KV format (``kvq``): bf16 [Hkv, max_len, 256], fp8 codes
              [Hkv, max_len, 256], or fp4 nibbles [Hkv, max_len, 128] + E4M3 scales [Hkv, max_len, 16].
              K is stored after qk-norm + RoPE (+ the per-head Hadamard rotation for quantized formats).
  scalars:    cur_len = committed tokens, n_commit = tokens of the last step accepted by the host

Backends: the CUDA extension on GPU; on CPU either ``cpu`` (fast, int8 weights, runs the 9B model)
or ``emu`` (the exact fp32 oracle the CUDA kernels are tested against).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import kvq
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


def _mm_f32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 x @ w^T for bf16 operands (fp32 output: logits must not be rounded to bf16)."""
    if x.is_cuda:
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    return x.float() @ w.float().t()


class QLin:
    """A device-resident packed linear layer."""

    def __init__(self, sd: dict, prefix: str, bits: int, N: int, K: int, device, ext=None):
        self.bits, self.N, self.K = bits, N, K
        self.ext = ext
        key = "q8" if bits == 8 else "lo"
        self.lo = sd[f"{prefix}.{key}"].to(device)
        self.hi = sd[f"{prefix}.hi"].to(device) if bits in (5, 6) else torch.empty(0, dtype=torch.uint8, device=device)
        self.scales = sd[f"{prefix}.scales"].to(device)
        assert self.scales.shape == (N, K // GROUP), (prefix, self.scales.shape)

    def nbytes(self):
        return self.lo.numel() + self.hi.numel() + self.scales.numel() * 2

    @property
    def _x(self):
        return self.ext or _ext()

    def gemv(self, pro, x, y, epi, z=None, head_dim=0, eps=1e-6):
        z = z if z is not None else torch.empty(0, device=x.device, dtype=torch.bfloat16)
        self._x.qgemv(self.lo, self.hi, self.scales, self.bits, self.N, self.K, pro, x, z, head_dim, eps, epi, y)

    def skinny(self, x, M, y, epi):
        self._x.qgemm_skinny(self.lo, self.hi, self.scales, self.bits, self.N, self.K, x, M, epi, y)

    def dequant(self, scratch):
        w = scratch[: self.N * self.K].view(self.N, self.K)
        self._x.dequant(self.lo, self.hi, self.scales, self.bits, self.N, self.K, w)
        return w

    def dequant_rows(self, r0, r1, scratch):
        """bf16 rows [r0, r1) of the matrix (lm_head in chunks)."""
        n, x = r1 - r0, self._x
        w = scratch[: n * self.K].view(n, self.K)
        if hasattr(x, "dequant_rows"):
            x.dequant_rows(self.lo, self.hi, self.scales, self.bits, self.N, self.K, r0, r1, w)
        else:
            hi = self.hi[r0:r1] if self.hi.numel() else self.hi
            x.dequant(self.lo[r0:r1], hi, self.scales[r0:r1], self.bits, n, self.K, w)
        return w


class Engine:
    def __init__(self, model_dir: str | Path, device: str = "cuda", max_len: int = 32768, attn_splits: int = 32,
                 prefill_chunk: int = 2048, use_graphs: bool = True, kv_format: str | None = None,
                 kv_rotate: bool | None = None, cpu_backend: str = "fast"):
        self.dir = Path(model_dir)
        man = json.loads((self.dir / "cck_manifest.json").read_text())
        assert man["format"] == "cck-v1", man["format"]
        self.manifest = man
        self.cfg = cfg = TextConfig.from_dict(man["config"])
        self.dev = torch.device(device)
        runtime = man.get("runtime", {})
        self.kv_format = kv_format or os.environ.get("CCK_KV") or runtime.get("kv_cache", "bf16")
        self.kfmt, self.vfmt = kvq.KV_FORMATS[self.kv_format]
        self.kv_rotate = (self.kv_format != "bf16") if kv_rotate is None else kv_rotate
        self.kv_signs = kvq.kv_signs(runtime.get("kv_rotate_seed", kvq.KV_SEED), cfg.head_dim).to(self.dev)
        if self.dev.type != "cuda" or os.environ.get("CCK_EMULATE") == "1":
            from . import cpu, emu

            self.ext = cpu if (self.dev.type == "cpu" and cpu_backend == "fast") else emu
            use_graphs = False
        else:
            self.ext = _ext()
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
                self.lin[name] = self._qlin(sd, name, bits[name], shapes[name])
            for k, v in sd.items():
                if k.split(".")[-1] in ("conv_w", "A_log", "dt_bias", "q_norm", "k_norm"):
                    self.small[k] = v.to(self.dev, torch.float32).contiguous()
        g = load_file(str(self.dir / "globals.safetensors"))
        self.lin["lm_head"] = self._qlin(g, "lm_head", bits["lm_head"], shapes["lm_head"])
        self.embed = g["embed"].to(self.dev)
        del g

        self._alloc_state()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def _qlin(self, sd, name, bits, shape) -> QLin:
        q = QLin(sd, name, bits, *shape, self.dev, ext=self.ext)
        if hasattr(self.ext, "prepare"):
            self.ext.prepare(q)
        return q

    # ------------------------------------------------------------------------------------------
    def _alloc_state(self):
        c, d = self.cfg, self.dev
        bf, f32 = torch.bfloat16, torch.float32
        C, Hv = c.gdn_conv_dim, c.linear_num_value_heads
        self.ring = {i: torch.zeros(C, RING, dtype=bf, device=d) for i in c.gdn_layers}
        self.S = {i: torch.zeros(Hv, 128, 128, dtype=f32, device=d) for i in c.gdn_layers}
        self.pend_u = {i: torch.zeros(MAXM, Hv, 128, dtype=f32, device=d) for i in c.gdn_layers}
        self.pend_g = {i: torch.zeros(MAXM, Hv, 4, dtype=f32, device=d) for i in c.gdn_layers}
        Hkv, D = c.num_key_value_heads, c.head_dim
        self._alloc_kv()
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
        # two dequant buffers: a block's input and output projections are resident together
        self.scratch = [torch.empty(biggest, dtype=bf, device=d), torch.empty(biggest, dtype=bf, device=d)]

    def _alloc_kv(self):
        c, d = self.cfg, self.dev
        Hkv, D = c.num_key_value_heads, c.head_dim
        self.kc, self.ks, self.vc, self.vs = {}, {}, {}, {}
        for i in c.attn_layers:
            self.kc[i], self.ks[i] = kvq.alloc(self.kfmt, Hkv, self.max_len, D, d)
            self.vc[i], self.vs[i] = kvq.alloc(self.vfmt, Hkv, self.max_len, D, d)

    def set_kv_format(self, kv_format: str, kv_rotate: bool | None = None):
        """Switch the KV cache format (re-allocates the cache and drops captured graphs)."""
        self.kv_format = kv_format
        self.kfmt, self.vfmt = kvq.KV_FORMATS[kv_format]
        self.kv_rotate = (kv_format != "bf16") if kv_rotate is None else kv_rotate
        self.kc = self.ks = self.vc = self.vs = None
        self.graphs.clear()
        self._alloc_kv()
        self.reset()

    def reset(self):
        for t in list(self.ring.values()) + list(self.S.values()):
            t.zero_()
        self.cur_len.zero_()
        self.n_commit.zero_()
        self.len_host = 0
        self._kv_changed()

    def _kv_changed(self):
        f = getattr(self.ext, "kv_changed", None)
        if f is not None:
            f()

    def vram_report(self) -> str:
        w = sum(q.nbytes() for q in self.lin.values())
        e = self.embed.numel() * 2
        kv = self.kv_bytes()
        st = sum(t.numel() * 4 for t in self.S.values())
        return (f"weights {w / 2**30:.2f} GiB | embed {e / 2**30:.2f} GiB | KV cache {self.kv_format} ({self.max_len} tok, "
                f"{self.kv_bytes_per_token()} B/tok) {kv / 2**30:.2f} GiB | GDN state {st / 2**20:.0f} MiB"
                + (f" | allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB" if self.dev.type == "cuda" else ""))

    def kv_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for dd in (self.kc, self.ks, self.vc, self.vs) for t in dd.values())

    def kv_bytes_per_token(self) -> int:
        c = self.cfg
        return kvq.kv_bytes_per_token(self.kv_format, len(c.attn_layers), c.num_key_value_heads, c.head_dim)

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
        c, ext, eps = self.cfg, self.ext, self.cfg.rms_norm_eps
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
                              self.kc[i], self.ks[i], self.vc[i], self.vs[i], self.kv_signs, self.cur_len, M, H, Hkv,
                              eps, self.kfmt, self.vfmt, self.kv_rotate)
                ext.attn_decode(self.qbuf, self.kc[i], self.ks[i], self.vc[i], self.vs[i], self.kv_signs, self.proj_a,
                                self.part_acc, self.part_ml, self.counters, self.attn_out, self.cur_len, M, H, Hkv,
                                self.NS, self.kfmt, self.vfmt, self.kv_rotate)
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
    # persistent sessions (prefix cache, DeepSeek-V4.1 sec. 3.2.1): KV prefix + recurrent state
    # ------------------------------------------------------------------------------------------
    def session_state(self) -> dict:
        """Everything needed to resume the current sequence: the committed KV prefix (in cache format),
        the GDN state S, the conv ring, the deferred-commit buffers and the scalars. CPU tensors."""
        L = self.len_host
        st = {"len": L, "n_commit": int(self.n_commit.item()), "kv_format": self.kv_format,
              "kv_rotate": bool(self.kv_rotate)}
        for name, dd in (("kc", self.kc), ("ks", self.ks), ("vc", self.vc), ("vs", self.vs)):
            for i, t in dd.items():
                st[f"{name}.{i}"] = t[:, :L].cpu().clone() if t.numel() else t.cpu()
        for name, dd in (("S", self.S), ("ring", self.ring), ("pu", self.pend_u), ("pg", self.pend_g)):
            for i, t in dd.items():
                st[f"{name}.{i}"] = t.cpu().clone()
        return st

    def load_session_state(self, st: dict):
        assert st["kv_format"] == self.kv_format and st["kv_rotate"] == bool(self.kv_rotate), "KV format mismatch"
        L = st["len"]
        assert L <= self.max_len
        for name, dd in (("kc", self.kc), ("ks", self.ks), ("vc", self.vc), ("vs", self.vs)):
            for i, t in dd.items():
                if t.numel():
                    t[:, :L].copy_(st[f"{name}.{i}"])
        for name, dd in (("S", self.S), ("ring", self.ring), ("pu", self.pend_u), ("pg", self.pend_g)):
            for i, t in dd.items():
                t.copy_(st[f"{name}.{i}"])  # in place: CUDA graphs hold these addresses
        self.len_host = L
        self.cur_len.fill_(L)
        self.n_commit.fill_(st["n_commit"])
        self._kv_changed()

    def save_session(self, path: str | Path):
        torch.save(self.session_state(), str(path))

    def load_session(self, path: str | Path):
        self.load_session_state(torch.load(str(path), map_location="cpu"))

    # ------------------------------------------------------------------------------------------
    # step-cost profile T(M, ctx) for the confidence-scheduled speculative decoding
    # ------------------------------------------------------------------------------------------
    def _sync(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize()

    def profile_costs(self, ctxs=(256, 4096), ms=range(1, MAXM + 1), reps: int = 3, cache: bool = True) -> dict:
        """Time verify steps of M tokens at a few context lengths and fit T = a + b*M + c*M*ctx
        (least squares). The KV cache content does not matter for timing, so no prefill is needed.
        Cached per device / KV format in the model directory."""
        name = f"step_cost_{self.dev.type}_{self.kv_format}.json"
        path = self.dir / name
        if cache and path.exists():
            return json.loads(path.read_text())
        saved = self.session_state() if self.len_host else None
        ctxs = [c for c in ctxs if c + MAXM < self.max_len] or [0]
        rows, ys, samples = [], [], []
        for ctx in ctxs:
            for M in ms:
                self.len_host = ctx
                self.cur_len.fill_(ctx)
                self.n_commit.zero_()
                self.step([0] * M)  # warm up / capture the graph
                self._sync()
                t0 = time.perf_counter()
                for _ in range(reps):
                    self.step([0] * M)
                self._sync()
                t = (time.perf_counter() - t0) / reps
                rows.append([1.0, M, M * ctx])
                ys.append(t)
                samples.append({"ctx": ctx, "M": M, "s": t})
        A, y = torch.tensor(rows, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)
        coef = torch.linalg.lstsq(A, y[:, None]).solution[:, 0].tolist()
        prof = {"a": coef[0], "b": coef[1], "c": coef[2], "device": self.dev.type, "kv_format": self.kv_format,
                "samples": samples}
        self.reset()
        if saved is not None:
            self.load_session_state(saved)
        if cache:
            try:
                path.write_text(json.dumps(prof, indent=1))
            except OSError:
                pass
        return prof

    # ------------------------------------------------------------------------------------------
    # prefill (torch + cuBLAS on dequantized weights)
    # ------------------------------------------------------------------------------------------

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
        """Process prompt tokens (committed). Returns fp32 logits [V] of the last token.
        Backends that set ``small_prefill`` (the CPU backend, where dequantizing every matrix costs
        more than streaming a few tokens through the int8 decode path) run short inputs as steps."""
        if len(ids) <= getattr(self.ext, "small_prefill", 0):
            for s in range(0, len(ids), MAXM):
                chunk = ids[s:s + MAXM]
                lg = self.step(chunk)
                self.commit(len(chunk))
            return lg[len(chunk) - 1]
        self._commit_torch()
        self._prefill(ids)
        return self.logits[0]

    @torch.no_grad()
    def score(self, ids: list[int], fn=None, block: int = 512):
        """Log-probabilities for every position of a fresh sequence (calibration / ppl / evaluation).
        Without ``fn`` returns fp32 [n, V]. With ``fn`` calls ``fn(start, logprobs_block)`` for blocks of
        ``block`` positions and returns None (bounded memory for long sequences). lm_head runs as
        row-chunked dequant + GEMM, like the prefill."""
        self.reset()
        h = self._prefill(ids, keep_hidden=True)
        self.reset()
        eps, V = self.cfg.rms_norm_eps, self.cfg.vocab_size
        xn = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
        del h
        lm = self.lin["lm_head"]
        rows = max(1, self.scratch[0].numel() // lm.K)
        out = None if fn is not None else torch.empty(len(ids), V, dtype=torch.float32, device=self.dev)
        for s in range(0, len(ids), block):
            x = xn[s:s + block]
            logits = torch.empty(x.shape[0], V, dtype=torch.float32, device=self.dev)
            for r0 in range(0, V, rows):
                r1 = min(V, r0 + rows)
                logits[:, r0:r1] = _mm_f32(x, lm.dequant_rows(r0, r1, self.scratch[0]))
            lp = torch.log_softmax(logits, dim=-1)
            del logits
            if fn is not None:
                fn(s, lp)
            else:
                out[s:s + x.shape[0]] = lp
        return out

    def _prefill(self, ids: list[int], keep_hidden: bool = False):
        """Layer-major prefill: every weight matrix is dequantized once per call (not per chunk), then
        the prompt is streamed through it in sub-chunks of ``prefill_chunk`` tokens (bounded
        activation memory). GDN state, conv ring and KV cache advance chunk by chunk."""
        c, eps, dev = self.cfg, self.cfg.rms_norm_eps, self.dev
        n, P0, CH = len(ids), self.len_host, self.prefill_chunk
        assert P0 + n <= self.max_len
        h = self.embed[torch.tensor(ids, device=dev)].float()
        rms = lambda t: (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)  # noqa: E731
        for i, lt in enumerate(c.layer_types):
            p = f"layers.{i}."
            names = ("in_proj", "out_proj") if lt == "linear_attention" else ("qkv_proj", "o_proj")
            W_in = self.lin[p + names[0]].dequant(self.scratch[0])
            W_out = self.lin[p + names[1]].dequant(self.scratch[1])
            for s in range(0, n, CH):
                e = min(n, s + CH)
                x = rms(h[s:e])
                if lt == "linear_attention":
                    mix = self._gdn_prefill(i, x @ W_in.t(), P0 + s)
                else:
                    mix = self._attn_prefill(i, x @ W_in.t(), P0 + s)
                h[s:e] += (mix @ W_out.t()).float()
            W_gu = self.lin[p + "gate_up"].dequant(self.scratch[0])
            W_dn = self.lin[p + "down"].dequant(self.scratch[1])
            for s in range(0, n, CH):
                e = min(n, s + CH)
                gu = rms(h[s:e]) @ W_gu.t()
                hm = (T.bf16r(F.silu(gu[:, 0::2].float())) * gu[:, 1::2].float()).to(torch.bfloat16)
                h[s:e] += (hm @ W_dn.t()).float()
        self.len_host += n
        self.cur_len.fill_(self.len_host)
        self.n_commit.zero_()
        self._kv_changed()
        if keep_hidden:
            return h
        self.resid[0].copy_(h[-1])
        self.lin["lm_head"].gemv(PRO_RMSNORM, self.resid[0], self.logits[0], EPI_F32, eps=eps)
        return None

    def _gdn_prefill(self, i: int, proj: torch.Tensor, P0: int) -> torch.Tensor:
        """GDN mixer for tokens at positions P0..P0+n-1; returns the gated-norm output (bf16)."""
        c, eps, dev = self.cfg, self.cfg.rms_norm_eps, self.dev
        p, n = f"layers.{i}.", proj.shape[0]
        Hk, Hv, C, Vd, kd = c.linear_num_key_heads, c.linear_num_value_heads, c.gdn_conv_dim, c.gdn_value_dim, c.gdn_key_dim
        mixed_in, z = proj[:, :C], proj[:, C:C + Vd]
        b, a = proj[:, C + Vd:C + Vd + Hv].float(), proj[:, C + Vd + Hv:].float()
        hp = torch.arange(P0 - 3, P0, device=dev)
        hist = self.ring[i][:, hp & (RING - 1)].t().float() * (hp >= 0).float()[:, None]
        mixed = T.conv_silu(hist, mixed_in.float(), self.small[p + "conv_w"])
        keep = min(3, n)
        slots = torch.arange(P0 + n - keep, P0 + n, device=dev) & (RING - 1)
        self.ring[i][:, slots] = mixed_in[n - keep:].t()
        rep = Hv // Hk
        q = (T.l2norm(mixed[:, :kd].reshape(n, Hk, 128)) * 128 ** -0.5).repeat_interleave(rep, dim=1)
        k = T.l2norm(mixed[:, kd:2 * kd].reshape(n, Hk, 128)).repeat_interleave(rep, dim=1)
        v = mixed[:, 2 * kd:].reshape(n, Hv, 128)
        g = -self.small[p + "A_log"].exp() * F.softplus(a + self.small[p + "dt_bias"])
        o, S_new = T.gdn_chunk(q, k, v, g, torch.sigmoid(b), self.S[i])
        self.S[i].copy_(S_new)  # in place: graphs hold the buffer address
        return T.gated_head_norm(o.reshape(n, Vd).to(torch.bfloat16), z, 128, eps)

    def _attn_prefill(self, i: int, proj: torch.Tensor, P0: int) -> torch.Tensor:
        """Gated attention for tokens at positions P0..P0+n-1 (writes the KV cache); returns bf16."""
        c, eps, dev = self.cfg, self.cfg.rms_norm_eps, self.dev
        p, n = f"layers.{i}.", proj.shape[0]
        H, Hkv, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        pos = torch.arange(P0, P0 + n, device=dev)
        qg = proj[:, :c.attn_q_dim].reshape(n, H, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(n, H * D)
        k = proj[:, c.attn_q_dim:c.attn_q_dim + Hkv * D].reshape(n, Hkv, D)
        v = proj[:, c.attn_q_dim + Hkv * D:].reshape(n, Hkv, D)
        q = T.rope_bf16(T.rms_norm_w(q, self.small[p + "q_norm"], eps), pos, self.inv_freq)
        k = T.rope_bf16(T.rms_norm_w(k, self.small[p + "k_norm"], eps), pos, self.inv_freq)
        Ls = P0 + n
        mask = torch.arange(Ls, device=dev)[None, :] <= pos[:, None]
        if self.kv_format == "bf16" and not self.kv_rotate:
            self.kc[i][:, P0:P0 + n] = k.transpose(0, 1).to(torch.bfloat16)
            self.vc[i][:, P0:P0 + n] = v.transpose(0, 1)
            K, V = self.kc[i][None, :, :Ls], self.vc[i][None, :, :Ls]
        else:
            # quantize into the cache first, then attend over the dequantized cache: prefill, decode and
            # verify all read exactly the same cached values
            if self.kv_rotate:
                q, k, v = kvq.rotate(q, self.kv_signs), kvq.rotate(k, self.kv_signs), kvq.rotate(v, self.kv_signs)
            for f, x, data, scale in ((self.kfmt, k, self.kc[i], self.ks[i]), (self.vfmt, v, self.vc[i], self.vs[i])):
                d_, s_ = kvq.encode(f, x.transpose(0, 1))
                data[:, P0:P0 + n] = d_
                if s_ is not None:
                    scale[:, P0:P0 + n] = s_
            K = kvq.decode(self.kfmt, self.kc[i][:, :Ls], self.ks[i][:, :Ls] if self.kfmt == kvq.FP4 else None)
            V = kvq.decode(self.vfmt, self.vc[i][:, :Ls], self.vs[i][:, :Ls] if self.vfmt == kvq.FP4 else None)
            K, V = K[None].to(torch.bfloat16), V[None].to(torch.bfloat16)
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None].to(torch.bfloat16), K, V, attn_mask=mask,
                                           enable_gqa=True)
        o = o[0].transpose(0, 1).float()  # [n, H, D]
        if self.kv_rotate:
            o = kvq.unrotate(o, self.kv_signs)
        o = o.reshape(n, H * D)
        return (T.bf16r(o) * T.bf16r(torch.sigmoid(gate.float()))).to(torch.bfloat16)
