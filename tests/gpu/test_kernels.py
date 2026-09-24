"""Each CUDA kernel against its torch emulation (cckernel/emu.py) at the model's real shapes."""

import pytest
import torch

from cckernel import emu, quant

C_ = pytest.importorskip("cckernel._C")
dev = torch.device("cuda")


def _qlin(N, K, bits, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=g) * 0.02
    q = quant.QLinear.from_weight(w, bits, clip_grid=4)
    lo = (q.planes["q8"] if bits == 8 else q.planes["lo"]).to(dev)
    hi = q.planes["hi"].to(dev) if bits in (5, 6) else torch.empty(0, dtype=torch.uint8, device=dev)
    return lo, hi, q.scales.to(dev)


def _rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6))


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("N,K", [(4096, 4096), (4096, 12288), (12352, 4096)])
@pytest.mark.parametrize("pro,epi", [(0, 0), (1, 0), (1, 1), (0, 2), (1, 3), (2, 2)])
def test_qgemv(bits, N, K, pro, epi):
    if pro == 2 and K != 4096:
        pytest.skip("gated prologue only used with K = Hv*dv")
    lo, hi, sc = _qlin(N, K, bits)
    x = torch.randn(K, device=dev) * (1.0 if pro == 1 else 0.5)
    x = x if pro == 1 else x.to(torch.bfloat16)
    z = torch.randn(K, device=dev).to(torch.bfloat16) if pro == 2 else torch.empty(0, device=dev, dtype=torch.bfloat16)
    ydt = torch.float32 if epi in (1, 2) else torch.bfloat16
    ny = N // 2 if epi == 3 else N
    y0 = torch.randn(ny, device=dev).to(ydt)
    y1 = y0.clone()
    C_.qgemv(lo, hi, sc, bits, N, K, pro, x, z, 128, 1e-6, epi, y0)
    emu.qgemv(lo, hi, sc, bits, N, K, pro, x, z, 128, 1e-6, epi, y1)
    torch.cuda.synchronize()
    assert _rel(y0, y1) < (2e-2 if ydt == torch.bfloat16 else 2e-3)


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("M", [2, 3, 5, 8])
@pytest.mark.parametrize("epi", [0, 1, 2])
def test_skinny(bits, M, epi):
    N, K = 10240, 4096
    lo, hi, sc = _qlin(N, K, bits, seed=1)
    x = (torch.randn(8, K, device=dev) * 0.5).to(torch.bfloat16)
    ydt = torch.bfloat16 if epi == 0 else torch.float32
    y0 = torch.randn(8, N, device=dev).to(ydt)
    y1 = y0.clone()
    C_.qgemm_skinny(lo, hi, sc, bits, N, K, x, M, epi, y0)
    emu.qgemm_skinny(lo, hi, sc, bits, N, K, x, M, epi, y1)
    torch.cuda.synchronize()
    assert _rel(y0[:M], y1[:M]) < (2e-2 if epi == 0 else 2e-3)
    assert torch.equal(y0[M:], y1[M:])  # rows beyond M untouched


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_dequant(bits):
    N, K = 512, 4096
    lo, hi, sc = _qlin(N, K, bits, seed=2)
    a = torch.empty(N * K, dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    C_.dequant(lo, hi, sc, bits, N, K, a)
    emu.dequant(lo, hi, sc, bits, N, K, b)
    assert torch.equal(a, b)


def _gdn_state(seed=0, Hk=16, Hv=32):
    g = torch.Generator(device=dev).manual_seed(seed)
    C, Vd = 2 * Hk * 128 + Hv * 128, Hv * 128
    r = lambda *s, sc=1.0: torch.randn(*s, generator=g, device=dev) * sc  # noqa: E731
    return dict(
        proj=r(8, C + Vd + 2 * Hv).to(torch.bfloat16), ring=r(C, 32).to(torch.bfloat16), conv_w=r(C, 4, sc=0.4),
        A_log=r(Hv, sc=0.5), dt_bias=r(Hv, sc=0.5), S=r(Hv, 128, 128, sc=0.1), pend_u=r(8, Hv, 128, sc=0.2),
        pend_g=(-torch.rand(8, Hv, 1, generator=g, device=dev)).expand(8, Hv, 4).contiguous(),
        out=torch.zeros(8, Vd, dtype=torch.bfloat16, device=dev), C=C, Hk=Hk, Hv=Hv)


@pytest.mark.parametrize("M", [1, 2, 5, 8])
@pytest.mark.parametrize("L,a", [(0, 0), (7, 0), (100, 1), (100, 4), (37, 8)])
def test_gdn_decode(M, L, a):
    s0 = _gdn_state(seed=M + L)
    s1 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s0.items()}
    cur = torch.tensor([L], dtype=torch.int32, device=dev)
    nc = torch.tensor([a], dtype=torch.int32, device=dev)
    for mod, s in ((C_, s0), (emu, s1)):
        mod.gdn_decode(s["proj"], s["ring"], s["conv_w"], s["A_log"], s["dt_bias"], s["S"], s["pend_u"], s["pend_g"],
                       s["out"], cur, nc, M, s["Hk"], s["Hv"], s["C"])
    torch.cuda.synchronize()
    assert _rel(s0["out"][:M], s1["out"][:M]) < 2e-2
    assert _rel(s0["S"], s1["S"]) < 1e-4
    assert _rel(s0["pend_u"][:M], s1["pend_u"][:M]) < 1e-3
    assert _rel(s0["pend_g"][:M], s1["pend_g"][:M]) < 1e-4
    assert torch.equal(s0["ring"], s1["ring"])


@pytest.mark.parametrize("M", [1, 4, 8])
@pytest.mark.parametrize("L", [0, 5, 300, 2000])
def test_attention(M, L):
    H, Hkv, D, max_len, NS = 16, 4, 256, 4096, 32
    g = torch.Generator(device=dev).manual_seed(L + M)
    proj = (torch.randn(8, H * 2 * D + 2 * Hkv * D, generator=g, device=dev)).to(torch.bfloat16)
    qn = 1 + 0.1 * torch.randn(D, generator=g, device=dev)
    kn = 1 + 0.1 * torch.randn(D, generator=g, device=dev)
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, device=dev).float() / 64))
    kc = (torch.randn(Hkv, max_len, D, generator=g, device=dev)).to(torch.bfloat16)
    vc = (torch.randn(Hkv, max_len, D, generator=g, device=dev)).to(torch.bfloat16)
    cur = torch.tensor([L], dtype=torch.int32, device=dev)
    outs = []
    for mod in (C_, emu):
        q = torch.zeros(8, H, D, device=dev)
        k2, v2 = kc.clone(), vc.clone()
        out = torch.zeros(8, H * D, dtype=torch.bfloat16, device=dev)
        pa = torch.zeros(8, NS, H, D, device=dev)
        pm = torch.zeros(8, NS, H, 2, device=dev)
        cnt = torch.zeros(8, Hkv, dtype=torch.int32, device=dev)
        mod.attn_prep(proj, qn, kn, inv, q, k2, v2, cur, M, H, Hkv, 1e-6)
        mod.attn_decode(q, k2, v2, proj, pa, pm, cnt, out, cur, M, H, Hkv, NS)
        torch.cuda.synchronize()
        outs.append((q, k2, v2, out, cnt))
    (q0, k0, v0, o0, c0), (q1, k1, v1, o1, _) = outs
    assert _rel(q0[:M], q1[:M]) < 1e-2
    assert _rel(k0, k1) < 1e-2 and torch.equal(v0, v1)
    assert _rel(o0[:M], o1[:M]) < 2e-2
    assert int(c0.abs().sum()) == 0  # tickets re-armed
