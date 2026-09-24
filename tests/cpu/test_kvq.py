"""Quantized KV cache (kvq): number formats, scale search, rotation, and the engine with a quantized
cache (oracle agreement, prefill/decode consistency, exact speculative verification, CPU backend)."""

import pytest
import torch

from cckernel import kvq, quant
from cckernel.config import TextConfig
from cckernel.engine import Engine
from cckernel.folded import FoldedCache, FoldedModel
from cckernel.quant_io import write_cck
from cckernel.reference import random_weights
from cckernel.spec import accept

MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
EVEN = {0.0, 1.0, 2.0, 4.0}  # E2M1 values with a zero mantissa bit


def _e2m1_brute(y: float) -> float:
    a = min(abs(y), 6.0)
    d = [abs(a - m) for m in MAGS]
    best = min(d)
    cands = [m for m, di in zip(MAGS, d) if di == best]
    m = cands[0] if len(cands) == 1 else [c for c in cands if c in EVEN][0]
    return -m if y < 0 else m


def test_e2m1_round_to_nearest_even():
    ys = torch.tensor([0.0, 0.1, 0.25, 0.26, 0.5, 0.74, 0.75, 1.0, 1.25, 1.3, 1.75, 2.2, 2.5, 2.6, 3.5, 4.9, 5.0, 5.01,
                       6.0, 7.5, 100.0] + torch.linspace(-8, 8, 4001).tolist())
    got = kvq.e2m1_decode(kvq.e2m1_encode(ys))
    want = torch.tensor([_e2m1_brute(float(y)) for y in ys])
    assert torch.equal(got.abs(), want.abs())
    assert torch.equal(torch.signbit(got[want != 0]), torch.signbit(want[want != 0]))


def test_e4m3_codes_roundtrip():
    codes = torch.arange(0, kvq.E4M3_MAX_CODE + 1, dtype=torch.uint8)
    vals = kvq.e4m3_decode(codes)
    assert torch.equal(kvq.e4m3_encode(vals), codes)
    assert float(vals.max()) == 448.0 and torch.all(vals[1:] > vals[:-1])
    assert float(kvq.e4m3_decode(kvq.e4m3_encode(torch.tensor([1e6])))) == 448.0  # saturation


def test_fp4_scale_search_beats_absmax_and_error_level():
    torch.manual_seed(0)
    x = torch.randn(512, 256) * torch.rand(512, 1) * 3
    packed, sc = kvq.fp4_encode(x)
    assert packed.shape == (512, 128) and sc.shape == (512, 16)
    xh = kvq.fp4_decode(packed, sc)
    g = x.reshape(512, 16, 16)
    err = (xh.reshape(512, 16, 16) - g).pow(2).sum(-1)
    s0 = kvq.e4m3_decode(kvq.e4m3_encode(g.abs().amax(-1) / 6.0))
    q0 = kvq.e2m1_decode(kvq.e2m1_encode(g / s0[..., None])) * s0[..., None]
    err0 = (q0 - g).pow(2).sum(-1)
    assert torch.all(err <= err0 + 1e-9)
    rel = float((xh - x).pow(2).sum() / x.pow(2).sum())
    assert rel < 0.011, rel  # NVFP4-level error on Gaussian data (absmax scaling alone: ~0.012)
    assert torch.equal(kvq.fp4_decode(*kvq.fp4_encode(torch.zeros(3, 256))), torch.zeros(3, 256))


def test_rotation_is_exact():
    torch.manual_seed(1)
    s = kvq.kv_signs()
    q, k = torch.randn(10, 256, dtype=torch.float64), torch.randn(10, 256, dtype=torch.float64)
    qr, kr = kvq.rotate(q.float(), s).double(), kvq.rotate(k.float(), s).double()
    assert torch.allclose((qr * kr).sum(-1), (q * k).sum(-1), atol=1e-4)
    assert torch.allclose(kvq.unrotate(kvq.rotate(q.float(), s), s).double(), q, atol=1e-5)


def test_rotated_fp4_error_is_structure_independent():
    """After the rotation the FP4 error is the Gaussian level (~0.74% relative MSE) whatever the channel
    structure. (With 16-channel groups and the MSE scale search, plain FP4 handles these synthetic
    outlier patterns about as well; whether the rotation pays off is measured on the real model.)"""
    torch.manual_seed(2)
    s = kvq.kv_signs()
    rel = lambda y: float((kvq.fake_quant(kvq.FP4, y) - y).pow(2).sum() / y.pow(2).sum())  # noqa: E731
    for idx in (slice(0, 8), slice(0, None, 16), slice(0, 0)):
        x = torch.randn(256, 256)
        x[:, idx] *= 30
        assert 0.006 < rel(kvq.rotate(x, s)) < 0.0085


# ------------------------------------------------------------------------------------------ engine
CFG = TextConfig.tiny(hidden_size=256, intermediate_size=512, vocab_size=600, num_attention_heads=4,
                      num_key_value_heads=1, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=4,
                      linear_key_head_dim=128, linear_value_head_dim=128, rope_theta=1e7)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(0)
    w = random_weights(CFG, seed=5, scale=0.06)
    out = tmp_path_factory.mktemp("cckq")
    write_cck(CFG, w.__getitem__, out, {n: 8 for n, _, _ in quant.matrix_list(CFG)}, seed=3)
    return out


def _oracle(eng: Engine) -> FoldedModel:
    t = {name: quant.QLinear(q.bits, q.N, q.K, {"q8": q.lo}, q.scales).dequant() for name, q in eng.lin.items()}
    t.update(eng.small)
    t["embed"] = eng.embed.float()
    return FoldedModel(CFG, t, gdn_algo="recurrent", kv_format=eng.kv_format, kv_rotate=eng.kv_rotate,
                       kv_signs=eng.kv_signs)


def _rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


@pytest.mark.parametrize("fmt", ["fp8", "k8v4", "fp4"])
def test_engine_quantized_kv_matches_oracle(model_dir, fmt):
    eng = Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4, prefill_chunk=16, kv_format=fmt)
    assert eng.kv_rotate
    ref = _oracle(eng)
    ids = torch.randint(0, CFG.vocab_size, (26,), generator=torch.Generator().manual_seed(3)).tolist()
    cache = FoldedCache()
    assert _rel(eng.prefill(ids[:21]), ref.forward(torch.tensor(ids[:21]), cache)[-1]) < 3e-2
    for t in ids[21:]:
        lg = eng.step([t])[0]
        eng.commit(1)
        assert _rel(lg, ref.forward(torch.tensor([t]), cache)[-1]) < 3e-2


def test_quantized_kv_changes_little_vs_bf16(model_dir):
    ids = torch.randint(0, CFG.vocab_size, (40,), generator=torch.Generator().manual_seed(4)).tolist()
    lp = {}
    for fmt in ("bf16", "fp8", "fp4"):
        eng = Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4, kv_format=fmt)
        lp[fmt] = eng.score(ids)
    kl = lambda a, b: float((a.exp() * (a - b)).sum(-1).mean())  # noqa: E731
    assert kl(lp["bf16"], lp["fp8"]) < kl(lp["bf16"], lp["fp4"]) < 5e-2


def test_fp4_speculative_verify_equals_greedy(model_dir):
    prompt = torch.randint(0, CFG.vocab_size, (12,), generator=torch.Generator().manual_seed(5)).tolist()

    def mk():
        return Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4, kv_format="fp4")

    eng = mk()
    tok = int(eng.prefill(prompt).argmax())
    want = [tok]
    for _ in range(13):
        tok = int(eng.step([tok])[0].argmax())
        eng.commit(1)
        want.append(tok)
    eng = mk()
    tok = int(eng.prefill(prompt).argmax())
    got = [tok]
    rng = torch.Generator().manual_seed(7)
    while len(got) < 14:
        k = int(torch.randint(1, 5, (1,), generator=rng))
        draft = [want[len(got) + j] if len(got) + j < len(want) and torch.rand(1, generator=rng) < 0.7
                 else int(torch.randint(0, CFG.vocab_size, (1,), generator=rng)) for j in range(k)]
        em = accept(eng.step([tok] + draft), draft)
        eng.commit(len(em))
        got += em
        tok = got[-1]
    assert got[:14] == want


@pytest.mark.parametrize("fmt", ["bf16", "fp4"])
def test_cpu_fast_backend_matches_emu(model_dir, fmt, monkeypatch):
    from cckernel import cpu

    monkeypatch.setattr(cpu, "small_prefill", 0)  # same (layer-major) prefill path in both engines
    ids = torch.randint(0, CFG.vocab_size, (20,), generator=torch.Generator().manual_seed(6)).tolist()
    a = Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4, kv_format=fmt)
    b = Engine(model_dir, device="cpu", cpu_backend="fast", max_len=96, attn_splits=4, kv_format=fmt)
    assert b.lin["layers.0.in_proj"].lo.dtype == torch.int8  # converted to the group-major int8 layout
    assert _rel(b.prefill(ids[:16]), a.prefill(ids[:16])) < 1e-2
    for t in ids[16:]:
        la, lb = a.step([t])[0], b.step([t])[0]
        a.commit(1)
        b.commit(1)
        assert _rel(lb, la) < 2e-2
    # M-invariance of the fast backend: verifying 4 tokens == 4 single steps, bit for bit
    st = b.session_state()
    lm = b.step(ids[:4]).clone()
    b.load_session_state(st)
    rows = []
    for t in ids[:4]:
        rows.append(b.step([t])[0].clone())
        b.commit(1)
    assert torch.equal(lm, torch.stack(rows))
    lpa, lpb = a.score(ids), b.score(ids)
    assert _rel(lpb, lpa) < 1e-2
