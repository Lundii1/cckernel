"""Quantizer, packing layout and kernel-decode emulation."""

import pytest
import torch

from cckernel import quant
from cckernel.quant import PERM64, QLinear, dequant_rtn, pack, quantize_rtn, unpack

torch.manual_seed(0)


def test_perm_is_involution():
    assert torch.equal(PERM64[PERM64], torch.arange(64))
    assert sorted(PERM64.tolist()) == list(range(64))


@pytest.mark.parametrize("bits", quant.SUPPORTED_BITS)
def test_pack_roundtrip(bits):
    u = torch.randint(0, 2 ** bits, (24, 256), dtype=torch.uint8)
    p = pack(u, bits)
    assert torch.equal(unpack(p, bits, 256), u)
    expected = {4: 128, 5: 128 + 32, 6: 128 + 64, 8: 256}[bits]
    assert sum(t.shape[1] for t in p.values()) == expected


@pytest.mark.parametrize("bits", quant.SUPPORTED_BITS)
def test_emulated_gemv_matches_dequant(bits):
    w = torch.randn(40, 384) * 0.02
    q = QLinear.from_weight(w, bits)
    x = torch.randn(384)
    y_ref = q.dequant() @ x
    torch.testing.assert_close(quant.emulate_gemv(q, x), y_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bits", quant.SUPPORTED_BITS)
def test_mma_fragment_emulation(bits):
    w = torch.randn(32, 128) * 0.02
    q = QLinear.from_weight(w, bits)
    u = unpack(q.planes, bits, 128).to(torch.int64) - 2 ** (bits - 1)
    for kblk in range(2):
        for ks in range(4):
            for lane in (0, 5, 13, 31):
                g, c = lane >> 2, lane & 3
                regs = quant.emulate_mma_a_fragment(q, 16, kblk, ks, lane)
                k0 = kblk * 64 + 16 * ks
                want = [
                    (u[16 + g, k0 + 2 * c], u[16 + g, k0 + 2 * c + 1]),
                    (u[16 + g + 8, k0 + 2 * c], u[16 + g + 8, k0 + 2 * c + 1]),
                    (u[16 + g, k0 + 2 * c + 8], u[16 + g, k0 + 2 * c + 9]),
                    (u[16 + g + 8, k0 + 2 * c + 8], u[16 + g + 8, k0 + 2 * c + 9]),
                ]
                assert [tuple(int(a) for a in r) for r in regs] == [tuple(int(a) for a in r) for r in want]


def test_gaussian_mse_matches_theory():
    """After the Hadamard rotation weights are ~N(0, s^2); RTN MSE at b bits must be within the
    classic uniform-quantizer bound Delta^2/12 of the chosen step and fall ~4x per extra bit."""
    w = torch.randn(256, 1024)
    errs = {}
    for b in (4, 5, 6, 8):
        u, s = quantize_rtn(w, b)
        errs[b] = quant.rel_mse(w, dequant_rtn(u, s, b))
    assert 3.0 < errs[4] / errs[5] < 5.0
    assert 3.0 < errs[5] / errs[6] < 5.0
    assert 10.0 < errs[6] / errs[8] < 20.0
    assert errs[8] < 2e-4  # INT8 with MSE clipping: t^2 ~ 1e-4 on Gaussian groups of 128


def test_mse_clipping_beats_absmax():
    w = torch.randn(64, 1024)
    u1, s1 = quantize_rtn(w, 4, clip_grid=1)
    u2, s2 = quantize_rtn(w, 4, clip_grid=20)
    assert quant.rel_mse(w, dequant_rtn(u2, s2, 4)) < 0.9 * quant.rel_mse(w, dequant_rtn(u1, s1, 4))


def test_hadamard_gaussianizes_outliers():
    """Incoherence processing (QuIP# Lemma 3.1): a heavy-tailed matrix quantizes far better after RHT."""
    from cckernel.hadamard import random_signs, rotate_reader

    w = torch.randn(128, 1024) * 0.02
    w[:, 7] *= 60.0  # an outlier input channel, as seen in LLM activations/weights
    s = random_signs(1024, 1)
    wr = rotate_reader(w, s)
    e_plain = quant.rel_mse(w, dequant_rtn(*quantize_rtn(w, 4), 4))
    e_rot = quant.rel_mse(wr, dequant_rtn(*quantize_rtn(wr, 4), 4))
    assert e_rot < 0.5 * e_plain
