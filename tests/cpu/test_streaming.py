"""Row-chunked (memory-bounded) quantization is bit-identical to whole-matrix quantization."""

import torch

from cckernel import quant
from cckernel.config import TextConfig
from cckernel.hadamard import random_signs
from cckernel.quant_io import quantize_globals
from cckernel.reference import random_weights


def _same(a: quant.QLinear, b: quant.QLinear):
    assert a.bits == b.bits and a.N == b.N and a.K == b.K
    assert a.planes.keys() == b.planes.keys()
    for k in a.planes:
        assert torch.equal(a.planes[k], b.planes[k]), k
    assert torch.equal(a.scales, b.scales)


def test_chunked_from_weight_is_identical():
    w = torch.randn(1000, 256) * 0.02
    for bits in quant.SUPPORTED_BITS:
        whole = quant.QLinear.from_weight(w, bits, clip_grid=5, row_chunk=10 ** 9)
        chunked = quant.QLinear.from_weight(w, bits, clip_grid=5, row_chunk=97)
        _same(whole, chunked)
        t2 = quant.rel_mse(w, whole.dequant())
        assert abs(chunked.t2 - t2) < 1e-6 * max(t2, 1e-12) + 1e-12
        torch.testing.assert_close(chunked.dequant(100, 300), whole.dequant()[100:300])


def test_concat_equals_stacked_quantization():
    a, b = torch.randn(300, 128) * 0.02, torch.randn(200, 128) * 0.05
    for bits in (4, 6, 8):
        _same(quant.QLinear.concat([quant.QLinear.from_weight(a, bits, clip_grid=4),
                                    quant.QLinear.from_weight(b, bits, clip_grid=4)]),
              quant.QLinear.from_weight(torch.cat([a, b]), bits, clip_grid=4))


def test_chunked_globals_match_whole():
    cfg = TextConfig.tiny(vocab_size=1000)
    w = random_weights(cfg, seed=2)
    s = random_signs(cfg.hidden_size, 4)
    seen = []
    emb, lm = quantize_globals(cfg, lambda n, a, b: w[n][a:b], w["norm.weight"], s, 8, clip_grid=4, row_chunk=128,
                               on_chunk=lambda r0, r1, raw, ql: seen.append((r0, r1, ql.N)))
    g = quant.fold_globals(cfg, w.__getitem__, s)
    _same(lm, quant.QLinear.from_weight(g["lm_head"], 8, clip_grid=4))
    assert torch.equal(emb, g["embed"].to(torch.bfloat16))
    assert [r for r in seen][0] == (0, 128, 128) and seen[-1][1] == 1000


def test_checkpoint_get_rows(tmp_path):
    from safetensors.torch import save_file

    from cckernel.loader import Checkpoint

    t = torch.randn(50, 16).to(torch.bfloat16)
    save_file({"model.language_model.embed_tokens.weight": t}, str(tmp_path / "model.safetensors"))
    ck = Checkpoint(tmp_path)
    assert torch.equal(ck.get_rows("embed_tokens.weight", 7, 19, dtype=None), t[7:19])
    assert ck.shape("embed_tokens.weight") == [50, 16]
