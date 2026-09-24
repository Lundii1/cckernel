"""Folding norms + residual Hadamard keeps the network function (tiny random model, fp64)."""

import torch

from cckernel.config import TextConfig
from cckernel.folded import FoldedCache, FoldedModel, fold_all
from cckernel.hadamard import fwht, random_signs, rotate_reader, rotate_writer
from cckernel.reference import RefCache, RefModel, random_weights


def test_fwht_orthonormal():
    x = torch.randn(5, 64, dtype=torch.float64)
    y = fwht(x)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(fwht(y), x)


def test_reader_writer_identity():
    s = random_signs(64, 3).double()
    A = torch.randn(10, 64, dtype=torch.float64)
    B = torch.randn(64, 12, dtype=torch.float64)
    torch.testing.assert_close(rotate_reader(A, s) @ rotate_writer(B, s), A @ B)


def _check(rotate: bool, gdn_algo: str = "recurrent"):
    cfg = TextConfig.tiny()
    w = {k: v.double() for k, v in random_weights(cfg, seed=1).items()}
    ids = torch.randint(0, cfg.vocab_size, (11,))
    ref = RefModel(cfg, w, dtype=torch.float64)
    c1 = RefCache(cfg)
    l_ref = torch.cat([ref.forward(ids[:7], c1), ref.forward(ids[7:], c1)])
    signs = random_signs(cfg.hidden_size, 7).double() if rotate else None
    fw = fold_all(cfg, w, signs, dtype=torch.float64)
    fm = FoldedModel(cfg, fw, dtype=torch.float64, gdn_algo=gdn_algo)
    c2 = FoldedCache()
    l_fold = torch.cat([fm.forward(ids[:7], c2), fm.forward(ids[7:], c2)])
    torch.testing.assert_close(l_fold.double(), l_ref, rtol=1e-6, atol=1e-6)


def test_fold_norms_only():
    _check(rotate=False)


def test_fold_with_hadamard():
    _check(rotate=True)


def test_fold_with_chunked_gdn():
    _check(rotate=True, gdn_algo="chunked")
