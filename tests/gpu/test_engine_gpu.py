"""GPU engine (CUDA kernels + graphs) vs the CPU emulation engine on a small real-shaped model."""

import pytest
import torch

from cckernel import quant
from cckernel.config import TextConfig
from cckernel.engine import Engine
from cckernel.quant_io import write_cck
from cckernel.reference import random_weights
from cckernel.spec import accept

CFG = TextConfig.tiny(hidden_size=512, intermediate_size=1024, vocab_size=2048, num_hidden_layers=8,
                      layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2, num_attention_heads=4,
                      num_key_value_heads=1, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=4,
                      linear_key_head_dim=128, linear_value_head_dim=128, rope_theta=1e7)


@pytest.fixture(scope="module", params=[8, 6, 4])
def model_dir(request, tmp_path_factory):
    w = random_weights(CFG, seed=5, scale=0.05)
    out = tmp_path_factory.mktemp(f"cck{request.param}")
    write_cck(CFG, w.__getitem__, out, {n: request.param for n, _, _ in quant.matrix_list(CFG)}, seed=3)
    return out


def test_gpu_matches_emulation(model_dir):
    ids = torch.randint(0, CFG.vocab_size, (40,), generator=torch.Generator().manual_seed(0)).tolist()
    g = Engine(model_dir, device="cuda", max_len=256, attn_splits=8, prefill_chunk=16)
    c = Engine(model_dir, device="cpu", max_len=256, attn_splits=8, prefill_chunk=16)
    lg, lc = g.prefill(ids[:30]).cpu(), c.prefill(ids[:30])
    assert (lg - lc).abs().max() / lc.abs().max() < 2e-2
    for t in ids[30:]:
        a = g.step([t])[0].cpu()
        b = c.step([t])[0]
        g.commit(1)
        c.commit(1)
        assert (a - b).abs().max() / b.abs().max() < 3e-2


def test_gpu_speculation_exact(model_dir):
    prompt = list(range(20))
    eng = Engine(model_dir, device="cuda", max_len=256, attn_splits=8)
    t = int(eng.prefill(prompt).argmax())
    want = [t]
    for _ in range(15):
        t = int(eng.step([t])[0].argmax())
        eng.commit(1)
        want.append(t)
    eng2 = Engine(model_dir, device="cuda", max_len=256, attn_splits=8)
    t = int(eng2.prefill(prompt).argmax())
    got = [t]
    k = 0
    while len(got) < 16:
        k = k % 7 + 1
        draft = (want[len(got):len(got) + k] + [0] * k)[:k]
        if k % 3 == 0 and draft:
            draft[-1] = (draft[-1] + 1) % CFG.vocab_size  # inject a wrong draft token
        em = accept(eng2.step([t] + draft), draft)
        eng2.commit(len(em))
        got += em
        t = got[-1]
    assert got[:16] == want
