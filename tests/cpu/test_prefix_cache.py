"""Session snapshots and the persistent prefix cache: restoring a prefix and prefilling the rest gives
the same logits as processing everything, for bf16 and fp4 KV caches, including after decode steps
(pending deferred-commit state)."""

import pytest
import torch

from cckernel import quant
from cckernel.config import TextConfig
from cckernel.engine import Engine
from cckernel.prefix_cache import PrefixCache
from cckernel.quant_io import write_cck
from cckernel.reference import random_weights

CFG = TextConfig.tiny(hidden_size=256, intermediate_size=512, vocab_size=600, num_attention_heads=4,
                      num_key_value_heads=1, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=4,
                      linear_key_head_dim=128, linear_value_head_dim=128, rope_theta=1e7)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(0)
    w = random_weights(CFG, seed=5, scale=0.06)
    out = tmp_path_factory.mktemp("cckp")
    write_cck(CFG, w.__getitem__, out, {n: 8 for n, _, _ in quant.matrix_list(CFG)}, seed=3)
    return out


def _eng(d, fmt):
    return Engine(d, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4, prefill_chunk=16, kv_format=fmt)


@pytest.mark.parametrize("fmt", ["bf16", "fp4"])
def test_session_roundtrip_after_decode(model_dir, fmt, tmp_path):
    ids = torch.randint(0, CFG.vocab_size, (30,), generator=torch.Generator().manual_seed(1)).tolist()
    a = _eng(model_dir, fmt)
    a.prefill(ids[:10])
    for t in ids[10:14]:
        a.step([t])
        a.commit(1)  # leaves a pending (deferred) GDN commit in the state
    a.save_session(tmp_path / "s.pt")
    lg_a = a.step(ids[14:17]).clone()
    b = _eng(model_dir, fmt)
    b.load_session(tmp_path / "s.pt")
    assert torch.equal(b.step(ids[14:17]), lg_a)


@pytest.mark.parametrize("fmt", ["bf16", "fp4"])
def test_prefix_cache_longest_hit(model_dir, fmt, tmp_path):
    ids = torch.randint(0, CFG.vocab_size, (40,), generator=torch.Generator().manual_seed(2)).tolist()
    pc = PrefixCache(tmp_path / "pc")
    a = _eng(model_dir, fmt)
    lg20 = a.prefill(ids[:20]).clone()
    pc.save(a, ids[:20], lg20)
    a.prefill(ids[20:30])
    pc.save(a, ids[:30])
    b = _eng(model_dir, fmt)
    n, lg = pc.restore(b, ids[:20])  # exact hit with logits: nothing to prefill
    assert n == 20 and torch.equal(lg, lg20)
    n, lg = pc.restore(b, ids)  # longest prefix (30 tokens, no logits needed since tokens remain)
    assert n == 30 and lg is None
    full = _eng(model_dir, fmt).prefill(ids)
    rest = b.prefill(ids[30:])
    assert float((rest - full).abs().max() / full.abs().max()) < 1e-2
    assert pc.restore(b, [ids[0] + 1] + ids[1:])[0] == 0  # no shared prefix -> miss
    other = _eng(model_dir, "fp8")
    assert pc.restore(other, ids)[0] == 0  # snapshots are per KV format
