"""End-to-end engine test on CPU through the torch emulation of the CUDA ops.

Exercises the real engine code paths: quantized checkpoint writer, chunked prefill (ring buffer,
chunked Gated DeltaNet, KV cache), kernel-semantics decode, deferred GDN commit, speculative
M-token verification + rollback, and multi-turn prefill after decode.
"""

import pytest
import torch

from cckernel import quant
from cckernel.config import TextConfig
from cckernel.engine import Engine
from cckernel.folded import FoldedCache, FoldedModel
from cckernel.quant_io import write_cck
from cckernel.reference import random_weights
from cckernel.spec import NGramDrafter, accept

CFG = TextConfig.tiny(hidden_size=256, intermediate_size=512, vocab_size=600, num_attention_heads=4,
                      num_key_value_heads=1, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=4,
                      linear_key_head_dim=128, linear_value_head_dim=128, rope_theta=1e7)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(0)
    w = random_weights(CFG, seed=5, scale=0.06)
    out = tmp_path_factory.mktemp("cck")
    bits = {n: 8 for n, _, _ in quant.matrix_list(CFG)}
    write_cck(CFG, w.__getitem__, out, bits, seed=3)
    return out


def _oracle(eng: Engine) -> FoldedModel:
    t = {}
    for name, q in eng.lin.items():
        t[name] = quant.QLinear(q.bits, q.N, q.K, {("q8" if q.bits == 8 else "lo"): q.lo, **({"hi": q.hi} if q.bits in (5, 6) else {})},
                                q.scales).dequant()
    t.update(eng.small)
    t["embed"] = eng.embed.float()
    return FoldedModel(CFG, t, gdn_algo="recurrent")


def _close(a, b, tol=3e-2):
    """bf16-dataflow engine vs fp32 oracle: compare magnitudes, not argmax (random logits tie)."""
    a, b = a.float(), b.float()
    err = (a - b).abs().max() / b.abs().max()
    assert err < tol, f"relative max error {err:.3e}"


def test_prefill_and_decode_match_oracle(model_dir):
    eng = Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4, prefill_chunk=16)
    ref = _oracle(eng)
    ids = torch.randint(0, CFG.vocab_size, (26,)).tolist()
    prompt, rest = ids[:21], ids[21:]
    lg = eng.prefill(prompt)
    cache = FoldedCache()
    ref_lg = ref.forward(torch.tensor(prompt), cache)
    _close(lg, ref_lg[-1])
    for t in rest:  # teacher-forced decode steps
        lg = eng.step([t])[0]
        eng.commit(1)
        _close(lg, ref.forward(torch.tensor([t]), cache)[-1])


def test_speculative_verify_equals_greedy(model_dir):
    torch.manual_seed(1)
    prompt = torch.randint(0, CFG.vocab_size, (12,)).tolist()

    def greedy(n):
        eng = Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4)
        tok = int(eng.prefill(prompt).argmax())
        out = [tok]
        for _ in range(n - 1):
            lg = eng.step([tok])[0]
            eng.commit(1)
            tok = int(lg.argmax())
            out.append(tok)
        return out

    want = greedy(14)
    eng = Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4)
    tok = int(eng.prefill(prompt).argmax())
    got = [tok]
    rng = torch.Generator().manual_seed(7)
    while len(got) < 14:
        # drafts: a mix of correct continuations (from the known answer) and random wrong tokens
        k = int(torch.randint(1, 5, (1,), generator=rng))
        draft = [want[len(got) + j] if len(got) + j < len(want) and torch.rand(1, generator=rng) < 0.7
                 else int(torch.randint(0, CFG.vocab_size, (1,), generator=rng)) for j in range(k)]
        lg = eng.step([tok] + draft)
        emitted = accept(lg, draft)
        eng.commit(len(emitted))
        got += emitted
        tok = got[-1]
    assert got[:14] == want


def test_multiturn_prefill_after_decode(model_dir):
    torch.manual_seed(2)
    ids = torch.randint(0, CFG.vocab_size, (30,)).tolist()
    a = Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4)
    a.prefill(ids[:10])
    for t in ids[10:14]:
        a.step([t])
        a.commit(1)
    lg_a = a.prefill(ids[14:])  # second turn after decode steps (pending commit must be applied)
    b = Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4, prefill_chunk=7)
    lg_b = b.prefill(ids)
    _close(lg_a, lg_b, tol=1e-2)


def test_ngram_drafter_in_loop(model_dir):
    """The drafter never changes the output (exactness), only the number of steps."""
    prompt = [5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7]
    eng = Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4)
    tok = int(eng.prefill(prompt).argmax())
    d = NGramDrafter()
    d.reset(prompt + [tok])
    out = [tok]
    while len(out) < 10:
        draft = d.propose()[:7]
        lg = eng.step([tok] + draft)
        em = accept(lg, draft)
        eng.commit(len(em))
        d.update_stats(len(draft), len(em) - 1)
        d.extend(em)
        out += em
        tok = out[-1]
    ref = Engine(model_dir, device="cpu", cpu_backend="emu", max_len=96, attn_splits=4)
    t = int(ref.prefill(prompt).argmax())
    want = [t]
    while len(want) < 10:
        t = int(ref.step([t])[0].argmax())
        ref.commit(1)
        want.append(t)
    assert out[:10] == want
