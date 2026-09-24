"""Bit allocator optimality and the quantize CLI on a fake HF checkpoint."""

import itertools
import json
import subprocess
import sys
from pathlib import Path

import torch

from cckernel.alloc import allocate, predicted_loss
from cckernel.config import TextConfig
from cckernel.engine import Engine
from cckernel.quant import bits_per_weight
from cckernel.reference import random_weights

ROOT = Path(__file__).resolve().parents[2]


def test_knapsack_matches_bruteforce():
    torch.manual_seed(0)
    items = []
    for i in range(5):
        numel = int(torch.randint(1, 5, (1,))) * 2 ** 20
        base = float(torch.rand(1)) * 1e-2
        t2 = {4: base * 16, 5: base * 4, 6: base, 8: base / 16}
        items.append((f"m{i}", numel, t2, float(torch.rand(1)) + 0.5))
    budget = sum(n for _, n, _, _ in items) * bits_per_weight(5.6) / 8
    got = allocate(items, budget, unit_bytes=2 ** 14)
    best, best_bits = float("inf"), None
    for combo in itertools.product((4, 5, 6, 8), repeat=len(items)):
        cost = sum(n * bits_per_weight(b) / 8 for (_, n, _, _), b in zip(items, combo))
        if cost > budget:
            continue
        bits = {name: b for (name, *_), b in zip(items, combo)}
        loss = predicted_loss(items, bits)
        if loss < best:
            best, best_bits = loss, bits
    assert abs(predicted_loss(items, got) - best) <= 1e-12 + 1e-6 * best, (got, best_bits)


def _fake_hf(path: Path, cfg: TextConfig):
    from safetensors.torch import save_file

    w = random_weights(cfg, seed=11, scale=0.06)
    sd = {("lm_head.weight" if k == "lm_head.weight" else f"model.language_model.{k}"): v.to(torch.bfloat16)
          for k, v in w.items()}
    sd["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(8, 8, dtype=torch.bfloat16)  # must be skipped
    save_file(sd, str(path / "model.safetensors"))
    tc = cfg.to_dict()
    tc["rope_parameters"] = {"rope_theta": tc.pop("rope_theta"), "partial_rotary_factor": tc.pop("partial_rotary_factor"),
                             "rope_type": "default"}
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": tc, "tie_word_embeddings": False}))
    (path / "tokenizer_config.json").write_text("{}")


def test_quantize_cli_balanced(tmp_path):
    cfg = TextConfig.tiny(hidden_size=256, intermediate_size=512, vocab_size=600, num_attention_heads=4,
                          num_key_value_heads=1, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=4,
                          linear_key_head_dim=128, linear_value_head_dim=128, rope_theta=1e7)
    src, out = tmp_path / "hf", tmp_path / "cck"
    src.mkdir()
    _fake_hf(src, cfg)
    r = subprocess.run([sys.executable, str(ROOT / "tools/quantize.py"), "--model", str(src), "--out", str(out),
                        "--preset", "balanced", "--device", "cpu", "--clip-grid", "6"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    man = json.loads((out / "cck_manifest.json").read_text())
    assert man["bits"]["lm_head"] >= 8
    assert set(man["bits"].values()) <= {4, 5, 6, 8}
    assert (out / "tokenizer_config.json").exists()
    eng = Engine(out, device="cpu", max_len=64, attn_splits=4)
    lg = eng.prefill(list(range(10)))
    assert torch.isfinite(lg).all()


def test_calibrate_alpha_cli(tmp_path):
    cfg = TextConfig.tiny(hidden_size=256, intermediate_size=512, vocab_size=600, num_attention_heads=4,
                          num_key_value_heads=1, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=4,
                          linear_key_head_dim=128, linear_value_head_dim=128, rope_theta=1e7)
    src, out = tmp_path / "hf", tmp_path / "cck"
    src.mkdir()
    _fake_hf(src, cfg)
    r = subprocess.run([sys.executable, str(ROOT / "tools/quantize.py"), "--model", str(src), "--out", str(out),
                        "--device", "cpu", "--clip-grid", "4"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, str(ROOT / "tools/calibrate_alpha.py"), "--model", str(out), "--out",
                        str(tmp_path / "a.json"), "--device", "cpu", "--num-seqs", "1", "--seq-len", "12"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    alphas = json.loads((tmp_path / "a.json").read_text())
    assert len(alphas) == 4 * 4 + 1 and all(v > 0 for v in alphas.values())
