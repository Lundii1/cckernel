"""Optional parity check against Hugging Face transformers on the real checkpoint.

  CCK_HF_MODEL=/models/MiMo-V2.6-Distill-Qwen-9B CCK_MODEL=/models/mimo-cck-q8 pytest tests/gpu/test_hf_parity.py -s

HF runs in bf16 with device_map="auto" (CPU offload if needed). Checks next-token agreement and KL.
"""

import os

import pytest
import torch

from cckernel.engine import Engine

HF, CCK = os.environ.get("CCK_HF_MODEL"), os.environ.get("CCK_MODEL")
pytestmark = pytest.mark.skipif(not (HF and CCK), reason="set CCK_HF_MODEL and CCK_MODEL")

TEXT = ("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# The Gated DeltaNet recurrence updates a matrix-valued state")


def test_next_token_logits_vs_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF)
    ids = tok(TEXT)["input_ids"]
    eng = Engine(CCK, max_len=4096)
    ours = eng.score(ids).cpu()
    del eng
    torch.cuda.empty_cache()
    try:
        model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map="auto")
    except Exception:  # multimodal wrapper checkpoint
        from transformers import AutoModelForMultimodalLM

        model = AutoModelForMultimodalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map="auto")
    with torch.no_grad():
        ref = torch.log_softmax(model(torch.tensor([ids]).to(model.device)).logits[0].float(), -1).cpu()
    kl = (ref.exp() * (ref - ours)).sum(-1)
    agree = (ref.argmax(-1) == ours.argmax(-1)).float().mean()
    print(f"mean KL {kl.mean():.4f}  max KL {kl.max():.4f}  top-1 agreement {agree:.3f}")
    assert agree > 0.9 and kl.mean() < 0.05
