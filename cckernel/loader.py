"""Streaming safetensors loader for the HF checkpoint (text decoder only).

Names were verified against ``model.safetensors.index.json`` of MiMo-V2.6-Distill-Qwen-9B:
``model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight`` etc. and ``lm_head.weight``.
The vision tower (``model.visual.*``) is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

TEXT_PREFIXES = ("model.language_model.", "model.")


def canonical_name(name: str) -> str | None:
    """Map a checkpoint key to a short text-decoder name, or None if it is not part of it."""
    if name.startswith("model.visual.") or name.startswith("visual."):
        return None
    if name.startswith("mtp."):
        return name
    if name == "lm_head.weight":
        return name
    for p in TEXT_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return None


class Checkpoint:
    """Lazy access to the text-decoder tensors of a sharded safetensors checkpoint."""

    def __init__(self, path: str | Path):
        from safetensors import safe_open  # local import: optional dependency at import time

        self.path = Path(path)
        idx = self.path / "model.safetensors.index.json"
        if idx.exists():
            weight_map = json.loads(idx.read_text())["weight_map"]
        else:
            files = sorted(self.path.glob("*.safetensors"))
            weight_map = {}
            for f in files:
                with safe_open(str(f), framework="pt") as h:
                    for k in h.keys():
                        weight_map[k] = f.name
        self._map: dict[str, tuple[str, str]] = {}
        for full, fname in weight_map.items():
            c = canonical_name(full)
            if c is not None:
                self._map[c] = (full, fname)
        self._handles: dict[str, object] = {}
        self._safe_open = safe_open

    def keys(self):
        return self._map.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._map

    def get(self, name: str, dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
        full, fname = self._map[name]
        h = self._handles.get(fname)
        if h is None:
            h = self._safe_open(str(self.path / fname), framework="pt")
            self._handles[fname] = h
        t = h.get_tensor(full)
        return t if dtype is None else t.to(dtype)

    def layer(self, i: int, sub: str, dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
        return self.get(f"layers.{i}.{sub}", dtype)
