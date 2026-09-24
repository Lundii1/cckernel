"""Persistent prefix cache (DeepSeek-V4.1-Flash, arXiv 2609.19969, sec. 3.2.1).

The paper persists global KV for long-lived prefix reuse and snapshots the sliding-window state at
two points, the end of the prompt and the end of the output, which is what regeneration and
multi-turn sessions hit. Our hybrid model has the same split: the 8 attention layers carry an
append-only KV cache (stored in the engine's KV format, so FP4 snapshots are 3.6x smaller than
bf16), and the 24 Gated DeltaNet layers carry a fixed-size recurrent state (48 MiB, like the paper's
SWA state it cannot be truncated, only snapshotted).

A snapshot holds the committed tokens, the engine session state and optionally the logits of the
last token (so an exact hit needs no prefill at all). A request restores the longest snapshot whose
tokens are a prefix of its own and prefills only the rest.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch


def _key(tokens: list[int]) -> str:
    return hashlib.sha1(json.dumps(tokens).encode()).hexdigest()[:20]


class PrefixCache:
    def __init__(self, directory: str | Path, max_entries: int = 32):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.index_path = self.dir / "index.json"
        self.index: list[dict] = json.loads(self.index_path.read_text()) if self.index_path.exists() else []

    def _write_index(self):
        self.index_path.write_text(json.dumps(self.index))

    def lookup(self, ids: list[int], kv_format: str) -> dict | None:
        best = None
        for e in self.index:
            n = e["n"]
            if e["kv_format"] == kv_format and n <= len(ids) and (best is None or n > best["n"]) \
                    and e["tokens"] == ids[:n] and (n < len(ids) or e["has_logits"]):
                best = e
        return best

    def restore(self, eng, ids: list[int]) -> tuple[int, torch.Tensor | None]:
        """Load the longest usable snapshot into ``eng``. Returns (restored tokens, last logits or None)."""
        e = self.lookup(ids, eng.kv_format)
        if e is None:
            return 0, None
        snap = torch.load(self.dir / e["file"], map_location="cpu", weights_only=False)
        eng.load_session_state(snap["state"])
        e["used"] = time.time()
        self._write_index()
        lg = snap.get("logits")
        return e["n"], (lg.to(eng.dev) if lg is not None else None)

    def save(self, eng, tokens: list[int], logits: torch.Tensor | None = None):
        """Snapshot the engine, whose committed sequence must be exactly ``tokens``."""
        assert eng.len_host == len(tokens), (eng.len_host, len(tokens))
        k = _key(tokens) + f"_{eng.kv_format}"
        fname = f"{k}.pt"
        torch.save({"state": eng.session_state(), "logits": None if logits is None else logits.detach().float().cpu()},
                   self.dir / fname)
        self.index = [e for e in self.index if e["file"] != fname]
        self.index.append({"file": fname, "n": len(tokens), "tokens": list(tokens), "kv_format": eng.kv_format,
                           "has_logits": logits is not None, "used": time.time()})
        while len(self.index) > self.max_entries:  # LRU eviction
            old = min(self.index, key=lambda e: e["used"])
            self.index.remove(old)
            (self.dir / old["file"]).unlink(missing_ok=True)
        self._write_index()
