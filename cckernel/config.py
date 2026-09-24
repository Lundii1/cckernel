"""Text-decoder configuration for Qwen3.5 / MiMo-V2.6-Distill-Qwen-9B.

Only the fields the kernels need are kept. ``from_hf`` reads the ``text_config`` block of a
Hugging Face ``config.json`` (the checkpoint is a ``Qwen3_5ForConditionalGeneration`` wrapper).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TextConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    layer_types: list[str] = field(default_factory=lambda: (["linear_attention"] * 3 + ["full_attention"]) * 8)
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    # gated full attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    # gated deltanet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    eos_token_id: int | list[int] | None = 248044
    tie_word_embeddings: bool = False

    # ---- derived sizes -------------------------------------------------------------------
    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def gdn_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def gdn_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def gdn_conv_dim(self) -> int:
        return 2 * self.gdn_key_dim + self.gdn_value_dim

    @property
    def gdn_in_dim(self) -> int:
        """Rows of the fused GDN input projection [qkv | z | b | a]."""
        return self.gdn_conv_dim + self.gdn_value_dim + 2 * self.linear_num_value_heads

    @property
    def attn_q_dim(self) -> int:
        """q_proj rows: per head [query | gate]."""
        return self.num_attention_heads * self.head_dim * 2

    @property
    def attn_kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attn_in_dim(self) -> int:
        """Rows of the fused attention input projection [q(+gate) | k | v]."""
        return self.attn_q_dim + 2 * self.attn_kv_dim

    @property
    def attn_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def gdn_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    # ---- io ------------------------------------------------------------------------------
    @classmethod
    def from_hf(cls, path: str | Path) -> "TextConfig":
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        raw = json.loads(path.read_text())
        tc = raw.get("text_config", raw)
        rope = tc.get("rope_parameters") or {}
        return cls(
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            layer_types=list(tc["layer_types"]),
            vocab_size=tc["vocab_size"],
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            partial_rotary_factor=rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)),
            rope_theta=rope.get("rope_theta", tc.get("rope_theta", 10000.0)),
            linear_num_key_heads=tc["linear_num_key_heads"],
            linear_num_value_heads=tc["linear_num_value_heads"],
            linear_key_head_dim=tc["linear_key_head_dim"],
            linear_value_head_dim=tc["linear_value_head_dim"],
            linear_conv_kernel_dim=tc["linear_conv_kernel_dim"],
            eos_token_id=tc.get("eos_token_id"),
            tie_word_embeddings=raw.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TextConfig":
        return cls(**d)

    @classmethod
    def tiny(cls, **overrides) -> "TextConfig":
        """A small config with the same structure, for CPU tests."""
        base = dict(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            vocab_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            partial_rotary_factor=0.25,
            rope_theta=10000.0,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            eos_token_id=None,
        )
        base.update(overrides)
        return cls(**base)
