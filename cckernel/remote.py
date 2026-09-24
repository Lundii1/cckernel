"""Stream safetensors tensors from the Hugging Face Hub with HTTP range requests.

The checkpoint's four shards interleave layers, so layer-by-layer processing would otherwise need all
18.8 GB on disk. Instead we read each shard's safetensors header (8-byte little-endian length + JSON
with dtype/shape/data_offsets) and then fetch exactly the bytes of a tensor, or of a contiguous block
of its rows, with ranged GETs (split into parallel parts). Nothing is written to disk.

Interface mirrors ``cckernel.loader.Checkpoint``: ``keys()``, ``get(name)``, ``get_rows(name, r0, r1)``,
``shape(name)``, with the same canonical (text-decoder) names.
"""

from __future__ import annotations

import http.client
import json
import os
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch

from .loader import canonical_name

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
          "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
          "BOOL": torch.bool}


class RangeUnsupported(RuntimeError):
    pass


class RemoteCheckpoint:
    def __init__(self, repo: str, revision: str = "main", token: str | None = None,
                 endpoint: str = "https://huggingface.co", part_bytes: int = 16 << 20, workers: int = 8,
                 retries: int = 6, timeout: float = 120.0):
        self.repo, self.revision = repo, revision
        self.endpoint = endpoint.rstrip("/")
        self.base = f"{self.endpoint}/{repo}/resolve/{revision}/"
        self.token = token if token is not None else os.environ.get("HF_TOKEN")
        self.part_bytes, self.workers, self.retries, self.timeout = part_bytes, workers, retries, timeout
        self.bytes_fetched = 0
        self._headers: dict[str, tuple[int, dict]] = {}
        try:
            weight_map = json.loads(self.fetch("model.safetensors.index.json"))["weight_map"]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            weight_map = {k: "model.safetensors" for k in self._header("model.safetensors")[1]}
        self._map = {}
        for full, fname in weight_map.items():
            c = canonical_name(full)
            if c is not None:
                self._map[c] = (full, fname)

    # ------------------------------------------------------------------------------------------------ http
    def _request(self, fname: str, rng: tuple[int, int] | None = None) -> bytes:
        headers = {"User-Agent": "cckernel/0.1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if rng is not None:
            headers["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
        err = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.base + fname, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if rng is not None and r.status != 206:
                        raise RangeUnsupported(f"server ignored Range for {fname} (HTTP {r.status})")
                    data = r.read()
                if rng is not None and len(data) != rng[1] - rng[0]:
                    raise IOError(f"short read {len(data)} != {rng[1] - rng[0]}")
                self.bytes_fetched += len(data)
                return data
            except RangeUnsupported:
                raise
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404):
                    raise
                err = e
            except (http.client.HTTPException, OSError) as e:  # IncompleteRead, resets, timeouts, URLError
                err = e
            time.sleep(min(30.0, 2.0 ** attempt))
        raise IOError(f"failed to fetch {fname} {rng}: {err}")

    def fetch(self, fname: str) -> bytes:
        """A whole (small) file: config, index, tokenizer..."""
        return self._request(fname)

    def _range(self, fname: str, start: int, end: int) -> bytearray:
        n = end - start
        if n <= self.part_bytes:
            return bytearray(self._request(fname, (start, end)))
        parts = [(a, min(end, a + self.part_bytes)) for a in range(start, end, self.part_bytes)]
        out = bytearray(n)
        with ThreadPoolExecutor(self.workers) as ex:
            for (a, b), data in zip(parts, ex.map(lambda p: self._request(fname, p), parts)):
                out[a - start: b - start] = data
        return out

    def _header(self, fname: str) -> tuple[int, dict]:
        if fname not in self._headers:
            (n,) = struct.unpack("<Q", self._request(fname, (0, 8)))
            hdr = json.loads(self._request(fname, (8, 8 + n)))
            hdr.pop("__metadata__", None)
            self._headers[fname] = (8 + n, hdr)
        return self._headers[fname]

    # ------------------------------------------------------------------------------------------------ api
    def keys(self):
        return self._map.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._map

    def _meta(self, name: str):
        full, fname = self._map[name]
        data_start, hdr = self._header(fname)
        m = hdr[full]
        return fname, data_start, DTYPES[m["dtype"]], list(m["shape"]), m["data_offsets"]

    def shape(self, name: str) -> list[int]:
        return self._meta(name)[3]

    def get(self, name: str, dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
        fname, ds, dt, shape, (a, b) = self._meta(name)
        t = torch.frombuffer(self._range(fname, ds + a, ds + b), dtype=dt).reshape(shape)
        return t if dtype is None else t.to(dtype)

    def get_rows(self, name: str, r0: int, r1: int, dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
        fname, ds, dt, shape, (a, _) = self._meta(name)
        row = dt.itemsize
        for s in shape[1:]:
            row *= s
        t = torch.frombuffer(self._range(fname, ds + a + r0 * row, ds + a + r1 * row), dtype=dt)
        t = t.reshape([r1 - r0] + shape[1:])
        return t if dtype is None else t.to(dtype)

    def list_files(self) -> list[str]:
        """Repository file names via the Hub API (falls back to the common small files)."""
        try:
            req = urllib.request.Request(f"{self.endpoint}/api/models/{self.repo}/revision/{self.revision}",
                                         headers={"Authorization": f"Bearer {self.token}"} if self.token else {})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return [s["rfilename"] for s in json.loads(r.read())["siblings"]]
        except Exception:  # noqa: BLE001
            return ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                    "chat_template.jinja", "special_tokens_map.json", "vocab.json", "merges.txt"]
