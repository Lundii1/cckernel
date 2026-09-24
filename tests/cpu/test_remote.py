"""RemoteCheckpoint against a local HTTP server that implements Range requests (and a redirect)."""

import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from cckernel.loader import Checkpoint
from cckernel.remote import RangeUnsupported, RemoteCheckpoint


class RangeHandler(SimpleHTTPRequestHandler):
    ranges = True

    def log_message(self, *a):
        pass

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/redirect/"):  # emulate the Hub -> CDN redirect
            self.send_response(302)
            self.send_header("Location", self.path[len("/redirect"):])
            self.end_headers()
            return
        parts = self.path.split("/resolve/main/")
        if len(parts) != 2:
            self.send_error(404)
            return
        f = Path(self.directory) / parts[1]
        if not f.exists():
            self.send_error(404)
            return
        data = f.read_bytes()
        rng = self.headers.get("Range")
        if rng and self.ranges:
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), int(b)
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def server(tmp_path):
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    t1 = {"model.language_model.embed_tokens.weight": torch.randn(300, 64, generator=g).to(torch.bfloat16),
          "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(32, 64, generator=g).to(torch.bfloat16)}
    t2 = {"lm_head.weight": torch.randn(300, 64, generator=g).to(torch.bfloat16),
          "model.language_model.norm.weight": torch.randn(64, generator=g),
          "model.visual.blocks.0.attn.qkv.weight": torch.zeros(4, 4)}
    save_file(t1, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file(t2, str(tmp_path / "model-00002-of-00002.safetensors"))
    wm = {k: "model-00001-of-00002.safetensors" for k in t1} | {k: "model-00002-of-00002.safetensors" for k in t2}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    (tmp_path / "config.json").write_text("{}")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(tmp_path)))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()
    RangeHandler.ranges = True


def test_remote_matches_local(server):
    url, path = server
    rc = RemoteCheckpoint("org/model", endpoint=url, part_bytes=1000, workers=4)
    lc = Checkpoint(path)
    assert set(rc.keys()) == set(lc.keys())
    assert "visual" not in " ".join(rc.keys())
    for k in rc.keys():
        assert torch.equal(rc.get(k, dtype=None), lc.get(k, dtype=None)), k
    assert torch.equal(rc.get_rows("embed_tokens.weight", 37, 211, dtype=None), lc.get("embed_tokens.weight", dtype=None)[37:211])
    assert rc.shape("lm_head.weight") == [300, 64]
    assert json.loads(rc.fetch("config.json")) == {}


def test_remote_follows_redirect(server):
    url, _ = server
    rc = RemoteCheckpoint("org/model", endpoint=url + "/redirect")
    assert rc.get("norm.weight").shape == (64,)


def test_remote_detects_missing_range_support(server):
    url, _ = server
    RangeHandler.ranges = False
    with pytest.raises(RangeUnsupported):
        RemoteCheckpoint("org/model", endpoint=url).get("norm.weight")
