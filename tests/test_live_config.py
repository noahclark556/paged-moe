# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Live settings (live_config.py) and the /config route on the real mlx-lm
handler.

A live change has to be all or nothing: a key that needs a restart rejects
the whole request, so the caller restarts instead of running half the new
settings. The prune head bakes PRUNE in at load, so it blocks live changes.
Both live keys change logits, so a change drops the KV cached under the old
values and a request it lands in the middle of stores none.

Run:
    python tests/test_live_config.py
"""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx_lm.server as _mlx_server  # noqa: E402

from pagedmoe import config, live_config, server  # noqa: E402


def check_apply():
    config.PRUNE, config.WAIT_ABOVE = 0.0, 0.0
    r = live_config.apply({"PAGED_MOE_PRUNE": "0.5", "PAGED_MOE_WAIT_ABOVE": "0.3"})
    assert r["rejected"] == {}, r
    assert config.PRUNE == 0.5 and config.WAIT_ABOVE == 0.3
    assert r["applied"] == {"PAGED_MOE_PRUNE": 0.5, "PAGED_MOE_WAIT_ABOVE": 0.3}

    r = live_config.apply({"PAGED_MOE_PRUNE": "0.2", "PAGED_MOE_FLOW": "1"})
    assert "PAGED_MOE_FLOW" in r["rejected"], r
    assert config.PRUNE == 0.5, "a rejected request must not apply anything"

    for bad in ("1.5", "-0.1", "abc"):
        r = live_config.apply({"PAGED_MOE_PRUNE": bad})
        assert r["rejected"], bad
        assert config.PRUNE == 0.5

    r = live_config.apply({"PAGED_MOE_PRUNE": ""})
    assert r["rejected"] == {} and config.PRUNE == 0.0

    r = live_config.apply({"PAGED_MOE_PRUNE": "0.0"})
    assert r["applied"] == {}, "no change reports nothing applied"

    config.SIDECAR, config.SIDECAR_HEAD_PRUNE = True, True
    try:
        r = live_config.apply({"PAGED_MOE_PRUNE": "0.4"})
        assert r["rejected"] == {} and config.PRUNE == 0.4, "PRUNE stays live with the head on"
        r = live_config.apply({"PAGED_MOE_PRUNE": "0.0"})
        r = live_config.apply({"PAGED_MOE_FIDELITY_TOOL_SCALE": "0"})
        assert r["rejected"] == {} and config.FIDELITY_TOOL_SCALE == 0.0, "tool scale stays live with the head on"
    finally:
        config.SIDECAR, config.SIDECAR_HEAD_PRUNE = False, False
    print("ok apply")


def check_epoch():
    """KV made under one setting is never served or stored under another."""
    from types import SimpleNamespace

    from mlx_lm.models.cache import LRUPromptCache

    config.PRUNE = 0.0
    live_config.begin_request()
    assert not live_config.begin_request(), "no change, nothing to drop"
    assert not live_config.kv_stale()

    live_config.apply({"PAGED_MOE_PRUNE": "0.0"})
    assert not live_config.kv_stale(), "a no-op change must not drop caches"

    live_config.apply({"PAGED_MOE_PRUNE": "0.4"})
    assert live_config.kv_stale(), "change mid-request: its KV mixes both"
    toks = list(range(80))
    kv = [SimpleNamespace(offset=len(toks))]
    pin = server._Pin()
    assert not pin.put("m", toks, kv), "stale request stored its KV"
    assert live_config.begin_request(), "next request must drop old KV"
    assert not live_config.kv_stale()
    assert pin.put("m", toks, kv)

    store = LRUPromptCache()
    store.insert_cache("m", toks, [SimpleNamespace(nbytes=10, is_trimmable=lambda: False)])
    server._stable_pin.put("m", toks, kv)
    server._drop_settings_kv(store)
    assert len(store) == 0 and store.nbytes == 0
    assert server._stable_pin.get("m") is None
    config.PRUNE = 0.0
    live_config.begin_request()
    print("ok epoch")


class _Args:
    allowed_origins = ["*"]


class _Gen:
    cli_args = _Args()


def check_route():
    server._install_config_route()
    server._install_config_route()  # idempotent

    class Handler(_mlx_server.APIHandler):
        def __init__(self, *a, **kw):
            super().__init__(_Gen(), *a, system_fingerprint="t", **kw)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"

    def post(payload):
        req = urllib.request.Request(
            base + "/config",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    try:
        config.PRUNE = 0.0
        status, body = post({"env": {"PAGED_MOE_PRUNE": "0.7"}})
        assert status == 200 and config.PRUNE == 0.7, (status, body)

        status, body = post({"env": {"PAGED_MOE_CACHE_GB": "30"}})
        assert status == 409 and "PAGED_MOE_CACHE_GB" in body["rejected"], (status, body)
        assert config.PRUNE == 0.7

        status, _ = post({"nope": 1})
        assert status == 400

        with urllib.request.urlopen(base + "/config", timeout=5) as resp:
            got = json.loads(resp.read())
        assert got["values"]["PAGED_MOE_PRUNE"] == 0.7, got
        assert "PAGED_MOE_PRUNE" in got["live"]

        try:
            urllib.request.urlopen(base + "/nope", timeout=5)
            raise AssertionError("unknown GET path must 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()
        config.PRUNE = 0.0
    print("ok route")


if __name__ == "__main__":
    check_apply()
    check_epoch()
    check_route()
    print("all ok")
