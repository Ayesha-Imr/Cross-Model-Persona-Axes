"""Lightweight unit tests (config + checkpoint) plus a marker for the full smoke run."""

from __future__ import annotations

import json
from pathlib import Path

from mpa.checkpoint import append_jsonl, filter_pending, load_done_keys
from mpa.config import load_config


def test_load_default_config():
    cfg = load_config("config.yaml")
    assert len(cfg.axes) >= 2
    assert any(m.enabled for m in cfg.models)


def test_load_smoke_config():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.prompts.n == 5
    assert len(cfg.axes) == 2


def test_checkpoint_roundtrip(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    for i in range(3):
        append_jsonl(p, {"k": str(i)})
    done = load_done_keys(p, key_fn=lambda r: r["k"])
    assert done == {"0", "1", "2"}
    pending = filter_pending(
        [{"k": "1"}, {"k": "3"}], done, key_fn=lambda r: r["k"]
    )
    assert pending == [{"k": "3"}]


def test_short_hash_stable():
    a = load_config("config.yaml").short_hash()
    b = load_config("config.yaml").short_hash()
    assert a == b and len(a) == 8
