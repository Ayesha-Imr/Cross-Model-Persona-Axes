from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable


def load_done_keys(path: Path, key_fn: Callable[[dict], str]) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(key_fn(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return keys


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def filter_pending(items: Iterable[dict], done: set[str], key_fn: Callable[[dict], str]) -> list[dict]:
    return [x for x in items if key_fn(x) not in done]
