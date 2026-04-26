from __future__ import annotations

import datetime as dt
from pathlib import Path

from .config import Config


def latest_run_dir(cfg: Config) -> Path | None:
    root = Path(cfg.output_root)
    if not root.exists():
        return None
    h = cfg.short_hash()
    candidates = sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith(f"_{h}")])
    return candidates[-1] if candidates else None


def make_run_dir(cfg: Config) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.output_root) / f"{ts}_{cfg.short_hash()}"
    for sub in ("data", "data/responses", "data/contrastive", "data/raw_api_calls",
                "artifacts", "figures", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def get_or_make_run_dir(cfg: Config) -> Path:
    return latest_run_dir(cfg) or make_run_dir(cfg)
