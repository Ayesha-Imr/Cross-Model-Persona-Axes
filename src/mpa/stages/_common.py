from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config, load_config
from ..paths import get_or_make_run_dir, make_run_dir


def base_parser(stage: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"mpa.stages.{stage}")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def resolve_run_dir(cfg: Config, run_dir: str | None, fresh: bool = False) -> Path:
    if run_dir is not None:
        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d
    if fresh:
        return make_run_dir(cfg)
    return get_or_make_run_dir(cfg)


def snapshot_config(cfg_path: str, run_dir: Path) -> None:
    dst = run_dir / "config.yaml"
    if not dst.exists():
        dst.write_text(Path(cfg_path).read_text())


def load_cfg_and_run_dir(args, fresh: bool = False) -> tuple[Config, Path]:
    cfg = load_config(args.config)
    run_dir = resolve_run_dir(cfg, args.run_dir, fresh=fresh)
    snapshot_config(args.config, run_dir)
    return cfg, run_dir
