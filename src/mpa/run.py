"""End-to-end orchestrator: runs all stages in order."""

from __future__ import annotations

import argparse
import sys

from .logging import setup_logging
from .paths import make_run_dir
from .stages import extract_vectors, generate_responses, project, sample_prompts, visualize
from .stages._common import snapshot_config
from .config import load_config


def main():
    p = argparse.ArgumentParser(prog="mpa.run")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--from", dest="from_stage", default=None,
                   choices=["sample", "extract", "generate", "project", "visualize"])
    p.add_argument("--only", dest="only_stage", default=None,
                   choices=["sample", "extract", "generate", "project", "visualize"])
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.run_dir:
        from pathlib import Path
        run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    else:
        from .paths import latest_run_dir
        run_dir = latest_run_dir(cfg) or make_run_dir(cfg)
    snapshot_config(args.config, run_dir)
    log = setup_logging(run_dir, "run")
    log.info("Run dir: %s", run_dir)

    stages = [
        ("sample", sample_prompts),
        ("extract", extract_vectors),
        ("generate", generate_responses),
        ("project", project),
        ("visualize", visualize),
    ]
    started = args.from_stage is None
    base_argv = ["--config", args.config, "--run-dir", str(run_dir)]
    if args.dry_run:
        base_argv.append("--dry-run")
    if args.force:
        base_argv.append("--force")

    for name, mod in stages:
        if args.only_stage is not None and name != args.only_stage:
            continue
        if args.only_stage is None and not started:
            if name == args.from_stage:
                started = True
            else:
                continue
        log.info("=== stage: %s ===", name)
        sys.argv = [f"mpa.stages.{name}", *base_argv]
        mod.main()


if __name__ == "__main__":
    main()
