from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(run_dir: Path, stage: str, level: int = logging.INFO) -> logging.Logger:
    log_file = run_dir / "logs" / f"{stage}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    handlers = [
        RichHandler(rich_tracebacks=True, show_time=False, show_path=False),
        logging.FileHandler(log_file),
    ]
    handlers[1].setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    for h in handlers:
        root.addHandler(h)
    return logging.getLogger(stage)
