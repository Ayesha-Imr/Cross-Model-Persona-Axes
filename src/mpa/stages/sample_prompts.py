from __future__ import annotations

from ..checkpoint import append_jsonl
from ..logging import setup_logging
from ..prompts import sample_prompts as _sample_prompts
from ._common import base_parser, load_cfg_and_run_dir


def main():
    args = base_parser("sample_prompts").parse_args()
    cfg, run_dir = load_cfg_and_run_dir(args, fresh=True)
    log = setup_logging(run_dir, "sample_prompts")
    out = run_dir / "data" / "prompts.jsonl"
    if out.exists() and not args.force:
        log.info("prompts.jsonl exists; skipping (use --force).")
        return
    if out.exists():
        out.unlink()
    prompts = _sample_prompts(cfg.prompts, cfg.seed)
    for p in prompts:
        append_jsonl(out, p)
    log.info("Wrote %d prompts -> %s", len(prompts), out)


if __name__ == "__main__":
    main()
