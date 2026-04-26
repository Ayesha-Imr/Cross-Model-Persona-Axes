from __future__ import annotations

import time

from tqdm import tqdm

from ..checkpoint import append_jsonl, load_done_keys, read_jsonl
from ..config import Config, ModelEntry
from ..cost import CostTracker
from ..logging import setup_logging
from ..providers import make_generator
from ._common import base_parser, load_cfg_and_run_dir


def _key(rec: dict) -> str:
    return f"{rec['model']}|{rec['prompt_id']}"


def _process_model(model: ModelEntry, prompts: list[dict], cfg: Config, run_dir, log, costs: CostTracker):
    out_path = run_dir / "data" / "responses" / f"{model.name}.jsonl"
    audit_path = run_dir / "data" / "raw_api_calls" / f"{model.provider}.jsonl"
    done = load_done_keys(out_path, _key)
    pending = [p for p in prompts if f"{model.name}|{p['id']}" not in done]
    if not pending:
        log.info("[%s] all %d prompts already done.", model.name, len(prompts))
        return

    log.info("[%s] generating %d responses...", model.name, len(pending))
    gen = make_generator(model)
    try:
        for p in tqdm(pending, desc=model.name):
            t0 = time.time()
            r = gen.generate(p["text"])
            latency = time.time() - t0
            rec = {
                "model": model.name,
                "provider": model.provider,
                "model_id": model.model_id,
                "prompt_id": p["id"],
                "prompt": p["text"],
                "response": r.text,
                "in_tok": r.in_tok,
                "out_tok": r.out_tok,
                "latency_s": latency,
                "gen": model.gen.model_dump(),
            }
            append_jsonl(out_path, rec)
            append_jsonl(audit_path, {**rec, "ts": time.time()})
            if model.provider != "hf_local":
                costs.add(model.provider, r.in_tok, r.out_tok, cfg)
                costs.check_cap()
    finally:
        gen.close()
    log.info("[%s] done. cost=%s", model.name, costs.summary())


def main():
    args = base_parser("generate_responses").parse_args()
    cfg, run_dir = load_cfg_and_run_dir(args)
    log = setup_logging(run_dir, "generate_responses")
    prompts = read_jsonl(run_dir / "data" / "prompts.jsonl")
    if not prompts:
        raise SystemExit("No prompts found. Run sample_prompts first.")

    if args.dry_run:
        for m in cfg.enabled_models():
            log.info("[dry] %s: %d prompts to generate", m.name, len(prompts))
        return

    costs = CostTracker(cap_usd=cfg.cost_cap_usd, log_path=run_dir / "data" / "raw_api_calls" / "_costs.jsonl")
    # Local first (sequential, GPU-bound), then APIs.
    locals_, apis = [], []
    for m in cfg.enabled_models():
        (locals_ if m.provider == "hf_local" else apis).append(m)
    for m in locals_ + apis:
        _process_model(m, prompts, cfg, run_dir, log, costs)


if __name__ == "__main__":
    main()
