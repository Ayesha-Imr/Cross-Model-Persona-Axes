from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm

from ..checkpoint import read_jsonl
from ..logging import setup_logging
from ..prober import make_prober
from ._common import base_parser, load_cfg_and_run_dir


def main():
    args = base_parser("project").parse_args()
    cfg, run_dir = load_cfg_and_run_dir(args)
    log = setup_logging(run_dir, "project")

    pv_path = run_dir / "artifacts" / "persona_vectors.pt"
    out_dir = run_dir / "artifacts" / "projections_per_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = run_dir / "artifacts" / "projections.parquet"
    if not pv_path.exists():
        raise SystemExit("persona_vectors.pt not found; run extract_vectors first.")
    pv = torch.load(pv_path, map_location="cpu", weights_only=False)
    directions = pv["directions"]
    axes_names = pv["axes"]

    response_files = sorted((run_dir / "data" / "responses").glob("*.jsonl"))
    pending = []
    for rf in response_files:
        per_model_path = out_dir / f"{rf.stem}.parquet"
        if per_model_path.exists() and not args.force:
            log.info("[proj] %s already projected; skipping.", rf.stem)
            continue
        pending.append(rf)

    if args.dry_run:
        n_resp = sum(len(read_jsonl(rf)) for rf in pending)
        log.info("[dry] would project %d responses across %d pending models",
                 n_resp, len(pending))
        return

    if pending:
        prober = make_prober(cfg.prober)
        try:
            for rf in pending:
                model_name = rf.stem
                recs = read_jsonl(rf)
                if not recs:
                    continue
                log.info("[proj] %s: %d responses", model_name, len(recs))
                bs = cfg.prober.batch_size
                rows = []
                for i in tqdm(range(0, len(recs), bs), desc=model_name):
                    chunk = recs[i:i + bs]
                    texts = [r["response"] for r in chunk]
                    act = prober.encode(texts)
                    scores = torch.einsum("blh,alh->bal", act, directions)
                    for bi, r in enumerate(chunk):
                        for ai, axis_name in enumerate(axes_names):
                            for li in range(scores.shape[2]):
                                rows.append({
                                    "model": model_name,
                                    "prompt_id": r["prompt_id"],
                                    "axis": axis_name,
                                    "layer": li,
                                    "score": float(scores[bi, ai, li]),
                                })
                pd.DataFrame(rows).to_parquet(out_dir / f"{model_name}.parquet", index=False)
        finally:
            prober.close()
    else:
        log.info("[proj] all models already projected.")

    parts = [pd.read_parquet(p) for p in sorted(out_dir.glob("*.parquet"))]
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(merged_path, index=False)
        log.info("Merged -> %s", merged_path)


if __name__ == "__main__":
    main()
