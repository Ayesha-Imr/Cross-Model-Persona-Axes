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
    out_path = run_dir / "artifacts" / "projections.parquet"
    if not pv_path.exists():
        raise SystemExit("persona_vectors.pt not found; run extract_vectors first.")
    pv = torch.load(pv_path, map_location="cpu", weights_only=False)
    directions = pv["directions"]  # (A, L, H)
    axes_names = pv["axes"]

    if args.dry_run:
        n_resp = sum(len(read_jsonl(p)) for p in (run_dir / "data" / "responses").glob("*.jsonl"))
        log.info("[dry] would project %d responses through prober", n_resp)
        return

    prober = make_prober(cfg.prober)
    try:
        rows = []
        for resp_file in sorted((run_dir / "data" / "responses").glob("*.jsonl")):
            model_name = resp_file.stem
            recs = read_jsonl(resp_file)
            if not recs:
                continue
            log.info("[proj] %s: %d responses", model_name, len(recs))
            bs = cfg.prober.batch_size
            for i in tqdm(range(0, len(recs), bs), desc=model_name):
                chunk = recs[i:i + bs]
                texts = [r["response"] for r in chunk]
                act = prober.encode(texts)  # (b, L, H)
                # scores: (b, A, L) = einsum("blh, alh -> bal")
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
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        log.info("Wrote %s rows=%d", out_path, len(df))
    finally:
        prober.close()


if __name__ == "__main__":
    main()
