"""Goblin-axis analysis: direction cosine similarities + per-model scores.

Standalone script. Loads persona_vectors.pt and projections.parquet from a run
dir, computes pairwise cosine similarity between the creature_whimsy direction
and all other persona directions, and prints per-model creature_whimsy scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    p = argparse.ArgumentParser(prog="mpa.stages.goblin_analysis")
    p.add_argument("--run-dir", required=True, type=str)
    p.add_argument("--target-axis", default="creature_whimsy")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    pv = torch.load(run_dir / "artifacts" / "persona_vectors.pt",
                     map_location="cpu", weights_only=False)
    directions = pv["directions"]  # (A, L, H)
    axes = pv["axes"]
    target = args.target_axis

    if target not in axes:
        raise SystemExit(f"Axis '{target}' not found in persona_vectors.pt. Available: {axes}")

    ti = axes.index(target)

    # --- Direction cosine similarity (mean over layers) ---
    print(f"\n{'='*60}")
    print(f"Direction cosine similarity: {target} vs all other axes")
    print(f"(averaged over {directions.shape[1]} layers)")
    print(f"{'='*60}\n")

    target_dir = directions[ti]  # (L, H)
    rows = []
    for ai, name in enumerate(axes):
        if ai == ti:
            continue
        # per-layer cosine, then mean
        cos = torch.nn.functional.cosine_similarity(target_dir, directions[ai], dim=-1)
        rows.append({"axis": name, "mean_cos": float(cos.mean()),
                      "std_cos": float(cos.std()), "max_cos": float(cos.max())})
    df_cos = pd.DataFrame(rows).sort_values("mean_cos", ascending=False)
    for _, r in df_cos.iterrows():
        print(f"  {r['axis']:<25s}  mean={r['mean_cos']:+.4f}  std={r['std_cos']:.4f}  max={r['max_cos']:+.4f}")

    # --- Per-model creature_whimsy scores ---
    proj_path = run_dir / "artifacts" / "projections.parquet"
    if not proj_path.exists():
        print(f"\nNo projections.parquet found; skipping per-model scores.")
        return

    df = pd.read_parquet(proj_path)
    sub = df[df["axis"] == target]
    if sub.empty:
        print(f"\nNo projections for axis '{target}'; rerun project stage with updated vectors.")
        return

    by_model = sub.groupby("model")["score"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)

    print(f"\n{'='*60}")
    print(f"Per-model {target} projection (mean over layers + prompts)")
    print(f"{'='*60}\n")
    for model, r in by_model.iterrows():
        bar = "█" * int(max(0, r["mean"]) * 10) if r["mean"] > 0 else ""
        print(f"  {model:<35s}  {r['mean']:+.4f}  (±{r['std']:.4f}, n={int(r['count'])})  {bar}")

    # --- Per-layer discriminability for target axis ---
    layer_stats = sub.groupby("layer").apply(
        lambda g: g.groupby("model")["score"].mean().std(), include_groups=False,
    ).reset_index()
    layer_stats.columns = ["layer", "between_model_std"]
    peak = layer_stats.loc[layer_stats["between_model_std"].idxmax()]
    print(f"\n  Peak discriminability at layer {int(peak['layer'])} "
          f"(between-model std={peak['between_model_std']:.4f})")

    # --- Top-scoring individual responses ---
    resp_dir = run_dir / "data" / "responses"
    if resp_dir.exists():
        from ..checkpoint import read_jsonl
        print(f"\n{'='*60}")
        print(f"Highest-scoring individual responses on {target}")
        print(f"{'='*60}")
        prompt_scores = sub.groupby(["model", "prompt_id"])["score"].mean().reset_index()
        top = prompt_scores.nlargest(5, "score")
        for _, row in top.iterrows():
            rfile = resp_dir / f"{row['model']}.jsonl"
            if not rfile.exists():
                continue
            recs = [r for r in read_jsonl(rfile) if r["prompt_id"] == row["prompt_id"]]
            if not recs:
                continue
            r = recs[0]
            print(f"\n  [{row['model']}] score={row['score']:+.4f}")
            print(f"  prompt: {r['prompt'][:120]}...")
            print(f"  response: {r['response'][:200]}...")


if __name__ == "__main__":
    main()
