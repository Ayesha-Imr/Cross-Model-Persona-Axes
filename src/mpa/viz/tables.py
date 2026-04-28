from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..checkpoint import read_jsonl


def _resp_loader(responses_dir: Path):
    cache: dict[str, list[dict]] = {}

    def _resp(model_name: str) -> list[dict]:
        if model_name not in cache:
            p = responses_dir / f"{model_name}.jsonl"
            cache[model_name] = read_jsonl(p) if p.exists() else []
        return cache[model_name]

    def lookup(model_name: str, prompt_id: str | None) -> tuple[str, str]:
        if prompt_id is None:
            return "", ""
        for r in _resp(model_name):
            if r.get("prompt_id") == prompt_id:
                return r.get("prompt", ""), r.get("response", "")
        return "", ""

    return lookup


def _representative_pid(sub: pd.DataFrame) -> tuple[str | None, float | None]:
    if sub.empty:
        return None, None
    median = sub["score"].median()
    idx = (sub["score"] - median).abs().idxmin()
    row = sub.loc[idx]
    return row["prompt_id"], float(row["score"])


def axis_emergence_table(
    df_proj: pd.DataFrame,
    responses_dir: Path,
    axes_order: list[str],
    family_of: dict[str, str],
    out_path: Path,
):
    """Per (axis, layer) discriminability + per-axis HIGH/LOW exemplar with full text."""
    lookup = _resp_loader(responses_dir)
    aggr = df_proj.groupby(["model", "prompt_id", "axis"])["score"].mean().reset_index()

    rows = []
    for axis_name in axes_order:
        sub = df_proj[df_proj["axis"] == axis_name]
        per_layer = sub.groupby("layer")
        layer_disc = {}
        for layer, g in per_layer:
            per_model = g.groupby("model")["score"]
            within = per_model.std().mean()
            between = per_model.mean().std()
            layer_disc[int(layer)] = float(between / max(within, 1e-6))

        sub_aggr = aggr[aggr["axis"] == axis_name]
        if sub_aggr.empty:
            continue
        hi = sub_aggr.loc[sub_aggr["score"].idxmax()]
        lo = sub_aggr.loc[sub_aggr["score"].idxmin()]

        for label, r in [("HIGH", hi), ("LOW", lo)]:
            prompt, response = lookup(r["model"], r["prompt_id"])
            rows.append({
                "axis": axis_name,
                "extreme": label,
                "model": r["model"],
                "family": family_of.get(r["model"], ""),
                "prompt_id": r["prompt_id"],
                "score": float(r["score"]),
                "prompt": prompt,
                "response": response,
                "layer_discriminability": ";".join(
                    f"{k}:{v:.4f}" for k, v in sorted(layer_disc.items())
                ),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def trait_extremes_table(
    df_full: pd.DataFrame,
    by_model_axis: pd.DataFrame,
    responses_dir: Path,
    axes_order: list[str],
    family_of: dict[str, str],
    out_path: Path,
):
    """Per axis: highest- and lowest-mean model with representative response (full text)."""
    lookup = _resp_loader(responses_dir)
    rows = []
    for axis_name in axes_order:
        sub_means = by_model_axis[by_model_axis["axis"] == axis_name].sort_values("score")
        if sub_means.empty:
            continue
        sub_full = df_full[df_full["axis"] == axis_name]
        for side, mrow in [("HIGHEST", sub_means.iloc[-1]), ("LOWEST", sub_means.iloc[0])]:
            model = mrow["model"]
            pid, sample_score = _representative_pid(sub_full[sub_full["model"] == model])
            prompt, response = lookup(model, pid)
            rows.append({
                "axis": axis_name,
                "side": side,
                "model": model,
                "family": family_of.get(model, ""),
                "mean_projection": float(mrow["score"]),
                "representative_prompt_id": pid,
                "representative_sample_score": sample_score,
                "prompt": prompt,
                "response": response,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return float((a.mean() - b.mean()) / np.sqrt(max(s2, 1e-12)))


def family_signatures_table(
    df_full: pd.DataFrame,
    by_model_axis: pd.DataFrame,
    responses_dir: Path,
    axes_order: list[str],
    family_of: dict[str, str],
    out_path: Path,
):
    """Per family: signature axis (largest mean |d| vs other families) + most extreme in-family response."""
    lookup = _resp_loader(responses_dir)
    df = df_full.assign(family=df_full["model"].map(family_of))
    families = [f for f in sorted(df["family"].dropna().unique()) if f]
    cross_model_mean = by_model_axis.groupby("axis")["score"].mean().to_dict()

    rows = []
    for family in families:
        others = [f for f in families if f != family]
        best_axis, best_score, best_signed = axes_order[0], -1.0, 0.0
        for axis in axes_order:
            sub = df[df["axis"] == axis]
            ours = sub[sub["family"] == family]["score"].values
            ds = []
            for other in others:
                theirs = sub[sub["family"] == other]["score"].values
                d = _cohens_d(ours, theirs)
                if not np.isnan(d):
                    ds.append(d)
            if not ds:
                continue
            magnitude = float(np.mean(np.abs(ds)))
            if magnitude > best_score:
                best_score = magnitude
                best_axis = axis
                best_signed = float(np.mean(ds))

        sub = df[(df["axis"] == best_axis) & (df["family"] == family)]
        if sub.empty:
            continue
        center = cross_model_mean.get(best_axis, 0.0)
        if best_signed >= 0:
            best = sub.loc[(sub["score"] - center).idxmax()]
        else:
            best = sub.loc[(sub["score"] - center).idxmin()]
        prompt, response = lookup(best["model"], best["prompt_id"])
        rows.append({
            "family": family,
            "signature_axis": best_axis,
            "mean_abs_cohens_d": best_score,
            "signed_mean_d": best_signed,
            "direction": "above_peers" if best_signed >= 0 else "below_peers",
            "model": best["model"],
            "score": float(best["score"]),
            "cross_model_mean": center,
            "prompt_id": best["prompt_id"],
            "prompt": prompt,
            "response": response,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def axis_ladder_table(
    df_full: pd.DataFrame,
    by_model_axis: pd.DataFrame,
    responses_dir: Path,
    axes_order: list[str],
    family_of: dict[str, str],
    out_path: Path,
):
    """Per (axis, model): mean score, rank, representative response (full text)."""
    lookup = _resp_loader(responses_dir)
    rows = []
    for axis_name in axes_order:
        ranking = by_model_axis[by_model_axis["axis"] == axis_name].sort_values(
            "score", ascending=False,
        ).reset_index(drop=True)
        sub_full = df_full[df_full["axis"] == axis_name]
        for rank, mrow in ranking.iterrows():
            model = mrow["model"]
            pid, sample_score = _representative_pid(sub_full[sub_full["model"] == model])
            prompt, response = lookup(model, pid)
            rows.append({
                "axis": axis_name,
                "rank": int(rank) + 1,
                "model": model,
                "family": family_of.get(model, ""),
                "mean_score": float(mrow["score"]),
                "representative_prompt_id": pid,
                "representative_sample_score": sample_score,
                "prompt": prompt,
                "response": response,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
