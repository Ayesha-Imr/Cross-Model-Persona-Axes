from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from ..checkpoint import read_jsonl
from .theme import FAMILY_COLORS, INK, LINE, MUTED, PAPER, add_explainer, save_fig

EXPLAINER = (
    "For each model family we identify its 'signature axis' — the persona axis where "
    "this family differs most from all others (largest mean |Cohen's d| against other "
    "families).\n\n"
    "We then show the most extreme real response within the family on that axis (the "
    "response furthest from the cross-model mean in the family's direction).\n\n"
    "This answers: 'What does this family do distinctively, and what does it actually "
    "look like?' — one card per family."
)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def _truncate(text: str, n: int) -> str:
    text = (text or "").strip().replace("\r", "").replace("\n\n\n", "\n\n")
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return float((a.mean() - b.mean()) / np.sqrt(max(s2, 1e-12)))


def _signature_axis(df_per_prompt: pd.DataFrame, family: str, axes_order: list[str]) -> tuple[str, float]:
    """Pick the axis with the largest mean |d| of `family` vs each other family."""
    others = [f for f in df_per_prompt["family"].unique() if f != family]
    best_axis, best_score, best_signed = axes_order[0], -1.0, 0.0
    for axis in axes_order:
        sub = df_per_prompt[df_per_prompt["axis"] == axis]
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
            best_signed = float(np.mean(ds))  # signed direction
    return best_axis, best_signed


def family_signatures(
    df_full: pd.DataFrame,                   # long: model, prompt_id, axis, score (layer-aggregated)
    by_model_axis: pd.DataFrame,             # mean per (model, axis)
    responses_dir: Path,
    axes_order: list[str],
    family_of: dict[str, str],
    out_path,
):
    df = df_full.assign(family=df_full["model"].map(family_of))
    families = [f for f in sorted(df["family"].dropna().unique()) if f]
    if not families:
        return

    cross_model_mean = by_model_axis.groupby("axis")["score"].mean().to_dict()

    n = len(families)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(7.0 * cols, 3.6 * rows + 1.4))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.10,
                  left=0.04, right=0.98, top=0.93, bottom=0.04)

    resp_cache: dict[str, list[dict]] = {}

    def _resp(model_name: str) -> list[dict]:
        if model_name not in resp_cache:
            p = responses_dir / f"{model_name}.jsonl"
            resp_cache[model_name] = read_jsonl(p) if p.exists() else []
        return resp_cache[model_name]

    def _lookup(model_name: str, prompt_id: str) -> tuple[str, str]:
        for r in _resp(model_name):
            if r.get("prompt_id") == prompt_id:
                return r.get("prompt", ""), r.get("response", "")
        return "", ""

    for i, family in enumerate(families):
        axis, signed_d = _signature_axis(df, family, axes_order)
        # find the most extreme in-family response on that axis (in the family's direction)
        sub = df[(df["axis"] == axis) & (df["family"] == family)]
        if sub.empty:
            continue
        center = cross_model_mean.get(axis, 0.0)
        if signed_d >= 0:
            best = sub.loc[(sub["score"] - center).idxmax()]
        else:
            best = sub.loc[(sub["score"] - center).idxmin()]
        prompt_text, response_text = _lookup(best["model"], best["prompt_id"])
        ax = fig.add_subplot(gs[i // cols, i % cols])
        _draw_card(
            ax,
            family=family,
            axis=axis,
            signed_d=signed_d,
            model=best["model"],
            score=float(best["score"]),
            cross_mean=center,
            prompt_text=prompt_text,
            response_text=response_text,
        )

    fig.suptitle("Family signatures — strongest distinguishing axis per family",
                 fontsize=14, color=INK, weight="bold", y=0.98)
    add_explainer(fig, EXPLAINER, loc="bottom", height_frac=0.10)
    save_fig(fig, out_path)
    plt.close(fig)


def _draw_card(
    ax,
    *,
    family: str,
    axis: str,
    signed_d: float,
    model: str,
    score: float,
    cross_mean: float,
    prompt_text: str,
    response_text: str,
):
    color = FAMILY_COLORS.get(family, MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                facecolor=PAPER, edgecolor=LINE, linewidth=0.6))
    # left family-color stripe
    ax.add_patch(plt.Rectangle((0, 0), 0.012, 1, transform=ax.transAxes,
                                facecolor=color, edgecolor="none"))

    direction = "↑ above peers" if signed_d >= 0 else "↓ below peers"
    ax.text(0.04, 0.95, family.title(), transform=ax.transAxes,
            fontsize=14, color=color, weight="bold", va="top")
    ax.text(0.04, 0.86, f"signature axis  ·  {_pretty(axis)}  ({direction}, mean d = {signed_d:+.2f})",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="top")

    ax.text(0.04, 0.76, f"{model}", transform=ax.transAxes,
            fontsize=10.5, color=INK, weight="bold", va="top")
    ax.text(0.04, 0.69, f"projection score  {score:+.2f}   (vs cross-model mean {cross_mean:+.2f})",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")

    prompt_line = "User: " + _truncate(prompt_text, 130) if prompt_text else "—"
    ax.text(0.04, 0.59, prompt_line, transform=ax.transAxes,
            fontsize=8.8, color=MUTED, style="italic", va="top", wrap=True)

    ax.add_patch(plt.Rectangle((0.04, 0.49), 0.92, 0.0025, transform=ax.transAxes,
                                facecolor=LINE, edgecolor="none"))

    ax.text(0.04, 0.46, _truncate(response_text, 540) if response_text else "(no response)",
            transform=ax.transAxes, fontsize=9, color=INK, va="top",
            wrap=True, linespacing=1.4)
