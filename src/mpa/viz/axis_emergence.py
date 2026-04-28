from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from ..checkpoint import read_jsonl
from .theme import FAMILY_COLORS, INK, LINE, MUTED, add_explainer, save_fig

EXPLAINER = (
    "Top: at each Gemma layer, we measure how strongly the prober separates models on "
    "this persona axis (between-model spread / within-prompt noise). Tall peaks = the "
    "axis is geometrically 'visible' at that layer.\n\n"
    "Bottom: real model responses scoring at the extremes — the highest-projecting and "
    "lowest-projecting response across all evaluation prompts. Model name colored by family.\n\n"
    "Together: the curve says where in Gemma the trait lives, and the boxes show what "
    "high/low actually look like in text."
)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def _truncate(text: str, n: int = 480) -> str:
    text = text.strip().replace("\n\n", "\n")
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def _discriminability(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer between-model spread / within-model spread. Returns (layers, signal)."""
    by_layer = sub.groupby("layer")
    layers = []
    signals = []
    for layer, g in by_layer:
        per_model = g.groupby("model")["score"]
        means = per_model.mean()
        within = per_model.std().mean()
        between = means.std()
        signals.append(between / max(within, 1e-6))
        layers.append(layer)
    order = np.argsort(layers)
    return np.array(layers)[order], np.array(signals)[order]


def axis_emergence(
    df_proj: pd.DataFrame,             # long projections (with layer)
    responses_dir: Path,               # run_dir/data/responses
    axes_order: list[str],
    family_of: dict[str, str],
    out_path,
):
    n = len(axes_order)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    # 2 rows per axis grid cell: top = curve, bottom = exemplars
    fig = plt.figure(figsize=(5.6 * cols, 4.4 * rows + 0.8))
    gs = GridSpec(rows * 2, cols, figure=fig, height_ratios=[1.0, 1.1] * rows,
                  hspace=0.55, wspace=0.28)

    # mean-over-layers per (model, prompt, axis) for picking exemplars
    aggr = df_proj.groupby(["model", "prompt_id", "axis"])["score"].mean().reset_index()

    # cache responses per model
    resp_cache: dict[str, list[dict]] = {}

    def _resp(model_name: str) -> list[dict]:
        if model_name not in resp_cache:
            p = responses_dir / f"{model_name}.jsonl"
            resp_cache[model_name] = read_jsonl(p) if p.exists() else []
        return resp_cache[model_name]

    def _find_response(model_name: str, prompt_id: str) -> tuple[str, str]:
        for r in _resp(model_name):
            if r.get("prompt_id") == prompt_id:
                return r.get("prompt", ""), r.get("response", "")
        return "", ""

    for idx, axis_name in enumerate(axes_order):
        r, c = idx // cols, idx % cols
        ax_top = fig.add_subplot(gs[2 * r, c])
        ax_bot = fig.add_subplot(gs[2 * r + 1, c])

        sub = df_proj[df_proj["axis"] == axis_name]
        layers, signal = _discriminability(sub)

        ax_top.fill_between(layers, 0, signal, color=FAMILY_COLORS.get("claude"), alpha=0.18)
        ax_top.plot(layers, signal, color=FAMILY_COLORS.get("claude"), linewidth=2.0,
                    marker="o", markersize=3.5)
        ax_top.axhline(0, color=LINE, linewidth=0.8)
        ax_top.set_title(_pretty(axis_name), color=INK)
        ax_top.set_xlabel("Gemma layer", color=MUTED)
        ax_top.set_ylabel("Discriminability", color=MUTED)
        ax_top.tick_params(axis="both", colors=MUTED)

        # exemplars: top + bottom by mean score on this axis
        sub_aggr = aggr[aggr["axis"] == axis_name]
        if sub_aggr.empty:
            ax_bot.axis("off")
            continue
        hi = sub_aggr.loc[sub_aggr["score"].idxmax()]
        lo = sub_aggr.loc[sub_aggr["score"].idxmin()]
        _draw_exemplars(ax_bot, axis_name, hi, lo, family_of, _find_response)

    fig.suptitle("Where each persona axis lives in Gemma — and what it looks like in text",
                 fontsize=14, color=INK, y=1.0, weight="bold")
    add_explainer(fig, EXPLAINER, loc="bottom", height_frac=0.13)
    save_fig(fig, out_path)
    plt.close(fig)


def _draw_exemplars(ax, axis_name, hi_row, lo_row, family_of, lookup):
    ax.axis("off")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for i, (label, row, y0, y1) in enumerate([
        ("HIGH", hi_row, 0.52, 0.98),
        ("LOW", lo_row, 0.02, 0.48),
    ]):
        prompt, resp = lookup(row["model"], row["prompt_id"])
        fam = family_of.get(row["model"], "other")
        color = FAMILY_COLORS.get(fam, "#888")
        ax.add_patch(plt.Rectangle((0.0, y0), 1.0, y1 - y0,
                                    facecolor="#FBF8F2", edgecolor=LINE, linewidth=0.6,
                                    transform=ax.transAxes))
        ax.text(0.02, y1 - 0.02, label, transform=ax.transAxes,
                fontsize=8.5, color=MUTED, weight="bold", va="top")
        ax.text(0.13, y1 - 0.02,
                f"{row['model']}  ", transform=ax.transAxes,
                fontsize=9, color=color, weight="bold", va="top")
        ax.text(0.55, y1 - 0.02, f"score = {row['score']:+.1f}",
                transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")
        body = _truncate(resp, n=380)
        prompt_short = _truncate(prompt, n=140)
        ax.text(0.02, y1 - 0.13, f"User: {prompt_short}", transform=ax.transAxes,
                fontsize=7.8, color=MUTED, va="top", style="italic", wrap=True)
        ax.text(0.02, y1 - 0.24, body, transform=ax.transAxes,
                fontsize=8.2, color=INK, va="top", wrap=True, linespacing=1.35)
