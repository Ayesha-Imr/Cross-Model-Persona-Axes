from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import FAMILY_COLORS, INK, LINE, MUTED, PALETTE, add_explainer, save_fig

EXPLAINER_RAW = (
    "Each panel shows how strongly each model's responses project onto a persona axis "
    "as we move through the prober's layers (Gemma-4).\n\n"
    "• Curves trace mean projection across prompts at each layer.\n"
    "• If lines stack on top of each other, it means the prober's per-layer geometry "
    "dominates over model-to-model differences.\n\n"
    "See the residual variant (04b) for the model-specific signal alone."
)

EXPLAINER_RESIDUAL = (
    "Same as the raw per-layer plot, but with the cross-model mean subtracted at every "
    "(axis, layer). What remains is the model-specific signal.\n\n"
    "• Lines above zero → the model is above peers at that layer for that axis.\n"
    "• Below zero → below peers.\n\n"
    "Look for layers where models fan out (high discriminability) — those are the "
    "layers carrying family-level signal. Flat-near-zero layers carry no information."
)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def per_layer_lines(
    df: pd.DataFrame,
    axes_order: list[str],
    models_order: list[str],
    family_of: dict[str, str],
    out_path,
    *,
    residual: bool = False,
):
    grouped = df.groupby(["model", "axis", "layer"])["score"].mean().reset_index()
    if residual:
        layer_means = grouped.groupby(["axis", "layer"])["score"].transform("mean")
        grouped = grouped.assign(score=grouped["score"] - layer_means)

    n = len(axes_order)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 2.9 * rows + 0.8), squeeze=False)
    color_map = _color_per_model(models_order, family_of)

    for idx, axis_name in enumerate(axes_order):
        ax = axes[idx // cols][idx % cols]
        sub = grouped[grouped["axis"] == axis_name]
        for m in models_order:
            mdf = sub[sub["model"] == m].sort_values("layer")
            if mdf.empty:
                continue
            c = color_map[m]
            ax.plot(mdf["layer"], mdf["score"], color=c, linewidth=1.7,
                    marker="o", markersize=3.5, alpha=0.9, label=m)
        ax.axhline(0, color=LINE, linewidth=1.0, zorder=0)
        ax.set_title(_pretty(axis_name), color=INK)
        ax.set_xlabel("Gemma layer", color=MUTED)
        ylabel = "Residual projection" if residual else "Projection score"
        ax.set_ylabel(ylabel, color=MUTED)
        ax.tick_params(axis="both", colors=MUTED)

    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)),
                   bbox_to_anchor=(0.5, -0.04), frameon=False, title="Model")

    title = ("Per-layer model-specific signal (residual after subtracting cross-model mean)"
             if residual else "Per-layer persona-axis projections")
    fig.suptitle(title, fontsize=14, color=INK, y=1.0, weight="bold")
    fig.tight_layout()
    add_explainer(fig, EXPLAINER_RESIDUAL if residual else EXPLAINER_RAW,
                  loc="right", width_frac=0.20)
    save_fig(fig, out_path)
    plt.close(fig)


def _color_per_model(models, family_of):
    by_family: dict[str, list[str]] = {}
    for m in models:
        by_family.setdefault(family_of.get(m, "other"), []).append(m)
    out: dict[str, str] = {}
    fallback = itertools.cycle(PALETTE)
    for fam, names in by_family.items():
        base = FAMILY_COLORS.get(fam)
        if base and len(names) == 1:
            out[names[0]] = base
        else:
            for n in names:
                out[n] = next(fallback)
    return out
