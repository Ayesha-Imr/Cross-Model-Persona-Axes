from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import FAMILY_COLORS, INK, LINE, MUTED, PALETTE, save_fig


def per_layer_lines(
    df: pd.DataFrame,
    axes_order: list[str],
    models_order: list[str],
    family_of: dict[str, str],
    out_path,
):
    n = len(axes_order)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 2.8 * rows + 0.8), squeeze=False)

    grouped = df.groupby(["model", "axis", "layer"])["score"].mean().reset_index()
    # Distinct color per model: family color, with a palette fallback for diversity within family.
    color_map = _color_per_model(models_order, family_of)

    for idx, axis_name in enumerate(axes_order):
        ax = axes[idx // cols][idx % cols]
        sub = grouped[grouped["axis"] == axis_name]
        for m in models_order:
            mdf = sub[sub["model"] == m].sort_values("layer")
            if mdf.empty:
                continue
            c = color_map[m]
            ax.plot(mdf["layer"], mdf["score"], color=c, linewidth=2.0,
                    marker="o", markersize=4.5, markeredgecolor=c,
                    markerfacecolor=c, alpha=0.95, label=m)
        ax.axhline(0, color=LINE, linewidth=1.0, zorder=0)
        ax.set_title(_pretty(axis_name), color=INK)
        ax.set_xlabel("Gemma layer", color=MUTED)
        ax.set_ylabel("Projection score", color=MUTED)
        ax.tick_params(axis="both", colors=MUTED)

    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(5, len(labels)), bbox_to_anchor=(0.5, -0.04),
                   frameon=False, title="Model")
    fig.suptitle("Per-layer persona-axis projections", fontsize=14, color=INK,
                 y=1.0, weight="bold")
    fig.tight_layout()
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
        if not base or len(names) > 1:
            for n in names:
                out[n] = next(fallback)
        else:
            out[names[0]] = base
    return out


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()
