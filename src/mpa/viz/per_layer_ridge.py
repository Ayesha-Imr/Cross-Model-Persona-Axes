from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import FAMILY_COLORS, INK, LINE, save_fig


def per_layer_lines(
    df: pd.DataFrame,  # long: model, axis, layer, score
    axes_order: list[str],
    models_order: list[str],
    family_of: dict[str, str],
    out_path,
):
    """Small-multiples: one panel per axis; lines = per-model mean projection across layers."""
    n = len(axes_order)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.6 * rows + 0.6), squeeze=False)

    grouped = df.groupby(["model", "axis", "layer"])["score"].mean().reset_index()
    for idx, axis_name in enumerate(axes_order):
        ax = axes[idx // cols][idx % cols]
        sub = grouped[grouped["axis"] == axis_name]
        for m in models_order:
            mdf = sub[sub["model"] == m].sort_values("layer")
            if mdf.empty:
                continue
            c = FAMILY_COLORS.get(family_of.get(m, "other"), "#888")
            ax.plot(mdf["layer"], mdf["score"], color=c, linewidth=1.2, alpha=0.9, label=m)
        ax.axhline(0, color=LINE, linewidth=0.8)
        ax.set_title(axis_name.replace("_", " "), color=INK)
        ax.set_xlabel("layer")
        ax.set_ylabel("projection")

    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
                   bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Per-layer projections by model", fontsize=12, color=INK, y=1.0)
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
