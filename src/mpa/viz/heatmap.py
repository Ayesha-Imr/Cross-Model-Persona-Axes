from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import DIVERGING, INK, MUTED, save_fig


def model_axis_heatmap(df: pd.DataFrame, axes_order: list[str], models_order: list[str], out_path):
    pivot = df.pivot(index="model", columns="axis", values="score").reindex(
        index=models_order, columns=axes_order,
    )
    z = (pivot - pivot.mean(axis=0)) / pivot.std(axis=0).replace(0, 1)

    fig, ax = plt.subplots(figsize=(0.85 * len(axes_order) + 2.4,
                                     0.5 * len(models_order) + 1.6))
    vmax = float(np.nanmax(np.abs(z.values))) or 1.0
    im = ax.imshow(z.values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(axes_order)))
    ax.set_xticklabels([_pretty(a) for a in axes_order], rotation=30, ha="right", color=INK)
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels(models_order, color=INK)
    ax.set_title("Persona-axis projections by model (z-scored across models)",
                 color=INK, pad=10, weight="bold")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color=INK if abs(v) < 0.7 * vmax else "white", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED)
    cb.set_label("z-score", fontsize=9.5, color=MUTED)
    save_fig(fig, out_path)
    plt.close(fig)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()
