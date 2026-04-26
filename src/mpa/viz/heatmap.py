from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import DIVERGING, INK, save_fig


def model_axis_heatmap(df: pd.DataFrame, axes_order: list[str], models_order: list[str], out_path):
    """df: long projections (model, axis, score) — already mean-aggregated over layers + prompts."""
    pivot = df.pivot(index="model", columns="axis", values="score").reindex(
        index=models_order, columns=axes_order,
    )
    z = (pivot - pivot.mean(axis=0)) / pivot.std(axis=0).replace(0, 1)

    fig, ax = plt.subplots(figsize=(0.7 * len(axes_order) + 2.2, 0.45 * len(models_order) + 1.5))
    vmax = float(np.nanmax(np.abs(z.values))) or 1.0
    im = ax.imshow(z.values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(axes_order)))
    ax.set_xticklabels([a.replace("_", " ") for a in axes_order], rotation=30, ha="right")
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels(models_order)
    ax.set_title("Persona-axis projections (z-scored across models)", color=INK, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("z-score", fontsize=9, color=INK)
    save_fig(fig, out_path)
    plt.close(fig)
