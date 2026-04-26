from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import FAMILY_COLORS, INK, LINE, save_fig


def per_axis_bars(
    df_full: pd.DataFrame,  # long: model, prompt_id, axis, score (already aggregated over layers)
    axes_order: list[str],
    models_order: list[str],
    family_of: dict[str, str],
    out_path,
    n_boot: int = 1000,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n_axes = len(axes_order)
    cols = min(3, n_axes)
    rows = int(np.ceil(n_axes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 0.32 * len(models_order) * rows + 1.2),
                              squeeze=False)

    for idx, axis_name in enumerate(axes_order):
        ax = axes[idx // cols][idx % cols]
        sub = df_full[df_full["axis"] == axis_name]
        means, los, his, colors = [], [], [], []
        for m in models_order:
            vals = sub[sub["model"] == m]["score"].values
            if len(vals) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); colors.append("#888")
                continue
            boots = np.array([
                rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)
            ])
            means.append(vals.mean())
            los.append(np.percentile(boots, 2.5))
            his.append(np.percentile(boots, 97.5))
            colors.append(FAMILY_COLORS.get(family_of.get(m, "other"), "#888"))
        y = np.arange(len(models_order))
        means = np.asarray(means); los = np.asarray(los); his = np.asarray(his)
        ax.barh(y, means, color=colors, alpha=0.85, height=0.7)
        ax.errorbar(means, y, xerr=[means - los, his - means], fmt="none",
                    ecolor=INK, elinewidth=0.8, capsize=2)
        ax.axvline(0, color=LINE, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(models_order)
        ax.invert_yaxis()
        ax.set_title(axis_name.replace("_", " "), color=INK)
        ax.set_xlabel("projection")

    for k in range(n_axes, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Per-axis model projections (mean ± 95% bootstrap CI over prompts)",
                 fontsize=12, color=INK, y=1.0)
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
