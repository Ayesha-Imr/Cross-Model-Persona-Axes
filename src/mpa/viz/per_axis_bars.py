from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from .theme import FAMILY_COLORS, INK, LINE, MUTED, PALETTE, save_fig


def per_axis_bars(
    df_full: pd.DataFrame,
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
    fig, axes = plt.subplots(rows, cols,
                              figsize=(5.0 * cols, 0.42 * len(models_order) * rows + 1.4),
                              squeeze=False)

    color_map = _color_per_model(models_order, family_of)

    for idx, axis_name in enumerate(axes_order):
        ax = axes[idx // cols][idx % cols]
        sub = df_full[df_full["axis"] == axis_name]
        means, los, his = [], [], []
        for m in models_order:
            vals = sub[sub["model"] == m]["score"].values
            if len(vals) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            boots = np.array([
                rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)
            ])
            means.append(vals.mean())
            los.append(np.percentile(boots, 2.5))
            his.append(np.percentile(boots, 97.5))

        y = np.arange(len(models_order))
        means = np.asarray(means); los = np.asarray(los); his = np.asarray(his)

        for i, m in enumerate(models_order):
            c = color_map[m]
            fill = to_rgba(c, alpha=0.18)
            edge = c
            ax.barh(y[i], means[i], color=fill, edgecolor=edge, linewidth=1.6, height=0.62)
            ax.plot([means[i]], [y[i]], "o", color=c, markersize=5.5,
                    markeredgecolor=c, zorder=3)
        ax.errorbar(means, y, xerr=[means - los, his - means], fmt="none",
                    ecolor=INK, elinewidth=1.0, capsize=3, alpha=0.7, zorder=2)
        ax.axvline(0, color=LINE, linewidth=0.9, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(models_order, color=INK, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(_pretty(axis_name), color=INK)
        ax.set_xlabel("Projection score (mean ± 95% CI over prompts)", color=MUTED)
        ax.tick_params(axis="x", colors=MUTED)

    for k in range(n_axes, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Per-axis model projections", fontsize=14, color=INK, y=1.0, weight="bold")
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
