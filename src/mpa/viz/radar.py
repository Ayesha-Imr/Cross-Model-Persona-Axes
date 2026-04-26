from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import FAMILY_COLORS, INK, LINE, save_fig


def family_radar(df: pd.DataFrame, axes_order: list[str], family_of: dict[str, str], out_path):
    pivot = df.pivot(index="model", columns="axis", values="score").reindex(columns=axes_order)
    z = (pivot - pivot.mean(axis=0)) / pivot.std(axis=0).replace(0, 1)
    z["family"] = z.index.map(family_of)
    fam_means = z.groupby("family")[axes_order].mean()

    angles = np.linspace(0, 2 * np.pi, len(axes_order), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(6.4, 6.4), subplot_kw={"projection": "polar"})
    for fam, row in fam_means.iterrows():
        vals = list(row.values) + [row.values[0]]
        c = FAMILY_COLORS.get(fam, "#888")
        ax.plot(angles, vals, color=c, linewidth=2.0, marker="o", markersize=5,
                label=fam.title())
        ax.fill(angles, vals, color=c, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_pretty(a) for a in axes_order], fontsize=10, color=INK)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(LINE)
    ax.grid(color=LINE, linewidth=0.7)
    ax.set_title("Family-level persona signature (z-scored)", color=INK, pad=22,
                 fontsize=14, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1), title="Family")
    save_fig(fig, out_path)
    plt.close(fig)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()
