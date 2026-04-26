from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import FAMILY_COLORS, INK, save_fig


def family_radar(df: pd.DataFrame, axes_order: list[str], family_of: dict[str, str], out_path):
    pivot = df.pivot(index="model", columns="axis", values="score").reindex(columns=axes_order)
    z = (pivot - pivot.mean(axis=0)) / pivot.std(axis=0).replace(0, 1)
    z["family"] = z.index.map(family_of)
    fam_means = z.groupby("family")[axes_order].mean()

    angles = np.linspace(0, 2 * np.pi, len(axes_order), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for fam, row in fam_means.iterrows():
        vals = list(row.values) + [row.values[0]]
        c = FAMILY_COLORS.get(fam, "#888")
        ax.plot(angles, vals, color=c, linewidth=1.6, label=fam)
        ax.fill(angles, vals, color=c, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a.replace("_", " ") for a in axes_order], fontsize=9)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("#E8E4DC")
    ax.grid(color="#E8E4DC", linewidth=0.6)
    ax.set_title("Family-level persona signature (z-scored)", color=INK, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    save_fig(fig, out_path)
    plt.close(fig)
