from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import FAMILY_COLORS, INK, LINE, MUTED, add_explainer, save_fig

EXPLAINER = (
    "Each panel shows one model family's persona signature as a polygon over the seven axes.\n\n"
    "• Values are z-scored across models — 0 (dotted ring) is the cross-model average; "
    "outward = above average, inward = below.\n"
    "• A faint grey polygon shows the global cross-model mean for reference.\n\n"
    "Compare panels to see how each family leans differently. Look for axes where one "
    "family extends far out (signature trait) while others sit near the baseline."
)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def family_radar(df: pd.DataFrame, axes_order: list[str], family_of: dict[str, str], out_path):
    pivot = df.pivot(index="model", columns="axis", values="score").reindex(columns=axes_order)
    z = (pivot - pivot.mean(axis=0)) / pivot.std(axis=0).replace(0, 1)
    z["family"] = z.index.map(family_of)
    fam_means = z.groupby("family")[axes_order].mean()

    families = list(fam_means.index)
    n = len(families)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    angles = np.linspace(0, 2 * np.pi, len(axes_order), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    # global axis range so panels are comparable
    vmax = float(np.nanmax(np.abs(fam_means.values))) or 1.0
    rmax = max(1.2, vmax * 1.15)
    rmin = -rmax

    fig = plt.figure(figsize=(4.2 * cols + 0.6, 4.2 * rows + 1.2))
    for i, fam in enumerate(families):
        ax = fig.add_subplot(rows, cols, i + 1, projection="polar")
        # baseline (zero ring)
        ax.plot(angles_closed, [0] * len(angles_closed),
                color=MUTED, linewidth=0.7, linestyle=(0, (2, 3)), zorder=1)
        # global mean polygon (faint grey reference)
        global_mean = z[axes_order].mean(axis=0).values
        gvals = list(global_mean) + [global_mean[0]]
        ax.plot(angles_closed, gvals, color=MUTED, linewidth=1.1, alpha=0.45, zorder=2)
        ax.fill(angles_closed, gvals, color=MUTED, alpha=0.06, zorder=1)
        # family polygon
        row = fam_means.loc[fam].values
        vals = list(row) + [row[0]]
        c = FAMILY_COLORS.get(fam, "#888")
        ax.plot(angles_closed, vals, color=c, linewidth=2.3, marker="o",
                markersize=5, zorder=3)
        ax.fill(angles_closed, vals, color=c, alpha=0.22, zorder=2)

        ax.set_xticks(angles)
        ax.set_xticklabels([_pretty(a) for a in axes_order], fontsize=8.5, color=INK)
        ax.set_ylim(rmin, rmax)
        ax.set_yticks([0])
        ax.set_yticklabels([])
        ax.spines["polar"].set_color(LINE)
        ax.grid(color=LINE, linewidth=0.5)
        ax.set_title(fam.title(), color=c, pad=14, fontsize=12, weight="bold")

    fig.suptitle("Family-level persona signature (z-scored across models)",
                 fontsize=14, color=INK, weight="bold", y=1.0)
    fig.tight_layout()
    add_explainer(fig, EXPLAINER, loc="right", width_frac=0.22)
    save_fig(fig, out_path)
    plt.close(fig)
