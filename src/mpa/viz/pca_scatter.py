from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import FAMILY_COLORS, INK, LINE, MUTED, add_explainer, save_fig

EXPLAINER = (
    "Each model's seven-axis fingerprint is reduced to two principal components.\n\n"
    "• Dots = models, colored by family. Distance encodes overall persona similarity.\n"
    "• Arrows show how each persona axis loads onto PC1 / PC2 — a model on the warmth "
    "arrow leans warm; a model opposite leans cold.\n\n"
    "This is the interpretable analogue of lyra's cosine-similarity fingerprint plot: "
    "if families cluster, the prober is detecting real structure; the arrows tell you "
    "in human terms what each axis of the cluster is."
)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def model_pca_scatter(
    df: pd.DataFrame,                       # long: model, axis, score (already mean-aggregated)
    axes_order: list[str],
    models_order: list[str],
    family_of: dict[str, str],
    out_path,
):
    pivot = df.pivot(index="model", columns="axis", values="score").reindex(
        index=models_order, columns=axes_order,
    )
    Z = (pivot - pivot.mean(axis=0)) / pivot.std(axis=0).replace(0, 1)
    Z = Z.fillna(0.0)
    M = Z.values
    # PCA via SVD
    Mc = M - M.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    pc = U[:, :2] * S[:2]
    loadings = Vt[:2].T  # (n_axes, 2)
    explained = (S ** 2) / (S ** 2).sum()

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    # axis arrows (loadings) — scaled to match scatter range
    pc_range = max(pc[:, 0].ptp(), pc[:, 1].ptp(), 1e-6)
    arrow_scale = 0.45 * pc_range / max(np.abs(loadings).max(), 1e-6)
    for i, name in enumerate(axes_order):
        dx, dy = loadings[i] * arrow_scale
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0, alpha=0.7))
        ax.text(dx * 1.08, dy * 1.08, _pretty(name), fontsize=9, color=MUTED,
                ha="center", va="center")

    # model dots
    seen_fams: list[str] = []
    for m, (x, y) in zip(models_order, pc):
        fam = family_of.get(m, "other")
        c = FAMILY_COLORS.get(fam, "#888")
        if fam not in seen_fams:
            seen_fams.append(fam)
            ax.scatter([x], [y], color=c, s=110, edgecolor="white", linewidth=1.2,
                       label=fam.title(), zorder=3)
        else:
            ax.scatter([x], [y], color=c, s=110, edgecolor="white", linewidth=1.2,
                       zorder=3)
        ax.annotate(m, (x, y), xytext=(7, 5), textcoords="offset points",
                    fontsize=8.5, color=INK)

    ax.axhline(0, color=LINE, linewidth=0.8, zorder=0)
    ax.axvline(0, color=LINE, linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)", color=MUTED)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)", color=MUTED)
    ax.set_title("Model persona-fingerprint PCA", color=INK, weight="bold", pad=10)
    ax.legend(loc="best", title="Family")
    add_explainer(fig, EXPLAINER, loc="right", width_frac=0.26)
    save_fig(fig, out_path)
    plt.close(fig)
