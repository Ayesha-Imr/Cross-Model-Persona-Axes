from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import DIVERGING, INK, MUTED, add_explainer, save_fig

EXPLAINER = (
    "Each cell is the standardized mean difference (Cohen's d) between two families on "
    "one persona axis, computed over per-prompt projection scores.\n\n"
    "• Red → row-family scores higher than column-family on that axis.\n"
    "• Blue → row-family scores lower.\n"
    "• |d| ≈ 0.2 small, 0.5 medium, 0.8 large.\n\n"
    "Reads as: 'on axis X, family A is +d standard deviations higher than family B'. "
    "Big colored cells highlight the actual family-discriminating axes."
)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    s = np.sqrt(max(s2, 1e-12))
    return float((a.mean() - b.mean()) / s)


def family_effect_size(
    df_full: pd.DataFrame,                  # long: model, prompt_id, axis, score (layer-aggregated)
    axes_order: list[str],
    family_of: dict[str, str],
    out_path,
):
    df = df_full.assign(family=df_full["model"].map(family_of))
    families = sorted(df["family"].dropna().unique().tolist())
    pairs = list(combinations(families, 2))
    if not pairs:
        return

    matrix = np.zeros((len(axes_order), len(pairs)))
    for ai, axis in enumerate(axes_order):
        sub = df[df["axis"] == axis]
        for pi, (fa, fb) in enumerate(pairs):
            a = sub[sub["family"] == fa]["score"].values
            b = sub[sub["family"] == fb]["score"].values
            matrix[ai, pi] = _cohens_d(a, b)

    vmax = max(float(np.nanmax(np.abs(matrix))), 0.5)
    fig, ax = plt.subplots(figsize=(0.85 * len(pairs) + 4.0, 0.55 * len(axes_order) + 2.0))
    im = ax.imshow(matrix, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{a.title()}\nvs\n{b.title()}" for a, b in pairs], fontsize=9, color=INK)
    ax.set_yticks(range(len(axes_order)))
    ax.set_yticklabels([_pretty(a) for a in axes_order], color=INK)
    ax.set_title("Family-pairwise effect size (Cohen's d) per persona axis",
                 color=INK, weight="bold", pad=10)
    threshold = 0.7 * vmax
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v) or abs(v) < 0.1:
                continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > threshold else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED)
    cb.set_label("Cohen's d", fontsize=9.5, color=MUTED)
    add_explainer(fig, EXPLAINER, loc="right", width_frac=0.26)
    save_fig(fig, out_path)
    plt.close(fig)
