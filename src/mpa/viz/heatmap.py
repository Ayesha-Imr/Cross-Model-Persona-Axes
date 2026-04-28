from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from .theme import DIVERGING, FAMILY_COLORS, INK, MUTED, add_explainer, save_fig

EXPLAINER = (
    "Each cell shows how much a model deviates from the cross-model average on a given "
    "persona axis, expressed as a z-score (units of standard deviation).\n\n"
    "• Red → model scores higher than peers on that trait.\n"
    "• Blue → model scores lower than peers.\n"
    "• Near-white → typical / no separation.\n\n"
    "Rows are grouped by model family (left stripe). Reading down a column tells you "
    "which families lean which way; reading across a row gives that model's signature."
)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def model_axis_heatmap(
    df: pd.DataFrame,
    axes_order: list[str],
    models_order: list[str],
    family_of: dict[str, str],
    out_path,
):
    pivot = df.pivot(index="model", columns="axis", values="score").reindex(
        index=models_order, columns=axes_order,
    )
    z = (pivot - pivot.mean(axis=0)) / pivot.std(axis=0).replace(0, 1)

    n_models = len(models_order)
    n_axes = len(axes_order)
    fig = plt.figure(figsize=(0.95 * n_axes + 4.6, 0.5 * n_models + 1.6))
    gs = GridSpec(1, 2, width_ratios=[0.05, 1.0], wspace=0.04, figure=fig)
    ax_stripe = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    # family stripe
    fam_colors = [FAMILY_COLORS.get(family_of.get(m, "other"), "#888") for m in models_order]
    ax_stripe.imshow(np.arange(n_models)[:, None], aspect="auto",
                     cmap=plt.matplotlib.colors.ListedColormap(fam_colors))
    ax_stripe.set_xticks([])
    ax_stripe.set_yticks(range(n_models))
    ax_stripe.set_yticklabels(models_order, color=INK, fontsize=10)
    ax_stripe.tick_params(length=0)
    for s in ax_stripe.spines.values():
        s.set_visible(False)

    # heatmap
    vmax = float(np.nanmax(np.abs(z.values))) or 1.0
    im = ax.imshow(z.values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_axes))
    ax.set_xticklabels([_pretty(a) for a in axes_order], rotation=30, ha="right", color=INK)
    ax.set_yticks([])
    ax.set_title("Persona-axis projections by model (z-scored across models)",
                 color=INK, pad=10, weight="bold")
    threshold = 0.7 * vmax
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z.values[i, j]
            if np.isnan(v) or abs(v) < 0.4:
                continue
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if abs(v) > threshold else INK, fontsize=8.5)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED)
    cb.set_label("z-score", fontsize=9.5, color=MUTED)

    # family legend (under stripe)
    seen = []
    handles = []
    for m in models_order:
        fam = family_of.get(m, "other")
        if fam in seen:
            continue
        seen.append(fam)
        handles.append(plt.Rectangle((0, 0), 1, 1,
                                      facecolor=FAMILY_COLORS.get(fam, "#888")))
    fig.legend(handles, [f.title() for f in seen], loc="lower center",
               ncol=min(6, len(seen)), bbox_to_anchor=(0.5, -0.06),
               frameon=False, title="Family")

    add_explainer(fig, EXPLAINER, loc="right", width_frac=0.24)
    save_fig(fig, out_path)
    plt.close(fig)
