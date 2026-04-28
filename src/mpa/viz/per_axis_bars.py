from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import FAMILY_COLORS, INK, LINE, MUTED, PALETTE, add_explainer, save_fig

EXPLAINER = (
    "Each panel shows one persona axis. Dots mark each model's mean projection over "
    "all evaluation prompts; the line connects to zero (the cross-prompt midpoint).\n\n"
    "• Horizontal bars are 95% bootstrap confidence intervals (resampled over prompts).\n"
    "• Models are grouped by family (color); families separated by a thin gap.\n\n"
    "Look for: (a) which models are consistently far from zero on an axis, "
    "(b) whether models within a family cluster together, "
    "(c) axes where CIs overlap heavily — those don't discriminate models."
)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def _grouped_order(models: list[str], family_of: dict[str, str]) -> tuple[list[str], list[int]]:
    """Sort models by family, return ordered list + indices where families change."""
    sorted_models = sorted(models, key=lambda m: (family_of.get(m, "z"), m))
    breaks: list[int] = []
    last = None
    for i, m in enumerate(sorted_models):
        fam = family_of.get(m, "other")
        if last is not None and fam != last:
            breaks.append(i)
        last = fam
    return sorted_models, breaks


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
    sorted_models, fam_breaks = _grouped_order(models_order, family_of)
    color_map = _color_per_model(sorted_models, family_of)

    n_axes = len(axes_order)
    cols = min(3, n_axes)
    rows = int(np.ceil(n_axes / cols))
    n = len(sorted_models)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5.2 * cols, 0.42 * n * rows + 1.6),
        squeeze=False,
    )

    for idx, axis_name in enumerate(axes_order):
        ax = axes[idx // cols][idx % cols]
        sub = df_full[df_full["axis"] == axis_name]
        means, los, his = [], [], []
        for m in sorted_models:
            vals = sub[sub["model"] == m]["score"].values
            if len(vals) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            boots = np.array([
                rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)
            ])
            means.append(vals.mean())
            los.append(np.percentile(boots, 2.5))
            his.append(np.percentile(boots, 97.5))
        y = np.arange(n)
        means = np.asarray(means); los = np.asarray(los); his = np.asarray(his)

        # zero reference
        ax.axvline(0, color=LINE, linewidth=0.9, zorder=0)
        # family separators
        for b in fam_breaks:
            ax.axhline(b - 0.5, color=LINE, linewidth=0.6, zorder=0)
        # lollipops
        for i, m in enumerate(sorted_models):
            c = color_map[m]
            if np.isnan(means[i]):
                continue
            ax.hlines(y[i], 0, means[i], color=c, linewidth=2.0, alpha=0.85, zorder=2)
            ax.errorbar(means[i], y[i], xerr=[[means[i] - los[i]], [his[i] - means[i]]],
                        fmt="none", ecolor=c, elinewidth=1.0, capsize=2.5, alpha=0.5, zorder=2)
            ax.plot(means[i], y[i], "o", color=c, markersize=6.5,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels(sorted_models, color=INK, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_title(_pretty(axis_name), color=INK)
        ax.set_xlabel("Projection score", color=MUTED)
        ax.tick_params(axis="x", colors=MUTED)

    for k in range(n_axes, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.suptitle("Per-axis model projections (mean ± 95% bootstrap CI over prompts)",
                 fontsize=14, color=INK, y=1.0, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    add_explainer(fig, EXPLAINER, loc="bottom", height_frac=0.15)
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
        if base and len(names) == 1:
            out[names[0]] = base
        else:
            for n in names:
                out[n] = next(fallback)
    return out
