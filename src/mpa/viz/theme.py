"""Vibrant categorical palette + cream background, inspired by Anthropic research figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Background
CREAM = "#F4EFE6"
PAPER = "#FBF7EE"
INK = "#1F1A14"
MUTED = "#6E6356"
LINE = "#E0D7C4"

# Vibrant categorical palette — high-saturation but still warm/cream-friendly
GOLD = "#E8A33D"
TEAL = "#3D8E91"
INDIGO = "#3F4CA3"
CRIMSON = "#B23A48"
SAGE = "#6B9874"
SALMON = "#E07856"
PLUM = "#8E5A8C"
SLATE = "#5C6373"
ROSE = "#D88BA0"
MOSS = "#7E8C4F"

PALETTE = [GOLD, TEAL, INDIGO, CRIMSON, SAGE, SALMON, PLUM, SLATE, ROSE, MOSS]

# Family colors
FAMILY_COLORS = {
    "claude": CRIMSON,
    "gpt": INDIGO,
    "gemini": TEAL,
    "cohere": GOLD,
    "aya": SAGE,
    "open": SLATE,
    "other": PLUM,
    "prober": SALMON,
}

# Diverging colormap for heatmaps (centered at 0): indigo -> cream -> crimson
DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "mpa_div", [INDIGO, "#7480C0", PAPER, "#D77785", CRIMSON], N=256,
)


def _available_fonts(candidates: list[str], fallback: str) -> list[str]:
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chain = [c for c in candidates if c in installed]
    chain.append(fallback)
    return chain


def apply_theme() -> None:
    sans = _available_fonts(
        ["Inter", "Söhne", "Söhne Buch", "Helvetica Neue", "Helvetica",
         "Arial", "Liberation Sans", "DejaVu Sans"],
        "sans-serif",
    )
    plt.rcParams.update({
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "savefig.facecolor": CREAM,
        "font.family": sans,
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.titlecolor": INK,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.labelweight": "regular",
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "legend.title_fontsize": 10,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "image.cmap": "viridis",
        "figure.dpi": 110,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
    })


def color_for(family: str | None) -> str:
    return FAMILY_COLORS.get(family or "other", PLUM)


def save_fig(fig, path) -> None:
    p = str(path)
    fig.savefig(p + ".png")
    fig.savefig(p + ".pdf")


def add_explainer(
    fig,
    text: str,
    *,
    title: str = "How to read",
    loc: str = "right",
    width_frac: float = 0.22,
    height_frac: float = 0.20,
) -> None:
    """Attach a soft 'how to read' panel to fig.

    `loc='right'` reserves space on the right; `loc='bottom'` reserves a footer.
    """
    if loc == "right":
        fig.subplots_adjust(right=1.0 - width_frac - 0.02)
        ax_x = 1.0 - width_frac
        ax_y = 0.06
        ax_w = width_frac - 0.015
        ax_h = 0.88
    else:
        fig.subplots_adjust(bottom=height_frac + 0.04)
        ax_x = 0.04
        ax_y = 0.02
        ax_w = 0.92
        ax_h = height_frac

    ax = fig.add_axes([ax_x, ax_y, ax_w, ax_h])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        facecolor=PAPER, edgecolor=LINE, linewidth=0.7,
    ))
    ax.text(0.05, 0.93, title, transform=ax.transAxes,
            fontsize=10.5, color=INK, weight="bold", va="top")
    ax.text(0.05, 0.82, text, transform=ax.transAxes,
            fontsize=9, color=INK, va="top", ha="left",
            linespacing=1.5, wrap=True)
