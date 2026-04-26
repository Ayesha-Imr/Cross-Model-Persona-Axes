from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Core palette
CLAY = "#CC785C"          # primary warm clay
INDIGO = "#5C6BC4"         # secondary
SAGE = "#6B8E72"
OCHRE = "#B8895C"
INK = "#1F1F1E"
MUTED = "#6B6B68"
LINE = "#E8E4DC"
CREAM = "#F7F4EE"

PALETTE = [CLAY, INDIGO, SAGE, OCHRE, "#7A6E5D", "#A55C5C", "#4F7A6F", "#8D7BB8"]

# Family colors (extend as new families are added)
FAMILY_COLORS = {
    "claude": CLAY,
    "gpt": INDIGO,
    "gemini": SAGE,
    "cohere": OCHRE,
    "open": MUTED,
    "other": "#7A6E5D",
    "prober": "#A55C5C",
}

# Diverging palette for heatmaps (centered at 0)
DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "mpa_div", [INDIGO, "#FFFFFF", CLAY], N=256,
)


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "savefig.facecolor": CREAM,
        "font.family": ["Source Serif Pro", "Source Serif 4", "Georgia", "serif"],
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "regular",
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "image.cmap": "viridis",
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def color_for(model_name: str, family: str | None = None) -> str:
    if family and family in FAMILY_COLORS:
        return FAMILY_COLORS[family]
    return CLAY


def save_fig(fig, path) -> None:
    path = str(path)
    fig.savefig(path + ".png")
    fig.savefig(path + ".pdf")
