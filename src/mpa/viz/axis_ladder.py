from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from ..checkpoint import read_jsonl
from .theme import FAMILY_COLORS, INK, LINE, MUTED, PAPER, add_explainer, save_fig

EXPLAINER = (
    "One panel per persona axis. Within each panel, all models are stacked top-to-bottom "
    "by their mean projection score on that axis (highest at the top).\n\n"
    "Each row shows: family color stripe, model name, mean score, and a one-line excerpt "
    "from that model's most representative response (closest to its own median score).\n\n"
    "Scan a column to compare models on a single trait; scan across columns to see how "
    "a single model's behavior shifts axis to axis."
)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def _truncate(text: str, n: int) -> str:
    text = (text or "").strip().replace("\r", "").replace("\n", " ")
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def _representative_pid(sub_axis_model: pd.DataFrame) -> str | None:
    if sub_axis_model.empty:
        return None
    median = sub_axis_model["score"].median()
    idx = (sub_axis_model["score"] - median).abs().idxmin()
    return sub_axis_model.loc[idx, "prompt_id"]


def axis_ladder(
    df_full: pd.DataFrame,                   # long: model, prompt_id, axis, score (layer-aggregated)
    by_model_axis: pd.DataFrame,             # mean per (model, axis)
    responses_dir: Path,
    axes_order: list[str],
    family_of: dict[str, str],
    out_path,
):
    n_axes = len(axes_order)
    cols = min(2, n_axes)
    rows = int(np.ceil(n_axes / cols))
    n_models = by_model_axis["model"].nunique()
    panel_h = 0.42 * n_models + 0.6
    fig = plt.figure(figsize=(8.5 * cols, panel_h * rows + 1.4))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.18, wspace=0.10,
                  left=0.03, right=0.98, top=0.94, bottom=0.05)

    resp_cache: dict[str, list[dict]] = {}

    def _resp(model_name: str) -> list[dict]:
        if model_name not in resp_cache:
            p = responses_dir / f"{model_name}.jsonl"
            resp_cache[model_name] = read_jsonl(p) if p.exists() else []
        return resp_cache[model_name]

    def _lookup(model_name: str, prompt_id: str | None) -> str:
        if prompt_id is None:
            return ""
        for r in _resp(model_name):
            if r.get("prompt_id") == prompt_id:
                return r.get("response", "")
        return ""

    for i, axis_name in enumerate(axes_order):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ranking = by_model_axis[by_model_axis["axis"] == axis_name].sort_values(
            "score", ascending=False,
        )
        models_sorted = ranking["model"].tolist()

        sub_full = df_full[df_full["axis"] == axis_name]
        rows_data = []
        for model in models_sorted:
            mdf = sub_full[sub_full["model"] == model]
            pid = _representative_pid(mdf)
            mean_score = float(ranking[ranking["model"] == model]["score"].iloc[0])
            snippet = _truncate(_lookup(model, pid), 130)
            rows_data.append((model, mean_score, snippet))

        _draw_ladder(ax, axis_name, rows_data, family_of)

    fig.suptitle("Axis ladder — every model ranked on every axis, with a representative response",
                 fontsize=14, color=INK, weight="bold", y=0.98)
    add_explainer(fig, EXPLAINER, loc="bottom", height_frac=0.07)
    save_fig(fig, out_path)
    plt.close(fig)


def _draw_ladder(ax, axis_name, rows_data, family_of):
    n = len(rows_data)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(_pretty(axis_name), color=INK, loc="left", weight="bold", pad=8)

    # background card
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                facecolor=PAPER, edgecolor=LINE, linewidth=0.6))

    if n == 0:
        return
    cell_h = 1.0 / n
    for i, (model, score, snippet) in enumerate(rows_data):
        y0 = 1.0 - (i + 1) * cell_h
        # alternating row tint
        if i % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y0), 1, cell_h, transform=ax.transAxes,
                                        facecolor="#F2EBDC", edgecolor="none"))
        # family stripe (left)
        fam = family_of.get(model, "other")
        c = FAMILY_COLORS.get(fam, MUTED)
        ax.add_patch(plt.Rectangle((0, y0), 0.014, cell_h, transform=ax.transAxes,
                                    facecolor=c, edgecolor="none"))

        text_y = y0 + cell_h * 0.62
        # model name + score
        ax.text(0.030, text_y, model, transform=ax.transAxes,
                fontsize=9.5, color=c, weight="bold", va="center")
        ax.text(0.30, text_y, f"{score:+.2f}", transform=ax.transAxes,
                fontsize=9, color=MUTED, va="center", family="monospace")
        # response snippet
        ax.text(0.36, text_y, f"\u201C{snippet}\u201D" if snippet else "—",
                transform=ax.transAxes, fontsize=8.8, color=INK, va="center",
                style="italic" if snippet else "normal")

        # subtle row separator
        if i < n - 1:
            ax.add_patch(plt.Rectangle((0.014, y0), 1 - 0.014, 0.0015,
                                        transform=ax.transAxes,
                                        facecolor=LINE, edgecolor="none"))
