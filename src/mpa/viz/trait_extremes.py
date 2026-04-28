from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from ..checkpoint import read_jsonl
from .theme import FAMILY_COLORS, INK, LINE, MUTED, PAPER, add_explainer, save_fig

EXPLAINER = (
    "For each persona axis, we picked the model with the highest mean projection "
    "and the lowest mean projection across all evaluation prompts.\n\n"
    "Each card shows that model's most representative response on this axis — the "
    "single response whose score is closest to that model's median (so the example is "
    "typical, not an outlier).\n\n"
    "This is the qualitative companion to the heatmap: there you saw who scored "
    "high/low; here you see what 'high' and 'low' actually sound like in practice."
)


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def _truncate(text: str, n: int) -> str:
    text = (text or "").strip().replace("\r", "").replace("\n\n\n", "\n\n")
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def _representative_response(
    df_axis_model: pd.DataFrame, model: str,
) -> tuple[str | None, float | None]:
    """Return (prompt_id, score) of the response closest to this model's median score on the axis."""
    sub = df_axis_model[df_axis_model["model"] == model]
    if sub.empty:
        return None, None
    median = sub["score"].median()
    idx = (sub["score"] - median).abs().idxmin()
    row = sub.loc[idx]
    return row["prompt_id"], float(row["score"])


def trait_extremes(
    df_full: pd.DataFrame,                     # long: model, prompt_id, axis, score (layer-aggregated)
    by_model_axis: pd.DataFrame,               # mean projection per (model, axis)
    responses_dir: Path,
    axes_order: list[str],
    family_of: dict[str, str],
    out_path,
):
    n_axes = len(axes_order)
    fig = plt.figure(figsize=(13.5, 2.7 * n_axes + 1.6))
    gs = GridSpec(
        n_axes, 2, figure=fig, hspace=0.55, wspace=0.10,
        left=0.04, right=0.98, top=0.96, bottom=0.04,
    )

    resp_cache: dict[str, list[dict]] = {}

    def _resp(model_name: str) -> list[dict]:
        if model_name not in resp_cache:
            p = responses_dir / f"{model_name}.jsonl"
            resp_cache[model_name] = read_jsonl(p) if p.exists() else []
        return resp_cache[model_name]

    def _lookup(model_name: str, prompt_id: str) -> tuple[str, str]:
        for r in _resp(model_name):
            if r.get("prompt_id") == prompt_id:
                return r.get("prompt", ""), r.get("response", "")
        return "", ""

    for r, axis_name in enumerate(axes_order):
        # rank models by mean projection on this axis
        sub_means = by_model_axis[by_model_axis["axis"] == axis_name].sort_values("score")
        if sub_means.empty:
            continue
        lo_row = sub_means.iloc[0]
        hi_row = sub_means.iloc[-1]

        sub_full = df_full[df_full["axis"] == axis_name]
        hi_pid, hi_repr = _representative_response(sub_full, hi_row["model"])
        lo_pid, lo_repr = _representative_response(sub_full, lo_row["model"])

        ax_hi = fig.add_subplot(gs[r, 0])
        ax_lo = fig.add_subplot(gs[r, 1])

        _draw_card(
            ax_hi, axis_name=axis_name, side="HIGHEST",
            model=hi_row["model"], mean_score=float(hi_row["score"]),
            response_score=hi_repr,
            family_color=FAMILY_COLORS.get(family_of.get(hi_row["model"], "other"), MUTED),
            prompt_text=_lookup(hi_row["model"], hi_pid)[0] if hi_pid else "",
            response_text=_lookup(hi_row["model"], hi_pid)[1] if hi_pid else "",
        )
        _draw_card(
            ax_lo, axis_name=axis_name, side="LOWEST",
            model=lo_row["model"], mean_score=float(lo_row["score"]),
            response_score=lo_repr,
            family_color=FAMILY_COLORS.get(family_of.get(lo_row["model"], "other"), MUTED),
            prompt_text=_lookup(lo_row["model"], lo_pid)[0] if lo_pid else "",
            response_text=_lookup(lo_row["model"], lo_pid)[1] if lo_pid else "",
        )

    fig.suptitle("Trait extremes — what 'highest' and 'lowest' actually sound like",
                 fontsize=14, color=INK, weight="bold", y=0.995)
    add_explainer(fig, EXPLAINER, loc="bottom", height_frac=0.10)
    save_fig(fig, out_path)
    plt.close(fig)


def _draw_card(
    ax,
    *,
    axis_name: str,
    side: str,
    model: str,
    mean_score: float,
    response_score: float | None,
    family_color: str,
    prompt_text: str,
    response_text: str,
):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # card background
    ax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax.transAxes,
                                facecolor=PAPER, edgecolor=LINE, linewidth=0.6))
    # family-colored top stripe
    ax.add_patch(plt.Rectangle((0.0, 0.965), 1.0, 0.035, transform=ax.transAxes,
                                facecolor=family_color, edgecolor="none"))

    # axis label (top-left, muted)
    ax.text(0.018, 0.99, _pretty(axis_name).upper(), transform=ax.transAxes,
            fontsize=8.5, color="white", weight="bold", va="top", ha="left")
    # side label (HIGHEST / LOWEST)
    ax.text(0.985, 0.99, side, transform=ax.transAxes,
            fontsize=8.5, color="white", weight="bold", va="top", ha="right")

    # model name + score row
    ax.text(0.025, 0.90, model, transform=ax.transAxes,
            fontsize=11.5, color=family_color, weight="bold", va="top")
    score_text = f"mean projection  {mean_score:+.2f}"
    if response_score is not None:
        score_text += f"   ·   sample score  {response_score:+.2f}"
    ax.text(0.025, 0.81, score_text, transform=ax.transAxes,
            fontsize=8.5, color=MUTED, va="top")

    # prompt
    prompt_line = "User: " + _truncate(prompt_text, 140) if prompt_text else "—"
    ax.text(0.025, 0.71, prompt_line, transform=ax.transAxes,
            fontsize=8.5, color=MUTED, style="italic", va="top", wrap=True)

    # divider
    ax.add_patch(plt.Rectangle((0.025, 0.605), 0.95, 0.002, transform=ax.transAxes,
                                facecolor=LINE, edgecolor="none"))

    # response body
    body = _truncate(response_text, 520) if response_text else "(no response)"
    ax.text(0.025, 0.58, body, transform=ax.transAxes,
            fontsize=9, color=INK, va="top", wrap=True, linespacing=1.4)
