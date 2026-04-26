from __future__ import annotations

import pandas as pd

from ..logging import setup_logging
from ..viz.heatmap import model_axis_heatmap
from ..viz.per_axis_bars import per_axis_bars
from ..viz.per_layer_ridge import per_layer_lines
from ..viz.radar import family_radar
from ..viz.theme import apply_theme
from ._common import base_parser, load_cfg_and_run_dir


def main():
    args = base_parser("visualize").parse_args()
    cfg, run_dir = load_cfg_and_run_dir(args)
    log = setup_logging(run_dir, "visualize")
    apply_theme()

    proj_path = run_dir / "artifacts" / "projections.parquet"
    if not proj_path.exists():
        raise SystemExit("projections.parquet not found; run project first.")

    df = pd.read_parquet(proj_path)
    axes_order = [a.name for a in cfg.axes]
    family_of = {m.name: m.family for m in cfg.models}
    models_order = [m.name for m in cfg.enabled_models() if m.name in df["model"].unique()]
    models_order.sort(key=lambda n: (family_of.get(n, "z"), n))

    # mean over layers for axis-level views
    by_model_axis = df.groupby(["model", "axis"])["score"].mean().reset_index()
    by_model_axis_prompt = df.groupby(["model", "axis", "prompt_id"])["score"].mean().reset_index()

    fig_dir = run_dir / "figures"
    log.info("Writing figures -> %s", fig_dir)
    model_axis_heatmap(by_model_axis, axes_order, models_order, fig_dir / "01_heatmap")
    family_radar(by_model_axis, axes_order, family_of, fig_dir / "02_family_radar")
    per_axis_bars(by_model_axis_prompt, axes_order, models_order, family_of,
                  fig_dir / "03_per_axis_bars", seed=cfg.seed)
    per_layer_lines(df, axes_order, models_order, family_of, fig_dir / "04_per_layer_lines")
    log.info("Done.")


if __name__ == "__main__":
    main()
