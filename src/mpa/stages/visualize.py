from __future__ import annotations

import pandas as pd

from ..logging import setup_logging
from ..viz.family_effect_size import family_effect_size
from ..viz.heatmap import model_axis_heatmap
from ..viz.pca_scatter import model_pca_scatter
from ..viz.per_axis_bars import per_axis_bars
from ..viz.per_layer_ridge import per_layer_lines
from ..viz.radar import family_radar
from ..viz.tables import (
    axis_emergence_table,
    axis_ladder_table,
    family_signatures_table,
    trait_extremes_table,
)
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

    by_model_axis = df.groupby(["model", "axis"])["score"].mean().reset_index()
    by_model_axis_prompt = df.groupby(["model", "axis", "prompt_id"])["score"].mean().reset_index()

    fig_dir = run_dir / "figures"
    tbl_dir = run_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing figures -> %s", fig_dir)
    log.info("Writing tables  -> %s", tbl_dir)

    model_axis_heatmap(by_model_axis, axes_order, models_order, family_of,
                       fig_dir / "01_heatmap")
    family_radar(by_model_axis, axes_order, family_of, fig_dir / "02_family_radar")
    per_axis_bars(by_model_axis_prompt, axes_order, models_order, family_of,
                  fig_dir / "03_per_axis_bars", seed=cfg.seed)
    per_layer_lines(df, axes_order, models_order, family_of,
                    fig_dir / "04a_per_layer_raw")
    per_layer_lines(df, axes_order, models_order, family_of,
                    fig_dir / "04b_per_layer_residual", residual=True)
    model_pca_scatter(by_model_axis, axes_order, models_order, family_of,
                      fig_dir / "06_pca_scatter")
    family_effect_size(by_model_axis_prompt, axes_order, family_of,
                       fig_dir / "07_family_effect_size")

    axis_emergence_table(df, run_dir / "data" / "responses", axes_order, family_of,
                         tbl_dir / "05_axis_emergence.csv")
    trait_extremes_table(by_model_axis_prompt, by_model_axis,
                         run_dir / "data" / "responses",
                         axes_order, family_of, tbl_dir / "08_trait_extremes.csv")
    family_signatures_table(by_model_axis_prompt, by_model_axis,
                            run_dir / "data" / "responses",
                            axes_order, family_of, tbl_dir / "09_family_signatures.csv")
    axis_ladder_table(by_model_axis_prompt, by_model_axis,
                      run_dir / "data" / "responses",
                      axes_order, family_of, tbl_dir / "10_axis_ladder.csv")
    log.info("Done.")


if __name__ == "__main__":
    main()
