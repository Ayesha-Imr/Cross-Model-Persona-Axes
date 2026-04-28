# Commands & resumption guide

All commands assume `cd cross-model-persona-axes`, `uv sync` is done, and `.env` is filled with the API keys for whichever models are `enabled: true` in the config. Replace `config.yaml` with `configs/smoke.yaml` for a fast end-to-end check.

## Quick paths

```bash
make smoke            # 2 small models × 5 prompts × 2 axes — full pipeline, ~5 min
make full             # config.yaml — full run
make dryrun           # cost preview without API spend
make figures          # re-run only the visualize stage
```

## End-to-end orchestrator (`mpa.run`)

```bash
uv run python -m mpa.run --config config.yaml
```

| Flag | Behavior |
|------|----------|
| `--config <path>` | required |
| `--only <stage>` | run exactly one stage; `<stage>` ∈ {sample, extract, generate, project, visualize} |
| `--from <stage>` | start at this stage and continue through visualize |
| `--run-dir <path>` | use a specific run directory (defaults to the latest matching the config hash; creates a new one otherwise) |
| `--reuse-vectors-from <run_dir>` | copy `persona_vectors.pt` + judge audit + contrastive completions from an old run dir into the current one. `extract_vectors` then short-circuits |
| `--force` | propagate `--force` to each stage (recompute even if outputs exist) |
| `--dry-run` | propagate `--dry-run`; print projected work/cost without API spend |

## Stage-by-stage CLI

Every stage accepts `--config <path> [--run-dir <path>] [--force] [--dry-run]`.

| Stage | Standalone command |
|-------|--------------------|
| Sample prompts | `python -m mpa.stages.sample_prompts --config config.yaml` |
| Extract persona vectors | `python -m mpa.stages.extract_vectors --config config.yaml` |
| Generate responses | `python -m mpa.stages.generate_responses --config config.yaml` |
| Project | `python -m mpa.stages.project --config config.yaml` |
| Visualize | `python -m mpa.stages.visualize --config config.yaml` |

### What each stage produces (under `runs/<run_dir>/`)

| Stage | Outputs |
|-------|---------|
| `sample_prompts` | `data/prompts.jsonl` |
| `extract_vectors` | `data/contrastive/<axis>/{pos,neg}.jsonl`, `artifacts/judge_scores_per_record/<axis>.jsonl`, `artifacts/judge_scores.parquet`, `artifacts/persona_vectors.pt` |
| `generate_responses` | `data/responses/<model>.jsonl`, `data/raw_api_calls/<provider>.jsonl` |
| `project` | `artifacts/projections_per_model/<model>.parquet`, `artifacts/projections.parquet` |
| `visualize` | `figures/*.png` + `figures/*.pdf` |

## Run-dir hashing — what does and doesn't bump the hash

`Config.short_hash()` (used as the `_<8 hex>` suffix on `runs/<…>`) hashes only the fields that actually invalidate shared artifacts: `seed`, `prober`, `judge`, `prompts`, `contrastive`, `axes`. The `models` list, `pricing`, `cost_cap_usd`, and `output_root` are deliberately excluded.

Practical consequence: toggling `enabled: true|false` on a model, adding a new model entry, removing one, or tweaking generation params **does not** create a new run dir. The same run dir is reused — already-done models stay skipped, newly-enabled ones get their responses generated.

Changing `prompts.source`, axes, prober, judge, contrastive seed prompts, or the global seed **does** bump the hash.

## Reusing persona vectors after an axis or prompt-set change

If a config change bumps the run-dir hash, the new run dir starts empty — but the existing persona vectors may still be valid (e.g. only `prompts.source` changed). To carry them over:

```bash
# 1. fresh prompt sample with the new dataset:
uv run python -m mpa.run --config config.yaml --only sample

# 2. import vectors from the previous run:
uv run python -m mpa.run --config config.yaml \
    --reuse-vectors-from runs/<old_run_dir> --only extract

# 3. continue:
uv run python -m mpa.run --config config.yaml --from generate
```

What gets copied: `artifacts/persona_vectors.pt`, `artifacts/judge_scores.parquet`, `artifacts/judge_scores_per_record/`, `data/contrastive/`. Anything already present in the new run dir is left alone.

## Failure modes & resumption

The pipeline is designed so that **rerunning the same command after any crash is safe** — every stage skips already-completed work using on-disk artifacts.

### Crash during `extract_vectors`

Persisted incrementally:
1. Each contrastive completion → `data/contrastive/<axis>/{pos,neg}.jsonl`, keyed by `(axis, polarity, seed_idx, sample_idx)`.
2. Each judge score → `artifacts/judge_scores_per_record/<axis>.jsonl`.
3. `persona_vectors.pt` is written only at the end.

Recovery: `uv run python -m mpa.run --config config.yaml --only extract`.

### Crash during `generate_responses`

Each `(model, prompt_id)` is appended to `data/responses/<model>.jsonl` immediately after generation. Models whose responses are 100% complete have `pending=[]` and don't even load — no GPU time spent on already-finished local models. The crashing model resumes at the first un-done prompt.

Recovery: `uv run python -m mpa.run --config config.yaml --only generate`.

### Crash during `project`

Per-model parquets at `artifacts/projections_per_model/<model>.parquet`. Already-projected models are skipped (the prober isn't loaded if everything's done); the crashing model is reprojected from its raw responses.

Recovery: `uv run python -m mpa.run --config config.yaml --only project`.

### Cost-cap halt

`cost_cap_usd` is checked after every API call. If it trips, the run halts cleanly with all artifacts on disk. Raise the cap in `config.yaml` and rerun the same command.

## Recompute one specific axis or model

There's no `--axis X` flag. Cleanest options:

- **One axis:** edit `config.yaml` to leave only the axis of interest, then `--only extract --force`. Persona vectors for the rest stay valid in their own run dir.
- **One model:** delete `data/responses/<model>.jsonl` and `artifacts/projections_per_model/<model>.parquet` from the run dir, then `--only generate` and `--only project`.

## Inspecting artifacts

```bash
# How many candidates the judge kept per (axis, polarity):
uv run python -c "import pandas as pd; df = pd.read_parquet('runs/<run_dir>/artifacts/judge_scores.parquet'); print(df.groupby(['axis','polarity'])['kept'].agg(['sum','count']))"

# Headline projections (mean over layers + prompts):
uv run python -c "import pandas as pd; df = pd.read_parquet('runs/<run_dir>/artifacts/projections.parquet'); print(df.groupby(['model','axis'])['score'].mean().unstack().round(2))"
```
