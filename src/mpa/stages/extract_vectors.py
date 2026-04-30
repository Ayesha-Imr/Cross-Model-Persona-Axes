from __future__ import annotations

import time

import torch
from tqdm import tqdm

from ..checkpoint import append_jsonl, load_done_keys, read_jsonl
from ..config import AxisEntry, Config, ModelEntry
from ..judge import score_batch
from ..logging import setup_logging
from ..prober import make_prober
from ..providers.hf_local import HFLocalGen
from ._common import base_parser, load_cfg_and_run_dir


def _gen_key(rec: dict) -> str:
    return f"{rec['axis']}|{rec['polarity']}|{rec['seed_idx']}|{rec['sample_idx']}"


def _cached_axis_names(run_dir) -> set[str]:
    cache_dir = run_dir / "artifacts" / "direction_cache"
    if not cache_dir.exists():
        return set()
    return {p.stem for p in cache_dir.glob("*.pt")}


def _generate_contrastive(cfg: Config, run_dir, axes_to_run: list[AxisEntry], log) -> None:
    if not axes_to_run:
        return
    prober_id = cfg.prober.model_id
    seeds = cfg.contrastive.seed_prompts[: cfg.contrastive.n_seed_prompts]
    if not seeds:
        raise SystemExit("contrastive.seed_prompts is empty in config.")

    gen_model = ModelEntry(
        name="prober_gen", provider="hf_local", model_id=prober_id, family="prober",
    )
    gen_model.gen.temperature = 1.0
    gen_model.gen.max_tokens = cfg.prober.max_response_tokens

    gen = HFLocalGen(gen_model)
    try:
        for axis in axes_to_run:
            for polarity, sysprompt in (("pos", axis.pos_system), ("neg", axis.neg_system)):
                out = run_dir / "data" / "contrastive" / axis.name / f"{polarity}.jsonl"
                done = load_done_keys(out, _gen_key)
                target = axis.n_candidates
                per_seed = max(1, target // len(seeds))
                pending = []
                for si, seed in enumerate(seeds):
                    for k in range(per_seed):
                        key = f"{axis.name}|{polarity}|{si}|{k}"
                        if key not in done:
                            pending.append((si, seed, k))
                if not pending:
                    continue
                log.info("[contrast] %s/%s: %d to generate", axis.name, polarity, len(pending))
                for si, seed, k in tqdm(pending, desc=f"{axis.name}/{polarity}"):
                    r = gen.generate(seed, system=sysprompt)
                    append_jsonl(out, {
                        "axis": axis.name, "polarity": polarity,
                        "seed_idx": si, "sample_idx": k,
                        "system": sysprompt, "prompt": seed, "completion": r.text,
                        "ts": time.time(),
                    })
    finally:
        gen.close()


def _judge_filter(cfg: Config, run_dir, axes_to_run: list[AxisEntry], log) -> dict[str, dict[str, list[dict]]]:
    """Score candidates and filter; only processes axes_to_run."""
    cache_dir = run_dir / "artifacts" / "judge_scores_per_record"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(rec: dict) -> str:
        return f"{rec['axis']}|{rec['polarity']}|{rec['seed_idx']}|{rec['sample_idx']}"

    kept: dict[str, dict[str, list[dict]]] = {}

    for axis in axes_to_run:
        kept[axis.name] = {"pos": [], "neg": []}
        cache_path = cache_dir / f"{axis.name}.jsonl"
        cached = {f"{r['axis']}|{r['polarity']}|{r['seed_idx']}|{r['sample_idx']}": r
                  for r in read_jsonl(cache_path)}

        for polarity in ("pos", "neg"):
            in_path = run_dir / "data" / "contrastive" / axis.name / f"{polarity}.jsonl"
            recs = read_jsonl(in_path)
            if not recs:
                continue
            pending = [r for r in recs if _cache_key({**r, "polarity": polarity}) not in cached]
            log.info("[judge] %s/%s: %d cached, %d to score",
                     axis.name, polarity, len(recs) - len(pending), len(pending))

            scored: list[dict] = []
            if pending:
                scored = score_batch(
                    [{"prompt": r["prompt"], "completion": r["completion"]} for r in pending],
                    trait_noun=axis.trait_noun,
                    model_id=cfg.judge.model,
                    min_prob=cfg.judge.min_prob,
                    max_workers=cfg.judge.max_workers,
                )
                with open(cache_path, "a") as f:
                    import json as _json
                    for rec, sc in zip(pending, scored):
                        row = {
                            "axis": axis.name, "polarity": polarity,
                            "seed_idx": rec["seed_idx"], "sample_idx": rec["sample_idx"],
                            "score": sc["score"], "in_tok": sc["in_tok"], "out_tok": sc["out_tok"],
                        }
                        cached[_cache_key(row)] = row
                        f.write(_json.dumps(row) + "\n")

            thr_keep = (axis.pos_threshold if polarity == "pos" else axis.neg_threshold)
            for rec in recs:
                key = f"{axis.name}|{polarity}|{rec['seed_idx']}|{rec['sample_idx']}"
                row = cached[key]
                s = row.get("score")
                if s is not None:
                    keep = (s >= thr_keep) if polarity == "pos" else (s <= thr_keep)
                    if keep:
                        kept[axis.name][polarity].append({**rec, "judge_score": s})
            log.info("  kept %d/%d", len(kept[axis.name][polarity]), len(recs))
    return kept


def _compute_vectors(cfg: Config, run_dir, kept, log) -> None:
    cache_dir = run_dir / "artifacts" / "direction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # load existing per-axis caches
    cached_dirs: dict[str, dict] = {}
    for pt_file in cache_dir.glob("*.pt"):
        d = torch.load(pt_file, map_location="cpu", weights_only=False)
        cached_dirs[d["axis"]] = d

    new_axes = [a for a in cfg.axes if a.name not in cached_dirs]
    if new_axes:
        log.info("Need prober for %d new axis direction(s): %s",
                 len(new_axes), [a.name for a in new_axes])
        prober = make_prober(cfg.prober)
        try:
            L, H = prober.num_layers, prober.hidden_dim
            for axis in new_axes:
                pos_texts = [r["completion"] for r in kept[axis.name]["pos"]]
                neg_texts = [r["completion"] for r in kept[axis.name]["neg"]]
                if not pos_texts or not neg_texts:
                    log.warning("[vec] %s: no kept samples (pos=%d neg=%d); zero direction.",
                                axis.name, len(pos_texts), len(neg_texts))
                    continue
                pos = prober.encode(pos_texts).mean(dim=0)
                neg = prober.encode(neg_texts).mean(dim=0)
                d = pos - neg
                d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                entry = {"axis": axis.name, "direction": d,
                         "n_pos": len(pos_texts), "n_neg": len(neg_texts),
                         "L": int(d.shape[0]), "H": int(d.shape[1])}
                torch.save(entry, cache_dir / f"{axis.name}.pt")
                cached_dirs[axis.name] = entry
                log.info("[vec] %s: pos=%d neg=%d (cached)", axis.name, len(pos_texts), len(neg_texts))
        finally:
            prober.close()
    else:
        log.info("All axis directions already cached; assembling from cache.")

    # assemble full tensor from per-axis caches
    ref = next(iter(cached_dirs.values()))
    L, H = ref["L"], ref["H"]
    directions = torch.zeros(len(cfg.axes), L, H)
    meta = []
    for ai, axis in enumerate(cfg.axes):
        if axis.name not in cached_dirs:
            continue
        entry = cached_dirs[axis.name]
        directions[ai] = entry["direction"]
        meta.append({"axis": axis.name, "n_pos": entry["n_pos"], "n_neg": entry["n_neg"]})

    torch.save({
        "directions": directions,
        "axes": [a.name for a in cfg.axes],
        "num_layers": L, "hidden_dim": H,
        "prober_id": cfg.prober.model_id,
        "meta": meta,
    }, run_dir / "artifacts" / "persona_vectors.pt")
    log.info("Wrote persona_vectors.pt shape=(%d,%d,%d)", len(cfg.axes), L, H)


def _needs_recompute(cfg: Config, run_dir) -> bool:
    pv_path = run_dir / "artifacts" / "persona_vectors.pt"
    if not pv_path.exists():
        return True
    pv = torch.load(pv_path, map_location="cpu", weights_only=False)
    saved_axes = set(pv.get("axes", []))
    wanted_axes = {a.name for a in cfg.axes}
    return wanted_axes != saved_axes


def main():
    args = base_parser("extract_vectors").parse_args()
    cfg, run_dir = load_cfg_and_run_dir(args)
    log = setup_logging(run_dir, "extract_vectors")

    if not _needs_recompute(cfg, run_dir) and not args.force:
        log.info("persona_vectors.pt is up-to-date with config axes; skipping.")
        return

    cached = _cached_axis_names(run_dir) if not args.force else set()
    new_axes = [a for a in cfg.axes if a.name not in cached]
    if not new_axes and not args.force:
        log.info("All %d axes already have cached directions; assembling persona_vectors.pt.",
                 len(cfg.axes))
        _compute_vectors(cfg, run_dir, {}, log)
        return

    if args.dry_run:
        n = sum(a.n_candidates * 2 for a in new_axes)
        log.info("[dry] would generate %d contrastive completions + %d judge calls for %s",
                 n, n, [a.name for a in new_axes])
        return

    log.info("New axes to extract: %s (cached: %s)", [a.name for a in new_axes], sorted(cached))
    _generate_contrastive(cfg, run_dir, new_axes, log)
    kept = _judge_filter(cfg, run_dir, new_axes, log)
    _compute_vectors(cfg, run_dir, kept, log)


if __name__ == "__main__":
    main()
