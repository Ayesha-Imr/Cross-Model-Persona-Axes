"""One-shot: split an existing persona_vectors.pt into per-axis direction_cache/ files.

Run once on your existing run dir so that extract_vectors recognizes the 7
original axes as already cached and only encodes the new one through the prober.

Usage:
    python scripts/seed_direction_cache.py --run-dir runs/<your_run_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    pv_path = run_dir / "artifacts" / "persona_vectors.pt"
    pv = torch.load(pv_path, map_location="cpu", weights_only=False)

    cache_dir = run_dir / "artifacts" / "direction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    directions = pv["directions"]
    axes = pv["axes"]
    meta_list = {m["axis"]: m for m in pv.get("meta", [])}

    for ai, name in enumerate(axes):
        out = cache_dir / f"{name}.pt"
        if out.exists():
            print(f"  {name}: already cached, skipping")
            continue
        d = directions[ai]
        m = meta_list.get(name, {})
        torch.save({
            "axis": name, "direction": d,
            "n_pos": m.get("n_pos", 0), "n_neg": m.get("n_neg", 0),
            "L": int(d.shape[0]), "H": int(d.shape[1]),
        }, out)
        print(f"  {name}: cached ({d.shape[0]} layers, {d.shape[1]} hidden)")

    print(f"\nDone. {len(axes)} axes cached in {cache_dir}")


if __name__ == "__main__":
    main()
