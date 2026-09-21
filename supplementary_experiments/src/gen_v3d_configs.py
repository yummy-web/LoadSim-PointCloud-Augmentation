"""Generate V3D cross-domain budget-curve run configs.

Emits configs/V3D_<METHOD>_B<budget>_<seed>.json for every (method, budget,
seed). Self-contained (no A3/A4 contract dependency).

Budget grid: V3D is a single ALS scene tiled into blocks, so the train-pool
size (n_train blocks) is only known AFTER v3d_prepare.py runs. This generator
therefore:
  * reads ../v3d_experiment/manifest.json when present and drops any budget
    that exceeds the actual train-block count (keeping the largest = full pool);
  * otherwise falls back to a default grid and prints a reminder to re-run after
    preparing blocks.

Run:  python gen_v3d_configs.py [--manifest ../v3d_experiment/manifest.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
CONFIG_DIR = BASE / "configs"

METHODS = ("BASELINE", "TRADITIONAL", "LOADSIM", "LOADSIM_GEOM")
# Default log-spaced grid; clamped to the real train-pool size at generation
# time if a manifest is available.
# Sized for the real V3D train-pool (~68 blocks at 30 m tiles). The clamp below
# keeps every budget < n_train and always appends the full pool, so this yields
# a clean 4-point grid {10, 20, 40, n_train}. If you re-tile at a different
# block-size and n_train changes a lot, revisit these anchors.
DEFAULT_BUDGETS = (10, 20, 40)
SEEDS = {1: 20260911, 2: 20260912, 3: 20260913}

REQUIRED_ARTIFACTS = [
    "status.json", "resolved_config.json", "split_manifest.json",
    "history.json", "metrics.json", "provenance.json",
    "per_frame_block_metrics.jsonl", "checkpoints/best.pt",
]


def resolve_budgets(manifest_path: Path) -> tuple[tuple[int, ...], int | None]:
    """Return (budgets, n_train). Clamp the grid to n_train if manifest exists."""
    if not manifest_path.is_file():
        return DEFAULT_BUDGETS, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_train = int(manifest.get("split_counts", {}).get("train", 0))
    if n_train < 2:
        raise SystemExit(f"manifest train count too small: {n_train}")
    budgets = sorted({b for b in DEFAULT_BUDGETS if b < n_train} | {n_train})
    budgets = tuple(b for b in budgets if b >= 2)
    return budgets, n_train


def make_config(method: str, budget: int, seed: int) -> dict:
    run_id = f"V3D_{method}_B{budget}_{seed}"
    return {
        "schema_version": "lsda-revision-v2-run-v2",
        "run_id": run_id,
        "experiment": "V3D",
        "entrypoint": "v3d_budget_train.py",
        "method": method,
        "seed": seed,
        "paths": {
            "v3d_root": "../v3d_experiment",
            "v3d_config": "configs/v3d_v1.yaml",
        },
        "protocol": {
            "budget_B": budget,
            "max_updates_U": 20000,
            "num_points": 4096,
        },
        "training": {
            "epochs": 50,
            "batch_size": 8,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "lr_scheduler": "CosineAnnealingLR",
            "lr_step_size": 15,
            "lr_gamma": 0.5,
            "device": "cuda",
        },
        "model": {
            "backbone": "PointNet++",
            "num_classes": 2,
            "use_normals": False,
        },
        "output_contract": {"required_artifacts": REQUIRED_ARTIFACTS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=BASE.parent / "v3d_experiment" / "manifest.json",
                        help="prepared V3D manifest.json (to clamp budgets).")
    args = parser.parse_args()

    budgets, n_train = resolve_budgets(args.manifest)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for method in METHODS:
        for budget in budgets:
            for seed in SEEDS.values():
                cfg = make_config(method, budget, seed)
                path = CONFIG_DIR / f"{cfg['run_id']}.json"
                path.write_text(
                    json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
                written.append(path.name)
    print(f"wrote {len(written)} V3D configs to {CONFIG_DIR}")
    print(f"budget grid: {budgets}"
          + (f"  (clamped to n_train={n_train})" if n_train else
             "  (DEFAULT grid; manifest not found -- re-run after v3d_prepare.py "
             "to clamp to the real train-block count)"))


if __name__ == "__main__":
    main()
