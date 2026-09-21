"""Generate A4B budget-curve run configs (Reviewer-3 scheme 1).

Self-contained: emits configs/A4B_<METHOD>_B<budget>_<seed>.json for every
(method, budget, seed) in the grid. No dependency on the frozen A3/A4 contract
generator. Run: python gen_a4b_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
CONFIG_DIR = BASE / "configs"

METHODS = ("BASELINE", "TRADITIONAL", "LOADSIM")
BUDGETS = (100, 400, 800, 1600, 3200)
SEEDS = {1: 20260911, 2: 20260912, 3: 20260913}

REQUIRED_ARTIFACTS = [
    "status.json", "resolved_config.json", "split_manifest.json",
    "history.json", "metrics.json", "provenance.json",
    "per_frame_block_metrics.jsonl", "checkpoints/best.pt",
]


def make_config(method: str, budget: int, seed: int) -> dict:
    run_id = f"A4B_{method}_B{budget}_{seed}"
    return {
        "schema_version": "lsda-revision-v2-run-v2",
        "run_id": run_id,
        "experiment": "A4B",
        "entrypoint": "a4b_budget_train.py",
        "method": method,
        "seed": seed,
        "paths": {
            "kitti_root": "../kitti_experiment/data",
            "kitti_config": "configs/kitti_v1.yaml",
        },
        "protocol": {
            "budget_B": budget,
            "val_frames": 600,
            "test_frames": 741,
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
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for method in METHODS:
        for budget in BUDGETS:
            for seed in SEEDS.values():
                cfg = make_config(method, budget, seed)
                path = CONFIG_DIR / f"{cfg['run_id']}.json"
                path.write_text(
                    json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
                written.append(path.name)
    print(f"wrote {len(written)} A4B configs to {CONFIG_DIR}")
    for name in written:
        print(name)


if __name__ == "__main__":
    main()
