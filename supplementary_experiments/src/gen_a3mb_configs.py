"""Generate A3MB multi-backbone matched-budget run configs (reviewer supplement).

Emits configs/A3MB_<backbone>_<METHOD>_B150_s<n>.json for every
(backbone, method, seed): 5 backbones x 2 methods x 3 seeds = 30 configs.

Design (mirrors the FROZEN all-valid A3 B=150 setup so A3MB is a controlled
extension of the same matched-budget protocol):
  * M=9, B=150, U=1330 (locked), num_points=4096, use_normals=true, 2 classes.
  * methods: RAW_REPEAT (candidate_manifest=null, no eligibility_rule) and
    LOADSIM (all-valid loadsim bundle + eligibility_rule=all-technically-valid-v1
    + catalog_namespace=all_valid_v1) -- byte-identical data path to A3 all-valid.
  * TRAD is intentionally dropped (reviewer's 30-run design).

The candidate bundle (a3_assets/all_valid_v1/loadsim/) lives on the server as a
frozen A3 asset; these configs only reference its path.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
CONFIG_DIR = BASE / "configs"

BACKBONES = ("pointnet2", "pointnext", "kpconv", "ptv3", "randlanet")
METHODS = ("RAW_REPEAT", "LOADSIM")
SEEDS = {1: 20260911, 2: 20260912, 3: 20260913}
BUDGET_B = 150
MAX_UPDATES_U = 1330

# All-valid namespace paths (mirror configs_all_valid_v1/A3_LOADSIM_B150_s1.json).
LOADSIM_CANDIDATE = "a3_assets/all_valid_v1/loadsim/candidate_manifest.csv"
ELIGIBILITY_RULE = "all-technically-valid-v1"
CATALOG_NAMESPACE = "all_valid_v1"

REQUIRED_ARTIFACTS = [
    "resolved_config.json", "selection_manifest.csv", "per_source_exposure.csv",
    "history.json", "metrics.json", "provenance.json",
    "checkpoint_best.pt", "checkpoint_latest.pt",
]


def make_config(backbone: str, method: str, seed: int) -> dict:
    tag = {20260911: "s1", 20260912: "s2", 20260913: "s3"}[seed]
    run_id = f"A3MB_{backbone}_{method}_B{BUDGET_B}_{tag}"
    is_loadsim = method == "LOADSIM"
    cfg = {
        "schema_version": "a3mb-run-config-v1",
        "run_id": run_id,
        "experiment": "A3MB",
        "entrypoint": "a3mb_train.py",
        "backbone": backbone,
        "method": method,
        "seed": seed,
        # inputs (relative to revision_v2), identical to A3 all-valid B=150
        "project_root": "..",
        "splits": "manifest/splits_v2.json",
        "source_manifest": "manifest/source_scans_v2.csv",
        "pseudo_label_manifest": "a3_assets/original/pseudo_label_manifest_v3.csv",
        "coordinate_scale_manifest": "manifest/coordinate_scales_v2.csv",
        "candidate_manifest": LOADSIM_CANDIDATE if is_loadsim else None,
        "a3_protocol": "configs/a3_pointnetpp_mb_v1.json",
        "quality_config": "configs/quality_v1.yaml",
        "output_dir": f"runs/a3mb_v1/{run_id}",
        # locked matched-budget protocol
        "budget_B": BUDGET_B,
        "max_updates": MAX_UPDATES_U,
        "num_points": 4096,
        "batch_size": 8,
        "num_workers": 0,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "lr_min_ratio": 0.01,
        "eval_interval": 100,
        "checkpoint_interval": 100,
        "dev_crops_per_scan": 4,
        "cache_size": 4,
        "device": "cuda",
        "use_normals": True,
        "amp": True,
        "deterministic": True,
        "verify_hashes": True,
        "model_profile": "formal",
        "model": {"backbone": backbone, "num_classes": 2, "use_normals": True},
        "output_contract": {"required_artifacts": REQUIRED_ARTIFACTS},
    }
    if is_loadsim:
        cfg["eligibility_rule"] = ELIGIBILITY_RULE
        cfg["catalog_namespace"] = CATALOG_NAMESPACE
    return cfg


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for backbone in BACKBONES:
        for method in METHODS:
            for seed in SEEDS.values():
                cfg = make_config(backbone, method, seed)
                path = CONFIG_DIR / f"{cfg['run_id']}.json"
                path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
                written.append(path.name)
    print(f"wrote {len(written)} A3MB configs to {CONFIG_DIR}")
    print(f"backbones={BACKBONES} methods={METHODS} B={BUDGET_B} U={MAX_UPDATES_U} "
          f"seeds={sorted(SEEDS.values())}  => {len(written)} runs")


if __name__ == "__main__":
    main()
