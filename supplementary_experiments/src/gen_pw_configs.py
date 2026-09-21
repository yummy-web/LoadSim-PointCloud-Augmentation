"""Generate PointWOLF matched-budget run configs (reviewer supplement R2-3).

Emits configs/PW_POINTWOLF_B<budget>_s<n>.json for the preregistered grid:
  * budgets: 150 (primary), 50 (secondary)         -- both matched to A3
  * seeds:   20260911 / 20260912 / 20260913
=> primary block = 1 method x 1 budget(150) x 3 seeds = 3 runs.
   Passing --include-b50 adds the 3 secondary B=50 runs (6 total).

Design mirrors the FROZEN all-valid A3 B=150 setup so PointWOLF is a controlled
fourth method in the SAME matched-budget comparison:
  M=9, U=1330 (locked), num_points=4096, use_normals=true, 2 classes,
  PointNet++ ('formal'). Comparators RAW_REPEAT/TRAD/LOADSIM at the same
  (B, seed) were already run in A3 and are reused, not re-run.

The PointWOLF candidate bank (pw_assets/pw_bank_v1/) is produced offline by
build_pw_candidate_bank.py; these configs only reference its path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
CONFIG_DIR = BASE / "configs"

METHOD = "POINTWOLF"
SEEDS = {1: 20260911, 2: 20260912, 3: 20260913}
PRIMARY_BUDGET = 150
SECONDARY_BUDGET = 50
MAX_UPDATES_U = 1330

REQUIRED_ARTIFACTS = [
    "resolved_config.json", "selection_manifest.csv", "per_source_exposure.csv",
    "history.json", "metrics.json", "provenance.json",
    "checkpoint_best.pt", "checkpoint_latest.pt",
]


def make_config(budget: int, seed: int) -> dict:
    tag = {20260911: "s1", 20260912: "s2", 20260913: "s3"}[seed]
    run_id = f"PW_{METHOD}_B{budget}_{tag}"
    return {
        "schema_version": "pw-run-config-v1",
        "run_id": run_id,
        "experiment": "PW",
        "entrypoint": "pw_train.py",
        "method": METHOD,
        "budget_B": budget,
        "seed": seed,
        # inputs (relative to revision_v2), identical to A3 all-valid B=150
        "project_root": "..",
        "splits": "manifest/splits_v2.json",
        "source_manifest": "manifest/source_scans_v2.csv",
        "pseudo_label_manifest": "a3_assets/original/pseudo_label_manifest_v3.csv",
        "coordinate_scale_manifest": "manifest/coordinate_scales_v2.csv",
        "pw_bank_dir": "pw_assets/pw_bank_v1",
        "a3_protocol": "configs/a3_pointnetpp_mb_v1.json",
        "quality_config": "configs/quality_v1.yaml",
        "output_dir": f"runs/pw_v1/{run_id}",
        # locked matched-budget protocol
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
        "model": {"backbone": "pointnet2", "num_classes": 2, "use_normals": True},
        "output_contract": {"required_artifacts": REQUIRED_ARTIFACTS},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-b50", action="store_true",
                   help="Also emit the secondary B=50 runs (6 configs total).")
    args = p.parse_args()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    budgets = [PRIMARY_BUDGET] + ([SECONDARY_BUDGET] if args.include_b50 else [])
    written = []
    for budget in budgets:
        for seed in SEEDS.values():
            cfg = make_config(budget, seed)
            path = CONFIG_DIR / f"{cfg['run_id']}.json"
            path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            written.append(path.name)
    print(f"wrote {len(written)} PW configs to {CONFIG_DIR}")
    print(f"method={METHOD} budgets={budgets} U={MAX_UPDATES_U} "
          f"seeds={sorted(SEEDS.values())} => {len(written)} runs")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
