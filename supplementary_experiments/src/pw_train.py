"""CLI for PointWOLF matched-budget training (reviewer supplement R2-3).

Light process (same posture as A3MB/A4B/V3D): reads a plain JSON run config
emitted by gen_pw_configs.py and calls the isolated PointWOLF engine. PointWOLF
is an EXTRA exploratory supplement -- it does NOT carry A4's dual-factor
authorization / contract-SHA lock / independent-review ceremony. The scientific
red lines are kept: matched M/B/U (M=9, B in {50,150}, U=1330), fixed DEV,
labels propagate by identity (warp preserves point count/order), honest framing.

The comparators (RAW_REPEAT/TRAD/LOADSIM at the same B/seed) were already run in
A3 and are REUSED, not re-run.

Usage (per run):
  python pw_train.py --config configs/PW_POINTWOLF_B150_s1.json
  python pw_train.py --config configs/PW_POINTWOLF_B150_s1.json --debug
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Deterministic algorithms + CUDA matmul require this cuBLAS workspace setting;
# must be set before torch is imported (via the engine below). Mirrors a4_train.py L19.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from a3_io import A3Error, load_json
from pw_train_engine import PW_METHODS, PW_ALLOWED_BUDGETS, SEED_TAGS, train_pw

# Fields the engine consumes; copied from the flattened config verbatim.
ENGINE_FIELDS = {
    "project_root", "splits", "source_manifest", "pseudo_label_manifest",
    "coordinate_scale_manifest", "output_dir", "a3_protocol", "quality_config",
    "pw_bank_dir", "budget_B", "max_updates", "batch_size", "num_points",
    "num_workers", "learning_rate", "weight_decay", "lr_min_ratio",
    "eval_interval", "checkpoint_interval", "dev_crops_per_scan", "cache_size",
    "device", "use_normals", "amp", "deterministic", "verify_hashes",
    "model_profile", "method", "seed", "run_id",
}
PATH_FIELDS = {
    "project_root", "splits", "source_manifest", "pseudo_label_manifest",
    "coordinate_scale_manifest", "output_dir", "a3_protocol", "quality_config",
    "pw_bank_dir",
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="PointWOLF JSON run config (from gen_pw_configs.py).")
    p.add_argument("--method", choices=list(PW_METHODS))
    p.add_argument("--budget", type=int, choices=list(PW_ALLOWED_BUDGETS))
    p.add_argument("--seed", type=int, choices=list(SEED_TAGS))
    p.add_argument("--run-dir", type=Path, help="Override output_dir (must end in run_id).")
    p.add_argument("--resume", type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="Real-data short mode; metrics stay model-derived, never simulated.")
    return p


def _config(args: argparse.Namespace) -> dict[str, Any]:
    base = Path(__file__).resolve().parent
    raw = load_json(args.config.expanduser().resolve())
    if raw.get("schema_version") != "pw-run-config-v1":
        raise A3Error("pw_train accepts only schema_version=pw-run-config-v1 configs")
    if raw.get("experiment") != "PW":
        raise A3Error("pw_train accepts only experiment=PW configs")
    config = {key: raw[key] for key in ENGINE_FIELDS if key in raw}
    # Config paths are relative to revision_v2, never the caller's cwd.
    for key in PATH_FIELDS:
        value = config.get(key)
        if value is not None:
            path = Path(value).expanduser()
            config[key] = str(path.resolve() if path.is_absolute() else (base / path).resolve())
    # Identity CLI args, when supplied, must match the frozen config.
    for key, attr in (("method", "method"), ("budget_B", "budget"), ("seed", "seed")):
        cli_value = getattr(args, attr)
        if cli_value is not None and cli_value != config.get(key):
            raise A3Error(f"CLI {attr}={cli_value!r} does not match config {key}={config.get(key)!r}")
    if args.run_dir is not None:
        config["output_dir"] = str(args.run_dir.expanduser().resolve())
    if args.debug:
        config.update({
            "debug": True, "model_profile": "smoke", "max_updates": 2,
            "batch_size": 1, "num_points": 64, "eval_interval": 1,
            "checkpoint_interval": 1, "dev_crops_per_scan": 1,
            "cache_size": 1, "amp": False,
        })
        config.setdefault("device", "cuda")
    return config


def main() -> int:
    args = _parser().parse_args()
    try:
        result = train_pw(_config(args), resume=args.resume,
                          preflight_only=args.preflight_only)
    except (A3Error, ImportError, OSError, RuntimeError, ValueError) as exc:
        _parser().error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
