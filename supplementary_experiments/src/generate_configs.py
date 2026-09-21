"""Generate and validate the frozen A3/A4 run catalogue.

The JSON files and jobs.csv are generated together.  Paths stored in either
format are project-relative POSIX paths so the same bundle works on Windows
and Linux.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from a3_io import PSEUDO_LABEL_MANIFEST_FILENAME

SCHEMA_VERSION = "lsda-revision-v2-run-v2"
SEEDS = (20260911, 20260912, 20260913)
A3_METHOD_BUDGET_GRID = {
    "RAW_REPEAT": (11, 25, 50, 75, 100, 150),
    "TRAD": (11, 50, 150),
    "LOADSIM": (11, 25, 50, 75, 100, 150),
}
A3_BUDGETS = (11, 25, 50, 75, 100, 150)
A3_METHODS = tuple(A3_METHOD_BUDGET_GRID)
A3_ALLOWED_METHOD_BUDGETS = frozenset(
    (method, budget)
    for method, budgets in A3_METHOD_BUDGET_GRID.items()
    for budget in budgets
)
A3_ESTIMATED_HOURS = {
    11: 2.5, 25: 3.0, 50: 3.5, 75: 4.0, 100: 4.5, 150: 5.0,
}
A4_ESTIMATED_HOURS = 4.0
A3_RUN_COUNT = 45
A4_RUN_COUNT = 9
TOTAL_RUN_COUNT = 54
A4_METHODS = ("BASELINE", "TRADITIONAL", "LOADSIM")
A3_PSEUDO_LABEL_MANIFEST = (
    f"a3_assets/original/{PSEUDO_LABEL_MANIFEST_FILENAME}"
)
A3_COORDINATE_SCALE_MANIFEST = "manifest/coordinate_scales_v2.csv"
COMMON_CONFIG_KEYS = {
    "schema_version", "run_id", "experiment", "entrypoint", "method",
    "seed", "paths", "protocol", "training", "model", "output_contract",
}
A3_ENGINE_KEYS = {
    "project_root", "splits", "source_manifest", "pseudo_label_manifest",
    "coordinate_scale_manifest",
    "candidate_manifest", "output_dir", "a3_protocol", "quality_config", "budget_B",
    "max_updates", "batch_size", "num_points", "num_workers",
    "learning_rate", "weight_decay", "lr_min_ratio", "eval_interval", "checkpoint_interval",
    "dev_crops_per_scan", "cache_size", "device", "use_normals", "amp",
    "deterministic", "verify_hashes", "model_profile",
}
PATH_KEYS = {
    "A3": {"project_root", "splits", "source_manifest", "pseudo_label_manifest",
           "coordinate_scale_manifest",
           "candidate_manifest_trad", "candidate_manifest_loadsim", "a3_protocol",
           "quality_config"},
    "A4": {"kitti_root", "kitti_config"},
}
PROTOCOL_KEYS = {
    "A3": {"source_scans_M", "budget_B", "max_updates_U", "num_points",
           "allow_repeat"},
    "A4": {"train_frames", "val_frames", "test_frames", "max_updates_U",
           "num_points"},
}
TRAINING_KEYS = {
    "A3": {"max_updates", "batch_size", "optimizer", "learning_rate",
           "weight_decay", "lr_scheduler", "lr_min_ratio", "device"},
    "A4": {"epochs", "batch_size", "optimizer", "learning_rate",
           "weight_decay", "lr_scheduler", "lr_step_size", "lr_gamma", "device"},
}
MODEL_KEYS = {"backbone", "num_classes", "use_normals"}
OUTPUT_KEYS = {"required_artifacts"}
JOB_FIELDS = ("run_id", "experiment", "method", "budget_B", "seed",
              "config_path", "priority", "estimated_hours")


class ContractError(ValueError):
    """Raised when a run config or job row violates the frozen contract."""


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{where} fields mismatch; missing={sorted(expected-actual)}, "
            f"unknown={sorted(actual-expected)}"
        )
    return value

def _require_int(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, where: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ContractError(f"{where} must be finite and >= {minimum}")
    return value


def _portable_relative(value: Any, where: str, allow_parent: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a non-empty string")
    if "\\" in value or PurePosixPath(value).is_absolute() or re.match(r"^[A-Za-z]:", value):
        raise ContractError(f"{where} must be a relative POSIX path: {value!r}")
    parts = PurePosixPath(value).parts
    if any(part in {"", "."} for part in parts):
        raise ContractError(f"{where} is not normalized: {value!r}")
    if not allow_parent and ".." in parts:
        raise ContractError(f"{where} must not escape its root: {value!r}")
    return value


def validate_config(config: Any, expected_filename: str | None = None) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ContractError("config must be an object")
    experiment = config.get("experiment")
    if experiment not in {"A3", "A4"}:
        raise ContractError(f"unknown experiment: {experiment!r}")
    expected_keys = COMMON_CONFIG_KEYS | (A3_ENGINE_KEYS if experiment == "A3" else set())
    cfg = _exact_keys(config, expected_keys, "config")
    if cfg["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"unsupported schema_version: {cfg['schema_version']!r}")
    run_id = cfg["run_id"]
    # Canonical IDs use uppercase experiment/method tokens and a lowercase
    # seed suffix (s1/s2/s3). make_run_id() below is the single constructor.
    if not isinstance(run_id, str) or not re.fullmatch(
            r"A[34]_[A-Z0-9_]+_s[123]", run_id):
        raise ContractError(f"invalid run_id: {run_id!r}")
    if expected_filename and expected_filename != f"{run_id}.json":
        raise ContractError(f"filename/run_id mismatch: {expected_filename!r} != {run_id!r}.json")

    if cfg["entrypoint"] != f"{experiment.lower()}_train.py":
        raise ContractError("entrypoint does not match experiment")
    allowed_methods = A3_METHODS if experiment == "A3" else A4_METHODS
    if cfg["method"] not in allowed_methods:
        raise ContractError(f"invalid {experiment} method: {cfg['method']!r}")
    if cfg["seed"] not in SEEDS:
        raise ContractError(f"seed is not frozen: {cfg['seed']!r}")

    paths = _exact_keys(cfg["paths"], PATH_KEYS[experiment], "paths")
    for key, value in paths.items():
        _portable_relative(value, f"paths.{key}", allow_parent=True)
    protocol = _exact_keys(cfg["protocol"], PROTOCOL_KEYS[experiment], "protocol")
    for key, value in protocol.items():
        if key == "allow_repeat":
            if not isinstance(value, bool):
                raise ContractError("protocol.allow_repeat must be boolean")
        else:
            _require_int(value, f"protocol.{key}", 1)

    training = _exact_keys(cfg["training"], TRAINING_KEYS[experiment], "training")
    _require_int(training["batch_size"], "training.batch_size", 1)
    for key in ("learning_rate", "weight_decay"):
        _require_number(training[key], f"training.{key}", 0.0)
    if training["optimizer"] != "Adam" or training["device"] != "cuda":
        raise ContractError("only Adam on CUDA is allowed")
    if experiment == "A3":
        if (training["lr_scheduler"] != "CosineAnnealingLR" or
                training["max_updates"] != 1330 or
                training["lr_min_ratio"] != 0.01):
            raise ContractError("A3 fixed-update cosine schedule mismatch")
    else:
        for key in ("epochs", "lr_step_size"):
            _require_int(training[key], f"training.{key}", 1)
        _require_number(training["lr_gamma"], "training.lr_gamma", 0.0)
        if training["lr_scheduler"] != "StepLR":
            raise ContractError("A4 lr_scheduler must be StepLR")

    model = _exact_keys(cfg["model"], MODEL_KEYS, "model")
    if model["backbone"] != "PointNet++" or model["num_classes"] != 2:
        raise ContractError("model must be binary PointNet++")
    if not isinstance(model["use_normals"], bool):
        raise ContractError("model.use_normals must be boolean")
    output = _exact_keys(cfg["output_contract"], OUTPUT_KEYS, "output_contract")
    artifacts = output["required_artifacts"]
    if not isinstance(artifacts, list) or not artifacts or len(artifacts) != len(set(artifacts)):
        raise ContractError("required_artifacts must be a non-empty unique list")
    for index, value in enumerate(artifacts):
        _portable_relative(value, f"required_artifacts[{index}]")

    expected = make_run_id(experiment, cfg["method"], cfg["seed"], protocol.get("budget_B"))
    if run_id != expected:
        raise ContractError(f"run_id is not canonical: {run_id!r} != {expected!r}")
    if experiment == "A3":
        method_budget = (cfg["method"], protocol["budget_B"])
        if protocol["source_scans_M"] != 9 or method_budget not in A3_ALLOWED_METHOD_BUDGETS:
            allowed = ", ".join(
                f"{method}={list(budgets)}"
                for method, budgets in A3_METHOD_BUDGET_GRID.items()
            )
            raise ContractError(
                f"A3 requires M=9 and an allowed method-budget combination ({allowed})"
            )
        if protocol["max_updates_U"] != 1330 or protocol["num_points"] != 4096:
            raise ContractError("A3 requires U=1330 and num_points=4096")
        if protocol["allow_repeat"] != (cfg["method"] == "RAW_REPEAT"):
            raise ContractError("A3 allow_repeat is inconsistent with method")
        required = {
            "resolved_config.json", "selection_manifest.csv",
            "per_source_exposure.csv", "history.json", "metrics.json",
            "provenance.json", "checkpoint_best.pt", "checkpoint_latest.pt",
        }
        _validate_a3_engine_fields(cfg)
    else:
        expected_protocol = {
            "train_frames": 3200, "val_frames": 600, "test_frames": 741,
            "max_updates_U": 20000, "num_points": 4096,
        }
        if protocol != expected_protocol:
            raise ContractError("A4 protocol differs from the frozen chronology/update contract")
        if cfg["model"]["use_normals"] is not False:
            raise ContractError("A4 must not use normals")
        required = {
            "status.json", "resolved_config.json", "split_manifest.json",
            "history.json", "metrics.json", "provenance.json",
            "per_frame_block_metrics.jsonl", "checkpoints/best.pt",
        }
    if set(artifacts) != required:
        raise ContractError(
            f"required_artifacts mismatch; missing={sorted(required-set(artifacts))}, "
            f"unknown={sorted(set(artifacts)-required)}"
        )
    return cfg


def _validate_a3_engine_fields(cfg: dict[str, Any]) -> None:
    path_fields = {
        "project_root", "splits", "source_manifest", "pseudo_label_manifest",
        "coordinate_scale_manifest",
        "a3_protocol", "quality_config", "output_dir",
    }
    for key in path_fields:
        _portable_relative(cfg[key], key, allow_parent=key != "output_dir")
    candidate = cfg["candidate_manifest"]
    if cfg["method"] == "RAW_REPEAT":
        if candidate is not None:
            raise ContractError("RAW_REPEAT candidate_manifest must be null")
    else:
        _portable_relative(candidate, "candidate_manifest", allow_parent=True)
    integer_expectations = {
        "budget_B": cfg["protocol"]["budget_B"],
        "max_updates": cfg["protocol"]["max_updates_U"],
        "batch_size": cfg["training"]["batch_size"],
        "num_points": cfg["protocol"]["num_points"],
        "num_workers": 0,
        "eval_interval": 100,
        "checkpoint_interval": 100,
        "dev_crops_per_scan": 4,
        "cache_size": 4,
    }
    for key, expected in integer_expectations.items():
        if cfg[key] != expected:
            raise ContractError(f"A3 engine field {key} must be {expected!r}")
    if cfg["learning_rate"] != cfg["training"]["learning_rate"]:
        raise ContractError("A3 learning_rate mismatch")
    if cfg["weight_decay"] != cfg["training"]["weight_decay"]:
        raise ContractError("A3 weight_decay mismatch")
    if cfg["lr_min_ratio"] != 0.01:
        raise ContractError("A3 lr_min_ratio must match frozen protocol (0.01)")
    expected_flags = {
        "device": "cuda", "use_normals": True, "amp": True,
        "deterministic": True, "verify_hashes": True, "model_profile": "formal",
    }
    for key, expected in expected_flags.items():
        if cfg[key] != expected:
            raise ContractError(f"A3 engine field {key} must be {expected!r}")
    mapping = {
        "project_root": "project_root", "splits": "splits",
        "source_manifest": "source_manifest",
        "pseudo_label_manifest": "pseudo_label_manifest",
        "coordinate_scale_manifest": "coordinate_scale_manifest",
        "a3_protocol": "a3_protocol",
        "quality_config": "quality_config",
    }
    canonical_evidence = {
        "coordinate_scale_manifest": A3_COORDINATE_SCALE_MANIFEST,
    }
    for key, expected in canonical_evidence.items():
        if cfg[key] != expected or cfg["paths"][key] != expected:
            raise ContractError(f"A3 {key} must be canonical: {expected}")
    for engine_key, path_key in mapping.items():
        if cfg[engine_key] != cfg["paths"][path_key]:
            raise ContractError(f"A3 engine/path mismatch: {engine_key}")
    expected_candidate = None
    if cfg["method"] == "TRAD":
        expected_candidate = cfg["paths"]["candidate_manifest_trad"]
    elif cfg["method"] == "LOADSIM":
        expected_candidate = cfg["paths"]["candidate_manifest_loadsim"]
    if candidate != expected_candidate:
        raise ContractError("A3 candidate_manifest does not match method")

def make_run_id(experiment: str, method: str, seed: int, budget: int | None = None) -> str:
    seed_tag = f"s{SEEDS.index(seed) + 1}"
    if experiment == "A3":
        if budget is None:
            raise ContractError("A3 run_id requires budget_B")
        return f"A3_{method}_B{budget}_{seed_tag}"
    return f"A4_{method}_{seed_tag}"


def _base_training(epochs: int, step_size: int) -> dict[str, Any]:
    return {
        "epochs": epochs, "batch_size": 8, "optimizer": "Adam",
        "learning_rate": 0.001, "weight_decay": 0.0001,
        "lr_scheduler": "StepLR", "lr_step_size": step_size,
        "lr_gamma": 0.5, "device": "cuda",
    }


def generate_a3_configs() -> list[dict[str, Any]]:
    result = []
    for budget in A3_BUDGETS:
        for method in A3_METHODS:
            if budget not in A3_METHOD_BUDGET_GRID[method]:
                continue
            for seed in SEEDS:
                run_id = make_run_id("A3", method, seed, budget)
                paths = {
                    "project_root": "..",
                    "splits": "manifest/splits_v2.json",
                    "source_manifest": "manifest/source_scans_v2.csv",
                    "pseudo_label_manifest": A3_PSEUDO_LABEL_MANIFEST,
                    "coordinate_scale_manifest": A3_COORDINATE_SCALE_MANIFEST,
                    "candidate_manifest_trad": "a3_assets/trad/candidate_manifest.csv",
                    "candidate_manifest_loadsim": "a3_assets/loadsim/candidate_manifest.csv",
                    "a3_protocol": "configs/a3_pointnetpp_mb_v1.json",
                    "quality_config": "configs/quality_v1.yaml",
                }
                candidate = None
                if method == "TRAD":
                    candidate = paths["candidate_manifest_trad"]
                elif method == "LOADSIM":
                    candidate = paths["candidate_manifest_loadsim"]
                a3_training = {
                    "max_updates": 1330, "batch_size": 8,
                    "optimizer": "Adam", "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "lr_scheduler": "CosineAnnealingLR", "lr_min_ratio": 0.01,
                    "device": "cuda",
                }
                result.append({
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "experiment": "A3",
                    "entrypoint": "a3_train.py",
                    "method": method,
                    "seed": seed,
                    "paths": paths,
                    "protocol": {
                        "source_scans_M": 9, "budget_B": budget,
                        "max_updates_U": 1330, "num_points": 4096,
                        "allow_repeat": method == "RAW_REPEAT",
                    },
                    "training": a3_training,
                    "model": {
                        "backbone": "PointNet++", "num_classes": 2,
                        "use_normals": True,
                    },
                    "output_contract": {"required_artifacts": [
                        "resolved_config.json", "selection_manifest.csv",
                        "per_source_exposure.csv", "history.json", "metrics.json",
                        "provenance.json", "checkpoint_best.pt",
                        "checkpoint_latest.pt",
                    ]},
                    "project_root": paths["project_root"],
                    "splits": paths["splits"],
                    "source_manifest": paths["source_manifest"],
                    "pseudo_label_manifest": paths["pseudo_label_manifest"],
                    "coordinate_scale_manifest": paths["coordinate_scale_manifest"],
                    "candidate_manifest": candidate,
                    "output_dir": f"runs/formal/{run_id}",
                    "a3_protocol": paths["a3_protocol"],
                    "quality_config": paths["quality_config"],
                    "budget_B": budget,
                    "max_updates": 1330,
                    "batch_size": 8,
                    "num_points": 4096,
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
                })
    return result


def generate_a4_configs() -> list[dict[str, Any]]:
    result = []
    for method in A4_METHODS:
        for seed in SEEDS:
            run_id = make_run_id("A4", method, seed)
            result.append({
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "experiment": "A4",
                "entrypoint": "a4_train.py",
                "method": method,
                "seed": seed,
                "paths": {
                    "kitti_root": "../kitti_experiment/data",
                    "kitti_config": "configs/kitti_v1.yaml",
                },
                "protocol": {
                    "train_frames": 3200, "val_frames": 600,
                    "test_frames": 741, "max_updates_U": 20000,
                    "num_points": 4096,
                },
                "training": _base_training(50, 15),
                "model": {
                    "backbone": "PointNet++", "num_classes": 2,
                    "use_normals": False,
                },
                "output_contract": {"required_artifacts": [
                    "status.json", "resolved_config.json",
                    "split_manifest.json", "history.json", "metrics.json",
                    "provenance.json", "per_frame_block_metrics.jsonl",
                    "checkpoints/best.pt",
                ]},
            })
    return result

def _job_row(config: dict[str, Any], priority: int) -> dict[str, str]:
    budget = config["protocol"].get("budget_B", "")
    estimate = (
        A3_ESTIMATED_HOURS[budget]
        if config["experiment"] == "A3"
        else A4_ESTIMATED_HOURS
    )
    return {
        "run_id": config["run_id"], "experiment": config["experiment"],
        "method": config["method"], "budget_B": str(budget),
        "seed": str(config["seed"]),
        "config_path": f"configs/{config['run_id']}.json",
        "priority": str(priority), "estimated_hours": str(estimate),
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_catalog(base_dir: Path) -> None:
    configs = generate_a3_configs() + generate_a4_configs()
    for config in configs:
        validate_config(config)
    run_ids = [cfg["run_id"] for cfg in configs]
    if len(run_ids) != TOTAL_RUN_COUNT or len(set(run_ids)) != TOTAL_RUN_COUNT:
        raise ContractError(
            f"generator did not produce exactly {TOTAL_RUN_COUNT} unique run IDs"
        )
    if sum(cfg["experiment"] == "A3" for cfg in configs) != A3_RUN_COUNT:
        raise ContractError(
            f"generator did not produce exactly {A3_RUN_COUNT} A3 configs"
        )
    if sum(cfg["experiment"] == "A4" for cfg in configs) != A4_RUN_COUNT:
        raise ContractError(
            f"generator did not produce exactly {A4_RUN_COUNT} A4 configs"
        )

    config_dir = base_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{cfg['run_id']}.json" for cfg in configs}
    stale = sorted(
        path.name for path in config_dir.glob("A[34]_*.json")
        if path.name not in expected_names
    )
    if stale:
        raise ContractError(
            "stale generated configs found; do not delete implicitly: " + ", ".join(stale)
        )
    for config in configs:
        payload = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
        atomic_write_text(config_dir / f"{config['run_id']}.json", payload)

    jobs_path = base_dir / "jobs.csv"
    rows = [_job_row(config, priority)
            for priority, config in enumerate(configs, start=1)]
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=JOB_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(jobs_path, buffer.getvalue())


def validate_jobs_file(base_dir: Path, require_files: bool = True) -> list[dict[str, str]]:
    jobs_path = base_dir / "jobs.csv"
    with jobs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != JOB_FIELDS:
            raise ContractError(f"jobs.csv schema must be exactly {JOB_FIELDS}")
        rows = list(reader)
    if len(rows) != TOTAL_RUN_COUNT:
        raise ContractError(
            f"jobs.csv must contain {TOTAL_RUN_COUNT} rows, found {len(rows)}"
        )
    if len({row["run_id"] for row in rows}) != TOTAL_RUN_COUNT:
        raise ContractError("jobs.csv run_id values are not unique")

    config_ids: set[str] = set()
    counts = {"A3": 0, "A4": 0}
    for index, row in enumerate(rows, start=2):
        if set(row) != set(JOB_FIELDS):
            raise ContractError(f"jobs.csv row {index} has invalid fields")
        try:
            priority = int(row["priority"])
            estimate = float(row["estimated_hours"])
            seed = int(row["seed"])
        except ValueError as exc:
            raise ContractError(f"jobs.csv row {index} has invalid numeric fields") from exc
        if priority != index - 1 or estimate <= 0 or not math.isfinite(estimate):
            raise ContractError(f"jobs.csv row {index} has invalid priority/estimate")
        if seed not in SEEDS:
            raise ContractError(f"jobs.csv row {index} has invalid seed")
        config_rel = _portable_relative(row["config_path"], f"jobs row {index} config_path")
        config_path = base_dir.joinpath(*PurePosixPath(config_rel).parts)
        if require_files and not config_path.is_file():
            raise ContractError(f"missing config: {config_rel}")
        if require_files:
            with config_path.open("r", encoding="utf-8") as handle:
                config = validate_config(json.load(handle), config_path.name)
            for field in ("run_id", "experiment", "method"):
                if row[field] != str(config[field]):
                    raise ContractError(f"jobs/config mismatch at {row['run_id']}: {field}")
            if row["seed"] != str(config["seed"]):
                raise ContractError(f"jobs/config mismatch at {row['run_id']}: seed")
            expected_budget = str(config["protocol"].get("budget_B", ""))
            if row["budget_B"] != expected_budget:
                raise ContractError(f"jobs/config mismatch at {row['run_id']}: budget_B")
            expected_estimate = (
                A3_ESTIMATED_HOURS[config["protocol"]["budget_B"]]
                if config["experiment"] == "A3"
                else A4_ESTIMATED_HOURS
            )
            if estimate != expected_estimate:
                raise ContractError(
                    f"jobs/config mismatch at {row['run_id']}: estimated_hours"
                )
            config_ids.add(config["run_id"])
            counts[config["experiment"]] += 1
    expected_counts = {"A3": A3_RUN_COUNT, "A4": A4_RUN_COUNT}
    expected_method_counts = {
        ("A3", "RAW_REPEAT"): 18,
        ("A3", "TRAD"): 9,
        ("A3", "LOADSIM"): 18,
        ("A4", "BASELINE"): 3,
        ("A4", "TRADITIONAL"): 3,
        ("A4", "LOADSIM"): 3,
    }
    method_counts = Counter((row["experiment"], row["method"]) for row in rows)
    if require_files and (
        counts != expected_counts or config_ids != {row["run_id"] for row in rows}
    ):
        raise ContractError(
            f"catalog mismatch: counts={counts}, "
            f"matched={len(config_ids)}/{TOTAL_RUN_COUNT}"
        )
    if method_counts != expected_method_counts:
        raise ContractError(
            f"catalog method counts mismatch: {dict(method_counts)}")
    return rows

def validate_generated_catalog(base_dir: Path) -> None:
    """Require checked-in configs/jobs to equal current generator output exactly."""
    expected_configs = generate_a3_configs() + generate_a4_configs()
    expected_by_id = {config["run_id"]: config for config in expected_configs}
    for run_id, expected in expected_by_id.items():
        path = base_dir / "configs" / f"{run_id}.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                actual = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid generated config {path.name}: {exc}") from exc
        if actual != expected:
            raise ContractError(
                f"checked-in config differs from current generator output: {path.name}; "
                "run generate_configs.py before formal use")
    expected_rows = [_job_row(config, priority)
                     for priority, config in enumerate(expected_configs, start=1)]
    actual_rows = validate_jobs_file(base_dir)
    if actual_rows != expected_rows:
        raise ContractError(
            "checked-in jobs.csv differs from current generator output; "
            "run generate_configs.py before formal use")


def audit_a4_raw_data(base_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Read all frozen sequence-00 labels and verify the raw-ID mapping contract."""
    try:
        import numpy as np
        from a4_train import (
            CACHE_SCHEMA, FORMAL_SPLIT, GROUND_SEMANTIC_IDS, IGNORE_SEMANTIC_IDS,
            VALID_RAW_SEMANTIC_IDS, discover_frames, locate_sequence_dir,
        )
        from config_kitti import GROUND_CLASSES, IGNORE_CLASSES
        from kitti_dataset import load_bin, load_label, map_to_binary
    except (ImportError, AttributeError) as exc:
        raise ContractError(f"cannot import A4 raw-data audit implementation: {exc}") from exc
    a4_rows = [row for row in rows if row["experiment"] == "A4"]
    roots: set[str] = set()
    for row in a4_rows:
        config_path = base_dir.joinpath(*PurePosixPath(row["config_path"]).parts)
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        roots.add(cfg["paths"]["kitti_root"])
    if len(roots) != 1:
        raise ContractError(f"A4 configs must share one kitti_root, got {sorted(roots)}")
    rel = next(iter(roots))
    kitti_root = base_dir.joinpath(*PurePosixPath(rel).parts).resolve()
    seq_dir = locate_sequence_dir(kitti_root)
    frames = discover_frames(seq_dir)
    if sorted(GROUND_CLASSES) != GROUND_SEMANTIC_IDS:
        raise ContractError("runtime A4 ground IDs differ from the frozen contract")
    if sorted(IGNORE_CLASSES) != IGNORE_SEMANTIC_IDS:
        raise ContractError("runtime A4 ignore IDs differ from the frozen contract")
    observed: set[int] = set()
    mapped_counts = {name: {0: 0, 1: 0, 255: 0} for name in FORMAL_SPLIT}
    content_owners: dict[str, tuple[str, str]] = {}
    for frame in frames:
        split = next(name for name, (start, end) in FORMAL_SPLIT.items()
                     if start <= frame.index < end)
        points4 = load_bin(frame.bin_path)
        semantic = load_label(frame.label_path)
        if points4.ndim != 2 or points4.shape[1] != 4 or len(points4) != len(semantic):
            raise ContractError(f"invalid A4 raw frame shape/count: {frame.stem}")
        unknown = sorted(set(np.unique(semantic).tolist()) - set(VALID_RAW_SEMANTIC_IDS))
        if unknown:
            raise ContractError(
                f"A4 frame {frame.stem} contains unknown raw IDs: {unknown}")
        finite = np.isfinite(points4[:, :3]).all(axis=1)
        nonzero = np.any(points4[:, :3] != 0.0, axis=1)
        mapped = map_to_binary(semantic)
        if mapped.shape != semantic.shape or not np.isin(mapped, [0, 1, 255]).all():
            raise ContractError(f"invalid A4 mapping output in frame {frame.stem}")
        valid = finite & nonzero & (mapped != 255)
        if int(valid.sum()) < 100:
            raise ContractError(f"A4 frame {frame.stem} has fewer than 100 usable points")
        observed.update(int(value) for value in np.unique(semantic))
        for value in mapped_counts[split]:
            mapped_counts[split][value] += int(np.count_nonzero(mapped[finite & nonzero] == value))
        digest = hashlib.sha256()
        for path in (frame.bin_path, frame.label_path):
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
        identity = digest.hexdigest()
        if identity in content_owners:
            previous = content_owners[identity]
            raise ContractError(
                f"duplicate A4 raw frame content: {previous} and {(split, frame.stem)}")
        content_owners[identity] = (split, frame.stem)
    missing_ground = sorted(set(GROUND_SEMANTIC_IDS) - observed)
    missing_ignore = sorted(set(IGNORE_SEMANTIC_IDS) - observed)
    if missing_ground or missing_ignore:
        raise ContractError(
            f"A4 raw labels do not exercise the frozen mapping; "
            f"missing_ground={missing_ground}, missing_ignore={missing_ignore}")
    empty_splits = {name: counts for name, counts in mapped_counts.items()
                    if counts[0] == 0 or counts[1] == 0}
    if empty_splits:
        raise ContractError(f"A4 split-level mapped classes are empty: {empty_splits}")
    return {
        "frames": len(frames),
        "ground_ids": GROUND_SEMANTIC_IDS,
        "ignore_ids": IGNORE_SEMANTIC_IDS,
        "observed_raw_ids": sorted(observed),
        "mapped_counts": mapped_counts,
        "cache_schema": CACHE_SCHEMA,
        "cache_version": "v2",
    }


def _resolve_config_path(base_dir: Path, value: str) -> Path:
    return base_dir.joinpath(*PurePosixPath(value).parts).resolve()


def _audit_a3_source_evidence(
    base_dir: Path, rows: list[dict[str, str]], errors: list[str],
) -> None:
    """Validate the canonical scale evidence declared by all A3 configs."""
    try:
        from a3_data import load_frozen_sources
        from a3_io import A3Error, load_coordinate_scale_manifest
    except (ImportError, AttributeError) as exc:
        errors.append(f"P0-2 source-evidence blocker: cannot import validators: {exc}")
        return
    identities: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row["experiment"] != "A3":
            continue
        try:
            config = json.loads(
                _resolve_config_path(base_dir, row["config_path"]).read_text(encoding="utf-8"))
            paths = config["paths"]
            identities.add(tuple(paths[name] for name in (
                "project_root", "splits", "source_manifest",
                "coordinate_scale_manifest")))
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
            errors.append(f"P0-2 source-evidence blocker: invalid A3 config: {exc}")
            return
    if len(identities) != 1:
        errors.append(
            f"P0-2 source-evidence blocker: A3 configs must share one canonical scale manifest; "
            f"found={sorted(identities)}")
        return
    project_rel, splits_rel, source_rel, scale_rel = next(iter(identities))
    try:
        train, dev = load_frozen_sources(
            _resolve_config_path(base_dir, project_rel),
            _resolve_config_path(base_dir, splits_rel),
            _resolve_config_path(base_dir, source_rel))
        source_hashes = {source.scan_id: source.file_sha256 for source in train + dev}
    except (A3Error, OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"P0-2 source-evidence blocker: cannot load frozen TRAIN/DEV: {exc}")
        return
    checks = ((
        "coordinate-scale", scale_rel, load_coordinate_scale_manifest,
        "raw-coordinate-to-metre scale must be independently evidenced per source",
    ),)
    for label, relative, loader, requirement in checks:
        path = _resolve_config_path(base_dir, relative)
        try:
            loader(path, source_hashes)
        except (A3Error, OSError, UnicodeError, csv.Error) as exc:
            errors.append(
                f"P0-2 {label} evidence blocker: {requirement}; "
                f"required={relative}; detail={exc}")


def _audit_a3_pseudo_label_manifests(
    base_dir: Path, rows: list[dict[str, str]], errors: list[str],
) -> None:
    """Parse each distinct formal A3 pseudo-label input through the runtime v3 validator."""
    try:
        from a3_data import load_frozen_sources, load_label_manifest
        from a3_io import A3Error, PSEUDO_LABEL_MANIFEST_SCHEMA
    except (ImportError, AttributeError) as exc:
        errors.append(f"cannot import A3 pseudo-label v3 audit implementation: {exc}")
        return
    checked: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        if row["experiment"] != "A3":
            continue
        config_path = _resolve_config_path(base_dir, row["config_path"])
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            paths = config["paths"]
            identity = tuple(paths[name] for name in (
                "project_root", "splits", "source_manifest", "pseudo_label_manifest",
                "coordinate_scale_manifest"))
            if identity in checked:
                continue
            checked.add(identity)
            project_root, splits, source_manifest, pseudo_manifest, scale = (
                _resolve_config_path(base_dir, value) for value in identity)
            train, dev = load_frozen_sources(project_root, splits, source_manifest)
            load_label_manifest(
                pseudo_manifest, train + dev, verify_hashes=True,
                canonical_scale_manifest=scale)
        except (A3Error, TypeError, OSError, UnicodeError,
                json.JSONDecodeError, KeyError) as exc:
            errors.append(
                f"P0-2 pseudo-label data blocker ({PSEUDO_LABEL_MANIFEST_SCHEMA}): {exc}")


def _audit_a3_candidate_manifests(
    base_dir: Path, rows: list[dict[str, str]], errors: list[str],
) -> None:
    """Verify every configured candidate bundle against the configured canonical roots."""
    try:
        from a3_data import load_frozen_sources, load_candidates
        from a3_io import A3Error
        from quality_contract import load_quality_contract
    except (ImportError, AttributeError) as exc:
        errors.append(f"cannot import A3 candidate canonical-root audit: {exc}")
        return
    checked: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        if row["experiment"] != "A3" or row["method"] == "RAW_REPEAT":
            continue
        try:
            config = json.loads(
                _resolve_config_path(base_dir, row["config_path"]).read_text(encoding="utf-8"))
            paths = config["paths"]
            identity = (
                paths["project_root"], paths["splits"], paths["source_manifest"],
                config["candidate_manifest"], paths["quality_config"],
                paths["coordinate_scale_manifest"],
            )
            if identity in checked:
                continue
            checked.add(identity)
            project, splits, sources, candidate, quality, scale = (
                _resolve_config_path(base_dir, value) for value in identity)
            train, _ = load_frozen_sources(project, splits, sources)
            load_candidates(
                candidate, train, row["method"], True, load_quality_contract(quality),
                canonical_scale_manifest=scale)
        except (A3Error, TypeError, OSError, UnicodeError,
                json.JSONDecodeError, KeyError) as exc:
            errors.append(
                f"P0-2 {row['method']} canonical-evidence bundle blocker: {exc}")


def preflight(base_dir: Path, check_entrypoints: bool = True,
              check_data: bool = False) -> list[str]:
    validate_generated_catalog(base_dir)
    rows = validate_jobs_file(base_dir)
    errors: list[str] = []
    for entrypoint in ("a3_train.py", "a4_train.py"):
        if check_entrypoints and not (base_dir / entrypoint).is_file():
            errors.append(f"missing training entrypoint: {entrypoint}")
    if check_data:
        checked: set[str] = set()
        project_root = base_dir.parent.resolve()
        for row in rows:
            config_path = base_dir.joinpath(*PurePosixPath(row["config_path"]).parts)
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            for name, rel in config["paths"].items():
                key = f"{name}:{rel}"
                if key in checked:
                    continue
                checked.add(key)
                target = base_dir.joinpath(*PurePosixPath(rel).parts).resolve()
                try:
                    target.relative_to(project_root)
                except ValueError:
                    errors.append(f"required path escapes project root {name}: {rel}")
                    continue
                if not target.exists():
                    errors.append(f"missing required path {name}: {rel}")
        _audit_a3_source_evidence(base_dir, rows, errors)
        _audit_a3_pseudo_label_manifests(base_dir, rows, errors)
        _audit_a3_candidate_manifests(base_dir, rows, errors)
    return errors


def strict_preflight(base_dir: Path) -> list[dict[str, str]]:
    """Run the one formal queue gate for generated code, entrypoints and data.

    This function never starts training.  It raises ``ContractError`` unless the
    complete A3/A4 catalogue and the raw A4 audit are ready for formal use.
    """
    base_dir = Path(base_dir).resolve()
    errors = preflight(base_dir, check_entrypoints=True, check_data=True)
    if not (base_dir / "cloud_train.py").is_file():
        errors.append("missing queue entrypoint: cloud_train.py")
    if errors:
        raise ContractError("strict preflight failed:\n- " + "\n- ".join(errors))
    rows = validate_jobs_file(base_dir)
    try:
        audit_a4_raw_data(base_dir, rows)
    except ContractError:
        raise
    except (RuntimeError, ValueError, OSError, csv.Error) as exc:
        raise ContractError(f"strict A4 raw-data preflight failed: {exc}") from exc
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check-only", action="store_true",
                        help=f"validate the existing {TOTAL_RUN_COUNT} JSON files and jobs.csv")
    parser.add_argument("--check-entrypoints", action="store_true",
                        help="also require a3_train.py and a4_train.py")
    parser.add_argument("--check-data", action="store_true",
                        help="also require every configured data/config path")
    args = parser.parse_args(argv)
    base_dir = args.base_dir.resolve()
    try:
        if not args.check_only:
            write_catalog(base_dir)
        errors = preflight(base_dir, args.check_entrypoints, args.check_data)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        a4_audit = audit_a4_raw_data(base_dir, validate_jobs_file(base_dir)) \
            if args.check_data else None
        print(
            f"catalog valid: A3={A3_RUN_COUNT} "
            f"(RAW_REPEAT=18, LOADSIM=18, TRAD=9), A4={A4_RUN_COUNT}, "
            f"TOTAL={TOTAL_RUN_COUNT}, jobs/configs={TOTAL_RUN_COUNT}/{TOTAL_RUN_COUNT}, "
            "run IDs unique"
        )
        if a4_audit is not None:
            print(
                "A4 raw-data mapping PASS: "
                f"frames={a4_audit['frames']}, "
                f"raw ground IDs={'/'.join(map(str, a4_audit['ground_ids']))}, "
                f"ignore IDs={'/'.join(map(str, a4_audit['ignore_ids']))}, "
                f"cache_version={a4_audit['cache_version']}, "
                f"cache_schema={a4_audit['cache_schema']}"
            )
            print(
                "A4 cache note: this preflight validates the raw data and cache-v2 "
                "contract; actual cache creation/validation still requires the real CUDA debug smoke."
            )
        return 0
    except (ContractError, RuntimeError, OSError, json.JSONDecodeError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
