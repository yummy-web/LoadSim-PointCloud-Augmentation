"""Formal fixed-update PointNet++ training engine for matched-budget A3."""
from __future__ import annotations

import math
import os
import platform
import socket
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from a3_data import (
    IGNORE_INDEX, METHODS, CircularBatchSampler, PointCloudDataset, SampleRecord,
    build_selection,
    load_frozen_sources, load_label_manifest, make_original_records,
    source_exposure_rows, validate_records,
)
from a3_io import (
    A3Error, ALL_VALID_ELIGIBILITY_RULE, atomic_torch_save, atomic_write_csv,
    atomic_write_json, canonical_hash,
    capture_rng_state, load_coordinate_scale_manifest,
    load_json, package_versions, restore_rng_state, seed_everything,
    seed_worker, sha256_file, stable_seed,
)
from quality_contract import load_quality_contract

SCHEMA_VERSION = "a3-run-v1"


def _flatten_logits(logits: Any) -> Any:
    """Reshape per-point logits (B, C, N) -> (B*N, C) so CrossEntropyLoss uses the
    1-D ``nll_loss`` CUDA kernel, which HAS a deterministic implementation, instead
    of the K-dimensional ``nll_loss2d`` kernel (non-deterministic ``atomicAdd``
    backward) that a (B, C, N) target shape would dispatch to under
    ``torch.use_deterministic_algorithms(True)``.

    This is a pure reindexing: with ``reduction='mean'`` and a class-weighted,
    ``ignore_index``-masked loss, the value and gradients are the flattened
    computation's exact equivalents (same per-element losses, same weights, same
    normaliser), so it is mathematically identical to the (B, C, N) form while
    remaining bitwise-deterministic. ``target`` is flattened with ``.reshape(-1)``
    at the call site so the point ordering matches ``(B*N, C)`` row order.
    """
    return logits.permute(0, 2, 1).reshape(-1, logits.shape[1])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(config: dict[str, Any], name: str, kind: type) -> Any:
    value = config.get(name)
    if not isinstance(value, kind):
        raise A3Error(f"Config field {name!r} must be {kind.__name__}")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "project_root", "splits", "source_manifest", "pseudo_label_manifest",
        "coordinate_scale_manifest",
        "output_dir", "method", "budget_B", "seed", "a3_protocol",
        "quality_config", "run_id",
    }
    missing = sorted(required - set(config))
    if missing:
        raise A3Error(f"Missing config fields: {missing}")
    resolved = dict(config)
    method = str(resolved["method"]).upper()
    if method not in METHODS:
        raise A3Error(f"method must be one of {sorted(METHODS)}")
    resolved["method"] = method
    for name in (
            "project_root", "splits", "source_manifest", "pseudo_label_manifest",
            "coordinate_scale_manifest",
            "output_dir", "a3_protocol", "quality_config"):
        resolved[name] = str(Path(resolved[name]).expanduser().resolve())
    candidate = resolved.get("candidate_manifest")
    resolved["candidate_manifest"] = str(Path(candidate).expanduser().resolve()) if candidate else None
    protocol = load_json(resolved["a3_protocol"])
    if protocol.get("schema_version") != "a3-protocol-v1" or protocol.get("frozen") is not True:
        raise A3Error("Formal A3 requires a frozen a3-protocol-v1 file")
    debug = bool(resolved.get("debug", False))
    resolved["debug"] = debug
    if not debug:
        frozen_fields = {
            "max_updates": protocol.get("max_updates_U"),
            "batch_size": protocol.get("batch_size"),
            "num_points": protocol.get("num_points"),
            "use_normals": protocol.get("use_normals"),
            "model_profile": protocol.get("model_profile"),
            "deterministic": protocol.get("deterministic"),
            "verify_hashes": protocol.get("verify_hashes"),
            "learning_rate": protocol.get("learning_rate"),
            "weight_decay": protocol.get("weight_decay"),
            "lr_min_ratio": protocol.get("lr_min_ratio"),
            "amp": protocol.get("amp"),
            "eval_interval": protocol.get("eval_interval"),
            "checkpoint_interval": protocol.get("checkpoint_interval"),
            "dev_crops_per_scan": protocol.get("dev_crops_per_scan"),
            "cache_size": protocol.get("cache_size"),
            "num_workers": protocol.get("num_workers"),
        }
        for field, frozen_value in frozen_fields.items():
            if field in resolved and resolved[field] != frozen_value:
                raise A3Error(f"Formal A3 {field} is frozen at {frozen_value!r}, got {resolved[field]!r}")
            resolved[field] = frozen_value
        if method not in protocol.get("methods", []):
            raise A3Error(f"Method {method} is outside the frozen A3 protocol")
        if protocol.get("M") != 9 or protocol.get("ignore_index") != IGNORE_INDEX:
            raise A3Error("Frozen A3 protocol has invalid M or ignore_index")
        method_budget_grid = protocol.get("method_budget_grid")
        if not isinstance(method_budget_grid, dict):
            raise A3Error("Frozen A3 protocol has invalid method_budget_grid")
        allowed_budgets = method_budget_grid.get(method)
        if (not isinstance(allowed_budgets, list) or not allowed_budgets or
                any(isinstance(budget, bool) or not isinstance(budget, int)
                    for budget in allowed_budgets)):
            raise A3Error(
                f"Frozen A3 protocol has invalid method_budget_grid entry for {method}"
            )
        if resolved["budget_B"] not in allowed_budgets:
            raise A3Error(
                f"B={resolved['budget_B']} is not allowed for {method} by the frozen "
                f"A3 method_budget_grid; allowed={allowed_budgets}"
            )
        if resolved["seed"] not in protocol.get("seeds", []):
            raise A3Error(f"Seed {resolved['seed']} is outside the frozen A3 protocol")
        seed_tags = {20260911: "s1", 20260912: "s2", 20260913: "s3"}
        expected_run_id = (
            f"A3_{method}_B{resolved['budget_B']}_{seed_tags[resolved['seed']]}"
        )
        if resolved["run_id"] != expected_run_id:
            raise A3Error(
                "run_id is not canonical for the frozen method/budget/seed: "
                f"{resolved['run_id']!r} != {expected_run_id!r}"
            )
        if Path(resolved["output_dir"]).name != resolved["run_id"]:
            raise A3Error("formal output_dir leaf must equal run_id")
        requested_device = str(resolved.get("device", "cuda"))
        if not requested_device.startswith(str(protocol.get("device_type", "cuda"))):
            raise A3Error(f"Formal A3 device must be {protocol.get('device_type')}, got {requested_device}")
    integers = {
        "budget_B": resolved["budget_B"], "seed": resolved["seed"],
        "max_updates": resolved["max_updates"], "batch_size": resolved.get("batch_size", 8),
        "num_points": resolved.get("num_points", 4096), "num_workers": resolved.get("num_workers", 0),
        "eval_interval": resolved.get("eval_interval", 100),
        "checkpoint_interval": resolved.get("checkpoint_interval", 100),
        "dev_crops_per_scan": resolved.get("dev_crops_per_scan", 4),
        "cache_size": resolved.get("cache_size", 4),
    }
    for name, value in integers.items():
        if not isinstance(value, int) or (value <= 0 and name not in {"num_workers", "cache_size"}) or value < 0:
            raise A3Error(f"{name} has invalid integer value {value!r}")
        resolved[name] = value
    if method != "RAW_REPEAT" and not resolved["candidate_manifest"]:
        raise A3Error(f"{method} requires candidate_manifest")
    resolved["learning_rate"] = float(resolved.get("learning_rate", 1e-3))
    resolved["weight_decay"] = float(resolved.get("weight_decay", 1e-4))
    resolved["lr_min_ratio"] = float(resolved.get("lr_min_ratio", 0.01))
    if resolved["learning_rate"] <= 0 or resolved["weight_decay"] < 0:
        raise A3Error("Invalid optimizer settings")
    resolved["use_normals"] = bool(resolved.get("use_normals", True))
    resolved["amp"] = bool(resolved.get("amp", True))
    resolved["deterministic"] = bool(resolved.get("deterministic", True))
    resolved["verify_hashes"] = bool(resolved.get("verify_hashes", True))
    resolved["device"] = str(resolved.get("device", "cuda"))
    resolved["model_profile"] = str(resolved.get("model_profile", "formal"))
    if resolved["model_profile"] != "formal" and not resolved.get("debug", False):
        raise A3Error("Non-formal model profiles require debug=true")
    resolved["schema_version"] = SCHEMA_VERSION
    return resolved


def expand_for_full_batches(records: list[SampleRecord], batch_size: int) -> list[SampleRecord]:
    """Repeat the frozen B-vector to its shortest full-batch circular period."""
    if not records or batch_size <= 0:
        raise A3Error("Cannot build a full-batch stream from empty records")
    repeat_factor = batch_size // math.gcd(len(records), batch_size)
    return records * repeat_factor


def class_weights_from_counts(counts: dict[str, int], device: Any):
    import torch
    frequencies = torch.tensor([counts["0"], counts["1"]], dtype=torch.float64)
    if torch.any(frequencies <= 0):
        raise A3Error(f"Cannot derive class weights from {counts}")
    weights = frequencies.sum() / (2.0 * frequencies)
    weights = weights / weights.mean()
    return weights.to(dtype=torch.float32, device=device)


def confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (2, 2) or confusion.sum() == 0:
        raise A3Error("Cannot compute metrics from empty/invalid confusion matrix")
    true_positive = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - true_positive
    iou = np.divide(true_positive, union, out=np.zeros(2), where=union > 0)
    precision_den = confusion.sum(0)
    recall_den = confusion.sum(1)
    precision = np.divide(true_positive, precision_den, out=np.zeros(2), where=precision_den > 0)
    recall = np.divide(true_positive, recall_den, out=np.zeros(2), where=recall_den > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(2), where=(precision + recall) > 0)
    return {
        "mIoU": float(iou.mean()), "IoU_0": float(iou[0]), "IoU_1": float(iou[1]),
        "accuracy": float(true_positive.sum() / confusion.sum()),
        "precision_1": float(precision[1]), "recall_1": float(recall[1]),
        "f1_1": float(f1[1]), "valid_points": int(confusion.sum()),
        "confusion_matrix": confusion.tolist(),
    }


def update_confusion(confusion: np.ndarray, logits: Any, target: Any) -> None:
    import torch
    prediction = logits.argmax(1)
    valid = target != IGNORE_INDEX
    if not torch.any(valid):
        return
    encoded = target[valid].to(torch.int64) * 2 + prediction[valid].to(torch.int64)
    batch_confusion = torch.bincount(encoded, minlength=4).reshape(2, 2)
    confusion += batch_confusion.detach().cpu().numpy().astype(np.int64)


def evaluate(model: Any, loader: Any, criterion: Any, device: Any, amp_enabled: bool,
             autocast: Any) -> dict[str, Any]:
    import torch
    model.eval()
    total_loss = 0.0
    batches = 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    per_source: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for features, target, _, parents in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with autocast(enabled=amp_enabled):
                logits = model(features)
                loss = criterion(_flatten_logits(logits), target.reshape(-1))
            total_loss += float(loss.detach().cpu())
            batches += 1
            update_confusion(confusion, logits, target)
            predictions = logits.argmax(1)
            for row, parent in enumerate(parents):
                matrix = per_source.setdefault(parent, np.zeros((2, 2), dtype=np.int64))
                valid = target[row] != IGNORE_INDEX
                if torch.any(valid):
                    encoded = target[row][valid].to(torch.int64) * 2 + predictions[row][valid].to(torch.int64)
                    matrix += torch.bincount(encoded, minlength=4).reshape(2, 2).cpu().numpy()
    metrics = confusion_metrics(confusion)
    metrics["loss"] = total_loss / max(batches, 1)
    metrics["per_source"] = {parent: confusion_metrics(matrix) for parent, matrix in sorted(per_source.items())}
    return metrics


def selection_rows(selection: list[SampleRecord], vector: list[str]) -> list[dict[str, Any]]:
    rows = []
    for occurrence, (record, parent) in enumerate(zip(selection, vector)):
        if record.parent_scan_id != parent:
            raise A3Error("Internal parent-selection mismatch")
        rows.append({
            "occurrence_index": occurrence, "parent_scan_id": parent,
            "sample_id": record.sample_id, "method": record.method,
            "ply_path": str(record.path), "ply_sha256": record.file_sha256,
            "label_path": str(record.label_path), "label_sha256": record.label_sha256,
            "point_count": record.point_count, "generation_seed": record.generation_seed or "",
            "metadata_path": str(record.metadata_path) if record.metadata_path else "",
            "quality_config_sha256": record.quality_config_sha256 or "",
            "quality_report_sha256": record.quality_report_sha256 or "",
            "quality_Q": "" if record.quality_q is None else record.quality_q,
        })
    return rows


def checkpoint_payload(model: Any, optimizer: Any, scheduler: Any, scaler: Any,
                       config_hash: str, global_update: int, epoch: int,
                       batch_in_epoch: int, best: dict[str, Any], history: list[dict[str, Any]],
                       train_generator: Any, completed: bool,
                       interval_confusion: np.ndarray, interval_loss: float,
                       interval_batches: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "train_generator_state": train_generator.get_state(),
        "global_update": global_update,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "best": best,
        "history": history,
        "interval_confusion": np.asarray(interval_confusion, dtype=np.int64),
        "interval_loss": float(interval_loss),
        "interval_batches": int(interval_batches),
        "completed": completed,
        "saved_at": utc_now(),
    }


def restore_checkpoint(path: Path, model: Any, optimizer: Any, scheduler: Any,
                       scaler: Any, train_generator: Any, expected_hash: str) -> dict[str, Any]:
    import torch
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required = {"config_hash", "model_state", "optimizer_state", "scheduler_state",
                "scaler_state", "rng_state", "train_generator_state", "global_update",
                "epoch", "batch_in_epoch", "best", "history", "completed",
                "interval_confusion", "interval_loss", "interval_batches"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise A3Error(f"Checkpoint is incomplete: {path}")
    if checkpoint["config_hash"] != expected_hash:
        raise A3Error("Resume checkpoint config hash does not match resolved config")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    train_generator.set_state(checkpoint["train_generator_state"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def provenance(config: dict[str, Any], config_hash: str, inputs: dict[str, str],
               source_ids: dict[str, list[str]], class_weight_counts: dict[str, int],
               selected_class_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "config_hash": config_hash,
        "run_id": config["run_id"],
        "method": config["method"], "budget_B": config["budget_B"],
        "seed": config["seed"], "max_updates_U": config["max_updates"],
        "M": len(source_ids["TRAIN"]),
        "train_parent_scans": source_ids["TRAIN"],
        "dev_parent_scans": source_ids["DEV"],
        "dev_policy": "original frozen DEV scans only; no augmentation",
        "selection_policy": "shared balanced source-occurrence vector v1",
        "quality_filter": config["quality_contract"],
        "checkpoint_selection": "maximum DEV mIoU; no human reference labels",
        "class_weight_counts_from_frozen_train_originals": class_weight_counts,
        "selected_sample_class_counts": selected_class_counts,
        "input_hashes": inputs,
        "software": package_versions(),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "python_executable": sys.executable},
    }


def train_a3(raw_config: dict[str, Any], resume: Path | None = None,
             preflight_only: bool = False) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from a3_model import PointNetPPSeg

    config = validate_config(raw_config)
    output_dir = Path(config["output_dir"])
    latest_path = output_dir / "checkpoint_latest.pt"
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise A3Error(f"Output directory is not empty; use --resume or a new path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"], config["deterministic"])
    requested_device = config["device"]
    if requested_device.startswith("cuda") and not torch.cuda.is_available() and not preflight_only:
        raise A3Error(f"CUDA requested but unavailable: {requested_device}")
    device = torch.device("cpu" if preflight_only else requested_device)

    project_root = Path(config["project_root"])
    split_path = Path(config["splits"])
    source_manifest_path = Path(config["source_manifest"])
    label_manifest_path = Path(config["pseudo_label_manifest"])
    scale_manifest_path = Path(config["coordinate_scale_manifest"])
    train_sources, dev_sources = load_frozen_sources(
        project_root, split_path, source_manifest_path)
    all_source_hashes = {
        source.scan_id: source.file_sha256 for source in train_sources + dev_sources}
    load_coordinate_scale_manifest(scale_manifest_path, all_source_hashes)
    if not config["debug"]:
        protocol = load_json(config["a3_protocol"])
        actual_train = [item.scan_id for item in train_sources]
        actual_dev = [item.scan_id for item in dev_sources]
        if actual_train != protocol.get("train_parent_scans"):
            raise A3Error(f"Frozen TRAIN identities/order mismatch: {actual_train}")
        if actual_dev != protocol.get("dev_parent_scans"):
            raise A3Error(f"Frozen DEV identities/order mismatch: {actual_dev}")
    original_labels = load_label_manifest(
        label_manifest_path, train_sources + dev_sources, config["verify_hashes"],
        canonical_scale_manifest=scale_manifest_path)
    quality_contract = load_quality_contract(config["quality_config"])
    # For all-valid runs the activation trust-root chain MUST verify before any
    # bundle is consumed (and long before any model/optimizer is built).  The
    # all-valid loader itself runs the per-bundle fast verify.  This gate is
    # independent of verify_hashes, so --skip-hash-verification cannot reach it.
    eligibility_rule = config.get("eligibility_rule")
    if eligibility_rule == ALL_VALID_ELIGIBILITY_RULE:
        from a3_activation import verify_activation
        verify_activation(Path(__file__).resolve().parent)
    selection, vector = build_selection(
        config["method"], config["budget_B"], config["seed"], train_sources,
        original_labels, Path(config["candidate_manifest"]) if config["candidate_manifest"] else None,
        quality_contract, config["verify_hashes"],
        canonical_scale_manifest=scale_manifest_path,
        eligibility_rule=eligibility_rule)
    original_train_records = make_original_records(train_sources, original_labels)
    dev_base = make_original_records(dev_sources, original_labels)
    dev_records = [record for record in dev_base for _ in range(config["dev_crops_per_scan"])]
    selected_class_counts, _ = validate_records(selection, config["verify_hashes"])
    class_weight_counts, _ = validate_records(original_train_records, config["verify_hashes"])
    validate_records(dev_base, config["verify_hashes"])

    config.setdefault("run_id", f"A3_{config['method']}_B{config['budget_B']}_s{config['seed']}")
    config["train_parent_scans"] = [item.scan_id for item in train_sources]
    config["dev_parent_scans"] = [item.scan_id for item in dev_sources]
    config["M"] = len(train_sources)
    config["source_occurrence_vector_hash"] = canonical_hash(vector)
    selection_data = selection_rows(selection, vector)
    input_hashes = {
        "a3_protocol": sha256_file(config["a3_protocol"]),
        "quality_config": quality_contract["sha256"],
        "splits": sha256_file(split_path), "source_manifest": sha256_file(source_manifest_path),
        "coordinate_scale_manifest": sha256_file(scale_manifest_path),
        "pseudo_label_manifest": sha256_file(label_manifest_path),
    }
    config["quality_contract"] = {
        "rule": quality_contract["rule"],
        "threshold_Q": quality_contract["threshold_Q"],
        "weights": quality_contract["weights"],
        "applies_to": "TRAD/LOADSIM augmented candidates; not RAW_REPEAT originals",
    }
    if config["candidate_manifest"]:
        input_hashes["candidate_manifest"] = sha256_file(config["candidate_manifest"])
    implementation_dir = Path(__file__).resolve().parent
    for module_name in (
        "a3_io.py", "a3_model.py", "a3_data.py", "a3_candidate_bundle.py",
        "quality_contract.py", "a3_train_engine.py", "a3_train.py",
    ):
        input_hashes[f"implementation/{module_name}"] = sha256_file(implementation_dir / module_name)
    config["input_hashes"] = input_hashes
    config["selection_hash"] = canonical_hash(selection_data)
    config_hash = canonical_hash(config)
    audit_files = {
        "resolved_config.json": {**config, "config_hash": config_hash},
        "selection_manifest.csv": selection_data,
    }
    if resume is not None:
        resume_path = Path(resume).resolve()
        if not resume_path.is_file():
            raise A3Error(f"Missing resume checkpoint: {resume_path}")
        try:
            resume_header = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            resume_header = torch.load(resume_path, map_location="cpu")
        if not isinstance(resume_header, dict) or resume_header.get("config_hash") != config_hash:
            raise A3Error("Resume checkpoint is not bound to the current data/config selection")
        existing_config = load_json(output_dir / "resolved_config.json")
        if existing_config.get("config_hash") != config_hash:
            raise A3Error("Existing resolved_config.json does not match resume checkpoint")
        for required_artifact in ("selection_manifest.csv", "per_source_exposure.csv",
                                  "provenance.json", "history.json"):
            if not (output_dir / required_artifact).is_file():
                raise A3Error(f"Resume audit artifact is missing: {required_artifact}")
    else:
        atomic_write_json(output_dir / "resolved_config.json", audit_files["resolved_config.json"])
        atomic_write_csv(output_dir / "selection_manifest.csv", selection_data, list(selection_data[0]))
    exposure = source_exposure_rows(
        vector, config["method"], config["budget_B"], config["seed"],
        config["max_updates"], config["batch_size"])
    if resume is None:
        atomic_write_csv(output_dir / "per_source_exposure.csv", exposure, list(exposure[0]))
    prov = provenance(config, config_hash, input_hashes,
                      {"TRAIN": config["train_parent_scans"], "DEV": config["dev_parent_scans"]},
                      class_weight_counts, selected_class_counts)
    if resume is None:
        atomic_write_json(output_dir / "provenance.json", prov)
    if preflight_only:
        result = {"status": "preflight_ok", "config_hash": config_hash,
                  "selection_count": len(selection),
                  "class_weight_counts": class_weight_counts,
                  "selected_class_counts": selected_class_counts}
        atomic_write_json(output_dir / "metrics.json", result)
        return result

    # (all-valid activation was already verified before build_selection above,
    # i.e. before any bundle consumption or model/optimizer construction.)
    use_normals = config["use_normals"]
    train_dataset = PointCloudDataset(selection, config["num_points"], config["seed"],
                                      True, use_normals, config["cache_size"])
    dev_dataset = PointCloudDataset(dev_records, config["num_points"],
                                    stable_seed(config["seed"], "DEV"), False,
                                    use_normals, config["cache_size"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader_kwargs = dict(num_workers=config["num_workers"], worker_init_fn=seed_worker,
                         generator=generator, pin_memory=device.type == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_sampler=CircularBatchSampler(len(selection), config["batch_size"]),
        num_workers=config["num_workers"], worker_init_fn=seed_worker, generator=generator,
        pin_memory=device.type == "cuda")
    dev_generator = torch.Generator().manual_seed(stable_seed(config["seed"], "DEV-loader"))
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_size"], shuffle=False,
                            drop_last=False, num_workers=config["num_workers"],
                            worker_init_fn=seed_worker, generator=dev_generator,
                            pin_memory=device.type == "cuda")
    if len(train_loader) == 0 or len(dev_loader) == 0:
        raise A3Error("Training and DEV loaders must be nonempty")

    model = PointNetPPSeg(6 if use_normals else 3, 2, config["model_profile"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_updates"],
        eta_min=config["learning_rate"] * config["lr_min_ratio"])
    amp_enabled = config["amp"] and device.type == "cuda"
    # STATIC AMP loss scale (not the default dynamic 65536 with x2 growth). The
    # formal contract requires every one of the U updates to be a REAL optimizer
    # step (no skips), but dynamic scaling MUST occasionally skip+halve to
    # recalibrate and also GROWS back toward 65536, either of which would skip a
    # step and fail the run. A gradient diagnostic on the frozen configs showed
    # tiny, healthy gradients (fp32 grad-norm 2.5-8.7, max |grad| ~1.24); only the
    # default init_scale=65536 overflowed, and only at warmup step 1
    # (1.24*65536 > 65504=fp16 max). A FIXED scale of 4096 keeps AMP's speed/memory
    # benefit with a 13x overflow margin (65504/4096=16.0 vs observed 1.24) and,
    # with growth disabled, never drifts into an overflow mid-run -> zero skips,
    # fully deterministic. Loss scaling is mathematically transparent, so the
    # optimization trajectory is identical to any other non-overflowing scale;
    # amp stays True (frozen config unchanged).
    _AMP_STATIC_SCALE = 4096.0
    _AMP_NO_GROWTH = 10 ** 9  # growth_interval so large the scale never grows
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled, init_scale=_AMP_STATIC_SCALE,
            growth_interval=_AMP_NO_GROWTH)
        autocast = lambda enabled: torch.amp.autocast(device_type=device.type, enabled=enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled, init_scale=_AMP_STATIC_SCALE,
            growth_interval=_AMP_NO_GROWTH)
        autocast = torch.cuda.amp.autocast
    weights = class_weights_from_counts(class_weight_counts, device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)

    global_update, epoch, batch_in_epoch = 0, 0, 0
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"mIoU": -1.0, "global_update": 0, "metrics": None}
    interval_confusion = np.zeros((2, 2), dtype=np.int64)
    interval_loss = 0.0
    interval_batches = 0
    if resume is not None:
        resume = Path(resume).resolve()
        if resume != latest_path.resolve() and not resume.is_file():
            raise A3Error(f"Missing resume checkpoint: {resume}")
        checkpoint = restore_checkpoint(resume, model, optimizer, scheduler, scaler,
                                        generator, config_hash)
        if checkpoint["completed"]:
            raise A3Error("Checkpoint already completed max_updates")
        global_update = int(checkpoint["global_update"])
        epoch = int(checkpoint["epoch"])
        batch_in_epoch = int(checkpoint["batch_in_epoch"])
        best = checkpoint["best"]
        history = checkpoint["history"]
        interval_confusion = np.asarray(checkpoint["interval_confusion"], dtype=np.int64)
        if interval_confusion.shape != (2, 2):
            raise A3Error("Checkpoint interval confusion matrix is invalid")
        interval_loss = float(checkpoint["interval_loss"])
        interval_batches = int(checkpoint["interval_batches"])

    while global_update < config["max_updates"]:
        train_dataset.set_epoch(epoch)
        made_progress = False
        for batch_index, (features, target, _, _) in enumerate(train_loader):
            if batch_index < batch_in_epoch:
                continue
            made_progress = True
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if not torch.any(target != IGNORE_INDEX):
                raise A3Error(f"Training batch {batch_index} at epoch {epoch} contains only ignore labels")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                logits = model(features)
                loss = criterion(_flatten_logits(logits), target.reshape(-1))
            if not torch.isfinite(loss):
                raise A3Error(f"Non-finite loss at update {global_update + 1}")
            scaler.scale(loss).backward()
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if amp_enabled and scale_after < scale_before:
                raise A3Error(
                    f"AMP overflow skipped optimizer.step at requested update {global_update + 1}; "
                    "formal U counts only real optimizer updates")
            scheduler.step()
            global_update += 1
            interval_loss += float(loss.detach().cpu())
            interval_batches += 1
            update_confusion(interval_confusion, logits.detach(), target)

            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch >= len(train_loader):
                next_epoch, next_batch = epoch + 1, 0
            should_evaluate = (global_update % config["eval_interval"] == 0 or
                               global_update == config["max_updates"])
            if should_evaluate:
                train_metrics = confusion_metrics(interval_confusion)
                train_metrics["loss"] = interval_loss / max(interval_batches, 1)
                dev_metrics = evaluate(model, dev_loader, criterion, device, amp_enabled, autocast)
                entry = {
                    "global_update": global_update, "completed_epochs": next_epoch,
                    "batch_in_epoch": next_batch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train": train_metrics, "dev": dev_metrics,
                }
                history.append(entry)
                interval_confusion.fill(0)
                interval_loss = 0.0
                interval_batches = 0
                if dev_metrics["mIoU"] > best["mIoU"]:
                    best = {"mIoU": dev_metrics["mIoU"], "global_update": global_update,
                            "metrics": dev_metrics}
                    best_payload = checkpoint_payload(
                        model, optimizer, scheduler, scaler, config_hash, global_update,
                        next_epoch, next_batch, best, history, generator, False,
                        interval_confusion, interval_loss, interval_batches)
                    atomic_torch_save(output_dir / "checkpoint_best.pt", best_payload)
                atomic_write_json(output_dir / "history.json", {
                    "schema_version": SCHEMA_VERSION, "config_hash": config_hash,
                    "history": history, "best": best,
                })

            should_checkpoint = (global_update % config["checkpoint_interval"] == 0 or
                                 should_evaluate or global_update == config["max_updates"])
            epoch, batch_in_epoch = next_epoch, next_batch
            if should_checkpoint:
                payload = checkpoint_payload(
                    model, optimizer, scheduler, scaler, config_hash, global_update,
                    epoch, batch_in_epoch, best, history, generator,
                    global_update == config["max_updates"], interval_confusion,
                    interval_loss, interval_batches)
                atomic_torch_save(latest_path, payload)
            if global_update >= config["max_updates"]:
                break
        if not made_progress and batch_in_epoch >= len(train_loader):
            epoch, batch_in_epoch = epoch + 1, 0
        elif not made_progress:
            raise A3Error("Training loop made no progress")

    if best["metrics"] is None or not (output_dir / "checkpoint_best.pt").is_file():
        raise A3Error("No DEV evaluation produced a selectable checkpoint")
    result = {
        "schema_version": SCHEMA_VERSION, "status": "completed",
        "config_hash": config_hash, "run_id": config["run_id"],
        "method": config["method"], "budget_B": config["budget_B"],
        "max_updates_U": config["max_updates"], "global_update": global_update,
        "best": best, "last": history[-1], "class_weights": weights.detach().cpu().tolist(),
        "selection_manifest": "selection_manifest.csv",
        "per_source_exposure": "per_source_exposure.csv",
        "best_checkpoint": "checkpoint_best.pt", "latest_checkpoint": "checkpoint_latest.pt",
    }
    atomic_write_json(output_dir / "metrics.json", result)
    return result
