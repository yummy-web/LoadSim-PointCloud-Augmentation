"""PointWOLF matched-budget training engine (reviewer supplement R2-3).

WHY THIS FILE EXISTS
--------------------
R2-3 asked for a *recent point-cloud-specific* augmentation baseline (not just
rotation/scale/noise = TRAD).  This engine trains PointNet++ on PointWOLF
candidates under the SAME matched-budget protocol A3 already ran
(M=9 / U=1330 / seeds / PointNet++ / fixed DEV), so PointWOLF drops straight
into the finished A3 comparison as a fourth method:
    RAW_REPEAT  vs  TRAD  vs  LOADSIM  (already run in A3)  vs  POINTWOLF (here).

The completed A3 runs (RAW_REPEAT/TRAD/LOADSIM at matched budget) are REUSED as
comparators; only PointWOLF is new -> preregistered B=150 x 3 seeds = 3 runs
(optionally B=50 x 3 secondary).

DESIGN (isolated; A3's frozen code is NEVER modified) -- mirrors a3mb_train_engine
----------------------------------------------------------------------------------
Reuses A3's scientific core verbatim by import (class weighting, confusion/mIoU/F1
metrics, DEV evaluate, deterministic AMP, checkpointing).  Differs only in:
  1. validate_config: method locked to {POINTWOLF}, budgets from the frozen
     A3 grid, run_id scheme PW_<METHOD>_B<b>_s<n>.
  2. selection/dataset: PointWOLF candidate bank (pw_data) instead of the
     Q-v1/all-valid loaders.  Per-source exposure vector is A3's, so the budget
     is matched byte-for-byte.

HONEST FRAMING (reply letter)
-----------------------------
* PointWOLF is our fixed-parameter scene-segmentation adapter of the released
  classification method (pw_adapter); we state it is an adapted reimplementation,
  not the official released code, and that its labels propagate by identity
  because the warp preserves point count/order (verified in pw_tests).
* A3 has no held-out TEST group (REF groups are A1 human labels; protocol sets
  reference_labels_for_training_or_selection=false).  We report held-out DEV
  mIoU+F1 exactly as A3/A3MB do; selection-on-DEV bias is identical across all
  methods, so the POINTWOLF - RAW_REPEAT contrast is unbiased.
* Expected result given A3 (RAW_REPEAT >= TRAD ~ LOADSIM at every budget):
  PointWOLF is not expected to beat RAW_REPEAT either -> reinforces the
  immediate-augmentation-saturation finding.  We report whatever we observe.
"""
from __future__ import annotations

import platform
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np

from a3_data import (
    IGNORE_INDEX, CircularBatchSampler,
    load_frozen_sources, load_label_manifest, make_original_records,
    source_exposure_rows, validate_records,
)
from a3_io import (
    A3Error, atomic_torch_save, atomic_write_csv, atomic_write_json,
    canonical_hash, load_coordinate_scale_manifest, load_json, package_versions,
    seed_everything, seed_worker, sha256_file, stable_seed,
)
from quality_contract import load_quality_contract
from a3_train_engine import (
    _flatten_logits, utc_now, class_weights_from_counts, confusion_metrics,
    evaluate, selection_rows, checkpoint_payload, restore_checkpoint,
    update_confusion,
)
from pw_data import (
    PW_METHOD, PW_EXPECTED_TOTAL, PWPointCloudDataset,
    build_pw_selection, load_pw_manifest_rows,
)

SCHEMA_VERSION = "pw-run-v1"
PW_METHODS = ("POINTWOLF",)
SEED_TAGS = {20260911: "s1", 20260912: "s2", 20260913: "s3"}
# Preregistered budgets: B=150 primary (principal high-exposure point; all four
# methods exist there), B=50 secondary (TRAD also exists there).  Chosen BEFORE
# any PointWOLF run; never selected on a result.
PW_ALLOWED_BUDGETS = (50, 150)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "project_root", "splits", "source_manifest", "pseudo_label_manifest",
        "coordinate_scale_manifest", "output_dir", "method", "budget_B", "seed",
        "a3_protocol", "quality_config", "run_id", "pw_bank_dir",
    }
    missing = sorted(required - set(config))
    if missing:
        raise A3Error(f"Missing config fields: {missing}")
    resolved = dict(config)

    method = str(resolved["method"]).upper()
    if method not in PW_METHODS:
        raise A3Error(f"PW method must be one of {list(PW_METHODS)}")
    resolved["method"] = method

    for name in (
            "project_root", "splits", "source_manifest", "pseudo_label_manifest",
            "coordinate_scale_manifest", "output_dir", "a3_protocol",
            "quality_config", "pw_bank_dir"):
        resolved[name] = str(Path(resolved[name]).expanduser().resolve())

    protocol = load_json(resolved["a3_protocol"])
    if protocol.get("schema_version") != "a3-protocol-v1" or protocol.get("frozen") is not True:
        raise A3Error("PW requires the frozen a3-protocol-v1 file (matched-budget identities)")

    debug = bool(resolved.get("debug", False))
    resolved["debug"] = debug
    if not debug:
        frozen_fields = {
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
            "batch_size": protocol.get("batch_size"),
            "max_updates": protocol.get("max_updates_U"),
        }
        for field, frozen_value in frozen_fields.items():
            if field in resolved and resolved[field] != frozen_value:
                raise A3Error(f"PW {field} is frozen at {frozen_value!r}, got {resolved[field]!r}")
            resolved[field] = frozen_value
        if resolved["budget_B"] not in PW_ALLOWED_BUDGETS:
            raise A3Error(
                f"PW budget_B must be one of the preregistered {PW_ALLOWED_BUDGETS}, "
                f"got {resolved['budget_B']!r}")
        # B must also be valid for the matched comparators under the frozen grid
        # (TRAD anchors exist at 50 and 150, so both PW budgets are directly
        # comparable to an already-run A3 TRAD/LOADSIM/RAW_REPEAT number).
        grid = protocol.get("method_budget_grid", {})
        for comparator in ("TRAD", "LOADSIM", "RAW_REPEAT"):
            if resolved["budget_B"] not in (grid.get(comparator) or []):
                raise A3Error(
                    f"PW B={resolved['budget_B']} is not a matched A3 {comparator} budget")
        if protocol.get("M") != 9 or protocol.get("ignore_index") != IGNORE_INDEX:
            raise A3Error("Frozen A3 protocol has invalid M or ignore_index")
        if resolved["seed"] not in protocol.get("seeds", []):
            raise A3Error(f"Seed {resolved['seed']} is outside the frozen A3 protocol")
        expected_run_id = f"PW_{method}_B{resolved['budget_B']}_{SEED_TAGS[resolved['seed']]}"
        if resolved["run_id"] != expected_run_id:
            raise A3Error(f"run_id must be canonical PW id: {resolved['run_id']!r} != {expected_run_id!r}")
        if Path(resolved["output_dir"]).name != resolved["run_id"]:
            raise A3Error("formal output_dir leaf must equal run_id")
        requested_device = str(resolved.get("device", "cuda"))
        if not requested_device.startswith(str(protocol.get("device_type", "cuda"))):
            raise A3Error(f"PW device must be {protocol.get('device_type')}, got {requested_device}")

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


def provenance(config, config_hash, inputs, source_ids, class_weight_counts,
               selected_class_counts):
    return {
        "schema_version": SCHEMA_VERSION, "created_at": utc_now(),
        "config_hash": config_hash, "run_id": config["run_id"],
        "method": config["method"], "budget_B": config["budget_B"],
        "seed": config["seed"], "max_updates_U": config["max_updates"],
        "M": len(source_ids["TRAIN"]),
        "train_parent_scans": source_ids["TRAIN"],
        "dev_parent_scans": source_ids["DEV"],
        "dev_policy": "original frozen DEV scans only; no augmentation",
        "selection_policy": "shared balanced source-occurrence vector v1 (same as A3)",
        "augmentation": "PointWOLF (fixed-parameter scene-segmentation adapter, "
                        "pw-adapter-v1); labels propagate by identity (warp "
                        "preserves point count/order); normals recomputed via "
                        "kNN-PCA on the warped cloud, sign-oriented to parent.",
        "checkpoint_selection": "maximum DEV mIoU; no human reference labels",
        "eval_metric_note": "A3 has no held-out TEST group; reported metrics are "
                            "held-out DEV (frozen originals). Selection-on-DEV bias "
                            "is identical across methods, so POINTWOLF-RAW_REPEAT is unbiased.",
        "matched_comparators": "RAW_REPEAT/TRAD/LOADSIM at the same B/seed were run "
                               "in A3 (reused, not re-run).",
        "class_weight_counts_from_frozen_train_originals": class_weight_counts,
        "selected_sample_class_counts": selected_class_counts,
        "input_hashes": inputs, "software": package_versions(),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "python_executable": sys.executable},
    }


def train_pw(raw_config: dict[str, Any], resume: Path | None = None,
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
    pw_bank_dir = Path(config["pw_bank_dir"])
    train_sources, dev_sources = load_frozen_sources(
        project_root, split_path, source_manifest_path)
    all_source_hashes = {s.scan_id: s.file_sha256 for s in train_sources + dev_sources}
    load_coordinate_scale_manifest(scale_manifest_path, all_source_hashes)
    if not config["debug"]:
        protocol = load_json(config["a3_protocol"])
        if [s.scan_id for s in train_sources] != protocol.get("train_parent_scans"):
            raise A3Error("Frozen TRAIN identities/order mismatch")
        if [s.scan_id for s in dev_sources] != protocol.get("dev_parent_scans"):
            raise A3Error("Frozen DEV identities/order mismatch")
    original_labels = load_label_manifest(
        label_manifest_path, train_sources + dev_sources, config["verify_hashes"],
        canonical_scale_manifest=scale_manifest_path)
    quality_contract = load_quality_contract(config["quality_config"])

    selection, vector = build_pw_selection(
        config["budget_B"], config["seed"], train_sources, pw_bank_dir,
        config["verify_hashes"])
    pw_rows = load_pw_manifest_rows(pw_bank_dir)
    original_train_records = make_original_records(train_sources, original_labels)
    dev_base = make_original_records(dev_sources, original_labels)
    dev_records = [record for record in dev_base for _ in range(config["dev_crops_per_scan"])]

    # Validate PW selection labels via the PW dataset's own reader (validate_records
    # in a3_data assumes Q-v1/all-valid bindings, which PW records don't carry).
    selected_class_counts = _validate_pw_selection(selection, pw_rows, config["verify_hashes"])
    class_weight_counts, _ = validate_records(original_train_records, config["verify_hashes"])
    validate_records(dev_base, config["verify_hashes"])

    config["train_parent_scans"] = [s.scan_id for s in train_sources]
    config["dev_parent_scans"] = [s.scan_id for s in dev_sources]
    config["M"] = len(train_sources)
    config["source_occurrence_vector_hash"] = canonical_hash(vector)
    selection_data = selection_rows(selection, vector)
    input_hashes = {
        "a3_protocol": sha256_file(config["a3_protocol"]),
        "quality_config": quality_contract["sha256"],
        "splits": sha256_file(split_path), "source_manifest": sha256_file(source_manifest_path),
        "coordinate_scale_manifest": sha256_file(scale_manifest_path),
        "pseudo_label_manifest": sha256_file(label_manifest_path),
        "pw_candidate_manifest": sha256_file(pw_bank_dir / "candidate_manifest.csv"),
        "pw_bank_receipt": sha256_file(pw_bank_dir / "bank_receipt.json"),
    }
    config["quality_contract"] = {
        "rule": quality_contract["rule"], "threshold_Q": quality_contract["threshold_Q"],
        "weights": quality_contract["weights"],
        "applies_to": "PointWOLF candidates are all-valid by construction; Q not a gate.",
    }
    implementation_dir = Path(__file__).resolve().parent
    for module_name in (
        "a3_io.py", "a3_model.py", "a3_data.py", "quality_contract.py",
        "a3_train_engine.py", "pw_adapter.py", "pw_data.py",
        "pw_train_engine.py", "pw_train.py",
    ):
        mp = implementation_dir / module_name
        if mp.is_file():
            input_hashes[f"implementation/{module_name}"] = sha256_file(mp)
    config["input_hashes"] = input_hashes
    config["selection_hash"] = canonical_hash(selection_data)
    config_hash = canonical_hash(config)

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
    else:
        atomic_write_json(output_dir / "resolved_config.json", {**config, "config_hash": config_hash})
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

    use_normals = config["use_normals"]
    train_dataset = PWPointCloudDataset(selection, config["num_points"], config["seed"],
                                        True, use_normals, config["cache_size"], pw_rows)
    dev_dataset = PWPointCloudDataset(dev_records, config["num_points"],
                                      stable_seed(config["seed"], "DEV"), False,
                                      use_normals, config["cache_size"], pw_rows)
    # DEV records are frozen ORIGINALS (RAW_ORIGINAL); they are not PW candidates,
    # so the DEV dataset must read their ORIGINAL labels, not the PW schema. Use
    # the A3 dataset for DEV to stay byte-identical to A3's DEV pipeline.
    from a3_data import PointCloudDataset as _A3Dataset
    dev_dataset = _A3Dataset(dev_records, config["num_points"],
                             stable_seed(config["seed"], "DEV"), False,
                             use_normals, config["cache_size"])
    generator = torch.Generator().manual_seed(config["seed"])
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
    _AMP_STATIC_SCALE = 4096.0
    _AMP_NO_GROWTH = 10 ** 9
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
    best_dev = best["metrics"]
    result = {
        "schema_version": SCHEMA_VERSION, "status": "completed",
        "config_hash": config_hash, "run_id": config["run_id"],
        "method": config["method"], "budget_B": config["budget_B"],
        "max_updates_U": config["max_updates"], "global_update": global_update,
        "eval_split": "DEV",
        "dev_mIoU": float(best_dev["mIoU"]),
        "dev_f1_1": float(best_dev["f1_1"]),
        "dev_IoU_0": float(best_dev["IoU_0"]),
        "dev_IoU_1": float(best_dev["IoU_1"]),
        "best": best, "last": history[-1], "class_weights": weights.detach().cpu().tolist(),
        "selection_manifest": "selection_manifest.csv",
        "per_source_exposure": "per_source_exposure.csv",
        "best_checkpoint": "checkpoint_best.pt", "latest_checkpoint": "checkpoint_latest.pt",
    }
    atomic_write_json(output_dir / "metrics.json", result)
    return result


def _validate_pw_selection(selection, pw_rows, verify_hashes) -> dict[str, int]:
    """Validate PW selected records' labels via the PW label reader and return
    aggregate class counts (mirrors a3_data.validate_records semantics for PW)."""
    from pw_data import _pw_label_snapshot
    from a3_data import _load_ply_snapshot as _ply
    class_counts = {"0": 0, "1": 0, "255": 0}
    from collections import Counter as _Counter
    mult = _Counter((r.sample_id, str(r.path), str(r.label_path)) for r in selection)
    rep = {(r.sample_id, str(r.path), str(r.label_path)): r for r in selection}
    for identity, m in mult.items():
        record = rep[identity]
        declared = record.file_sha256 if verify_hashes else None
        points, _, _ = _ply(record.path, declared, "PW preflight PLY")
        if len(points) != record.point_count:
            raise A3Error(f"PW point count mismatch for {record.sample_id}")
        row = pw_rows.get(record.sample_id)
        if row is None:
            raise A3Error(f"PW preflight missing manifest row for {record.sample_id}")
        labels = _pw_label_snapshot(record.label_path, row, len(points))
        for label in (0, 1, 255):
            class_counts[str(label)] += m * int(np.count_nonzero(labels == label))
    if class_counts["0"] == 0 or class_counts["1"] == 0:
        raise A3Error(f"Both train classes must be represented, got {class_counts}")
    return class_counts
