"""A3MB multi-backbone matched-budget training engine (reviewer supplement).

WHY THIS FILE EXISTS
--------------------
A reviewer noted that A3 (PointNet++ only) cannot, on its own, establish the
original manuscript's *five-backbone* "augmentation-benefit vs geometric
inductive-bias" relationship.  To keep the multi-backbone conclusion we
supplement a matched-budget experiment on all five backbones
(PointNet++, PointNeXt, KPConv, RandLA-Net, PTv3): same M=9 / B=150 / U=1330 /
training strategy / seeds, contrasting RAW_REPEAT vs LOADSIM, 3 seeds ->
5 x 2 x 3 = 30 runs.  Metrics: mIoU + F1 per backbone (mean +/- SD) and the
LoadSim - RAW_REPEAT difference.

DESIGN (isolated; A3's frozen code is NEVER modified)
-----------------------------------------------------
This module reuses A3's scientific core *verbatim by import* from
``a3_train_engine`` -- the matched-budget data selection (``build_selection``),
class-weighting, confusion/mIoU/F1 metrics, DEV evaluation, deterministic AMP,
checkpointing and provenance are all the same functions A3 uses.  Only two
things differ:
  1. ``validate_config`` here is A3MB-specific: it requires a ``backbone`` field,
     restricts methods to {RAW_REPEAT, LOADSIM}, locks B=150 / U=1330, and uses
     the A3MB run_id scheme ``A3MB_<backbone>_<METHOD>_B150_s<n>`` (A3's engine
     hard-codes the ``A3_...`` canonical id and the PointNet++ backbone).
  2. the model-construction line calls ``build_backbone(...)`` instead of
     ``PointNetPPSeg(...)`` so the same engine drives any of the five backbones.

HONEST FRAMING (goes in the reply letter)
-----------------------------------------
* The four non-PointNet++ backbones are self-contained *simplified
  reimplementations* (pure PyTorch, no spconv/pointnet2_ops/Minkowski) -- the
  SAME code that produced the manuscript's original multi-backbone numbers, now
  re-run under the matched budget.  They are NOT the official reference
  implementations; this is stated plainly.
* A3 defines only TRAIN + DEV groups (no held-out TEST split; the REF groups are
  A1 human labels, and the protocol sets
  ``reference_labels_for_training_or_selection: false``).  A3MB therefore reports
  the SAME held-out **DEV** mIoU+F1 that A3 reports (frozen original scans, never
  augmented).  Checkpoint is selected on DEV mIoU, so DEV numbers are optimistic
  -- but that bias is identical across all five backbones and both methods, so
  the LoadSim - RAW_REPEAT contrast (the reviewer's actual question) is unbiased.
  We report DEV, not a nonexistent "test", and say so.
"""
from __future__ import annotations

import socket
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- reuse A3's scientific core verbatim (single source of truth) ------------
from a3_data import (
    IGNORE_INDEX, CircularBatchSampler, PointCloudDataset,
    build_selection, load_frozen_sources, load_label_manifest,
    make_original_records, source_exposure_rows, validate_records,
)
from a3_io import (
    A3Error, ALL_VALID_ELIGIBILITY_RULE, atomic_torch_save, atomic_write_csv,
    atomic_write_json, canonical_hash, load_coordinate_scale_manifest, load_json,
    package_versions, sha256_file, stable_seed, seed_everything, seed_worker,
)
from quality_contract import load_quality_contract
from a3_train_engine import (
    _flatten_logits, utc_now, class_weights_from_counts, confusion_metrics,
    evaluate, selection_rows, checkpoint_payload, restore_checkpoint,
    update_confusion, expand_for_full_batches,
)
from a3mb_backbones import BACKBONES, build_backbone

SCHEMA_VERSION = "a3mb-run-v1"
A3MB_METHODS = ("RAW_REPEAT", "LOADSIM")   # reviewer design: drop TRAD
LOCK_BUDGET_B = 150
LOCK_MAX_UPDATES_U = 1330
SEED_TAGS = {20260911: "s1", 20260912: "s2", 20260913: "s3"}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """A3MB config validation: same frozen-protocol identities/hyperparameters as
    A3, but backbone-aware, methods locked to {RAW_REPEAT, LOADSIM}, B=150,
    U=1330, and the A3MB run_id scheme."""
    required = {
        "project_root", "splits", "source_manifest", "pseudo_label_manifest",
        "coordinate_scale_manifest", "output_dir", "method", "budget_B", "seed",
        "a3_protocol", "quality_config", "run_id", "backbone",
    }
    missing = sorted(required - set(config))
    if missing:
        raise A3Error(f"Missing config fields: {missing}")
    resolved = dict(config)

    backbone = str(resolved["backbone"]).lower()
    if backbone not in BACKBONES:
        raise A3Error(f"backbone must be one of {list(BACKBONES)}, got {backbone!r}")
    resolved["backbone"] = backbone

    method = str(resolved["method"]).upper()
    if method not in A3MB_METHODS:
        raise A3Error(f"A3MB method must be one of {list(A3MB_METHODS)} (TRAD dropped by design)")
    resolved["method"] = method

    for name in (
            "project_root", "splits", "source_manifest", "pseudo_label_manifest",
            "coordinate_scale_manifest", "output_dir", "a3_protocol", "quality_config"):
        resolved[name] = str(Path(resolved[name]).expanduser().resolve())
    candidate = resolved.get("candidate_manifest")
    resolved["candidate_manifest"] = (
        str(Path(candidate).expanduser().resolve()) if candidate else None)

    protocol = load_json(resolved["a3_protocol"])
    if protocol.get("schema_version") != "a3-protocol-v1" or protocol.get("frozen") is not True:
        raise A3Error("A3MB requires the frozen a3-protocol-v1 file (matched-budget identities)")

    debug = bool(resolved.get("debug", False))
    resolved["debug"] = debug
    if not debug:
        # Lock the shared matched-budget hyperparameters to the frozen protocol,
        # exactly as A3 does -- this is what makes A3MB comparable to A3.
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
        }
        for field, frozen_value in frozen_fields.items():
            if field in resolved and resolved[field] != frozen_value:
                raise A3Error(f"A3MB {field} is frozen at {frozen_value!r}, got {resolved[field]!r}")
            resolved[field] = frozen_value
        # A3MB locks B=150 and U=1330 (reviewer's matched-budget design).
        if resolved["budget_B"] != LOCK_BUDGET_B:
            raise A3Error(f"A3MB budget_B is locked at {LOCK_BUDGET_B}, got {resolved['budget_B']!r}")
        if resolved.get("max_updates") != LOCK_MAX_UPDATES_U:
            raise A3Error(f"A3MB max_updates is locked at {LOCK_MAX_UPDATES_U}, got {resolved.get('max_updates')!r}")
        # B=150 must be a valid budget for this method under the frozen grid.
        method_budget_grid = protocol.get("method_budget_grid")
        if not isinstance(method_budget_grid, dict) or LOCK_BUDGET_B not in (method_budget_grid.get(method) or []):
            raise A3Error(f"B={LOCK_BUDGET_B} not allowed for {method} by frozen method_budget_grid")
        if protocol.get("M") != 9 or protocol.get("ignore_index") != IGNORE_INDEX:
            raise A3Error("Frozen A3 protocol has invalid M or ignore_index")
        if resolved["seed"] not in protocol.get("seeds", []):
            raise A3Error(f"Seed {resolved['seed']} is outside the frozen A3 protocol")
        expected_run_id = f"A3MB_{backbone}_{method}_B{LOCK_BUDGET_B}_{SEED_TAGS[resolved['seed']]}"
        if resolved["run_id"] != expected_run_id:
            raise A3Error(f"run_id must be canonical A3MB id: {resolved['run_id']!r} != {expected_run_id!r}")
        if Path(resolved["output_dir"]).name != resolved["run_id"]:
            raise A3Error("formal output_dir leaf must equal run_id")
        requested_device = str(resolved.get("device", "cuda"))
        if not requested_device.startswith(str(protocol.get("device_type", "cuda"))):
            raise A3Error(f"A3MB device must be {protocol.get('device_type')}, got {requested_device}")

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


def provenance(config, config_hash, inputs, source_ids, class_weight_counts,
               selected_class_counts):
    return {
        "schema_version": SCHEMA_VERSION, "created_at": utc_now(),
        "config_hash": config_hash, "run_id": config["run_id"],
        "backbone": config["backbone"],
        "method": config["method"], "budget_B": config["budget_B"],
        "seed": config["seed"], "max_updates_U": config["max_updates"],
        "M": len(source_ids["TRAIN"]),
        "train_parent_scans": source_ids["TRAIN"],
        "dev_parent_scans": source_ids["DEV"],
        "dev_policy": "original frozen DEV scans only; no augmentation",
        "selection_policy": "shared balanced source-occurrence vector v1 (same as A3)",
        "quality_filter": config["quality_contract"],
        "checkpoint_selection": "maximum DEV mIoU; no human reference labels",
        "eval_metric_note": "A3 has no held-out TEST group; reported metrics are "
                            "held-out DEV (frozen originals). Selection-on-DEV bias "
                            "is identical across backbones/methods, so LoadSim-"
                            "RAW_REPEAT is unbiased.",
        "backbone_impl_note": "non-PointNet++ backbones are self-contained simplified "
                              "reimplementations (same code as the manuscript's original "
                              "multi-backbone numbers); not official reference impls.",
        "class_weight_counts_from_frozen_train_originals": class_weight_counts,
        "selected_sample_class_counts": selected_class_counts,
        "input_hashes": inputs, "software": package_versions(),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "python_executable": sys.executable},
    }


def train_a3mb(raw_config: dict[str, Any], resume: Path | None = None,
               preflight_only: bool = False) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

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
    all_source_hashes = {s.scan_id: s.file_sha256 for s in train_sources + dev_sources}
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

    config.setdefault("run_id", f"A3MB_{config['backbone']}_{config['method']}_B{config['budget_B']}_s{config['seed']}")
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
        "rule": quality_contract["rule"], "threshold_Q": quality_contract["threshold_Q"],
        "weights": quality_contract["weights"],
        "applies_to": "TRAD/LOADSIM augmented candidates; not RAW_REPEAT originals",
    }
    if config["candidate_manifest"]:
        input_hashes["candidate_manifest"] = sha256_file(config["candidate_manifest"])
    implementation_dir = Path(__file__).resolve().parent
    for module_name in (
        "a3_io.py", "a3_model.py", "a3_data.py", "a3_candidate_bundle.py",
        "quality_contract.py", "a3_train_engine.py",
        "a3mb_backbones.py", "a3mb_train_engine.py", "a3mb_train.py",
    ):
        mp = implementation_dir / module_name
        if mp.is_file():
            input_hashes[f"implementation/{module_name}"] = sha256_file(mp)
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
                  "backbone": config["backbone"], "selection_count": len(selection),
                  "class_weight_counts": class_weight_counts,
                  "selected_class_counts": selected_class_counts}
        atomic_write_json(output_dir / "metrics.json", result)
        return result

    use_normals = config["use_normals"]
    train_dataset = PointCloudDataset(selection, config["num_points"], config["seed"],
                                      True, use_normals, config["cache_size"])
    dev_dataset = PointCloudDataset(dev_records, config["num_points"],
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

    # === THE ONLY MODEL-CONSTRUCTION DIFFERENCE FROM A3 ======================
    # A3 hard-codes PointNetPPSeg(...); A3MB dispatches to the requested backbone.
    model = build_backbone(config["backbone"], 6 if use_normals else 3, 2,
                           config["model_profile"]).to(device)
    # =========================================================================
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
        "backbone": config["backbone"],
        "method": config["method"], "budget_B": config["budget_B"],
        "max_updates_U": config["max_updates"], "global_update": global_update,
        # reviewer-requested headline metrics, surfaced flat for easy aggregation.
        # A3 has no TEST group -> these are held-out DEV (see provenance note).
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

