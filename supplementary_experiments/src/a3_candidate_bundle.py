"""Transactional publication of identity-bound A3 candidate bundles."""
from __future__ import annotations

import csv
import hashlib
import math
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from a3_io import (
    A3Error, CANDIDATE_LABEL_NPZ_FIELDS, CANDIDATE_MANIFEST_FIELDS,
    PSEUDO_LABEL_ASSET_SCHEMA, PSEUDO_LABEL_MANIFEST_SCHEMA,
    PSEUDO_LABEL_REPORT_SCHEMA, SCALE_IDENTITY_FIELDS, SCALE_SEMANTICS,
    atomic_save_npz, atomic_write_csv, atomic_write_json, copy_manifest_with_evidence,
    load_coordinate_scale_manifest, load_json, read_ply_xyzn, resolve_under,
    sha256_file, validate_coordinate_scale_source, validate_geometric_label_entry,
    validate_geometric_pseudo_payload, validate_pseudo_label_report_bindings,
    validate_scale_identity_fields, validate_scale_identity_match,
)
from quality_contract import load_quality_contract, load_quality_report

INVENTORY_FIELD_ORDER = (
    "sample_id", "parent_scan_id", "method", "ply_path", "point_count",
    "ply_sha256", "metadata_path", "metadata_sha256", "generation_seed",
)
INVENTORY_FIELDS = set(INVENTORY_FIELD_ORDER)
CANDIDATE_METADATA_SCHEMA = "a3-candidate-metadata-v3"
REQUIRED_PER_PARENT = math.ceil(150 / 9)


_LOADSIM_DIRECTION_POLICY = "seeded_random_per_operation_v3"
_TRAD_ALLOWED_METHODS = {"scale_and_rotation", "surface_noise", "rbf_deformation"}
_V3_TOP_FIELDS = {
    "schema_version", "id", "sample_id", "source", "data_type", "variant_id",
    "deform_params", "applied_methods", "scheme", "original_count", "final_count",
    "retention_ratio", "fallback", "parent_scan_id", "parent_ply_sha256", "method",
    "generation_seed", "generation_attempt", "generator", "generator_sha256",
    "coordinate_scale_manifest_sha256", *SCALE_IDENTITY_FIELDS,
}


def _finite_number(value: Any, where: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or (positive and float(value) <= 0)):
        qualifier = "positive and " if positive else ""
        raise A3Error(f"{where} must be {qualifier}finite")
    return float(value)


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise A3Error(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise A3Error(
            f"{where} fields mismatch; missing={sorted(expected-actual)}, "
            f"unknown={sorted(actual-expected)}")
    return value


def _strict_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise A3Error(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise A3Error(f"{where} must be >= {minimum}")
    return value


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise A3Error(f"{where} must be a non-empty string")
    return value


def _sha256(value: Any, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise A3Error(f"{where} must be a lowercase SHA-256")
    return value


def _unit_vector(value: Any, where: str) -> np.ndarray:
    if (not isinstance(value, list) or len(value) != 2 or
            any(isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value)):
        raise A3Error(f"{where} must be an explicit numeric pair")
    vector = np.asarray(value, dtype=np.float64)
    if (not np.isfinite(vector).all() or not math.isclose(
            float(np.linalg.norm(vector)), 1.0, rel_tol=0.0, abs_tol=1e-9)):
        raise A3Error(f"{where} must be a unit vector")
    return vector


def _validate_v3_common(metadata: dict[str, Any], method: str, sample_id: str) -> None:
    _exact_keys(metadata, _V3_TOP_FIELDS, f"Candidate {sample_id} metadata")
    if metadata["schema_version"] != CANDIDATE_METADATA_SCHEMA:
        raise A3Error(
            f"Candidate {sample_id} schema_version must be {CANDIDATE_METADATA_SCHEMA}")
    for key in ("id", "sample_id"):
        if metadata[key] != sample_id:
            raise A3Error(f"Candidate {sample_id} {key} mismatch")
    for key in ("source", "data_type", "parent_scan_id", "generator"):
        _nonempty_string(metadata[key], f"Candidate {sample_id} {key}")
    if metadata["generator"] != "step1_augmentation.py":
        raise A3Error(f"Candidate {sample_id} generator must be step1_augmentation.py")
    _strict_int(metadata["variant_id"], f"Candidate {sample_id} variant_id", minimum=0)
    expected_id = (
        f"{metadata['parent_scan_id']}_{method.lower()}_{metadata['variant_id']:03d}")
    if sample_id != expected_id:
        raise A3Error(
            f"Candidate {sample_id} does not bind parent/method/variant identity; "
            f"expected {expected_id}")
    original = _strict_int(
        metadata["original_count"], f"Candidate {sample_id} original_count", minimum=1)
    final = _strict_int(
        metadata["final_count"], f"Candidate {sample_id} final_count", minimum=1)
    retention = _finite_number(
        metadata["retention_ratio"], f"Candidate {sample_id} retention_ratio", positive=True)
    if retention > 1.0 or not math.isclose(
            retention, final / original, rel_tol=0.0, abs_tol=1e-12):
        raise A3Error(f"Candidate {sample_id} retention_ratio/count mismatch")
    if type(metadata["fallback"]) is not bool or metadata["fallback"]:
        raise A3Error(f"Candidate {sample_id} fallback must be false")
    _sha256(metadata["parent_ply_sha256"], f"Candidate {sample_id} parent_ply_sha256")
    if metadata["method"] != method:
        raise A3Error(f"Candidate {sample_id} method mismatch")
    _strict_int(metadata["generation_seed"], f"Candidate {sample_id} generation_seed", minimum=0)
    attempt = _strict_int(
        metadata["generation_attempt"], f"Candidate {sample_id} generation_attempt", minimum=0)
    if attempt >= 64:
        raise A3Error(f"Candidate {sample_id} generation_attempt must be < 64")
    _sha256(metadata["generator_sha256"], f"Candidate {sample_id} generator_sha256")


def _validate_v3_nested(
    metadata: dict[str, Any], method: str, sample_id: str,
) -> None:
    applied = metadata["applied_methods"]
    deform = metadata["deform_params"]
    if method == "TRAD":
        expected: set[str] = set()
        if "scale_and_rotation" in applied:
            expected.update({"linear_scale", "rotation_z_deg", "scale_semantics"})
        if "surface_noise" in applied:
            expected.add("surface_noise")
        if "rbf_deformation" in applied:
            expected.add("rbf")
        _exact_keys(deform, expected, f"Candidate {sample_id} TRAD deform_params")
        if "surface_noise" in applied:
            noise = _exact_keys(
                deform["surface_noise"], {"intensity", "axis_policy"},
                f"Candidate {sample_id} surface_noise")
            _finite_number(
                noise["intensity"], f"Candidate {sample_id} surface_noise intensity",
                positive=True)
            if noise["axis_policy"] != "z_only":
                raise A3Error(f"Surface noise must explicitly be z-only: {sample_id}")
        if "rbf_deformation" in applied:
            rbf = _exact_keys(
                deform["rbf"], {"n_ctrl", "intensity"},
                f"Candidate {sample_id} RBF deformation")
            _strict_int(rbf["n_ctrl"], f"Candidate {sample_id} RBF n_ctrl", minimum=1)
            _finite_number(
                rbf["intensity"], f"Candidate {sample_id} RBF intensity", positive=True)
        return
    _exact_keys(deform, {"loading_simulation"},
                f"Candidate {sample_id} LOADSIM deform_params")
    loading = _exact_keys(
        deform["loading_simulation"],
        {"version", "direction_policy", "n_unique_operation_directions",
         "coordinate_scale_to_m", "n_operations", "operations", "removal_ratio"},
        f"Candidate {sample_id} LOADSIM loading_simulation")
    if loading["version"] != "v3_seeded_random_direction_physical_units":
        raise A3Error(f"Candidate {sample_id} LOADSIM version is invalid")
    if loading["direction_policy"] != _LOADSIM_DIRECTION_POLICY:
        raise A3Error(f"Candidate {sample_id} LOADSIM direction policy is invalid")
    unique_count = _strict_int(
        loading["n_unique_operation_directions"],
        f"Candidate {sample_id} LOADSIM n_unique_operation_directions", minimum=1)
    _finite_number(loading["coordinate_scale_to_m"],
                   f"Candidate {sample_id} LOADSIM coordinate_scale_to_m", positive=True)
    operation_count = _strict_int(
        loading["n_operations"], f"Candidate {sample_id} LOADSIM n_operations", minimum=1)
    _finite_number(loading["removal_ratio"],
                   f"Candidate {sample_id} LOADSIM removal_ratio", positive=True)
    operations = loading["operations"]
    if (not isinstance(operations, list) or not operations or
            operation_count != len(operations) or unique_count != operation_count):
        raise A3Error(
            f"Candidate {sample_id} LOADSIM must record one unique seeded direction "
            "for every operation")
    for index, operation in enumerate(operations):
        operation = _exact_keys(
            operation, {"type", "angle_deg", "bucket_width_m",
                        "bucket_width_raw", "removed_points"},
            f"Candidate {sample_id} LOADSIM operation[{index}]")
        if operation["type"] != "advanced":
            raise A3Error(f"Candidate {sample_id} LOADSIM operation type is invalid")
        angle = _finite_number(
            operation["angle_deg"],
            f"Candidate {sample_id} LOADSIM operation[{index}] angle_deg")
        if not 0.0 <= angle < 360.0:
            raise A3Error(
                f"Candidate {sample_id} LOADSIM operation[{index}] angle_deg "
                "must satisfy 0 <= angle < 360")
        _finite_number(operation["bucket_width_m"],
                       f"Candidate {sample_id} LOADSIM bucket_width_m", positive=True)
        _finite_number(operation["bucket_width_raw"],
                       f"Candidate {sample_id} LOADSIM bucket_width_raw", positive=True)
        _strict_int(operation["removed_points"],
                    f"Candidate {sample_id} LOADSIM removed_points", minimum=1)


def validate_candidate_augmentation_metadata(
    metadata: dict[str, Any], method: str, sample_id: str,
) -> None:
    """Enforce the formal scale-only candidate metadata schema v3."""
    if not isinstance(metadata, dict):
        raise A3Error(f"Candidate augmentation metadata must be an object: {sample_id}")
    method = method.upper()
    if method not in {"TRAD", "LOADSIM"}:
        raise A3Error(f"Candidate method must be TRAD or LOADSIM: {sample_id}")
    _validate_v3_common(metadata, method, sample_id)
    applied = metadata["applied_methods"]
    deform = metadata["deform_params"]
    if (not isinstance(applied, list) or not applied or
            not all(isinstance(item, str) for item in applied) or
            len(applied) != len(set(applied))):
        raise A3Error(f"Candidate augmentation metadata is incomplete: {sample_id}")
    _validate_v3_nested(metadata, method, sample_id)
    validate_scale_identity_fields(metadata, f"Candidate {sample_id} metadata")
    hashes = (
        (metadata.get("coordinate_scale_manifest_sha256"),
         "coordinate_scale_manifest_sha256"),
        (metadata.get("scale_evidence_sha256"), "scale_evidence_sha256"),
    )
    for value, where in hashes:
        if (not isinstance(value, str) or len(value) != 64 or
                any(character not in "0123456789abcdef" for character in value)):
            raise A3Error(f"Candidate {sample_id} {where} must be a lowercase SHA-256")
    validate_coordinate_scale_source(
        metadata.get("coordinate_scale_source"),
        f"Candidate {sample_id} coordinate_scale_source")
    _finite_number(metadata.get("coordinate_scale_to_m"),
                   f"Candidate {sample_id} coordinate_scale_to_m", positive=True)

    method = method.upper()
    if method == "TRAD":
        if metadata.get("scheme") != "traditional_only" or not set(applied).issubset(
                _TRAD_ALLOWED_METHODS):
            raise A3Error(f"TRAD candidate uses a forbidden/non-propagating method: {sample_id}")
        rotation = deform.get("rotation_z_deg")
        linear_scale = deform.get("linear_scale")
        if "scale_and_rotation" in applied:
            _finite_number(linear_scale, f"Candidate {sample_id} linear_scale", positive=True)
            _finite_number(rotation, f"Candidate {sample_id} rotation_z_deg")
            if deform.get("scale_semantics") != SCALE_SEMANTICS:
                raise A3Error(f"Candidate transform has wrong scale_semantics: {sample_id}")
        elif linear_scale is not None or rotation is not None:
            raise A3Error(f"TRAD metadata records an unapplied scale/rotation: {sample_id}")
        if "surface_noise" in applied:
            noise = deform.get("surface_noise")
            if not isinstance(noise, dict) or noise.get("axis_policy") != "z_only":
                raise A3Error(f"Surface noise must explicitly be z-only: {sample_id}")
        elif "surface_noise" in deform:
            raise A3Error(f"TRAD metadata records unapplied surface noise: {sample_id}")
        if "rbf_deformation" in applied:
            rbf = deform.get("rbf")
            if not isinstance(rbf, dict):
                raise A3Error(f"TRAD metadata lacks applied RBF parameters: {sample_id}")
        elif "rbf" in deform:
            raise A3Error(f"TRAD metadata records unapplied RBF deformation: {sample_id}")
        return

    if method != "LOADSIM" or metadata.get("scheme") != "loading_only" or applied != [
            "loading_simulation"]:
        raise A3Error(f"LOADSIM candidate must contain loading simulation only: {sample_id}")
    loading = deform.get("loading_simulation")
    if not isinstance(loading, dict):
        raise A3Error(f"LOADSIM metadata lacks loading_simulation: {sample_id}")
    operations = loading.get("operations")
    operation_count = loading.get("n_operations")
    if (loading.get("version") != "v3_seeded_random_direction_physical_units" or
            loading.get("direction_policy") != _LOADSIM_DIRECTION_POLICY or
            not isinstance(operations, list) or not operations or
            operation_count != len(operations) or
            loading.get("n_unique_operation_directions") != operation_count):
        raise A3Error(
            f"LOADSIM must use one seeded random direction per operation: {sample_id}")
    loading_scale = _finite_number(
        loading.get("coordinate_scale_to_m"),
        f"Candidate {sample_id} loading coordinate_scale_to_m", positive=True)
    if not math.isclose(
            loading_scale, float(metadata["coordinate_scale_to_m"]),
            rel_tol=0.0, abs_tol=0.0):
        raise A3Error(f"LOADSIM inner/outer coordinate scale mismatch: {sample_id}")
    if (_finite_number(loading.get("removal_ratio"),
                       f"Candidate {sample_id} removal_ratio", positive=True) <= 0):
        raise A3Error(f"LOADSIM candidate lacks a nonzero operation: {sample_id}")


def _formal_candidate_generator() -> Path:
    """Return the one absolute generator path accepted by formal A3 replay."""
    return (Path(__file__).resolve().parent.parent / "step1_augmentation.py").resolve()


def _replay_candidate_asset(
    *, ply: Path, metadata_path: Path, source: Any,
    scale_row: dict[str, Any], scale_manifest_sha256: str,
    replay_verifier: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    """Invoke real replay by default; a callback exists only for isolated fixture tests."""
    if replay_verifier is None:
        parent = getattr(source, "path", None)
        if parent is None:
            # Compatibility is limited to structural unit fixtures. Formal loading
            # always has immutable SourceRecord parent paths and cannot take this branch.
            xyz_hash, count = _ply_identity(ply)
            return {"point_count": count, "canonical_xyz_sha256": xyz_hash}
        from build_a3_candidates import replay_candidate
        replay_verifier = replay_candidate
    else:
        parent = getattr(source, "path", None)
    result = replay_verifier(
        candidate_ply=ply, candidate_metadata=metadata_path,
        parent_ply=Path(parent).resolve() if parent is not None else None,
        generator_path=_formal_candidate_generator(), scale_row=scale_row,
        scale_manifest_sha256=scale_manifest_sha256)
    if not isinstance(result, dict):
        raise A3Error(f"Candidate replay verifier returned an invalid result: {ply}")
    return result


def _ply_identity(path: Path) -> tuple[str, int]:
    xyz, _ = read_ply_xyzn(path)
    canonical = np.ascontiguousarray(xyz.astype("<f4", copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest(), len(xyz)


def _validate_candidate_provenance(
    provenance: dict[str, Any], raw: dict[str, str], parent_sha256: str,
    metadata_sha256: str, metadata: dict[str, Any], sample_id: str,
    scale_row: dict[str, Any], scale_manifest_sha256: str,
) -> None:
    expected = {
        "parent_scan_id": raw["parent_scan_id"],
        "parent_source_ply_sha256": parent_sha256,
        "augmentation_metadata_sha256": metadata_sha256,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise A3Error(f"Candidate pseudo-label provenance {key} mismatch: {sample_id}")
    if provenance.get("point_count") != int(raw["point_count"]):
        raise A3Error(f"Candidate pseudo-label point-count mismatch: {sample_id}")
    if (metadata.get("sample_id") != sample_id or
            metadata.get("parent_scan_id") != raw["parent_scan_id"] or
            metadata.get("parent_ply_sha256") != parent_sha256):
        raise A3Error(f"Candidate augmentation metadata identity mismatch: {sample_id}")
    validate_scale_identity_match(
        provenance, scale_row, f"Candidate report provenance for {sample_id}")
    validate_scale_identity_match(
        metadata, scale_row, f"Candidate metadata for {sample_id}")
    if metadata.get("coordinate_scale_manifest_sha256") != scale_manifest_sha256:
        raise A3Error(f"Candidate metadata/report scale hash mismatch: {sample_id}")
    if (provenance.get("front_axis_computed") not in {"x", "y"} or
            provenance.get("front_sign_computed") != 1 or
            provenance.get("front_computation_source") != "geometry_internal"):
        raise A3Error(f"Candidate computed direction audit is invalid: {sample_id}")


def _replay_candidate_pseudo_report(
    report_path: Path, inventory_path: Path, parent_hashes: dict[str, str], *,
    config_path: Path, generator_path: Path, scale_manifest: Path,
) -> dict[str, Any]:
    """Deterministically replay every inventory candidate against one bound report."""
    from pseudo_label_replay import candidate_samples, replay_report
    samples = candidate_samples(inventory_path, parent_hashes, scale_manifest)
    return replay_report(
        report_path, samples, config_path=config_path,
        generator_path=generator_path, scale_manifest=scale_manifest)


def _copy_candidate_replay_inputs(
    inventory: Path, inventory_rows: list[dict[str, str]], staging_root: Path,
) -> Path:
    """Freeze the exact inventory bytes and all assets needed by standalone replay."""
    frozen_inventory = staging_root / "candidate_inventory.csv"
    shutil.copy2(inventory, frozen_inventory)
    if sha256_file(frozen_inventory) != sha256_file(inventory):
        raise A3Error("Frozen candidate replay inventory hash mismatch")
    for row in inventory_rows:
        sample_id = row["sample_id"]
        for field, declared_hash in (("ply_path", row["ply_sha256"]),
                                     ("metadata_path", row["metadata_sha256"])):
            source = resolve_under(inventory.parent, row[field])
            destination = resolve_under(staging_root, row[field])
            if destination.exists():
                raise A3Error(
                    f"Candidate replay asset path collision: {sample_id}/{field}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256_file(destination) != declared_hash:
                raise A3Error(
                    f"Frozen candidate replay asset hash mismatch: {sample_id}/{field}")
    return frozen_inventory


def stage_candidate_bundle(
    inventory: Path, labels_json: Path, quality_report: Path,
    quality_config: Path, train_sources: list[Any], source_manifest_path: Path,
    split_path: Path, staging_root: Path, *,
    canonical_scale_manifest: Path,
    pseudo_label_config: Path | None = None,
    pseudo_label_generator: Path | None = None,
    replay_verifier: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    """Validate every input and build a complete unpublished version directory."""
    bindings = {
        "canonical_scale_manifest": canonical_scale_manifest,
        "pseudo_label_config": pseudo_label_config,
        "pseudo_label_generator": pseudo_label_generator,
    }
    missing_bindings = [name for name, path in bindings.items() if path is None]
    if missing_bindings:
        raise A3Error(
            f"Formal candidate finalization requires explicit binding files: "
            f"{missing_bindings}")
    for path, name in ((inventory, "inventory"), (labels_json, "pseudo labels")):
        if not path.is_file():
            raise A3Error(f"Missing candidate {name}: {path}")
    if staging_root.exists():
        raise A3Error(f"Refusing to overwrite candidate staging/version directory: {staging_root}")
    contract = load_quality_contract(quality_config)
    quality_by_id, provenance = load_quality_report(quality_report, contract)
    with inventory.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_inventory_fields = reader.fieldnames
        if actual_inventory_fields != list(INVENTORY_FIELD_ORDER):
            actual = set(actual_inventory_fields or ())
            raise A3Error(
                "Candidate inventory fields mismatch; "
                f"missing={sorted(INVENTORY_FIELDS-actual)}, "
                f"unknown={sorted(actual-INVENTORY_FIELDS)}")
        inventory_rows = list(reader)
    if not inventory_rows or any(None in row for row in inventory_rows):
        raise A3Error("Invalid candidate inventory schema")
    ids = [row["sample_id"] for row in inventory_rows]
    if len(ids) != len(set(ids)) or set(ids) != set(quality_by_id):
        raise A3Error("Candidate inventory IDs must be unique and equal quality-report IDs")
    methods = {row["method"].upper() for row in inventory_rows}
    if len(methods) != 1 or methods.pop() not in {"TRAD", "LOADSIM"}:
        raise A3Error("A candidate bundle must contain exactly one method: TRAD or LOADSIM")
    method = inventory_rows[0]["method"].upper()
    if (provenance["candidate_inventory_sha256"] != sha256_file(inventory) or
            provenance["candidate_inventory_count"] != len(inventory_rows) or
            provenance["candidate_inventory_method"] != method):
        raise A3Error("Quality report candidate_inventory binding mismatch")
    train_ids = [source.scan_id for source in train_sources]
    sources_by_id = {source.scan_id: source for source in train_sources}
    parent_hashes = {source.scan_id: source.file_sha256 for source in train_sources}
    if set(row["parent_scan_id"] for row in inventory_rows) != set(train_ids):
        raise A3Error("Candidate parents must equal the frozen TRAIN-9 set")
    split_payload = load_json(split_path)
    if split_payload.get("source_manifest_sha256") != sha256_file(source_manifest_path):
        raise A3Error("Frozen split/source-manifest SHA-256 binding mismatch")
    payload = load_json(labels_json)
    labels_by_id = validate_geometric_pseudo_payload(payload, labels_json)
    pseudo_contract = validate_pseudo_label_report_bindings(
        payload, pseudo_label_config, pseudo_label_generator,
        canonical_scale_manifest)
    scale_manifest_hash = pseudo_contract["coordinate_scale_manifest_sha256"]
    scales = load_coordinate_scale_manifest(
        canonical_scale_manifest, parent_hashes, allow_extra=True)
    if set(labels_by_id) != set(ids):
        raise A3Error(
            "Candidate pseudo-label report IDs must exactly equal inventory IDs; "
            f"missing={sorted(set(ids)-set(labels_by_id))}, "
            f"extra={sorted(set(labels_by_id)-set(ids))}")
    pseudo_hash = sha256_file(labels_json)
    report_hash = sha256_file(quality_report)
    staging_root.mkdir(parents=True)
    try:
        frozen_quality_report = staging_root / "quality_report.json"
        frozen_pseudo_report = staging_root / "pseudo_label_report.json"
        shutil.copy2(quality_report, frozen_quality_report)
        shutil.copy2(labels_json, frozen_pseudo_report)
        if (sha256_file(frozen_quality_report) != report_hash or
                sha256_file(frozen_pseudo_report) != pseudo_hash):
            raise A3Error("Candidate report changed while it was frozen")
        evidence_dir = staging_root / "evidence"
        evidence_dir.mkdir()
        frozen_config = evidence_dir / "pseudo_label_config"
        frozen_generator = evidence_dir / "pseudo_label_generator.py"
        shutil.copy2(pseudo_label_config, frozen_config)
        shutil.copy2(pseudo_label_generator, frozen_generator)
        frozen_scale_manifest = copy_manifest_with_evidence(
            canonical_scale_manifest,
            evidence_dir / "scale" / "coordinate_scales_v2.csv", "scale")
        frozen_inventory = _copy_candidate_replay_inputs(
            inventory, inventory_rows, staging_root)
        if sha256_file(frozen_inventory) != provenance["candidate_inventory_sha256"]:
            raise A3Error("Frozen candidate inventory differs from quality-report binding")
        replayed_report = _replay_candidate_pseudo_report(
            frozen_pseudo_report, frozen_inventory, parent_hashes,
            config_path=frozen_config, generator_path=frozen_generator,
            scale_manifest=frozen_scale_manifest)
        labels_by_id = replayed_report["labels"]
        pseudo_contract = replayed_report["summary"]["algorithm_contract"]
        pseudo_hash = sha256_file(frozen_pseudo_report)
        report_hash = sha256_file(frozen_quality_report)
        scale_manifest_hash = pseudo_contract["coordinate_scale_manifest_sha256"]
        scales = load_coordinate_scale_manifest(
            frozen_scale_manifest, parent_hashes, allow_extra=True)
        inventory = frozen_inventory
        label_dir = staging_root / "labels"
        rows: list[dict[str, object]] = []
        seeds: set[int] = set()
        ply_hashes: set[str] = set()
        xyz_hashes: set[str] = set()
        for raw in inventory_rows:
            sample_id = raw["sample_id"]
            quality = quality_by_id[sample_id]
            try:
                count = int(raw["point_count"])
                seed = int(raw["generation_seed"])
            except ValueError as exc:
                raise A3Error(f"Invalid numeric inventory row: {sample_id}") from exc
            if seed in seeds:
                raise A3Error(f"Duplicate generation seed in bundle: {seed}")
            seeds.add(seed)
            ply = resolve_under(inventory.parent, raw["ply_path"])
            metadata = resolve_under(inventory.parent, raw["metadata_path"])
            actual_ply_hash = sha256_file(ply)
            actual_metadata_hash = sha256_file(metadata)
            if (actual_ply_hash != raw["ply_sha256"] or
                    actual_metadata_hash != raw["metadata_sha256"]):
                raise A3Error(f"Candidate asset hash mismatch: {sample_id}")
            if (quality["candidate_ply_sha256"] != raw["ply_sha256"] or
                    quality["source_ply_sha256"] != parent_hashes[raw["parent_scan_id"]]):
                raise A3Error(f"Quality report PLY identity mismatch: {sample_id}")
            if raw["ply_sha256"] in ply_hashes:
                raise A3Error(f"Duplicate candidate PLY bytes: {sample_id}")
            ply_hashes.add(raw["ply_sha256"])
            xyz_hash, actual_count = _ply_identity(ply)
            if actual_count != count:
                raise A3Error(
                    f"Candidate inventory point count differs from PLY: {sample_id}")
            if xyz_hash in xyz_hashes:
                raise A3Error(f"Duplicate canonical XYZ geometry: {sample_id}")
            xyz_hashes.add(xyz_hash)
            metadata_value = load_json(metadata)
            if metadata_value.get("generation_seed") != seed:
                raise A3Error(
                    f"Candidate inventory/metadata generation_seed mismatch: {sample_id}")
            parent_id = raw["parent_scan_id"]
            scale_row = scales[parent_id]
            replay = _replay_candidate_asset(
                ply=ply, metadata_path=metadata,
                source=sources_by_id[parent_id], scale_row=scale_row,
                scale_manifest_sha256=scale_manifest_hash,
                replay_verifier=replay_verifier)
            if (replay.get("point_count") != actual_count or
                    replay.get("canonical_xyz_sha256") != xyz_hash):
                raise A3Error(f"Candidate replay identity mismatch: {sample_id}")
            label, label_provenance = validate_geometric_label_entry(
                labels_by_id.get(sample_id), sample_id, count, raw["ply_sha256"],
                scale_row["scale_evidence_sha256"])
            _validate_candidate_provenance(
                label_provenance, raw, parent_hashes[parent_id],
                actual_metadata_hash, metadata_value, sample_id,
                scale_row, scale_manifest_hash)
            if not quality["passed"]:
                continue
            frozen_ply = resolve_under(staging_root, raw["ply_path"])
            frozen_metadata = resolve_under(staging_root, raw["metadata_path"])
            if (sha256_file(frozen_ply) != raw["ply_sha256"] or
                    sha256_file(frozen_metadata) != raw["metadata_sha256"]):
                raise A3Error(f"Frozen replay/manifest asset changed: {sample_id}")
            label_path = label_dir / f"{sample_id}.npz"
            atomic_save_npz(
                label_path, labels=label, sample_id=np.array(sample_id),
                source_scan_id=np.array(raw["parent_scan_id"]),
                source_ply_sha256=np.array(raw["ply_sha256"]),
                parent_source_ply_sha256=np.array(
                    parent_hashes[raw["parent_scan_id"]]),
                augmentation_metadata_sha256=np.array(actual_metadata_hash),
                schema_version=np.array(PSEUDO_LABEL_ASSET_SCHEMA),
                pseudo_label_report_sha256=np.array(pseudo_hash),
                config_sha256=np.array(pseudo_contract["config_sha256"]),
                generator_sha256=np.array(pseudo_contract["generator_sha256"]),
                coordinate_scale_manifest_sha256=np.array(scale_manifest_hash),
                front_axis_computed=np.array(
                    label_provenance["front_axis_computed"]),
                front_sign_computed=np.array(
                    label_provenance["front_sign_computed"], dtype=np.int8),
                front_computation_source=np.array("geometry_internal"),
                coordinate_scale_to_m=np.array(
                    label_provenance["coordinate_scale_to_m"]),
                coordinate_scale_source=np.array(
                    label_provenance["coordinate_scale_source"]),
                scale_evidence_type=np.array(
                    label_provenance["scale_evidence_type"]),
                scale_evidence_path=np.array(
                    label_provenance["scale_evidence_path"]),
                scale_evidence_sha256=np.array(
                    label_provenance["scale_evidence_sha256"]),
                scale_semantics=np.array(label_provenance["scale_semantics"]),
                effective_convex_radius_raw=np.array(
                    label_provenance["effective_convex_radius_raw"]),
            )
            rows.append({
                "schema_version": PSEUDO_LABEL_MANIFEST_SCHEMA,
                "manifest_schema": PSEUDO_LABEL_MANIFEST_SCHEMA,
                "sample_id": sample_id, "parent_scan_id": raw["parent_scan_id"],
                "method": method,
                "ply_path": frozen_ply.relative_to(staging_root).as_posix(),
                "point_count": count, "ply_sha256": raw["ply_sha256"],
                "canonical_xyz_sha256": xyz_hash,
                "parent_source_ply_sha256": parent_hashes[raw["parent_scan_id"]],
                "label_path": label_path.relative_to(staging_root).as_posix(),
                "label_sha256": sha256_file(label_path),
                "label_schema": PSEUDO_LABEL_ASSET_SCHEMA,
                "metadata_path": frozen_metadata.relative_to(staging_root).as_posix(),
                "metadata_sha256": actual_metadata_hash, "generation_seed": seed,
                "label_type": "geometric_pseudo_label",
                "pseudo_label_report_schema": PSEUDO_LABEL_REPORT_SCHEMA,
                "pseudo_label_report_path": "pseudo_label_report.json",
                "pseudo_label_config_path": "evidence/pseudo_label_config",
                "pseudo_label_generator_path": "evidence/pseudo_label_generator.py",
                "coordinate_scale_manifest_path":
                    "evidence/scale/coordinate_scales_v2.csv",
                "pseudo_label_source_json_sha256": pseudo_hash,
                "config_sha256": pseudo_contract["config_sha256"],
                "generator_sha256": pseudo_contract["generator_sha256"],
                "coordinate_scale_manifest_sha256": scale_manifest_hash,
                "front_axis_computed": label_provenance[
                    "front_axis_computed"],
                "front_sign_computed": label_provenance[
                    "front_sign_computed"],
                "front_computation_source": "geometry_internal",
                "coordinate_scale_to_m": label_provenance[
                    "coordinate_scale_to_m"],
                "coordinate_scale_source": label_provenance[
                    "coordinate_scale_source"],
                "scale_evidence_type": label_provenance[
                    "scale_evidence_type"],
                "scale_evidence_path": label_provenance[
                    "scale_evidence_path"],
                "scale_evidence_sha256": label_provenance[
                    "scale_evidence_sha256"],
                "scale_semantics": label_provenance["scale_semantics"],
                "effective_convex_radius_raw": label_provenance[
                    "effective_convex_radius_raw"],
                "strict_and_applied":
                    "true" if label_provenance["strict_and_applied"] else "false",
                "adaptive_relaxation_applied":
                    "true" if label_provenance["adaptive_relaxation_applied"] else "false",
                "quality_rule": provenance["quality_rule"],
                "quality_threshold_Q": provenance["quality_threshold_Q"],
                "quality_config_sha256": provenance["quality_config_sha256"],
                "quality_report_path": "quality_report.json",
                "quality_report_sha256": report_hash,
                "quality_F": quality["F"], "quality_P": quality["P"],
                "quality_D": quality["D"], "quality_Q": quality["Q"],
                "quality_passed": "true",
                "label_0": int(np.count_nonzero(label == 0)),
                "label_1": int(np.count_nonzero(label == 1)),
                "label_255": int(np.count_nonzero(label == 255)),
            })
        if not rows:
            raise A3Error("Q-only filtering retained no candidates")
        counts = {parent: sum(row["parent_scan_id"] == parent for row in rows)
                  for parent in train_ids}
        short = {parent: count for parent, count in counts.items()
                 if count < REQUIRED_PER_PARENT}
        if short:
            raise A3Error(
                f"Q-only retained {method} pool cannot support B=150 without reuse; "
                f"need {REQUIRED_PER_PARENT}/parent, short={short}")
        manifest = staging_root / "candidate_manifest.csv"
        atomic_write_csv(manifest, rows, list(CANDIDATE_MANIFEST_FIELDS))
        manifest_hash = sha256_file(manifest)
        audit = {
            "schema_version": "a3-quality-selection-audit-v3",
            **provenance, "method": method,
            "inventory_sha256": sha256_file(inventory),
            "pseudo_label_report_schema": PSEUDO_LABEL_REPORT_SCHEMA,
            "pseudo_label_report_sha256": pseudo_hash,
            "pseudo_label_config_sha256": pseudo_contract["config_sha256"],
            "pseudo_label_generator_sha256": pseudo_contract["generator_sha256"],
            "coordinate_scale_manifest_sha256": scale_manifest_hash,
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "splits_sha256": sha256_file(split_path),
            "train_parent_scans": train_ids,
            "inventory_count": len(inventory_rows), "retained_count": len(rows),
            "rejected_count": len(inventory_rows) - len(rows),
            "required_retained_per_parent_for_B150": REQUIRED_PER_PARENT,
            "retained_per_parent": counts,
            "retained_ids": [row["sample_id"] for row in rows],
            "candidate_manifest_sha256": manifest_hash,
            "quality_report_path": "quality_report.json",
        }
        atomic_write_json(staging_root / "quality_selection_audit.json", audit)
        verify_candidate_bundle(
            manifest, contract, train_sources, expected_method=method,
            canonical_scale_manifest=canonical_scale_manifest,
            replay_verifier=replay_verifier)
        return rows, audit
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def verify_candidate_bundle(
    manifest: Path, contract: dict[str, Any], train_sources: list[Any],
    expected_method: str, *, canonical_scale_manifest: Path,
    replay_verifier: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Re-read report/audit/assets and prove the consumed manifest is self-consistent."""
    if canonical_scale_manifest is None:
        raise A3Error("The canonical A3 coordinate-scale manifest must be supplied")
    canonical_scale_manifest = Path(canonical_scale_manifest)
    manifest = Path(manifest)
    manifest_hash = sha256_file(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_field_order = tuple(reader.fieldnames or ())
        rows = list(reader)
    if sha256_file(manifest) != manifest_hash:
        raise A3Error("Candidate manifest changed while it was loaded")
    if actual_field_order != CANDIDATE_MANIFEST_FIELDS:
        actual_fields = set(actual_field_order)
        expected_fields = set(CANDIDATE_MANIFEST_FIELDS)
        raise A3Error(
            "Candidate manifest fields/order mismatch; "
            f"missing={sorted(expected_fields-actual_fields)}, "
            f"unknown={sorted(actual_fields-expected_fields)}")
    if not rows:
        raise A3Error(f"Empty candidate manifest: {manifest}")
    report_paths = {row["quality_report_path"] for row in rows}
    if len(report_paths) != 1:
        raise A3Error("Candidate manifest must bind exactly one quality report path")
    report = resolve_under(manifest.parent, report_paths.pop())
    quality_by_id, provenance = load_quality_report(report, contract)
    if sha256_file(report) != rows[0]["quality_report_sha256"]:
        raise A3Error("Candidate manifest quality report hash mismatch")
    pseudo_paths = {row["pseudo_label_report_path"] for row in rows}
    config_paths = {row["pseudo_label_config_path"] for row in rows}
    generator_paths = {row["pseudo_label_generator_path"] for row in rows}
    scale_paths = {row["coordinate_scale_manifest_path"] for row in rows}
    if any(len(paths) != 1 for paths in (
            pseudo_paths, config_paths, generator_paths, scale_paths)):
        raise A3Error(
            "Candidate manifest must bind exactly one pseudo-label report and evidence set")
    pseudo_report = resolve_under(manifest.parent, pseudo_paths.pop())
    config_path = resolve_under(manifest.parent, config_paths.pop())
    generator_path = resolve_under(manifest.parent, generator_paths.pop())
    evidence_scale = resolve_under(manifest.parent, scale_paths.pop())
    pseudo_payload = load_json(pseudo_report)
    pseudo_labels = validate_geometric_pseudo_payload(pseudo_payload, pseudo_report)
    pseudo_contract = validate_pseudo_label_report_bindings(
        pseudo_payload, config_path, generator_path, evidence_scale)
    pseudo_hash = sha256_file(pseudo_report)
    retained_ids = {row["sample_id"] for row in rows}
    if (not retained_ids.issubset(pseudo_labels) or
            any(row["pseudo_label_source_json_sha256"] != pseudo_hash for row in rows)):
        raise A3Error("Candidate manifest pseudo-label report identity/hash mismatch")
    audit_path = manifest.parent / "quality_selection_audit.json"
    audit = load_json(audit_path)
    if audit.get("schema_version") != "a3-quality-selection-audit-v3":
        raise A3Error("Candidate bundle lacks a v3 quality selection audit")
    if audit.get("candidate_manifest_sha256") != manifest_hash:
        raise A3Error("Quality selection audit does not bind the consumed manifest")
    expected_method = expected_method.upper()
    train_ids = [source.scan_id for source in train_sources]
    sources_by_id = {source.scan_id: source for source in train_sources}
    source_hashes = {source.scan_id: source.file_sha256 for source in train_sources}
    scale_rows = load_coordinate_scale_manifest(
        evidence_scale, source_hashes, allow_extra=True)
    canonical_scale_rows = load_coordinate_scale_manifest(
        canonical_scale_manifest, source_hashes, allow_extra=True)
    if sha256_file(evidence_scale) != sha256_file(canonical_scale_manifest):
        raise A3Error(
            f"{expected_method} bundle scale manifest differs from configured canonical file")
    if scale_rows != canonical_scale_rows:
        raise A3Error(
            f"{expected_method} bundle scale evidence rows differ from canonical manifest")
    if (audit.get("method") != expected_method or
            audit.get("train_parent_scans") != train_ids or
            audit.get("retained_ids") != [row["sample_id"] for row in rows]):
        raise A3Error("Candidate selection audit identity/method/TRAIN order mismatch")
    if audit.get("quality_report_sha256") != provenance["quality_report_sha256"]:
        raise A3Error("Candidate selection audit quality report hash mismatch")
    if (audit.get("pseudo_label_report_schema") != PSEUDO_LABEL_REPORT_SCHEMA or
            audit.get("pseudo_label_report_sha256") != pseudo_hash or
            audit.get("pseudo_label_config_sha256") != pseudo_contract["config_sha256"] or
            audit.get("pseudo_label_generator_sha256") != pseudo_contract["generator_sha256"] or
            audit.get("coordinate_scale_manifest_sha256") !=
            pseudo_contract["coordinate_scale_manifest_sha256"]):
        raise A3Error("Candidate selection audit pseudo-label binding mismatch")
    replay_inventory = manifest.parent / "candidate_inventory.csv"
    if (audit.get("inventory_sha256") != sha256_file(replay_inventory) or
            provenance.get("candidate_inventory_sha256") !=
            audit.get("inventory_sha256")):
        raise A3Error("Candidate bundle replay inventory binding mismatch")
    replayed_report = _replay_candidate_pseudo_report(
        pseudo_report, replay_inventory, source_hashes,
        config_path=config_path, generator_path=generator_path,
        scale_manifest=evidence_scale)
    pseudo_labels = replayed_report["labels"]
    seen_ids: set[str] = set()
    seeds: set[int] = set()
    ply_hashes: set[str] = set()
    xyz_hashes: set[str] = set()
    counts = {parent: 0 for parent in train_ids}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen_ids or row["method"].upper() != expected_method:
            raise A3Error(f"Duplicate/wrong-method candidate: {sample_id}")
        seen_ids.add(sample_id)
        parent = row["parent_scan_id"]
        if parent not in counts:
            raise A3Error(f"Candidate parent outside frozen TRAIN: {parent}")
        counts[parent] += 1
        validate_scale_identity_match(
            row, scale_rows[parent], f"Candidate manifest row {sample_id}",
            allow_numeric_text=True)
        seed = int(row["generation_seed"])
        if seed in seeds:
            raise A3Error(f"Duplicate generation seed: {seed}")
        seeds.add(seed)
        if (row["schema_version"] != PSEUDO_LABEL_MANIFEST_SCHEMA or
                row["manifest_schema"] != PSEUDO_LABEL_MANIFEST_SCHEMA or
                row["label_schema"] != PSEUDO_LABEL_ASSET_SCHEMA or
                row["pseudo_label_report_schema"] != PSEUDO_LABEL_REPORT_SCHEMA or
                row["config_sha256"] != pseudo_contract["config_sha256"] or
                row["generator_sha256"] != pseudo_contract["generator_sha256"] or
                row["coordinate_scale_manifest_sha256"] !=
                pseudo_contract["coordinate_scale_manifest_sha256"] or
                row["scale_evidence_sha256"] !=
                scale_rows[parent]["scale_evidence_sha256"] or
                row["scale_semantics"] != SCALE_SEMANTICS or
                row["strict_and_applied"].strip().lower() not in ("true", "false") or
                row["adaptive_relaxation_applied"].strip().lower() not in ("true", "false") or
                (row["strict_and_applied"].strip().lower() ==
                 row["adaptive_relaxation_applied"].strip().lower())):
            raise A3Error(f"Candidate v3 pseudo-label contract mismatch: {sample_id}")
        if row["parent_source_ply_sha256"] != source_hashes[parent]:
            raise A3Error(f"Candidate parent-source identity mismatch: {sample_id}")
        quality = quality_by_id.get(sample_id)
        if quality is None or not quality["passed"]:
            raise A3Error(f"Manifest includes a Q-rejected/unknown candidate: {sample_id}")
        if (quality["candidate_ply_sha256"] != row["ply_sha256"] or
                quality["source_ply_sha256"] != source_hashes[parent]):
            raise A3Error(f"Candidate quality report PLY identity mismatch: {sample_id}")
        for key in ("F", "P", "D", "Q"):
            if not math.isclose(float(row[f"quality_{key}"]), quality[key],
                                rel_tol=0.0, abs_tol=1e-12):
                raise A3Error(f"Candidate quality {key} differs from report: {sample_id}")
        if (row["quality_rule"] != provenance["quality_rule"] or
                float(row["quality_threshold_Q"]) != provenance["quality_threshold_Q"] or
                row["quality_config_sha256"] != contract["sha256"] or
                row["quality_report_sha256"] != provenance["quality_report_sha256"] or
                row["quality_passed"].lower() != "true"):
            raise A3Error(f"Candidate quality provenance mismatch: {sample_id}")
        ply = resolve_under(manifest.parent, row["ply_path"])
        label = resolve_under(manifest.parent, row["label_path"])
        metadata = resolve_under(manifest.parent, row["metadata_path"])
        for path, expected in ((ply, row["ply_sha256"]),
                               (label, row["label_sha256"]),
                               (metadata, row["metadata_sha256"])):
            if sha256_file(path) != expected:
                raise A3Error(f"Candidate bundle asset hash mismatch: {path}")
        if row["ply_sha256"] in ply_hashes:
            raise A3Error(f"Duplicate candidate PLY bytes: {sample_id}")
        ply_hashes.add(row["ply_sha256"])
        xyz_hash, actual_count = _ply_identity(ply)
        if (actual_count != int(row["point_count"]) or
                xyz_hash != row["canonical_xyz_sha256"] or xyz_hash in xyz_hashes):
            raise A3Error(f"Canonical XYZ hash/count mismatch or duplicate: {sample_id}")
        xyz_hashes.add(xyz_hash)
        metadata_value = load_json(metadata)
        replay = _replay_candidate_asset(
            ply=ply, metadata_path=metadata, source=sources_by_id[parent],
            scale_row=scale_rows[parent],
            scale_manifest_sha256=pseudo_contract[
                "coordinate_scale_manifest_sha256"],
            replay_verifier=replay_verifier)
        if (replay.get("point_count") != actual_count or
                replay.get("canonical_xyz_sha256") != xyz_hash):
            raise A3Error(f"Candidate replay identity mismatch: {sample_id}")
        if (metadata_value.get("sample_id") != sample_id or
                metadata_value.get("parent_scan_id") != parent or
                metadata_value.get("parent_ply_sha256") != source_hashes[parent] or
                str(metadata_value.get("method", "")).upper() != expected_method or
                int(metadata_value.get("generation_seed", -1)) != seed):
            raise A3Error(f"Candidate metadata identity mismatch: {sample_id}")
        labels, label_provenance = validate_geometric_label_entry(
            pseudo_labels.get(sample_id), sample_id, actual_count, row["ply_sha256"],
            scale_rows[parent]["scale_evidence_sha256"])
        _validate_candidate_provenance(
            label_provenance, row, source_hashes[parent], row["metadata_sha256"],
            metadata_value, sample_id, scale_rows[parent],
            pseudo_contract["coordinate_scale_manifest_sha256"])
        if (row["front_axis_computed"] != label_provenance["front_axis_computed"] or
                int(row["front_sign_computed"]) !=
                label_provenance["front_sign_computed"] or
                row["front_computation_source"] != "geometry_internal" or
                row["coordinate_scale_source"] !=
                label_provenance["coordinate_scale_source"] or
                row["scale_evidence_sha256"] !=
                label_provenance["scale_evidence_sha256"] or
                row["scale_semantics"] != label_provenance["scale_semantics"] or
                not math.isclose(float(row["coordinate_scale_to_m"]),
                                 float(label_provenance["coordinate_scale_to_m"]),
                                 rel_tol=0.0, abs_tol=0.0) or
                not math.isclose(float(row["effective_convex_radius_raw"]),
                                 float(label_provenance["effective_convex_radius_raw"]),
                                 rel_tol=0.0, abs_tol=0.0)):
            raise A3Error(f"Candidate manifest geometry/scale provenance mismatch: {sample_id}")
        try:
            with np.load(label, allow_pickle=False) as archive:
                required_npz = set(CANDIDATE_LABEL_NPZ_FIELDS)
                if set(archive.files) != required_npz:
                    actual_npz = set(archive.files)
                    raise A3Error(
                        f"Candidate label NPZ v3 fields mismatch: {sample_id}; "
                        f"missing={sorted(required_npz-actual_npz)}, "
                        f"unknown={sorted(actual_npz-required_npz)}")
                scalar_bindings = {
                    "sample_id": sample_id, "source_scan_id": parent,
                    "source_ply_sha256": row["ply_sha256"],
                    "parent_source_ply_sha256": source_hashes[parent],
                    "augmentation_metadata_sha256": row["metadata_sha256"],
                    "schema_version": PSEUDO_LABEL_ASSET_SCHEMA,
                    "pseudo_label_report_sha256": pseudo_hash,
                    "config_sha256": pseudo_contract["config_sha256"],
                    "generator_sha256": pseudo_contract["generator_sha256"],
                    "coordinate_scale_manifest_sha256":
                        pseudo_contract["coordinate_scale_manifest_sha256"],
                    "scale_evidence_sha256":
                        scale_rows[parent]["scale_evidence_sha256"],
                    "scale_evidence_type": scale_rows[parent]["scale_evidence_type"],
                    "scale_evidence_path": scale_rows[parent]["scale_evidence_path"],
                    "front_axis_computed": label_provenance["front_axis_computed"],
                    "front_computation_source": "geometry_internal",
                    "coordinate_scale_source": label_provenance["coordinate_scale_source"],
                    "scale_semantics": SCALE_SEMANTICS,
                }
                for key, expected in scalar_bindings.items():
                    if str(np.asarray(archive[key]).reshape(()).item()) != expected:
                        raise A3Error(f"Candidate label NPZ {key} mismatch: {sample_id}")
                if (int(np.asarray(archive["front_sign_computed"]).reshape(()).item()) !=
                        label_provenance["front_sign_computed"] or
                        not np.array_equal(np.asarray(archive["labels"]), labels) or
                        float(np.asarray(archive["coordinate_scale_to_m"]).reshape(()).item()) !=
                        float(label_provenance["coordinate_scale_to_m"]) or
                        float(np.asarray(archive["effective_convex_radius_raw"]).reshape(()).item()) !=
                        float(label_provenance["effective_convex_radius_raw"])):
                    raise A3Error(f"Candidate label NPZ payload mismatch: {sample_id}")
        except (OSError, ValueError) as exc:
            raise A3Error(f"Invalid candidate label NPZ: {sample_id}") from exc
    short = {parent: count for parent, count in counts.items()
             if count < REQUIRED_PER_PARENT}
    if short or audit.get("retained_per_parent") != counts:
        raise A3Error(f"Candidate pool cannot support B=150 without reuse: {short}")
    if sha256_file(manifest) != manifest_hash:
        raise A3Error("Candidate manifest changed while its bundle was verified")
    return {"method": expected_method, "count": len(rows),
            "manifest_sha256": manifest_hash,
            "quality_report_sha256": provenance["quality_report_sha256"],
            "audit_sha256": sha256_file(audit_path)}


def publish_candidate_bundle(staging_root: Path, output_root: Path) -> Path:
    """Publish one complete directory atomically; never overwrite an existing bundle."""
    if output_root.exists():
        raise A3Error(f"Refusing to overwrite existing candidate bundle: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_root, output_root)
    return output_root / "candidate_manifest.csv"
