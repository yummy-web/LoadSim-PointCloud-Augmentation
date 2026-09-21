"""Deterministically build/finalize frozen TRAD or LOADSIM A3 candidates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from a3_candidate_bundle import (
    CANDIDATE_METADATA_SCHEMA, INVENTORY_FIELD_ORDER,
    publish_candidate_bundle, stage_candidate_bundle,
    validate_candidate_augmentation_metadata,
)
from a3_data import load_frozen_sources
from a3_io import (
    A3Error, SCALE_IDENTITY_FIELDS, SCALE_SEMANTICS, atomic_write_csv,
    atomic_write_json, load_coordinate_scale_manifest, load_json, read_ply_xyzn,
    resolve_under, sha256_file, stable_seed,
)
from quality_contract import load_quality_contract, load_quality_report

INVENTORY_FIELDS = list(INVENTORY_FIELD_ORDER)
_AUGMENTATION_MODULE_NAME = "_lsda_a3_step1_augmentation"

# Formal pool contract: TRAIN-9 x [0, FORMAL_VARIANTS_PER_SOURCE) candidates/method.
FORMAL_VARIANTS_PER_SOURCE = 32
MIN_VARIANTS_PER_SOURCE = math.ceil(150 / 9)  # >=17 before Q filtering


def _canonical_xyz_sha256(candidate_ply: Path) -> str:
    """Hash the canonical little-endian float32 XYZ bytes of a candidate PLY."""
    xyz, _ = read_ply_xyzn(candidate_ply)
    return hashlib.sha256(_float32_bytes(xyz)).hexdigest()


def _formal_generator_path() -> Path:
    return (Path(__file__).resolve().parent.parent / "step1_augmentation.py").resolve()


def _augmentation_api_from_path(generator_path: Path):
    """Load the declared generator by absolute path without consulting import caches."""
    generator_path = Path(generator_path)
    if not generator_path.is_absolute():
        raise A3Error(f"Candidate generator path must be absolute: {generator_path}")
    generator_path = generator_path.resolve()
    if not generator_path.is_file():
        raise A3Error(f"Missing step1_augmentation.py: {generator_path}")
    try:
        spec = importlib.util.spec_from_file_location(
            _AUGMENTATION_MODULE_NAME, generator_path)
        if spec is None or spec.loader is None:
            raise ImportError("no import loader was created")
        augmentation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(augmentation)
    except Exception as exc:
        raise A3Error(
            f"Cannot load step1_augmentation.py from {generator_path}: {exc}") from exc
    module_file = getattr(augmentation, "__file__", None)
    try:
        loaded_path = Path(module_file).resolve() if isinstance(module_file, str) else None
    except (OSError, RuntimeError):
        loaded_path = None
    if loaded_path != generator_path:
        raise A3Error(
            "step1_augmentation module file mismatch: "
            f"expected {generator_path}, loaded {module_file!r}")
    required = ["_load_ply_numpy", "preprocess", "get_data_config", "augment_one",
                "estimate_normals_numpy", "_save_ply_numpy"]
    missing = [name for name in required if not hasattr(augmentation, name)]
    if missing:
        raise A3Error(f"step1_augmentation API is missing: {missing}")
    if not getattr(augmentation, "HAS_SCIPY", False):
        raise A3Error("SciPy is required; candidate generation may not use degraded augmentation")
    return augmentation


def _augmentation_api(project_root: Path):
    augmentation_path = (Path(project_root).resolve() / "step1_augmentation.py").resolve()
    return _augmentation_api_from_path(augmentation_path)


def _atomic_save_ply(api: Any, path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        api._save_ply_numpy(temporary, points, normals)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def candidate_generation_seed(
    root_seed: int, method: str, parent_scan_id: str, variant_id: int,
) -> int:
    """Derive the identity-bound seed stored in candidate metadata."""
    if (isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0 or
            isinstance(variant_id, bool) or not isinstance(variant_id, int) or variant_id < 0):
        raise A3Error("Candidate root seed and variant_id must be non-negative integers")
    return stable_seed(
        "A3-candidate-v3", root_seed, method.upper(), parent_scan_id, variant_id)


def _attempt_seed(
    generation_seed: int, method: str, parent_scan_id: str,
    variant_id: int, attempt: int,
) -> int:
    return stable_seed(
        "A3-candidate-attempt-v3", generation_seed, method.upper(),
        parent_scan_id, variant_id, attempt)


def _trial_is_acceptable(
    method: str, points: np.ndarray, normals: np.ndarray | None,
    metadata: dict[str, Any],
) -> bool:
    loading = metadata.get("deform_params", {}).get("loading_simulation")
    common = (not metadata.get("fallback") and len(points) >= 50 and
              np.isfinite(points).all() and normals is not None and
              len(normals) == len(points) and np.isfinite(normals).all())
    applied = metadata.get("applied_methods", [])
    if method == "TRAD":
        valid_method = bool(applied) and set(applied).issubset(
            {"scale_and_rotation", "surface_noise", "rbf_deformation"})
        if "rbf_deformation" in applied:
            rbf = metadata.get("deform_params", {}).get("rbf")
            valid_method = valid_method and isinstance(rbf, dict)
    else:
        operations = loading.get("operations") if isinstance(loading, dict) else None
        valid_method = (
            applied == ["loading_simulation"] and
            isinstance(loading, dict) and
            loading.get("version") == "v3_seeded_random_direction_physical_units" and
            loading.get("direction_policy") == "seeded_random_per_operation_v3" and
            isinstance(operations, list) and bool(operations) and
            loading.get("n_operations") == len(operations) and
            loading.get("n_unique_operation_directions") == len(operations) and
            loading.get("removal_ratio", 0.0) > 0.0
        )
    return bool(common and valid_method)


def _candidate_metadata(
    raw_metadata: dict[str, Any], *, sample_id: str, parent_scan_id: str,
    parent_sha256: str, method: str, generation_seed: int, attempt: int,
    generator_path: Path, scale_row: dict[str, Any], scale_manifest_sha256: str,
) -> dict[str, Any]:
    metadata = dict(raw_metadata)
    metadata["fallback"] = False
    metadata.update({
        "schema_version": CANDIDATE_METADATA_SCHEMA,
        "id": sample_id, "sample_id": sample_id,
        "parent_scan_id": parent_scan_id,
        "parent_ply_sha256": parent_sha256,
        "method": method,
        "generation_seed": generation_seed,
        "generation_attempt": attempt,
        "generator": generator_path.name,
        "generator_sha256": sha256_file(generator_path),
        "coordinate_scale_manifest_sha256": scale_manifest_sha256,
        **{key: scale_row[key] for key in SCALE_IDENTITY_FIELDS},
    })
    metadata.pop("timestamp", None)
    return metadata


def _run_candidate_attempt(
    api: Any, points: np.ndarray, normals: np.ndarray, source_name: str,
    parent_scan_id: str, method: str, variant_id: int, generation_seed: int,
    attempt: int, settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    seed = _attempt_seed(
        generation_seed, method, parent_scan_id, variant_id, attempt)
    scale = settings.get("coordinate_scale_to_m")
    if (isinstance(scale, bool) or not isinstance(scale, (int, float)) or
            not math.isfinite(float(scale)) or float(scale) <= 0.0):
        raise A3Error(
            "Candidate generation requires positive finite coordinate_scale_to_m")
    mode = "traditional_only" if method == "TRAD" else "loading_only"
    rng = np.random.default_rng(seed)
    trial_points, trial_metadata = api.augment_one(
        points, normals, source_name, variant_id, settings, mode=mode, rng=rng)
    trial_normals = api.estimate_normals_numpy(trial_points, k=20)
    return trial_points, trial_metadata, trial_normals


def _canonical_ply_round_trip(
    api: Any, points: np.ndarray, normals: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply the formal writer/read normalization: ASCII %.6f then little-endian float32."""
    if normals is None:
        raise A3Error("Candidate replay did not produce normals")
    with tempfile.TemporaryDirectory(prefix="a3-candidate-replay-") as temporary:
        path = Path(temporary) / "candidate.ply"
        api._save_ply_numpy(
            path, np.asarray(points, dtype=np.float32),
            np.asarray(normals, dtype=np.float32))
        return read_ply_xyzn(path)


def _float32_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f4").tobytes(order="C")


def _metadata_bytes(value: dict[str, Any], *, pretty: bool) -> bytes:
    try:
        text = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
                if pretty else json.dumps(
                    value, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise A3Error(f"Candidate metadata is not deterministic JSON: {exc}") from exc
    return text.encode("utf-8")


def replay_candidate(
    *, candidate_ply: Path, candidate_metadata: Path | dict[str, Any],
    parent_ply: Path, generator_path: Path, scale_row: dict[str, Any],
    scale_manifest_sha256: str,
) -> dict[str, Any]:
    """Regenerate and exactly verify one formal candidate from parent+seed provenance."""
    candidate_ply = Path(candidate_ply).resolve()
    parent_ply = Path(parent_ply).resolve()
    generator_path = Path(generator_path)
    if not generator_path.is_absolute():
        raise A3Error(f"Candidate generator path must be absolute: {generator_path}")
    generator_path = generator_path.resolve()
    metadata_path = (Path(candidate_metadata).resolve()
                     if isinstance(candidate_metadata, (str, Path)) else None)
    metadata = load_json(metadata_path) if metadata_path is not None else candidate_metadata
    if not isinstance(metadata, dict):
        raise A3Error("Candidate replay metadata must be an object")
    if (metadata_path is not None and
            metadata_path.read_bytes() != _metadata_bytes(metadata, pretty=True)):
        raise A3Error(
            f"Candidate replay metadata JSON is not in deterministic canonical form: "
            f"{metadata_path}")
    sample_id = metadata.get("sample_id")
    method = str(metadata.get("method", "")).upper()
    if not isinstance(sample_id, str) or not sample_id:
        raise A3Error("Candidate replay metadata lacks sample_id")
    validate_candidate_augmentation_metadata(metadata, method, sample_id)
    if metadata["generator"] != generator_path.name:
        raise A3Error(f"Candidate replay generator name mismatch: {sample_id}")
    expected_generator = _formal_generator_path()
    if generator_path != expected_generator:
        raise A3Error(
            f"Candidate replay must use the fixed formal generator path: "
            f"expected {expected_generator}, got {generator_path}")
    actual_generator_hash = sha256_file(generator_path)
    if metadata["generator_sha256"] != actual_generator_hash:
        raise A3Error(f"Candidate replay generator SHA-256 mismatch: {sample_id}")
    if sha256_file(parent_ply) != metadata["parent_ply_sha256"]:
        raise A3Error(f"Candidate replay parent PLY SHA-256 mismatch: {sample_id}")

    api = _augmentation_api_from_path(generator_path)
    points, normals = api._load_ply_numpy(parent_ply)
    if points is None or len(points) == 0 or normals is None or len(normals) != len(points):
        raise A3Error(f"Candidate replay parent requires points and normals: {sample_id}")
    points, normals, _ = api.preprocess(points, normals)
    if len(points) < 100 or normals is None or len(normals) != len(points):
        raise A3Error(f"Candidate replay preprocessing invalidated parent: {sample_id}")
    settings = dict(api.get_data_config(parent_ply.name))
    settings["coordinate_scale_to_m"] = scale_row["coordinate_scale_to_m"]
    parent_id = metadata["parent_scan_id"]
    variant_id = metadata["variant_id"]
    generation_seed = metadata["generation_seed"]
    accepted_attempt = metadata["generation_attempt"]
    replay_points = replay_raw_metadata = replay_normals = None
    for attempt in range(accepted_attempt + 1):
        trial_points, trial_metadata, trial_normals = _run_candidate_attempt(
            api, points, normals, parent_ply.name, parent_id, method, variant_id,
            generation_seed, attempt, settings)
        accepted = _trial_is_acceptable(
            method, trial_points, trial_normals, trial_metadata)
        if accepted and attempt < accepted_attempt:
            raise A3Error(
                f"Candidate replay attempt semantics mismatch; earlier attempt {attempt} "
                f"was acceptable: {sample_id}")
        if attempt == accepted_attempt:
            if not accepted:
                raise A3Error(
                    f"Candidate replay declared attempt is not acceptable: {sample_id}")
            replay_points, replay_raw_metadata, replay_normals = (
                trial_points, trial_metadata, trial_normals)

    assert replay_points is not None and replay_raw_metadata is not None
    replay_xyz, replay_nrm = _canonical_ply_round_trip(
        api, replay_points, replay_normals)
    actual_xyz, actual_nrm = read_ply_xyzn(candidate_ply)
    if len(actual_xyz) != len(replay_xyz):
        raise A3Error(f"Candidate replay point-count mismatch: {sample_id}")
    if _float32_bytes(actual_xyz) != _float32_bytes(replay_xyz):
        raise A3Error(f"Candidate replay canonical float32 XYZ mismatch: {sample_id}")
    if (actual_nrm is None) != (replay_nrm is None):
        raise A3Error(f"Candidate replay normal-property presence mismatch: {sample_id}")
    if actual_nrm is not None and _float32_bytes(actual_nrm) != _float32_bytes(replay_nrm):
        raise A3Error(f"Candidate replay canonical float32 normals mismatch: {sample_id}")

    replay_metadata = _candidate_metadata(
        replay_raw_metadata, sample_id=sample_id, parent_scan_id=parent_id,
        parent_sha256=metadata["parent_ply_sha256"], method=method,
        generation_seed=generation_seed, attempt=accepted_attempt,
        generator_path=generator_path, scale_row=scale_row,
        scale_manifest_sha256=scale_manifest_sha256)
    validate_candidate_augmentation_metadata(replay_metadata, method, sample_id)
    if _metadata_bytes(replay_metadata, pretty=False) != _metadata_bytes(
            metadata, pretty=False):
        differing = sorted(
            key for key in set(replay_metadata) | set(metadata)
            if replay_metadata.get(key) != metadata.get(key))
        raise A3Error(
            f"Candidate replay exact v3 metadata mismatch: {sample_id}; fields={differing}")
    xyz_sha256 = hashlib.sha256(_float32_bytes(actual_xyz)).hexdigest()
    normal_sha256 = (hashlib.sha256(_float32_bytes(actual_nrm)).hexdigest()
                     if actual_nrm is not None else None)
    return {"sample_id": sample_id, "point_count": len(actual_xyz),
            "canonical_xyz_sha256": xyz_sha256,
            "canonical_normals_sha256": normal_sha256,
            "metadata": replay_metadata}


def _resolve_shard_scope(
    train_sources: list[Any], scan_ids: list[str] | None,
    variant_start: int, variant_end: int, variants_per_source: int,
) -> list[Any]:
    """Validate scan-id subset and variant slice against the declared full pool."""
    if variants_per_source < MIN_VARIANTS_PER_SOURCE:
        raise A3Error(
            "At least 17 generated candidates/source are required before Q filtering")
    for name, value in (("variant-start", variant_start), ("variant-end", variant_end)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise A3Error(f"{name} must be an integer")
    if not (0 <= variant_start < variant_end <= variants_per_source):
        raise A3Error(
            f"Variant slice must satisfy 0<=start<end<=pool({variants_per_source}); "
            f"got start={variant_start}, end={variant_end}")
    by_id = {source.scan_id: source for source in train_sources}
    if scan_ids is None:
        return list(train_sources)
    if not scan_ids:
        raise A3Error("At least one --scan-id is required when the flag is used")
    seen: set[str] = set()
    selected: list[Any] = []
    for scan_id in scan_ids:
        if scan_id not in by_id:
            raise A3Error(
                f"--scan-id must be a frozen TRAIN-9 parent; got {scan_id!r}")
        if scan_id in seen:
            raise A3Error(f"Duplicate --scan-id: {scan_id}")
        seen.add(scan_id)
        selected.append(by_id[scan_id])
    return selected


def _load_shard_inventory_rows(inventory: Path) -> dict[str, dict[str, str]]:
    """Read an existing per-shard inventory (if any) into a sample_id-keyed map."""
    if not inventory.is_file():
        return {}
    with inventory.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != INVENTORY_FIELDS:
            raise A3Error(
                f"Existing shard inventory has unexpected fields: {reader.fieldnames}")
        rows = list(reader)
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        if None in row:
            raise A3Error(f"Malformed shard inventory row in {inventory}")
        mapping[row["sample_id"]] = row
    return mapping


def _inventory_row(
    *, sample_id: str, parent_scan_id: str, method: str, ply_path: Path,
    point_count: int, metadata_path: Path, seed: int, output_dir: Path,
) -> dict[str, object]:
    return {
        "sample_id": sample_id, "parent_scan_id": parent_scan_id, "method": method,
        "ply_path": ply_path.relative_to(output_dir).as_posix(),
        "point_count": point_count, "ply_sha256": sha256_file(ply_path),
        "metadata_path": metadata_path.relative_to(output_dir).as_posix(),
        "metadata_sha256": sha256_file(metadata_path), "generation_seed": seed,
    }


def _write_shard_inventory(
    inventory: Path, output_dir: Path, rows: list[dict[str, object]],
) -> None:
    """Atomically rewrite the shard inventory so any interruption leaves it consistent."""
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    serializable = [{field: row[field] for field in INVENTORY_FIELDS} for row in ordered]
    atomic_write_csv(inventory, serializable, INVENTORY_FIELDS)


def _generate_one_candidate(
    *, api: Any, points: np.ndarray, normals: np.ndarray, source: Any, method: str,
    variant: int, seed: int, settings: dict[str, Any], scale_row: dict[str, Any],
    scale_manifest_hash: str, generator_path: Path, ply_path: Path,
    metadata_path: Path, output_dir: Path,
) -> dict[str, object]:
    """Generate one deterministic candidate, atomically write PLY+JSON, return its row."""
    augmented = raw_metadata = candidate_normals = None
    accepted_attempt = -1
    for attempt in range(64):
        trial_points, trial_metadata, trial_normals = _run_candidate_attempt(
            api, points, normals, source.path.name, source.scan_id, method,
            variant, seed, attempt, settings)
        if _trial_is_acceptable(method, trial_points, trial_normals, trial_metadata):
            augmented, raw_metadata, candidate_normals = (
                trial_points, trial_metadata, trial_normals)
            accepted_attempt = attempt
            break
    if augmented is None or raw_metadata is None or candidate_normals is None:
        raise A3Error(
            f"No valid deterministic {method} candidate after 64 attempts: "
            f"{source.scan_id}/{variant}")
    sample_id = f"{source.scan_id}_{method.lower()}_{variant:03d}"
    metadata = _candidate_metadata(
        raw_metadata, sample_id=sample_id, parent_scan_id=source.scan_id,
        parent_sha256=source.file_sha256, method=method, generation_seed=seed,
        attempt=accepted_attempt, generator_path=generator_path, scale_row=scale_row,
        scale_manifest_sha256=scale_manifest_hash)
    validate_candidate_augmentation_metadata(metadata, method, sample_id)
    _atomic_save_ply(api, ply_path, augmented.astype(np.float32),
                     candidate_normals.astype(np.float32))
    atomic_write_json(metadata_path, metadata)
    return _inventory_row(
        sample_id=sample_id, parent_scan_id=source.scan_id, method=method,
        ply_path=ply_path, point_count=len(augmented), metadata_path=metadata_path,
        seed=seed, output_dir=output_dir)


def _resume_existing_candidate(
    *, sample_id: str, method: str, parent: Any, ply_path: Path, metadata_path: Path,
    generator_path: Path, scale_row: dict[str, Any], scale_manifest_hash: str,
    prior_row: dict[str, str] | None, expected_seed: int, output_dir: Path,
) -> dict[str, object]:
    """Fail-closed three-tier verification before reusing an on-disk candidate."""
    if not ply_path.is_file() or not metadata_path.is_file():
        raise A3Error(
            f"Resume verification requires both PLY and metadata to exist: {sample_id}")
    # Tier 1: metadata structure + identity (generator/scale/parent + recomputed seed).
    metadata = load_json(metadata_path)
    validate_candidate_augmentation_metadata(metadata, method, sample_id)
    if metadata["parent_scan_id"] != parent.scan_id:
        raise A3Error(f"Resume metadata parent mismatch: {sample_id}")
    if metadata["parent_ply_sha256"] != parent.file_sha256:
        raise A3Error(f"Resume metadata parent SHA-256 mismatch: {sample_id}")
    if metadata["generator_sha256"] != sha256_file(generator_path):
        raise A3Error(f"Resume metadata generator SHA-256 mismatch: {sample_id}")
    if metadata["coordinate_scale_manifest_sha256"] != scale_manifest_hash:
        raise A3Error(f"Resume metadata scale-manifest SHA-256 mismatch: {sample_id}")
    for key in SCALE_IDENTITY_FIELDS:
        if metadata.get(key) != scale_row[key]:
            raise A3Error(f"Resume metadata scale identity mismatch: {sample_id} ({key})")
    # expected_seed was derived by the caller from root_seed+method+scan+variant, so
    # equality binds the full identity chain (variant_id also checked via sample_id).
    if int(metadata["variant_id"]) != int(sample_id.rsplit("_", 1)[1]):
        raise A3Error(f"Resume metadata variant_id mismatch: {sample_id}")
    if metadata["generation_seed"] != expected_seed:
        raise A3Error(f"Resume metadata generation_seed mismatch: {sample_id}")
    # Tier 2: SHA agreement with the in-shard inventory row (when one exists).
    actual_ply_hash = sha256_file(ply_path)
    actual_metadata_hash = sha256_file(metadata_path)
    if prior_row is not None and (
            prior_row.get("ply_sha256") != actual_ply_hash or
            prior_row.get("metadata_sha256") != actual_metadata_hash):
        raise A3Error(
            f"Resume inventory SHA-256 disagrees with on-disk asset: {sample_id}")
    # Tier 3: byte-exact replay from parent+seed provenance.
    replay = replay_candidate(
        candidate_ply=ply_path, candidate_metadata=metadata_path,
        parent_ply=parent.path, generator_path=generator_path, scale_row=scale_row,
        scale_manifest_sha256=scale_manifest_hash)
    if replay["sample_id"] != sample_id:
        raise A3Error(f"Resume replay sample mismatch: {sample_id}")
    return _inventory_row(
        sample_id=sample_id, parent_scan_id=parent.scan_id, method=method,
        ply_path=ply_path, point_count=replay["point_count"],
        metadata_path=metadata_path, seed=expected_seed, output_dir=output_dir)


def generate_candidates(
    project_root: Path, splits: Path, source_manifest: Path,
    output_dir: Path, method: str, variants_per_source: int,
    generation_seed: int, *, coordinate_scale_manifest: Path,
    scan_ids: list[str] | None = None, variant_start: int = 0,
    variant_end: int | None = None, resume_verified: bool = False,
) -> Path:
    method = method.upper()
    if method not in {"TRAD", "LOADSIM"}:
        raise A3Error("Candidate method must be TRAD or LOADSIM")
    if variant_end is None:
        variant_end = variants_per_source
    train_sources, _ = load_frozen_sources(project_root, splits, source_manifest)
    sources = _resolve_shard_scope(
        train_sources, scan_ids, variant_start, variant_end, variants_per_source)
    source_hashes = {source.scan_id: source.file_sha256 for source in train_sources}
    scales = load_coordinate_scale_manifest(
        coordinate_scale_manifest, source_hashes, allow_extra=True)
    scale_manifest_hash = sha256_file(coordinate_scale_manifest)
    api = _augmentation_api(project_root)
    generator_path = Path(api.__file__).resolve()
    if generator_path != _formal_generator_path():
        raise A3Error(
            f"Candidate generation must use the fixed formal generator path: "
            f"expected {_formal_generator_path()}, got {generator_path}")
    candidate_dir = output_dir / "candidates"
    metadata_dir = output_dir / "metadata"
    inventory = output_dir / "candidate_inventory.csv"
    prior_rows = _load_shard_inventory_rows(inventory)
    rows: list[dict[str, object]] = []
    for source in sources:
        points, normals = api._load_ply_numpy(source.path)
        if points is None or len(points) != source.point_count:
            raise A3Error(f"Source point count mismatch: {source.scan_id}")
        if normals is None or len(normals) != len(points):
            raise A3Error(f"Normals are required for strict candidate generation: {source.path}")
        points, normals, _ = api.preprocess(points, normals)
        if len(points) < 100 or normals is None or len(normals) != len(points):
            raise A3Error(f"Preprocessing invalidated source: {source.scan_id}")
        settings = dict(api.get_data_config(source.path.name))
        scale_row = scales[source.scan_id]
        settings["coordinate_scale_to_m"] = scale_row["coordinate_scale_to_m"]
        for variant in range(variant_start, variant_end):
            sample_id = f"{source.scan_id}_{method.lower()}_{variant:03d}"
            seed = candidate_generation_seed(
                generation_seed, method, source.scan_id, variant)
            ply_path = candidate_dir / f"{sample_id}.ply"
            metadata_path = metadata_dir / f"{sample_id}.json"
            if ply_path.exists() or metadata_path.exists():
                if not resume_verified:
                    raise A3Error(
                        f"Refusing to overwrite existing candidate assets without "
                        f"--resume-verified: {sample_id}")
                row = _resume_existing_candidate(
                    sample_id=sample_id, method=method, parent=source,
                    ply_path=ply_path, metadata_path=metadata_path,
                    generator_path=generator_path, scale_row=scale_row,
                    scale_manifest_hash=scale_manifest_hash,
                    prior_row=prior_rows.get(sample_id), expected_seed=seed,
                    output_dir=output_dir)
                rows.append(row)
                _write_shard_inventory(inventory, output_dir, rows)
                continue
            row = _generate_one_candidate(
                api=api, points=points, normals=normals, source=source, method=method,
                variant=variant, seed=seed, settings=settings, scale_row=scale_row,
                scale_manifest_hash=scale_manifest_hash, generator_path=generator_path,
                ply_path=ply_path, metadata_path=metadata_path, output_dir=output_dir)
            rows.append(row)
            _write_shard_inventory(inventory, output_dir, rows)
    _write_shard_inventory(inventory, output_dir, rows)
    return inventory


def _collect_shard_candidate(
    shard_dir: Path, row: dict[str, str], collected: dict[str, dict[str, Any]],
) -> None:
    """Resolve one shard inventory row to absolute assets, deduping byte-identical copies."""
    sample_id = row["sample_id"]
    ply = resolve_under(shard_dir, row["ply_path"])
    metadata = resolve_under(shard_dir, row["metadata_path"])
    if not ply.is_file() or not metadata.is_file():
        raise A3Error(f"Shard {shard_dir} references missing assets for {sample_id}")
    ply_hash = sha256_file(ply)
    metadata_hash = sha256_file(metadata)
    if ply_hash != row["ply_sha256"] or metadata_hash != row["metadata_sha256"]:
        raise A3Error(f"Shard inventory SHA-256 disagrees with assets: {sample_id}")
    if sample_id in collected:
        prior = collected[sample_id]
        if prior["ply_sha256"] != ply_hash or prior["metadata_sha256"] != metadata_hash:
            raise A3Error(
                f"Duplicate candidate {sample_id} differs across shards; refusing merge")
        return  # byte-identical duplicate: register once
    collected[sample_id] = {
        "sample_id": sample_id, "parent_scan_id": row["parent_scan_id"],
        "method": row["method"].upper(), "ply": ply, "metadata": metadata,
        "point_count": int(row["point_count"]), "ply_sha256": ply_hash,
        "metadata_sha256": metadata_hash, "generation_seed": int(row["generation_seed"]),
    }


def merge_candidates(
    shard_dirs: list[Path], output_dir: Path, method: str, generation_seed: int,
    project_root: Path, splits: Path, source_manifest: Path, *,
    coordinate_scale_manifest: Path, variants_per_source: int = FORMAL_VARIANTS_PER_SOURCE,
) -> Path:
    """Merge verified shards into one authoritative candidate pool inventory (no finalize)."""
    method = method.upper()
    if method not in {"TRAD", "LOADSIM"}:
        raise A3Error("Candidate method must be TRAD or LOADSIM")
    if not shard_dirs:
        raise A3Error("merge requires at least one --shard-dir")
    if output_dir.exists():
        raise A3Error(f"Refusing to write into existing merge output: {output_dir}")
    train_sources, _ = load_frozen_sources(project_root, splits, source_manifest)
    source_hashes = {source.scan_id: source.file_sha256 for source in train_sources}
    sources_by_id = {source.scan_id: source for source in train_sources}
    scales = load_coordinate_scale_manifest(
        coordinate_scale_manifest, source_hashes, allow_extra=True)
    scale_manifest_hash = sha256_file(coordinate_scale_manifest)
    generator_path = _formal_generator_path()
    if not generator_path.is_file():
        raise A3Error(f"Missing formal generator: {generator_path}")
    generator_hash = sha256_file(generator_path)
    collected: dict[str, dict[str, Any]] = {}
    for shard_dir in shard_dirs:
        shard_dir = Path(shard_dir).resolve()
        inventory = shard_dir / "candidate_inventory.csv"
        for row in _load_shard_inventory_rows(inventory).values():
            if row["method"].upper() != method:
                raise A3Error(
                    f"Shard {shard_dir} mixes methods; expected {method}, got {row['method']}")
            _collect_shard_candidate(shard_dir, row, collected)
    expected = {f"{scan}_{method.lower()}_{variant:03d}"
                for scan in sources_by_id for variant in range(variants_per_source)}
    actual = set(collected)
    if actual != expected:
        raise A3Error(
            f"Merged set must be exactly TRAIN-9x[0,{variants_per_source})="
            f"{len(expected)}; missing={sorted(expected-actual)[:8]}, "
            f"extra={sorted(actual-expected)[:8]}")
    return _publish_merged_pool(
        collected, output_dir, method, generation_seed, sources_by_id, scales,
        scale_manifest_hash, generator_path, generator_hash, coordinate_scale_manifest)


def _publish_merged_pool(
    collected: dict[str, dict[str, Any]], output_dir: Path, method: str,
    generation_seed: int, sources_by_id: dict[str, Any], scales: dict[str, Any],
    scale_manifest_hash: str, generator_path: Path, generator_hash: str,
    coordinate_scale_manifest: Path,
) -> Path:
    """Replay every candidate, dedup globally, and atomically stage+publish the pool."""
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    if staging.exists():
        raise A3Error(f"Merge staging directory already exists: {staging}")
    candidate_dir = staging / "candidates"
    metadata_dir = staging / "metadata"
    candidate_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    seeds: set[int] = set()
    xyz_hashes: set[str] = set()
    ply_hashes: set[str] = set()
    rows: list[dict[str, object]] = []
    try:
        for sample_id in sorted(collected):
            item = collected[sample_id]
            parent = sources_by_id[item["parent_scan_id"]]
            variant = int(sample_id.rsplit("_", 1)[1])
            expected_seed = candidate_generation_seed(
                generation_seed, method, parent.scan_id, variant)
            if item["generation_seed"] != expected_seed:
                raise A3Error(
                    f"Merged generation_seed not recomputable from root: {sample_id}")
            if expected_seed in seeds:
                raise A3Error(f"Duplicate generation seed across pool: {sample_id}")
            seeds.add(expected_seed)
            scale_row = scales[parent.scan_id]
            replay = replay_candidate(
                candidate_ply=item["ply"], candidate_metadata=item["metadata"],
                parent_ply=parent.path, generator_path=generator_path,
                scale_row=scale_row, scale_manifest_sha256=scale_manifest_hash)
            if replay["sample_id"] != sample_id:
                raise A3Error(f"Merged replay sample mismatch: {sample_id}")
            if replay["metadata"]["generator_sha256"] != generator_hash:
                raise A3Error(f"Merged generator SHA-256 mismatch: {sample_id}")
            if replay["metadata"]["coordinate_scale_manifest_sha256"] != scale_manifest_hash:
                raise A3Error(f"Merged scale-manifest SHA-256 mismatch: {sample_id}")
            xyz_hash = replay["canonical_xyz_sha256"]
            if xyz_hash in xyz_hashes:
                raise A3Error(f"Duplicate canonical XYZ geometry in pool: {sample_id}")
            xyz_hashes.add(xyz_hash)
            if item["ply_sha256"] in ply_hashes:
                raise A3Error(f"Duplicate candidate PLY bytes in pool: {sample_id}")
            ply_hashes.add(item["ply_sha256"])
            dst_ply = candidate_dir / f"{sample_id}.ply"
            dst_metadata = metadata_dir / f"{sample_id}.json"
            import shutil as _shutil
            _shutil.copy2(item["ply"], dst_ply)
            _shutil.copy2(item["metadata"], dst_metadata)
            if (sha256_file(dst_ply) != item["ply_sha256"] or
                    sha256_file(dst_metadata) != item["metadata_sha256"]):
                raise A3Error(f"Merged asset changed during copy: {sample_id}")
            rows.append(_inventory_row(
                sample_id=sample_id, parent_scan_id=parent.scan_id, method=method,
                ply_path=dst_ply, point_count=item["point_count"],
                metadata_path=dst_metadata, seed=expected_seed, output_dir=staging))
        ordered = sorted(rows, key=lambda row: row["sample_id"])
        serializable = [{field: row[field] for field in INVENTORY_FIELDS}
                        for row in ordered]
        atomic_write_csv(staging / "candidate_inventory.csv", serializable, INVENTORY_FIELDS)
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            import shutil as _shutil
            _shutil.rmtree(staging, ignore_errors=True)
    return output_dir / "candidate_inventory.csv"


def finalize_candidates(
    inventory: Path, labels_json: Path, quality_report: Path,
    quality_config: Path, output_manifest: Path, project_root: Path,
    splits: Path, source_manifest: Path, *, pseudo_label_config: Path,
    pseudo_label_generator: Path, canonical_scale_manifest: Path,
) -> Path:
    """Build, validate and atomically publish one immutable Q-only bundle."""
    if output_manifest.name != "candidate_manifest.csv":
        raise A3Error("Formal bundle entrypoint must be named candidate_manifest.csv")
    output_root = output_manifest.parent
    if output_root.exists():
        raise A3Error(
            f"Refusing to modify existing output bundle; choose a clean path: {output_root}")
    train_sources, _ = load_frozen_sources(project_root, splits, source_manifest)
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        raise A3Error(f"Staging directory already exists: {staging}")
    try:
        stage_candidate_bundle(
            inventory, labels_json, quality_report, quality_config,
            train_sources, source_manifest, splits, staging,
            canonical_scale_manifest=canonical_scale_manifest,
            pseudo_label_config=pseudo_label_config,
            pseudo_label_generator=pseudo_label_generator,
        )
        result = publish_candidate_bundle(staging, output_root)
    finally:
        if staging.exists():
            import shutil
            shutil.rmtree(staging, ignore_errors=True)
    return result


def main() -> int:
    base = Path(__file__).resolve().parent
    project = base.parent
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Build deterministic unlabeled candidates")
    generate.add_argument("--project-root", type=Path, default=project)
    generate.add_argument("--splits", type=Path, default=base / "manifest" / "splits_v2.json")
    generate.add_argument("--source-manifest", type=Path, default=base / "manifest" / "source_scans_v2.csv")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--method", required=True, choices=["TRAD", "LOADSIM"])
    generate.add_argument("--variants-per-source", type=int, default=32,
                          help="Generate a surplus pool; finalize still requires >=17 Q-passing candidates/source")
    generate.add_argument("--generation-seed", type=int, default=20260910)
    generate.add_argument("--coordinate-scale-manifest", type=Path, required=True)
    generate.add_argument("--scan-id", dest="scan_ids", action="append", default=None,
                          help="Restrict to one TRAIN-9 parent (repeatable); default all 9")
    generate.add_argument("--variant-start", type=int, default=0,
                          help="Inclusive variant slice start into [0, variants-per-source)")
    generate.add_argument("--variant-end", type=int, default=None,
                          help="Exclusive variant slice end; default variants-per-source")
    generate.add_argument("--resume-verified", action="store_true",
                          help="Reuse pre-existing candidates only after fail-closed "
                               "three-tier verification; conflicts abort")
    merge = subparsers.add_parser(
        "merge", help="Merge verified shards into one authoritative candidate pool")
    merge.add_argument("--shard-dir", dest="shard_dirs", action="append", required=True,
                       type=Path, help="A shard output-dir with candidate_inventory.csv (repeatable)")
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--method", required=True, choices=["TRAD", "LOADSIM"])
    merge.add_argument("--generation-seed", type=int, default=20260910)
    merge.add_argument("--variants-per-source", type=int, default=FORMAL_VARIANTS_PER_SOURCE)
    merge.add_argument("--coordinate-scale-manifest", type=Path, required=True)
    merge.add_argument("--project-root", type=Path, default=project)
    merge.add_argument("--splits", type=Path, default=base / "manifest" / "splits_v2.json")
    merge.add_argument("--source-manifest", type=Path,
                       default=base / "manifest" / "source_scans_v2.csv")
    finalize = subparsers.add_parser("finalize", help="Attach labels and enforce Q-only v1")
    finalize.add_argument("--inventory", type=Path, required=True)
    finalize.add_argument("--pseudo-label-json", type=Path, required=True)
    finalize.add_argument("--pseudo-label-config", type=Path, required=True)
    finalize.add_argument("--pseudo-label-generator", type=Path, required=True)
    finalize.add_argument("--coordinate-scale-manifest", type=Path, required=True)
    finalize.add_argument("--quality-report", type=Path, required=True)
    finalize.add_argument("--quality-config", type=Path,
                          default=base / "configs" / "quality_v1.yaml")
    finalize.add_argument("--project-root", type=Path, default=project)
    finalize.add_argument("--splits", type=Path,
                          default=base / "manifest" / "splits_v2.json")
    finalize.add_argument("--source-manifest", type=Path,
                          default=base / "manifest" / "source_scans_v2.csv")
    finalize.add_argument("--output-manifest", type=Path, required=True)
    # all-valid (all-technically-valid-v1) main-analysis publisher. Eligibility is
    # every technically valid candidate (288 = TRAIN-9 x 32); Q is report-only.
    # Evidence mode (migrated vs fresh deep replay) is auto-detected, NOT a flag,
    # and there is no force-migration or skip-hash option by design.
    finalize_all_valid_cmd = subparsers.add_parser(
        "finalize-all-valid",
        help="Publish an all-valid bundle (no Q gate; auto migrated/fresh evidence)")
    finalize_all_valid_cmd.add_argument("--inventory", type=Path, required=True)
    finalize_all_valid_cmd.add_argument("--pseudo-label-json", type=Path, required=True)
    finalize_all_valid_cmd.add_argument("--pseudo-label-config", type=Path, required=True)
    finalize_all_valid_cmd.add_argument("--pseudo-label-generator", type=Path, required=True)
    finalize_all_valid_cmd.add_argument("--coordinate-scale-manifest", type=Path, required=True)
    finalize_all_valid_cmd.add_argument("--quality-report", type=Path, required=True)
    finalize_all_valid_cmd.add_argument("--quality-config", type=Path,
                                        default=base / "configs" / "quality_v1.yaml")
    finalize_all_valid_cmd.add_argument("--method", required=True, choices=["TRAD", "LOADSIM"])
    finalize_all_valid_cmd.add_argument("--project-root", type=Path, default=project)
    finalize_all_valid_cmd.add_argument("--splits", type=Path,
                                        default=base / "manifest" / "splits_v2.json")
    finalize_all_valid_cmd.add_argument("--source-manifest", type=Path,
                                        default=base / "manifest" / "source_scans_v2.csv")
    finalize_all_valid_cmd.add_argument("--output-dir", type=Path, required=True)
    finalize_all_valid_cmd.add_argument(
        "--evidence-dir", type=Path, default=None,
        help="Optional read-only historical-producer evidence dir; a complete "
             "structured closure enables the migrated fast path, any gap falls back "
             "to one fresh deep replay. Cannot force the fast path.")
    verify_all_valid_cmd = subparsers.add_parser(
        "verify-all-valid",
        help="Fast-verify a published all-valid bundle (no geometric replay)")
    verify_all_valid_cmd.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            result = generate_candidates(args.project_root.resolve(), args.splits.resolve(),
                                         args.source_manifest.resolve(), args.output_dir.resolve(),
                                         args.method, args.variants_per_source,
                                         args.generation_seed,
                                         coordinate_scale_manifest=
                                         args.coordinate_scale_manifest.resolve(),
                                         scan_ids=args.scan_ids,
                                         variant_start=args.variant_start,
                                         variant_end=args.variant_end,
                                         resume_verified=args.resume_verified)
        elif args.command == "merge":
            result = merge_candidates(
                [shard.resolve() for shard in args.shard_dirs], args.output_dir.resolve(),
                args.method, args.generation_seed, args.project_root.resolve(),
                args.splits.resolve(), args.source_manifest.resolve(),
                coordinate_scale_manifest=args.coordinate_scale_manifest.resolve(),
                variants_per_source=args.variants_per_source)
        elif args.command == "finalize":
            result = finalize_candidates(
                args.inventory.resolve(), args.pseudo_label_json.resolve(),
                args.quality_report.resolve(), args.quality_config.resolve(),
                args.output_manifest.resolve(), args.project_root.resolve(),
                args.splits.resolve(), args.source_manifest.resolve(),
                pseudo_label_config=args.pseudo_label_config.resolve(),
                pseudo_label_generator=args.pseudo_label_generator.resolve(),
                canonical_scale_manifest=args.coordinate_scale_manifest.resolve())
        elif args.command == "finalize-all-valid":
            import a3_all_valid
            result = a3_all_valid.finalize_all_valid(
                inventory=args.inventory.resolve(),
                candidate_report=args.pseudo_label_json.resolve(),
                quality_report=args.quality_report.resolve(),
                quality_config=args.quality_config.resolve(),
                pseudo_label_config=args.pseudo_label_config.resolve(),
                pseudo_label_generator=args.pseudo_label_generator.resolve(),
                coordinate_scale_manifest=args.coordinate_scale_manifest.resolve(),
                output_dir=args.output_dir.resolve(), method=args.method,
                project_root=args.project_root.resolve(),
                splits=args.splits.resolve(),
                source_manifest=args.source_manifest.resolve(),
                evidence_dir=(args.evidence_dir.resolve()
                              if args.evidence_dir is not None else None))
        else:  # verify-all-valid
            import a3_all_valid
            receipt = a3_all_valid.verify_all_valid(args.bundle_dir.resolve())
            print(f"Verified all-valid bundle {args.bundle_dir.resolve()}: "
                  f"method={receipt['method']}, evidence_mode={receipt['evidence_mode']}, "
                  f"total={len(receipt['ordered_sample_ids'])}")
            return 0
    except (A3Error, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
