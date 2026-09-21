"""Strict I/O and reproducibility primitives for the formal A3 experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import numpy as np


class A3Error(RuntimeError):
    """Fail-closed A3 protocol or data error."""


PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    path = Path(path)
    if not path.is_file():
        raise A3Error(f"Missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    text = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise A3Error(f"Missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A3Error(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A3Error(f"Expected JSON object: {path}")
    return value


PSEUDO_LABEL_REPORT_SCHEMA = "geometric-pseudo-label-report-v3"
PSEUDO_LABEL_CONTRACT_VERSION = "pseudo-label-contract-v3"
PSEUDO_LABEL_MANIFEST_SCHEMA = "a3-pseudo-label-manifest-v3"
PSEUDO_LABEL_ASSET_SCHEMA = "a3-label-v3"
PSEUDO_LABEL_MANIFEST_FILENAME = "pseudo_label_manifest_v3.csv"
FRONT_COMPUTATION_FIELDS = (
    "front_axis_computed", "front_sign_computed", "front_computation_source",
)
SCALE_IDENTITY_FIELDS = (
    "coordinate_scale_to_m", "coordinate_scale_source", "scale_evidence_type",
    "scale_evidence_path", "scale_evidence_sha256", "scale_semantics",
)
ORIGINAL_LABEL_MANIFEST_FIELDS = (
    "schema_version", "manifest_schema", "sample_id", "parent_scan_id", "split",
    "point_count", "source_ply_sha256", "parent_source_ply_sha256", "label_path",
    "label_sha256", "label_schema", "label_type", "pseudo_label_report_schema",
    "pseudo_label_report_path", "pseudo_label_config_path",
    "pseudo_label_generator_path", "coordinate_scale_manifest_path",
    "pseudo_label_source_json_sha256", "config_sha256", "generator_sha256",
    "coordinate_scale_manifest_sha256", "front_axis_computed",
    "front_sign_computed", "front_computation_source", "coordinate_scale_to_m",
    "coordinate_scale_source", "scale_evidence_type", "scale_evidence_path",
    "scale_evidence_sha256", "scale_semantics", "effective_convex_radius_raw",
    "strict_and_applied", "adaptive_relaxation_applied", "label_0", "label_1",
    "label_255",
)
CANDIDATE_MANIFEST_FIELDS = (
    "schema_version", "manifest_schema", "sample_id", "parent_scan_id", "method",
    "ply_path", "point_count", "ply_sha256", "canonical_xyz_sha256",
    "parent_source_ply_sha256", "label_path", "label_sha256", "label_schema",
    "metadata_path", "metadata_sha256", "generation_seed", "label_type",
    "pseudo_label_report_schema", "pseudo_label_report_path",
    "pseudo_label_config_path", "pseudo_label_generator_path",
    "coordinate_scale_manifest_path", "pseudo_label_source_json_sha256",
    "config_sha256", "generator_sha256", "coordinate_scale_manifest_sha256",
    "front_axis_computed", "front_sign_computed", "front_computation_source",
    "coordinate_scale_to_m", "coordinate_scale_source", "scale_evidence_type",
    "scale_evidence_path", "scale_evidence_sha256", "scale_semantics",
    "effective_convex_radius_raw", "strict_and_applied",
    "adaptive_relaxation_applied", "quality_rule", "quality_threshold_Q",
    "quality_config_sha256", "quality_report_path", "quality_report_sha256",
    "quality_F", "quality_P", "quality_D", "quality_Q", "quality_passed",
    "label_0", "label_1", "label_255",
)
LABEL_NPZ_COMMON_FIELDS = (
    "labels", "sample_id", "source_scan_id", "source_ply_sha256",
    "schema_version", "pseudo_label_report_sha256", "config_sha256",
    "generator_sha256", "coordinate_scale_manifest_sha256",
    "front_axis_computed", "front_sign_computed", "front_computation_source",
    "coordinate_scale_to_m", "coordinate_scale_source", "scale_evidence_type",
    "scale_evidence_path", "scale_evidence_sha256", "scale_semantics",
    "effective_convex_radius_raw",
)
CANDIDATE_LABEL_NPZ_FIELDS = LABEL_NPZ_COMMON_FIELDS + (
    "parent_source_ply_sha256", "augmentation_metadata_sha256",
)
FRONT_DIRECTION_POLICY = (
    "geometry_internal_raw_frozen_normal_axis_positive_sign_no_external_evidence"
)
COORDINATE_SCALE_POLICY = (
    "v2_independently_evidenced_raw_to_m_per_source_and_candidate"
)
SCALE_SEMANTICS = "physical_geometry_scaling_same_coordinate_units"
# Adaptive relaxation policy restored per plan A (2026-09-11). Reproduces the
# pre-registered step3_labels.py:234-247 fallback: strict AND is attempted first,
# and only the deterministic, configuration-declared fallbacks below may fire.
ADAPTIVE_RELAXATION_POLICY = {
    "ratio_min": 0.01, "fallback_min": 0.005, "widen_slope_degrees": [10, 55],
    "overcap": 0.40, "trim_percentile": 50,
}
RELAXATION_STAGES = ("none", "drop_convex", "widen_slope", "overcap_trim")
MASK_COMBINATIONS = (
    "front_and_slope_and_convex", "front_and_slope", "front_and_widened_slope",
)

# ---------------------------------------------------------------------------
# all-valid v1 (main-analysis eligibility = all-technically-valid-v1). These
# schemas are DISTINCT from the Q-v1 candidate manifest above and MUST NOT be
# accepted by the Q-v1 parser, nor Q-v1 manifests by the all-valid parser.
# ---------------------------------------------------------------------------
ALL_VALID_ELIGIBILITY_RULE = "all-technically-valid-v1"
ALL_VALID_MANIFEST_SCHEMA = "a3-all-valid-candidate-manifest-v1"
ALL_VALID_RECEIPT_SCHEMA = "a3-all-valid-receipt-v1"
ALL_VALID_ACTIVATION_SCHEMA = "a3-all-valid-activation-v1"
ALL_VALID_CATALOG_SCHEMA = "a3-all-valid-catalog-v1"
ALL_VALID_CONTRACT_SCHEMA = "a3-all-valid-technical-contract-v1"
ALL_VALID_NPZ_SCHEMA = "a3-all-valid-label-v1"
ALL_VALID_EVIDENCE_MODES = ("migrated_atomic_success_v1", "fresh_deep_replay_v1")
ALL_VALID_EXPECTED_PER_PARENT = 32
ALL_VALID_EXPECTED_TOTAL = 288  # TRAIN-9 x 32

# Field order is exact and verified on read. quality_* columns are report-only
# and NEVER decide eligibility; technical_valid / eligibility_rule do.
ALL_VALID_MANIFEST_FIELDS = (
    "schema_version", "manifest_schema", "eligibility_rule", "technical_valid",
    "quality_role", "sample_id", "parent_scan_id", "method", "variant_id",
    "ply_path", "point_count", "ply_sha256", "canonical_xyz_sha256",
    "canonical_normals_sha256", "parent_source_ply_sha256", "label_path",
    "label_sha256", "label_schema", "metadata_path", "metadata_sha256",
    "generation_seed", "generation_attempt", "label_type",
    "pseudo_label_report_schema", "pseudo_label_report_path",
    "pseudo_label_report_sha256", "pseudo_label_config_path",
    "pseudo_label_generator_path", "coordinate_scale_manifest_path",
    "config_sha256", "generator_sha256", "coordinate_scale_manifest_sha256",
    "candidate_inventory_sha256", "front_axis_computed", "front_sign_computed",
    "front_computation_source", "coordinate_scale_to_m", "coordinate_scale_source",
    "scale_evidence_type", "scale_evidence_path", "scale_evidence_sha256",
    "scale_semantics", "effective_convex_radius_raw", "strict_and_applied",
    "adaptive_relaxation_applied", "quality_rule", "quality_threshold_Q",
    "quality_config_sha256", "quality_report_path", "quality_report_sha256",
    "quality_F", "quality_P", "quality_D", "quality_Q", "historical_q_v1_passed",
    "label_0", "label_1", "label_255",
)
COORDINATE_SCALE_MANIFEST_SCHEMA = "coordinate-scales-v2"
COORDINATE_SCALE_MANIFEST_FIELDS = (
    "schema_version", "scan_id", "source_ply_sha256", "coordinate_scale_to_m",
    "coordinate_scale_source", "scale_semantics", "scale_evidence_type",
    "scale_evidence_path", "scale_evidence_sha256", "frozen_at", "notes",
)
_FORBIDDEN_PROVENANCE_TERMS = (
    "bbox", "bounding_box", "bounding box", "heuristic", "guess", "label",
    "prediction", "normal",
)
_SENSITIVE_EVIDENCE_NAMES = {
    ".env", ".git-credentials", "authorized_keys", "id_rsa", "id_dsa",
    "id_ecdsa", "id_ed25519", "credentials", "credentials.json",
}
_SENSITIVE_EVIDENCE_SUFFIXES = {
    ".key", ".pem", ".p12", ".pfx", ".ppk", ".token",
}
_SHA256_HEX = set("0123456789abcdef")


def _required_sha256(value: Any, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in _SHA256_HEX for character in value)):
        raise A3Error(f"{where} must be a lowercase SHA-256 hex digest")
    return value


def _coerce_bool(value: Any, where: str) -> bool:
    """Accept a real bool or its canonical lowercase text form only."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in ("true", "false"):
        return value == "true"
    raise A3Error(f"{where} must be a boolean or 'true'/'false' text")


def validate_relaxation_fields(
    stage: Any, combination: Any, strict_applied: Any, adaptive_applied: Any,
    where: str,
) -> tuple[str, str, bool, bool]:
    """Validate the deterministic relaxation audit quartet shared across A3 assets.

    The relaxation record is a reproducible outcome of the frozen algorithm, not
    external evidence. It fails closed on unknown stages/combinations or on a
    strict/adaptive pair that is not mutually exclusive.
    """
    if not isinstance(stage, str):
        raise A3Error(f"{where} relaxation_stage must be text")
    parts = stage.split("+")
    if stage == "none":
        parts = ["none"]
    if not parts or any(part not in RELAXATION_STAGES for part in parts):
        raise A3Error(f"{where} relaxation_stage is invalid: {stage!r}")
    if "none" in parts and parts != ["none"]:
        raise A3Error(f"{where} relaxation_stage cannot mix 'none' with fallbacks")
    if combination not in MASK_COMBINATIONS:
        raise A3Error(f"{where} mask_combination is invalid: {combination!r}")
    strict = _coerce_bool(strict_applied, f"{where} strict_and_applied")
    adaptive = _coerce_bool(adaptive_applied, f"{where} adaptive_relaxation_applied")
    if strict == adaptive:
        raise A3Error(
            f"{where} strict_and_applied and adaptive_relaxation_applied must be "
            "mutually exclusive")
    if strict and (stage != "none" or combination != "front_and_slope_and_convex"):
        raise A3Error(f"{where} strict-AND record must have stage none and full AND mask")
    if adaptive and stage == "none":
        raise A3Error(f"{where} adaptive relaxation requires a non-none stage")
    return stage, combination, strict, adaptive


def _finite_positive(value: Any, where: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or float(value) <= 0):
        raise A3Error(f"{where} must be positive and finite")
    return float(value)


def _validate_independent_source(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise A3Error(f"{where} must be a non-empty independent evidence source")
    source = value.strip()
    lowered = source.lower()
    if any(term in lowered for term in _FORBIDDEN_PROVENANCE_TERMS):
        raise A3Error(
            f"{where} must be independently evidenced, not bbox/heuristic/guess/"
            "label/prediction/normals-derived")
    return source


def validate_coordinate_scale_source(value: Any, where: str) -> str:
    return _validate_independent_source(value, where)


def _validate_evidence_type(value: Any, where: str) -> str:
    return _validate_independent_source(value, where)


def validate_scale_identity_fields(value: dict[str, Any], where: str) -> dict[str, Any]:
    """Validate the exact six-field scale identity carried across A3 assets."""
    missing = set(SCALE_IDENTITY_FIELDS) - set(value)
    if missing:
        raise A3Error(f"{where} lacks scale identity fields: {sorted(missing)}")
    _finite_positive(value["coordinate_scale_to_m"], f"{where} coordinate_scale_to_m")
    validate_coordinate_scale_source(
        value["coordinate_scale_source"], f"{where} coordinate_scale_source")
    _validate_evidence_type(value["scale_evidence_type"], f"{where} scale_evidence_type")
    evidence_path = value["scale_evidence_path"]
    if (not isinstance(evidence_path, str) or not evidence_path.strip() or
            "\\" in evidence_path or Path(evidence_path).is_absolute() or
            ".." in Path(evidence_path).parts):
        raise A3Error(f"{where} scale_evidence_path must be a safe POSIX relative path")
    _required_sha256(value["scale_evidence_sha256"], f"{where} scale_evidence_sha256")
    if value["scale_semantics"] != SCALE_SEMANTICS:
        raise A3Error(f"{where} scale_semantics must be {SCALE_SEMANTICS!r}")
    return {key: value[key] for key in SCALE_IDENTITY_FIELDS}


def validate_scale_identity_match(
    value: dict[str, Any], expected: dict[str, Any], where: str, *,
    allow_numeric_text: bool = False,
) -> None:
    """Validate and exactly compare all six canonical scale identity fields."""
    actual = {key: value.get(key) for key in SCALE_IDENTITY_FIELDS}
    if allow_numeric_text:
        try:
            actual["coordinate_scale_to_m"] = float(actual["coordinate_scale_to_m"])
        except (TypeError, ValueError) as exc:
            raise A3Error(f"{where} coordinate_scale_to_m must be numeric") from exc
    validate_scale_identity_fields(actual, where)
    canonical = validate_scale_identity_fields(expected, f"{where} canonical row")
    for key in SCALE_IDENTITY_FIELDS:
        if actual[key] != canonical[key]:
            raise A3Error(
                f"{where} {key} differs from canonical scale row: "
                f"actual={actual[key]!r}, expected={canonical[key]!r}")


def _validate_frozen_at(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise A3Error(f"{where} must be a timezone-aware ISO timestamp")
    text = value.strip()
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise A3Error(f"{where} must be a timezone-aware ISO timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise A3Error(f"{where} must include a timezone offset")
    if timestamp.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise A3Error(f"{where} must not be in the future")
    return text


def validate_evidence_path(manifest_path: str | Path, value: Any, where: str) -> Path:
    if not isinstance(value, str):
        raise A3Error(f"{where} must be a safe POSIX relative path")
    manifest_path = Path(manifest_path).resolve()
    path = resolve_under(manifest_path.parent, value.strip())
    if path == manifest_path:
        raise A3Error(f"{where} cannot use its manifest as row evidence")
    lowered_parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if (name in _SENSITIVE_EVIDENCE_NAMES or path.suffix.lower() in _SENSITIVE_EVIDENCE_SUFFIXES or
            any(part in {".ssh", "credentials", "secrets", "tokens"}
                for part in lowered_parts) or
            any(token in name for token in ("private_key", "secret", "password", "token"))):
        raise A3Error(f"{where} names a forbidden credential/private-key/token file: {value!r}")
    if not path.is_file():
        raise A3Error(f"Missing evidence file for {where}: {path}")
    return path


def copy_manifest_with_evidence(
    source_manifest: str | Path, destination_manifest: str | Path,
    manifest_kind: str,
) -> Path:
    """Copy a validated scale manifest and its evidence into an immutable tree."""
    source_manifest, destination_manifest = Path(source_manifest), Path(destination_manifest)
    if manifest_kind != "scale":
        raise A3Error(f"Unknown evidence manifest kind: {manifest_kind!r}")
    rows = load_coordinate_scale_manifest(source_manifest)
    path_key = "scale_evidence_path"
    if destination_manifest.exists() or destination_manifest.parent.exists():
        raise A3Error(f"Refusing to overwrite evidence copy: {destination_manifest.parent}")
    destination_manifest.parent.mkdir(parents=True)
    try:
        shutil.copy2(source_manifest, destination_manifest)
        for row in rows.values():
            relative = row[path_key]
            source = validate_evidence_path(source_manifest, relative, path_key)
            destination = resolve_under(destination_manifest.parent, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256_file(destination) != sha256_file(source):
                    raise A3Error(f"Evidence destination collision with different bytes: {relative}")
            else:
                shutil.copy2(source, destination)
            if sha256_file(destination) != row[path_key.replace("_path", "_sha256")]:
                raise A3Error(f"Copied evidence hash mismatch: {relative}")
    except Exception:
        shutil.rmtree(destination_manifest.parent, ignore_errors=True)
        raise
    return destination_manifest


def _read_manifest_rows(
    path: str | Path, required: tuple[str, ...], name: str, schema: str,
) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise A3Error(f"Missing {name}: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise A3Error(f"Invalid {name} {path}: {exc}") from exc
    if fields != required:
        raise A3Error(
            f"{name} fields mismatch; missing={sorted(set(required)-set(fields))}, "
            f"unknown={sorted(set(fields)-set(required))}, order={list(fields)}")
    if not rows:
        raise A3Error(f"{name} is empty")
    for line, row in enumerate(rows, start=2):
        if None in row or set(row) != set(required):
            raise A3Error(f"{name} line {line} is malformed")
        if row["schema_version"].strip() != schema:
            raise A3Error(
                f"{name} line {line} schema_version={row['schema_version']!r}; "
                f"expected {schema!r}")
    return rows


def _validate_expected_manifest_sources(
    result: dict[str, dict[str, Any]], expected_sources: dict[str, str], name: str,
    allow_extra: bool,
) -> None:
    missing = set(expected_sources) - set(result)
    extra = set(result) - set(expected_sources)
    if missing or (extra and not allow_extra):
        raise A3Error(
            f"{name} IDs do not cover the requested source identities; "
            f"missing={sorted(missing)}, extra={sorted(extra) if not allow_extra else []}")
    for sample_id, expected_hash in expected_sources.items():
        if result[sample_id]["source_ply_sha256"] != expected_hash:
            raise A3Error(f"{name} source PLY binding mismatch: {sample_id}")


def load_coordinate_scale_manifest(
    path: str | Path, expected_sources: dict[str, str] | None = None,
    *, allow_extra: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load and hash-check independently evidenced raw-coordinate-to-metre scales."""
    path = Path(path)
    rows = _read_manifest_rows(
        path, COORDINATE_SCALE_MANIFEST_FIELDS, "coordinate-scale manifest",
        COORDINATE_SCALE_MANIFEST_SCHEMA)
    result: dict[str, dict[str, Any]] = {}
    for line, row in enumerate(rows, start=2):
        sample_id = row["scan_id"].strip()
        if not sample_id or sample_id in result:
            raise A3Error(f"Coordinate-scale manifest line {line} has empty/duplicate scan_id")
        source_hash = _required_sha256(
            row["source_ply_sha256"].strip(), f"Scale {sample_id} source_ply_sha256")
        try:
            scale = float(row["coordinate_scale_to_m"])
        except ValueError as exc:
            raise A3Error(f"Scale {sample_id} coordinate_scale_to_m must be numeric") from exc
        scale = _finite_positive(scale, f"Scale {sample_id} coordinate_scale_to_m")
        source = validate_coordinate_scale_source(
            row["coordinate_scale_source"], f"Scale {sample_id} coordinate_scale_source")
        semantics = row["scale_semantics"].strip()
        if semantics != SCALE_SEMANTICS:
            raise A3Error(
                f"Scale {sample_id} scale_semantics={semantics!r}, expected {SCALE_SEMANTICS!r}")
        evidence_type = _validate_evidence_type(
            row["scale_evidence_type"], f"Scale {sample_id} scale_evidence_type")
        evidence_relative = row["scale_evidence_path"].strip()
        evidence_path = validate_evidence_path(
            path, evidence_relative, f"Scale {sample_id} scale_evidence_path")
        evidence_hash = _required_sha256(
            row["scale_evidence_sha256"].strip(), f"Scale {sample_id} scale_evidence_sha256")
        actual_evidence_hash = sha256_file(evidence_path)
        if evidence_hash != actual_evidence_hash:
            raise A3Error(
                f"Scale {sample_id} evidence hash mismatch: "
                f"declared={evidence_hash}, actual={actual_evidence_hash}")
        result[sample_id] = {
            "schema_version": COORDINATE_SCALE_MANIFEST_SCHEMA,
            "source_ply_sha256": source_hash, "coordinate_scale_to_m": scale,
            "coordinate_scale_source": source, "scale_semantics": semantics,
            "scale_evidence_type": evidence_type,
            "scale_evidence_path": evidence_relative,
            "scale_evidence_sha256": evidence_hash,
            "frozen_at": _validate_frozen_at(
                row["frozen_at"], f"Scale {sample_id} frozen_at"),
            "notes": row["notes"],
        }
    if expected_sources is not None:
        _validate_expected_manifest_sources(
            result, expected_sources, "Coordinate-scale manifest", allow_extra)
    return result


def _reject_external_direction_fields(value: Any, where: str) -> None:
    """Reject legacy/external direction evidence while allowing computed audit fields."""
    legacy_front_source = "front_direction" + "_source"
    forbidden = {
        "front_axis", "front_sign", "front_vector", "front_vector_xy",
        legacy_front_source, "front_direction_manifest",
        "front_direction_manifest_path", "front_direction_manifest_sha256",
        "direction_evidence", "direction_evidence_type", "direction_evidence_path",
        "direction_evidence_sha256",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            legacy_direction = (
                ("front_axis" in key and key != "front_axis_computed") or
                ("front_sign" in key and key != "front_sign_computed") or
                "front_vector" in key or "front_direction_manifest" in key or
                "direction_evidence" in key)
            if key in forbidden or legacy_direction:
                raise A3Error(f"{where} contains forbidden external direction field: {key}")
            _reject_external_direction_fields(child, where)
    elif isinstance(value, list):
        for child in value:
            _reject_external_direction_fields(child, where)


def validate_geometric_pseudo_payload(payload: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Require a v3 strict-AND report with geometry-internal direction audit."""
    if payload.get("schema_version") != PSEUDO_LABEL_REPORT_SCHEMA:
        raise A3Error(
            f"Pseudo-label report must use {PSEUDO_LABEL_REPORT_SCHEMA}; "
            f"legacy pseudo-label JSON is not formal evidence: {path}")
    _reject_external_direction_fields(payload, f"Pseudo-label report {path}")
    summary = payload.get("summary")
    labels = payload.get("labels")
    if not isinstance(summary, dict) or not isinstance(labels, dict) or not labels:
        raise A3Error(f"Pseudo-label report lacks non-empty summary/labels objects: {path}")
    contract = summary.get("algorithm_contract")
    expected = {
        "version": PSEUDO_LABEL_CONTRACT_VERSION,
        "combination": "strict_AND_with_preregistered_relaxation",
        "adaptive_relaxation": True,
        "adaptive_relaxation_policy": ADAPTIVE_RELAXATION_POLICY,
        "slope_definition": "step3_abs_up_component_degrees",
        "normalize_input_normals": False,
        "front_direction_policy": FRONT_DIRECTION_POLICY,
        "coordinate_scale_policy": COORDINATE_SCALE_POLICY,
        "source_identity_policy": "ply_sha256_and_ordered_point_count",
        "convex_radius_m": 0.5, "front_percentile": 0.70,
        "slope_range_degrees": [20.0, 40.0], "k_normal": 20,
    }
    if not isinstance(contract, dict):
        raise A3Error(f"Pseudo-label report lacks algorithm_contract: {path}")
    for key, value in expected.items():
        if contract.get(key) != value:
            raise A3Error(
                f"Pseudo-label contract {key}={contract.get(key)!r}, expected {value!r}: {path}")
    for key in ("config_sha256", "generator_sha256", "coordinate_scale_manifest_sha256"):
        _required_sha256(contract.get(key), f"Pseudo-label contract {key}")
    return labels


def validate_geometric_label_entry(
    entry: Any, sample_id: str, expected_count: int, expected_ply_sha256: str,
    expected_scale_evidence_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate one v3 label vector and its scale-bound, internal-direction provenance."""
    if not isinstance(entry, dict) or "labels" not in entry:
        raise A3Error(f"Pseudo-label report misses sample {sample_id}")
    raw = np.asarray(entry["labels"])
    if (raw.ndim != 1 or len(raw) != expected_count or
            not np.issubdtype(raw.dtype, np.integer)):
        raise A3Error(f"Invalid pseudo-label vector for {sample_id}: {raw.shape}")
    values = raw.astype(np.int64, copy=False)
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 1, 255}) or not np.any(values != 255):
        raise A3Error(f"Invalid pseudo-label values for {sample_id}: {sorted(unique)}")
    provenance = entry.get("provenance")
    if not isinstance(provenance, dict):
        raise A3Error(f"Pseudo-label sample lacks provenance: {sample_id}")
    if provenance.get("sample_id") != sample_id:
        raise A3Error(f"Pseudo-label sample_id provenance mismatch: {sample_id}")
    if provenance.get("source_ply_sha256") != expected_ply_sha256:
        raise A3Error(f"Pseudo-label source PLY provenance mismatch: {sample_id}")
    if provenance.get("point_count") != expected_count:
        raise A3Error(f"Pseudo-label point-count provenance mismatch: {sample_id}")
    _reject_external_direction_fields(provenance, f"Pseudo-label provenance for {sample_id}")
    if provenance.get("front_axis_computed") not in {"x", "y"}:
        raise A3Error(f"Pseudo-label computed front axis is invalid: {sample_id}")
    if provenance.get("front_sign_computed") != 1:
        raise A3Error(f"Pseudo-label computed front sign must be +1: {sample_id}")
    if provenance.get("front_computation_source") != "geometry_internal":
        raise A3Error(f"Pseudo-label front computation must be geometry_internal: {sample_id}")
    scale = provenance.get("coordinate_scale_to_m")
    radius_raw = provenance.get("effective_convex_radius_raw")
    _finite_positive(scale, f"Pseudo-label coordinate_scale_to_m for {sample_id}")
    _finite_positive(radius_raw, f"Pseudo-label effective_convex_radius_raw for {sample_id}")
    if not math.isclose(float(scale) * float(radius_raw), 0.5,
                        rel_tol=0.0, abs_tol=1e-12):
        raise A3Error(f"Pseudo-label effective radius is not exactly 0.5 m: {sample_id}")
    validate_coordinate_scale_source(
        provenance.get("coordinate_scale_source"),
        f"Pseudo-label coordinate_scale_source for {sample_id}")
    _validate_evidence_type(
        provenance.get("scale_evidence_type"),
        f"Pseudo-label scale_evidence_type for {sample_id}")
    evidence_path = provenance.get("scale_evidence_path")
    if (not isinstance(evidence_path, str) or not evidence_path.strip() or
            "\\" in evidence_path or Path(evidence_path).is_absolute() or
            ".." in Path(evidence_path).parts):
        raise A3Error(
            f"Pseudo-label scale_evidence_path must be a safe relative path: {sample_id}")
    if provenance.get("scale_semantics") != SCALE_SEMANTICS:
        raise A3Error(
            f"Pseudo-label scale_semantics must be {SCALE_SEMANTICS!r}: {sample_id}")
    scale_evidence_hash = _required_sha256(
        provenance.get("scale_evidence_sha256"),
        f"Pseudo-label scale_evidence_sha256 for {sample_id}")
    if scale_evidence_hash != expected_scale_evidence_sha256:
        raise A3Error(
            f"Pseudo-label coordinate-scale evidence does not bind the supplied row evidence: "
            f"{sample_id}")
    validate_relaxation_fields(
        provenance.get("relaxation_stage"), provenance.get("mask_combination"),
        provenance.get("strict_and_applied"),
        provenance.get("adaptive_relaxation_applied"),
        f"Pseudo-label provenance for {sample_id}")
    return values.astype(np.uint8), provenance


def validate_pseudo_label_report_bindings(
    payload: dict[str, Any], config_path: str | Path,
    generator_path: str | Path, coordinate_scale_manifest_path: str | Path,
) -> dict[str, Any]:
    """Bind report declarations to the algorithm and canonical scale evidence."""
    contract = payload.get("summary", {}).get("algorithm_contract", {})
    bindings = {
        "config_sha256": Path(config_path),
        "generator_sha256": Path(generator_path),
        "coordinate_scale_manifest_sha256": Path(coordinate_scale_manifest_path),
    }
    for key, path in bindings.items():
        if not path.is_file():
            raise A3Error(f"Missing pseudo-label binding file {key}: {path}")
        actual = sha256_file(path)
        if contract.get(key) != actual:
            raise A3Error(
                f"Pseudo-label report {key} does not bind supplied file: "
                f"declared={contract.get(key)!r}, actual={actual}")
    return contract


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def atomic_save_npz(path: str | Path, **arrays: Any) -> None:
    """Write a deterministic, compressed NPZ with fixed ZIP metadata."""
    import io
    import zipfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asanyarray(arrays[name]),
                                          allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                                 compresslevel=6)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: str | Path, value: Any) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_under(root: str | Path, value: str) -> Path:
    if not value or "\\" in value:
        raise A3Error(f"Manifest path must be a POSIX relative path: {value!r}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise A3Error(f"Unsafe manifest path: {value!r}")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise A3Error(f"Manifest path escapes root: {value!r}") from exc
    return result


# ---------------------------------------------------------------------------
# Stable-snapshot / TOCTOU-safe reads and durable atomic publication (all-valid).
# Reject symlinks/hardlinks/non-regular files; prove one immutable pathname
# version is hashed, parsed and consumed. Kept self-contained so a3_io stays a
# leaf module (a3_data has private analogues it does not export).
# ---------------------------------------------------------------------------
StableFingerprint = tuple[int, int, int, int]
_Parsed = TypeVar("_Parsed")


def _stable_fingerprint(value: os.stat_result) -> StableFingerprint:
    return (int(value.st_dev), int(value.st_ino), int(value.st_size),
            int(value.st_mtime_ns))


def stable_path_fingerprint(path: str | Path, asset: str) -> StableFingerprint:
    """Stat a path, rejecting symlinks, hardlinks and non-regular files."""
    path = Path(path)
    try:
        lst = path.lstat()
    except OSError as exc:
        raise A3Error(f"Cannot lstat {asset} {path}: {exc}") from exc
    if stat.S_ISLNK(lst.st_mode):
        raise A3Error(f"{asset} must not be a symlink: {path}")
    try:
        value = path.stat()
    except OSError as exc:
        raise A3Error(f"Cannot stat {asset} {path}: {exc}") from exc
    if not stat.S_ISREG(value.st_mode):
        raise A3Error(f"{asset} is not a regular file: {path}")
    if value.st_nlink != 1:
        raise A3Error(f"{asset} must not be a hard link (nlink={value.st_nlink}): {path}")
    return _stable_fingerprint(value)


def read_stable_bytes(path: str | Path, asset: str) -> tuple[bytes, StableFingerprint]:
    """Read one immutable pathname version; reject replacement/mutation around it."""
    path = Path(path)
    before_path = stable_path_fingerprint(path, asset)
    try:
        with path.open("rb") as handle:
            before_handle = _stable_fingerprint(os.fstat(handle.fileno()))
            if before_handle != before_path:
                raise A3Error(f"{asset} changed while it was being opened: {path}")
            if os.fstat(handle.fileno()).st_nlink != 1:
                raise A3Error(f"{asset} became a hard link during open: {path}")
            payload = handle.read()
            after_handle = _stable_fingerprint(os.fstat(handle.fileno()))
    except A3Error:
        raise
    except OSError as exc:
        raise A3Error(f"Cannot read {asset} {path}: {exc}") from exc
    after_path = stable_path_fingerprint(path, asset)
    if not (before_path == before_handle == after_handle == after_path):
        raise A3Error(f"{asset} changed while it was being read: {path}")
    if len(payload) != before_path[2]:
        raise A3Error(f"{asset} byte count changed while it was being read: {path}")
    return payload, before_path


def stable_digest_parse(
    path: str | Path, declared_sha256: str | None, asset: str,
    parser: Callable[[bytes, Path], _Parsed],
) -> tuple[_Parsed, str, StableFingerprint]:
    """Digest and parse the exact same immutable snapshot, then recheck the pathname."""
    path = Path(path)
    payload, fingerprint = read_stable_bytes(path, asset)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if declared_sha256 is not None and actual_sha256 != declared_sha256:
        raise A3Error(
            f"{asset} hash mismatch: {path}; declared={declared_sha256}, "
            f"actual={actual_sha256}")
    parsed = parser(payload, path)
    after = stable_path_fingerprint(path, asset)
    if after != fingerprint:
        raise A3Error(f"{asset} changed during its stable read: {path}")
    return parsed, actual_sha256, fingerprint


def _fsync_dir(directory: Path) -> None:
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_atomic_write_bytes(path: str | Path, payload: bytes) -> str:
    """Write bytes, flush+fsync file, fsync parent, atomic rename, fsync parent again."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".dtmp-{os.getpid()}")
    try:
        fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(path.parent)
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def durable_publish_dir(staging_root: str | Path, output_root: str | Path) -> Path:
    """fsync every file+dir in staging, then atomically rename; never overwrite."""
    staging_root = Path(staging_root)
    output_root = Path(output_root)
    if output_root.exists():
        raise A3Error(f"Refusing to overwrite existing output: {output_root}")
    for current, _dirs, files in os.walk(staging_root):
        current_path = Path(current)
        for name in files:
            fpath = current_path / name
            if fpath.is_symlink():
                raise A3Error(f"Refusing to publish a symlink: {fpath}")
            fd = os.open(str(fpath), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(current_path)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_root, output_root)
    _fsync_dir(output_root.parent)
    return output_root


def read_ply_xyzn(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read XYZ and optional normals in property order; reject malformed PLY files."""
    path = Path(path)
    if not path.is_file():
        raise A3Error(f"Missing PLY: {path}")
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise A3Error(f"Not a PLY file: {path}")
        fmt = None
        count = None
        properties: list[tuple[str, str]] = []
        current = None
        while True:
            raw = handle.readline()
            if not raw:
                raise A3Error(f"PLY header has no end_header: {path}")
            try:
                parts = raw.decode("ascii").strip().split()
            except UnicodeDecodeError as exc:
                raise A3Error(f"Non-ASCII PLY header: {path}") from exc
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                fmt = parts[1]
            elif parts[0] == "element" and len(parts) == 3:
                current = parts[1]
                if current == "vertex":
                    count = int(parts[2])
            elif parts[0] == "property" and current == "vertex":
                if len(parts) != 3 or parts[1] == "list" or parts[1] not in PLY_TYPES:
                    raise A3Error(f"Unsupported PLY vertex property in {path}: {' '.join(parts)}")
                properties.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                break
        if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise A3Error(f"Unsupported PLY format {fmt!r}: {path}")
        if count is None or count <= 0:
            raise A3Error(f"Invalid vertex count in {path}")
        names = [name.lower() for name, _ in properties]
        if any(name not in names for name in ("x", "y", "z")):
            raise A3Error(f"PLY lacks XYZ properties: {path}")
        wanted = [names.index(name) for name in ("x", "y", "z")]
        normal_indices = [names.index(name) for name in ("nx", "ny", "nz")] \
            if all(name in names for name in ("nx", "ny", "nz")) else None
        if fmt == "ascii":
            xyz = np.empty((count, 3), dtype=np.float32)
            normals = np.empty((count, 3), dtype=np.float32) if normal_indices else None
            for row in range(count):
                values = handle.readline().split()
                if len(values) < len(properties):
                    raise A3Error(f"Truncated PLY at vertex {row}/{count}: {path}")
                try:
                    xyz[row] = [float(values[index]) for index in wanted]
                    if normals is not None:
                        normals[row] = [float(values[index]) for index in normal_indices]
                except (ValueError, OverflowError) as exc:
                    raise A3Error(f"Invalid PLY vertex {row}: {path}") from exc
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(name, endian + PLY_TYPES[kind]) for name, kind in properties])
            payload = handle.read(dtype.itemsize * count)
            if len(payload) != dtype.itemsize * count:
                raise A3Error(f"Truncated binary PLY: {path}")
            records = np.frombuffer(payload, dtype=dtype, count=count)
            xyz = np.column_stack([records[properties[index][0]] for index in wanted]).astype(np.float32)
            normals = (np.column_stack([records[properties[index][0]] for index in normal_indices])
                       .astype(np.float32)) if normal_indices else None
    if not np.isfinite(xyz).all() or (normals is not None and not np.isfinite(normals).all()):
        raise A3Error(f"PLY contains NaN or infinity: {path}")
    return xyz, normals


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, Torch, CUDA and deterministic backend controls."""
    if seed < 0:
        raise A3Error("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError as exc:
        raise A3Error("PyTorch is required for formal A3 training") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker hook based on Torch's per-worker initial seed."""
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state() -> dict[str, Any]:
    import torch
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    import torch
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise A3Error("Checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise A3Error("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def package_versions() -> dict[str, Any]:
    import platform
    try:
        import torch
        torch_info: dict[str, Any] = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    except ImportError:
        torch_info = {"torch": None, "cuda_runtime": None, "cudnn": None}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, **torch_info}


def ensure_unique(values: Iterable[str], name: str) -> list[str]:
    result = list(values)
    if len(result) != len(set(result)):
        raise A3Error(f"{name} contains duplicates")
    return result
