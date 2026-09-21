"""Strict matched-budget selection and point-cloud datasets for A3."""
from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import stat
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from a3_candidate_bundle import (
    validate_candidate_augmentation_metadata, verify_candidate_bundle,
)
from a3_io import (
    A3Error, ALL_VALID_ELIGIBILITY_RULE, ALL_VALID_EXPECTED_PER_PARENT,
    ALL_VALID_EXPECTED_TOTAL, ALL_VALID_MANIFEST_FIELDS,
    ALL_VALID_MANIFEST_SCHEMA, ALL_VALID_NPZ_SCHEMA, CANDIDATE_LABEL_NPZ_FIELDS,
    CANDIDATE_MANIFEST_FIELDS, FRONT_COMPUTATION_FIELDS, LABEL_NPZ_COMMON_FIELDS,
    ORIGINAL_LABEL_MANIFEST_FIELDS, PLY_TYPES, PSEUDO_LABEL_ASSET_SCHEMA,
    PSEUDO_LABEL_MANIFEST_SCHEMA, PSEUDO_LABEL_REPORT_SCHEMA,
    SCALE_IDENTITY_FIELDS, SCALE_SEMANTICS, _coerce_bool, ensure_unique,
    load_coordinate_scale_manifest, load_json, resolve_under, sha256_file,
    stable_seed, validate_coordinate_scale_source, validate_geometric_label_entry,
    validate_geometric_pseudo_payload, validate_pseudo_label_report_bindings,
)
from quality_contract import EXPECTED_THRESHOLD

METHODS = {"RAW_REPEAT", "TRAD", "LOADSIM"}
EXPECTED_TRAIN = 9
EXPECTED_DEV = 2
IGNORE_INDEX = 255


@dataclass(frozen=True)
class SourceRecord:
    scan_id: str
    split: str
    path: Path
    point_count: int
    file_sha256: str
    ordered_xyz_sha256: str | None


@dataclass(frozen=True)
class LabelBinding:
    path: Path
    label_sha256: str
    report_sha256: str
    config_sha256: str
    generator_sha256: str
    scale_manifest_sha256: str
    front_axis_computed: str
    front_sign_computed: int
    front_computation_source: str
    coordinate_scale_to_m: float
    coordinate_scale_source: str
    scale_evidence_type: str
    scale_evidence_path: str
    scale_evidence_sha256: str
    scale_semantics: str
    effective_convex_radius_raw: float
    parent_source_ply_sha256: str
    metadata_sha256: str | None = None
    label_counts: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    parent_scan_id: str
    method: str
    path: Path
    point_count: int
    file_sha256: str
    label_path: Path
    label_sha256: str
    metadata_path: Path | None = None
    generation_seed: int | None = None
    quality_config_sha256: str | None = None
    quality_report_sha256: str | None = None
    quality_q: float | None = None
    label_binding: LabelBinding | None = None
    # All-valid (all-technically-valid-v1) TRAD/LOADSIM records carry NO Q-v1
    # LabelBinding (verified via _all_valid_label_snapshot + receipt closure, not
    # a LabelBinding object). To let the dataset/preflight consumers re-verify the
    # label NPZ from disk on every read with the SAME strictness, they carry their
    # manifest row here as a hashable, sorted key/value tuple (the record stays a
    # frozen, hashable dataclass; a raw dict would break hashing). Exactly one of
    # label_binding / all_valid_label_row is set for a label-bearing record.
    all_valid_label_row: tuple[tuple[str, str], ...] | None = None


_SHA256_HEX = set("0123456789abcdef")
ORIGINAL_LABEL_FIELDS = set(ORIGINAL_LABEL_MANIFEST_FIELDS)

_StatFingerprint = tuple[int, int, int, int]
_Parsed = TypeVar("_Parsed")


def _fingerprint(value: os.stat_result) -> _StatFingerprint:
    """Identity/version tuple required for fail-closed asset reads."""
    return (int(value.st_dev), int(value.st_ino), int(value.st_size),
            int(value.st_mtime_ns))


def _path_fingerprint(path: Path, asset: str) -> _StatFingerprint:
    try:
        value = path.stat()
    except OSError as exc:
        raise A3Error(f"Cannot stat {asset} {path}: {exc}") from exc
    if not stat.S_ISREG(value.st_mode):
        raise A3Error(f"{asset} is not a regular file: {path}")
    return _fingerprint(value)


def _assert_fingerprint(path: Path, expected: _StatFingerprint, asset: str) -> None:
    actual = _path_fingerprint(path, asset)
    if actual != expected:
        raise A3Error(
            f"{asset} changed during or after its stable read: {path}; "
            f"expected stat={expected}, actual={actual}")


def _verify_cached_snapshot(
    path: Path, declared_sha256: str, expected: _StatFingerprint, asset: str,
) -> None:
    payload, actual = _read_stable_bytes(path, asset)
    if actual != expected:
        raise A3Error(
            f"{asset} changed since it was cached: {path}; "
            f"expected stat={expected}, actual={actual}")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != declared_sha256:
        raise A3Error(
            f"{asset} hash changed since it was cached: {path}; "
            f"declared={declared_sha256}, actual={actual_sha256}")
    _assert_fingerprint(path, expected, asset)


def _read_stable_bytes(path: Path, asset: str) -> tuple[bytes, _StatFingerprint]:
    """Read one pathname version; reject replacement or mutation around the read."""
    path = Path(path)
    before_path = _path_fingerprint(path, asset)
    try:
        with path.open("rb") as handle:
            before_handle = _fingerprint(os.fstat(handle.fileno()))
            if before_handle != before_path:
                raise A3Error(f"{asset} changed while it was being opened: {path}")
            payload = handle.read()
            after_handle = _fingerprint(os.fstat(handle.fileno()))
    except A3Error:
        raise
    except OSError as exc:
        raise A3Error(f"Cannot read {asset} {path}: {exc}") from exc
    after_path = _path_fingerprint(path, asset)
    if not (before_path == before_handle == after_handle == after_path):
        raise A3Error(f"{asset} changed while it was being read: {path}")
    if len(payload) != before_path[2]:
        raise A3Error(f"{asset} byte count changed while it was being read: {path}")
    return payload, before_path


def _stable_digest_parse(
    path: Path, declared_sha256: str | None, asset: str,
    parser: Callable[[bytes, Path], _Parsed],
) -> tuple[_Parsed, _StatFingerprint]:
    """Digest and parse the exact same immutable snapshot, then recheck pathname."""
    payload, fingerprint = _read_stable_bytes(path, asset)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if declared_sha256 is not None and actual_sha256 != declared_sha256:
        raise A3Error(
            f"{asset} hash mismatch: {path}; "
            f"declared={declared_sha256}, actual={actual_sha256}")
    parsed = parser(payload, path)
    _assert_fingerprint(path, fingerprint, asset)
    return parsed, fingerprint


def _parse_ply_bytes(payload: bytes, path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse XYZ/optional normals from an immutable PLY byte snapshot."""
    handle = io.BytesIO(payload)
    if handle.readline().strip() != b"ply":
        raise A3Error(f"Not a PLY file: {path}")
    fmt, count, current = None, None, None
    properties: list[tuple[str, str]] = []
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
                try:
                    count = int(parts[2])
                except ValueError as exc:
                    raise A3Error(f"Invalid vertex count in {path}") from exc
        elif parts[0] == "property" and current == "vertex":
            if len(parts) != 3 or parts[1] == "list" or parts[1] not in PLY_TYPES:
                raise A3Error(
                    f"Unsupported PLY vertex property in {path}: {' '.join(parts)}")
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
    normal_indices = ([names.index(name) for name in ("nx", "ny", "nz")]
                      if all(name in names for name in ("nx", "ny", "nz")) else None)
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
        dtype = np.dtype([(name, endian + PLY_TYPES[kind])
                          for name, kind in properties])
        binary = handle.read(dtype.itemsize * count)
        if len(binary) != dtype.itemsize * count:
            raise A3Error(f"Truncated binary PLY: {path}")
        records = np.frombuffer(binary, dtype=dtype, count=count)
        xyz = np.column_stack(
            [records[properties[index][0]] for index in wanted]).astype(np.float32)
        normals = (np.column_stack(
            [records[properties[index][0]] for index in normal_indices]).astype(np.float32)
            if normal_indices else None)
    if not np.isfinite(xyz).all() or (normals is not None and
                                      not np.isfinite(normals).all()):
        raise A3Error(f"PLY contains NaN or infinity: {path}")
    return xyz, normals


def _load_ply_snapshot(
    path: Path, declared_sha256: str | None, asset: str = "PLY",
) -> tuple[np.ndarray, np.ndarray | None, _StatFingerprint]:
    parsed, fingerprint = _stable_digest_parse(
        Path(path), declared_sha256, asset, _parse_ply_bytes)
    points, normals = parsed
    return points, normals, fingerprint


def _required_sha256(value: Any, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in _SHA256_HEX for character in value)):
        raise A3Error(f"{where} must be a lowercase SHA-256 hex digest")
    return value


def _scalar(archive: Any, key: str, path: Path) -> Any:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise A3Error(f"{path}: {key} must be a scalar")
    return value.item()


def _parse_int(value: Any, where: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise A3Error(f"{where} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise A3Error(f"{where} must be an integer") from exc
    if str(value).strip() != str(result):
        raise A3Error(f"{where} must use canonical integer syntax")
    if minimum is not None and result < minimum:
        raise A3Error(f"{where} must be >= {minimum}")
    return result


def _parse_float(value: Any, where: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise A3Error(f"{where} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise A3Error(f"{where} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise A3Error(f"{where} must be {'positive and ' if positive else ''}finite")
    return result


def _label_binding(row: dict[str, str], root: Path, where: str,
                   metadata_sha256: str | None = None) -> LabelBinding:
    if row.get("schema_version") != PSEUDO_LABEL_MANIFEST_SCHEMA or \
            row.get("manifest_schema") != PSEUDO_LABEL_MANIFEST_SCHEMA:
        actual = row.get("manifest_schema") or row.get("schema_version") or "unversioned"
        raise A3Error(
            f"{where} uses legacy/unsupported manifest schema {actual!r}; "
            f"formal A3 requires {PSEUDO_LABEL_MANIFEST_SCHEMA}")
    if row.get("label_schema") != PSEUDO_LABEL_ASSET_SCHEMA:
        raise A3Error(
            f"{where} uses legacy/unsupported label schema {row.get('label_schema')!r}; "
            f"formal A3 requires {PSEUDO_LABEL_ASSET_SCHEMA}")
    if row.get("pseudo_label_report_schema") != PSEUDO_LABEL_REPORT_SCHEMA:
        raise A3Error(f"{where} does not bind {PSEUDO_LABEL_REPORT_SCHEMA}")
    if row.get("label_type") != "geometric_pseudo_label":
        raise A3Error(f"{where} label_type is not geometric_pseudo_label")
    strict_flag = _coerce_bool(row.get("strict_and_applied"),
                               f"{where} strict_and_applied")
    adaptive_flag = _coerce_bool(row.get("adaptive_relaxation_applied"),
                                 f"{where} adaptive_relaxation_applied")
    if strict_flag == adaptive_flag:
        raise A3Error(
            f"{where} strict_and_applied/adaptive_relaxation_applied must be "
            "mutually exclusive (report provenance carries the full stage record)")
    axis = row.get("front_axis_computed", "")
    sign = _parse_int(row.get("front_sign_computed"),
                      f"{where} front_sign_computed")
    if axis not in {"x", "y"} or sign != 1:
        raise A3Error(f"{where} computed front axis/sign is invalid")
    direction_source = row.get("front_computation_source", "").strip()
    if direction_source != "geometry_internal":
        raise A3Error(f"{where} front_computation_source must be geometry_internal")
    scale = _parse_float(row.get("coordinate_scale_to_m"),
                         f"{where} coordinate_scale_to_m", positive=True)
    scale_source = validate_coordinate_scale_source(
        row.get("coordinate_scale_source"), f"{where} coordinate_scale_source")
    scale_manifest_hash = _required_sha256(
        row.get("coordinate_scale_manifest_sha256"),
        f"{where} coordinate_scale_manifest_sha256")
    scale_evidence_hash = _required_sha256(
        row.get("scale_evidence_sha256"),
        f"{where} scale_evidence_sha256")
    scale_evidence_type = row.get("scale_evidence_type", "").strip()
    scale_evidence_path = row.get("scale_evidence_path", "").strip()
    if not scale_evidence_type:
        raise A3Error(f"{where} scale_evidence_type must be non-empty")
    if (not scale_evidence_path or "\\" in scale_evidence_path or
            Path(scale_evidence_path).is_absolute() or
            ".." in Path(scale_evidence_path).parts):
        raise A3Error(f"{where} scale_evidence_path must be a safe relative path")
    if row.get("scale_semantics") != SCALE_SEMANTICS:
        raise A3Error(f"{where} has wrong scale_semantics")
    radius = _parse_float(row.get("effective_convex_radius_raw"),
                          f"{where} effective_convex_radius_raw", positive=True)
    if not math.isclose(scale * radius, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise A3Error(f"{where} effective convex radius is not exactly 0.5 m")
    counts = tuple(_parse_int(row.get(f"label_{label}"),
                              f"{where} label_{label}", minimum=0)
                   for label in (0, 1, 255))
    return LabelBinding(
        path=resolve_under(root, row.get("label_path", "")),
        label_sha256=_required_sha256(row.get("label_sha256"), f"{where} label_sha256"),
        report_sha256=_required_sha256(
            row.get("pseudo_label_source_json_sha256"), f"{where} report SHA-256"),
        config_sha256=_required_sha256(row.get("config_sha256"), f"{where} config_sha256"),
        generator_sha256=_required_sha256(
            row.get("generator_sha256"), f"{where} generator_sha256"),
        scale_manifest_sha256=scale_manifest_hash,
        front_axis_computed=axis, front_sign_computed=sign,
        front_computation_source=direction_source,
        coordinate_scale_to_m=scale,
        coordinate_scale_source=scale_source,
        scale_evidence_type=scale_evidence_type,
        scale_evidence_path=scale_evidence_path,
        scale_evidence_sha256=scale_evidence_hash,
        scale_semantics=SCALE_SEMANTICS,
        effective_convex_radius_raw=radius,
        parent_source_ply_sha256=_required_sha256(
            row.get("parent_source_ply_sha256"),
            f"{where} parent_source_ply_sha256"),
        metadata_sha256=metadata_sha256,
        label_counts=counts,
    )


def _group_scans(splits: dict[str, Any], name: str, expected: int) -> list[str]:
    if splits.get("schema_version") != "scan-splits-v2" or splits.get("preregistered") is not True:
        raise A3Error("A3 requires frozen preregistered scan-splits-v2")
    group = splits.get("groups", {}).get(name)
    if not isinstance(group, dict) or group.get("count") != expected:
        raise A3Error(f"Split {name} must contain exactly {expected} scans")
    scans = group.get("scans")
    if not isinstance(scans, list) or not all(isinstance(item, str) and item for item in scans):
        raise A3Error(f"Invalid {name} scan list")
    return ensure_unique(scans, f"{name} scans")


def load_frozen_sources(project_root: Path, split_path: Path,
                        source_manifest_path: Path) -> tuple[list[SourceRecord], list[SourceRecord]]:
    project_root, split_path, source_manifest_path = map(Path, (project_root, split_path, source_manifest_path))
    splits = load_json(split_path)
    train_ids = _group_scans(splits, "TRAIN", EXPECTED_TRAIN)
    dev_ids = _group_scans(splits, "DEV", EXPECTED_DEV)
    if set(train_ids) & set(dev_ids):
        raise A3Error("TRAIN and DEV parent scans overlap")
    if splits.get("source_manifest") != source_manifest_path.name:
        raise A3Error("Split file is not bound to the supplied source manifest")
    if not source_manifest_path.is_file():
        raise A3Error(f"Missing source manifest: {source_manifest_path}")
    if splits.get("source_manifest_sha256") != sha256_file(source_manifest_path):
        raise A3Error("Split file source_manifest_sha256 does not match supplied manifest")
    with source_manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scan_id", "split", "file_path", "point_count", "file_sha256"}
    if not rows or not required.issubset(rows[0]):
        raise A3Error(f"Source manifest lacks fields {sorted(required)}")
    by_id: dict[str, SourceRecord] = {}
    for row in rows:
        scan_id = row["scan_id"]
        if scan_id in by_id:
            raise A3Error(f"Duplicate source manifest scan_id: {scan_id}")
        try:
            count = int(row["point_count"])
        except ValueError as exc:
            raise A3Error(f"Invalid point_count for {scan_id}") from exc
        path = resolve_under(project_root, row["file_path"])
        by_id[scan_id] = SourceRecord(scan_id, row["split"], path, count,
                                      row["file_sha256"], row.get("ordered_xyz_sha256") or None)
    expected_ids = train_ids + dev_ids
    missing = [scan_id for scan_id in expected_ids if scan_id not in by_id]
    if missing:
        raise A3Error(f"Source manifest misses frozen scans: {missing}")
    for split_name, ids in (("TRAIN", train_ids), ("DEV", dev_ids)):
        for scan_id in ids:
            record = by_id[scan_id]
            if record.split != split_name:
                raise A3Error(f"Source {scan_id} split is {record.split}, expected {split_name}")
            points, _, _ = _load_ply_snapshot(
                record.path, record.file_sha256, "Frozen source PLY")
            if len(points) != record.point_count:
                raise A3Error(f"Frozen source point count mismatch: {scan_id}")
    return [by_id[item] for item in train_ids], [by_id[item] for item in dev_ids]


def _label_payload_snapshot(path: Path, expected_sample: str, expected_parent: str,
                            expected_count: int, expected_ply_sha256: str,
                            binding: LabelBinding | None,
                            ) -> tuple[np.ndarray, _StatFingerprint]:
    """Read a manifest-bound a3-label-v3 NPZ from one stable snapshot."""
    if binding is None:
        raise A3Error(f"Formal A3 label lacks its v3 manifest binding: {path}")
    if path != binding.path:
        raise A3Error(f"Label path differs from its v3 manifest binding: {path}")
    if path.suffix.lower() != ".npz":
        raise A3Error(f"Formal A3 labels must be identity-bound NPZ files: {path}")
    payload, fingerprint = _read_stable_bytes(path, "Label NPZ")
    actual_label_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_label_sha256 != binding.label_sha256:
        raise A3Error(
            f"Label NPZ hash differs from v3 manifest: {path}; "
            f"declared={binding.label_sha256}, actual={actual_label_sha256}")
    common = set(LABEL_NPZ_COMMON_FIELDS)
    expected_fields = (set(CANDIDATE_LABEL_NPZ_FIELDS)
                       if binding.metadata_sha256 is not None else common)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            actual_fields = set(archive.files)
            if "schema_version" not in actual_fields:
                raise A3Error(f"Label NPZ lacks schema_version: {path}")
            actual_schema = str(_scalar(archive, "schema_version", path))
            if actual_schema != PSEUDO_LABEL_ASSET_SCHEMA:
                raise A3Error(
                    f"{path}: legacy/unsupported label schema {actual_schema!r}; "
                    f"formal A3 requires {PSEUDO_LABEL_ASSET_SCHEMA}")
            if actual_fields != expected_fields:
                raise A3Error(
                    f"Label NPZ v3 fields mismatch for {path}; "
                    f"missing={sorted(expected_fields-actual_fields)}, "
                    f"unknown={sorted(actual_fields-expected_fields)}")
            labels = np.asarray(archive["labels"])
            string_bindings = {
                "sample_id": expected_sample,
                "source_scan_id": expected_parent,
                "source_ply_sha256": expected_ply_sha256,
                "schema_version": PSEUDO_LABEL_ASSET_SCHEMA,
                "pseudo_label_report_sha256": binding.report_sha256,
                "config_sha256": binding.config_sha256,
                "generator_sha256": binding.generator_sha256,
                "coordinate_scale_manifest_sha256": binding.scale_manifest_sha256,
                "scale_evidence_sha256": binding.scale_evidence_sha256,
                "scale_evidence_type": binding.scale_evidence_type,
                "scale_evidence_path": binding.scale_evidence_path,
                "front_axis_computed": binding.front_axis_computed,
                "front_computation_source": binding.front_computation_source,
                "coordinate_scale_source": binding.coordinate_scale_source,
                "scale_semantics": binding.scale_semantics,
            }
            if binding.metadata_sha256 is not None:
                string_bindings.update({
                    "parent_source_ply_sha256": binding.parent_source_ply_sha256,
                    "augmentation_metadata_sha256": binding.metadata_sha256,
                })
            for key, expected in string_bindings.items():
                actual = str(_scalar(archive, key, path))
                if actual != expected:
                    raise A3Error(f"{path}: {key}={actual!r}, expected {expected!r}")
            if _parse_int(_scalar(archive, "front_sign_computed", path),
                          f"{path} front_sign_computed") != binding.front_sign_computed:
                raise A3Error(f"{path}: front_sign_computed differs from manifest")
            scale = _parse_float(_scalar(archive, "coordinate_scale_to_m", path),
                                 f"{path} coordinate_scale_to_m", positive=True)
            radius = _parse_float(_scalar(archive, "effective_convex_radius_raw", path),
                                  f"{path} effective_convex_radius_raw", positive=True)
            if (not math.isclose(scale, binding.coordinate_scale_to_m,
                                 rel_tol=0.0, abs_tol=1e-12) or
                    not math.isclose(radius, binding.effective_convex_radius_raw,
                                     rel_tol=0.0, abs_tol=1e-12) or
                    not math.isclose(scale * radius, 0.5,
                                     rel_tol=0.0, abs_tol=1e-12)):
                raise A3Error(f"{path}: physical-scale provenance differs from manifest")
    except (OSError, ValueError) as exc:
        raise A3Error(f"Invalid label NPZ {path}: {exc}") from exc
    _assert_fingerprint(path, fingerprint, "Label NPZ")
    if labels.dtype != np.dtype(np.uint8):
        raise A3Error(f"Formal A3 labels must use uint8 storage: {path}")
    if labels.ndim != 1 or len(labels) != expected_count:
        raise A3Error(f"Label length mismatch for {expected_sample}: {labels.shape}, expected ({expected_count},)")
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1, IGNORE_INDEX}):
        raise A3Error(f"Invalid labels {sorted(unique)} in {path}")
    if not np.any(labels != IGNORE_INDEX):
        raise A3Error(f"All points are ignored: {path}")
    actual_counts = tuple(int(np.count_nonzero(labels == value))
                          for value in (0, 1, IGNORE_INDEX))
    if actual_counts != binding.label_counts or sum(actual_counts) != expected_count:
        raise A3Error(
            f"{path}: label counts {actual_counts} differ from manifest {binding.label_counts}")
    return labels, fingerprint


def _label_payload(path: Path, expected_sample: str, expected_parent: str,
                   expected_count: int, expected_ply_sha256: str,
                   binding: LabelBinding | None) -> np.ndarray:
    labels, _ = _label_payload_snapshot(
        path, expected_sample, expected_parent, expected_count,
        expected_ply_sha256, binding)
    return labels


def _record_label_snapshot(
    record: "SampleRecord", expected_count: int,
) -> tuple[np.ndarray, _StatFingerprint]:
    """Read+verify ONE record's label NPZ from a stable snapshot, dispatching by
    provenance so every consumer (dataset, preflight) applies the SAME strictness:

      * Q-v1 records (label_binding set)  -> _label_payload_snapshot (v3 binding).
      * all-valid records (all_valid_label_row set) -> _all_valid_label_snapshot,
        the loader's own strict verifier (NPZ hash vs manifest, schema, every
        string binding, counts, dtype, length, fingerprint). No Q-v1 binding
        exists for these, so this is the only faithful re-verification path.

    Fail-closed if a label-bearing record carries neither (prevents an unverified
    label from ever reaching training), and cross-checks the carried row against
    the record identity so a swapped row cannot redirect verification."""
    if record.label_binding is not None:
        return _label_payload_snapshot(
            record.label_path, record.sample_id, record.parent_scan_id,
            expected_count, record.file_sha256, record.label_binding)
    if record.all_valid_label_row is not None:
        row = dict(record.all_valid_label_row)
        # The carried row must describe THIS record (defense in depth): identity,
        # parent, PLY hash and the declared label hash must all agree before the
        # NPZ is trusted, so a mis-bound row can never point verification elsewhere.
        if (row.get("sample_id") != record.sample_id
                or row.get("parent_scan_id") != record.parent_scan_id
                or row.get("ply_sha256") != record.file_sha256
                or row.get("label_sha256") != record.label_sha256):
            raise A3Error(
                f"all-valid label row does not match record identity: {record.sample_id}")
        labels, _counts, fingerprint = _all_valid_label_snapshot(
            record.label_path, row, expected_count)
        return labels, fingerprint
    raise A3Error(
        f"Formal A3 label lacks its v3 manifest binding: {record.label_path}")


def _candidate_row(row: dict[str, str], manifest_root: Path) -> SampleRecord:
    actual = set(row)
    expected = set(CANDIDATE_MANIFEST_FIELDS)
    if actual != expected:
        raise A3Error(
            "Candidate manifest v3 fields mismatch; "
            f"missing={sorted(expected-actual)}, "
            f"unknown={sorted(actual-expected)}")
    sample_id = row.get("sample_id", "")
    if not sample_id:
        raise A3Error("Candidate manifest contains an empty sample_id")
    method = row["method"].upper()
    if method not in {"TRAD", "LOADSIM"}:
        raise A3Error(f"Invalid candidate method: {method}")
    point_count = _parse_int(row["point_count"],
                             f"Candidate {sample_id} point_count", minimum=1)
    generation_seed = _parse_int(row["generation_seed"],
                                 f"Candidate {sample_id} generation_seed", minimum=0)
    quality_q = _parse_float(row["quality_Q"], f"Candidate {sample_id} quality_Q")
    if (row["quality_rule"] != "Q-only" or
            _parse_float(row["quality_threshold_Q"],
                         f"Candidate {sample_id} quality_threshold_Q") != EXPECTED_THRESHOLD or
            row["quality_passed"] != "true" or quality_q < EXPECTED_THRESHOLD):
        raise A3Error(f"Candidate was not accepted by Q-only v1: {sample_id}")
    for name in ("quality_F", "quality_P", "quality_D", "quality_Q"):
        _parse_float(row[name], f"Candidate {sample_id} {name}")
    for name in ("ply_sha256", "canonical_xyz_sha256", "metadata_sha256",
                 "quality_config_sha256", "quality_report_sha256"):
        _required_sha256(row[name], f"Candidate {sample_id} {name}")
    metadata = resolve_under(manifest_root, row["metadata_path"])
    binding = _label_binding(
        row, manifest_root, f"Candidate manifest row {sample_id}", row["metadata_sha256"])
    if sum(binding.label_counts) != point_count:
        raise A3Error(f"Candidate manifest label counts do not sum to point_count: {sample_id}")
    return SampleRecord(
        sample_id=sample_id, parent_scan_id=row["parent_scan_id"], method=method,
        path=resolve_under(manifest_root, row["ply_path"]), point_count=point_count,
        file_sha256=row["ply_sha256"], label_path=binding.path,
        label_sha256=binding.label_sha256, metadata_path=metadata,
        generation_seed=generation_seed,
        quality_config_sha256=row["quality_config_sha256"],
        quality_report_sha256=row["quality_report_sha256"], quality_q=quality_q,
        label_binding=binding,
    )


def load_candidates(
    manifest_path: Path, sources: list[SourceRecord], method: str,
    verify_hashes: bool, quality_contract: dict[str, Any],
    *, canonical_scale_manifest: Path,
) -> dict[str, list[SampleRecord]]:
    if canonical_scale_manifest is None:
        raise A3Error("The canonical A3 coordinate-scale manifest must be supplied")
    train_ids = [source.scan_id for source in sources]
    source_hashes = {source.scan_id: source.file_sha256 for source in sources}
    method = method.upper()
    if method == "RAW_REPEAT":
        return {scan_id: [] for scan_id in train_ids}
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise A3Error(f"{method} requires a candidate manifest: {manifest_path}")
    verified = verify_candidate_bundle(
        manifest_path, quality_contract, sources, method,
        canonical_scale_manifest=canonical_scale_manifest)
    expected_quality_config_sha256 = quality_contract["sha256"]
    manifest_payload, manifest_fingerprint = _read_stable_bytes(
        manifest_path, "Candidate manifest")
    if hashlib.sha256(manifest_payload).hexdigest() != verified["manifest_sha256"]:
        raise A3Error("Candidate manifest changed after bundle replay verification")
    try:
        text = manifest_payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise A3Error(f"Invalid candidate manifest {manifest_path}: {exc}") from exc
    if fields != CANDIDATE_MANIFEST_FIELDS:
        actual = set(fields)
        expected = set(CANDIDATE_MANIFEST_FIELDS)
        raise A3Error(
            "Candidate manifest v3 fields/order mismatch; "
            f"missing={sorted(expected-actual)}, "
            f"unknown={sorted(actual-expected)}")
    by_parent = {scan_id: [] for scan_id in train_ids}
    seen: set[str] = set()
    quality_report_hashes: set[str] = set()
    for raw in rows:
        record = _candidate_row(raw, manifest_path.parent)
        if record.method != method:
            continue
        if record.sample_id in seen:
            raise A3Error(f"Duplicate candidate sample_id: {record.sample_id}")
        seen.add(record.sample_id)
        if record.parent_scan_id not in by_parent:
            raise A3Error(f"Candidate parent {record.parent_scan_id} is outside frozen TRAIN")
        if record.quality_config_sha256 != expected_quality_config_sha256:
            raise A3Error(
                f"Candidate quality config hash mismatch: {record.sample_id}")
        if not record.quality_report_sha256:
            raise A3Error(f"Candidate quality report hash is missing: {record.sample_id}")
        quality_report_hashes.add(record.quality_report_sha256)
        if raw.get("label_type") != "geometric_pseudo_label" or not raw.get("pseudo_label_source_json_sha256"):
            raise A3Error(f"Candidate label provenance is not geometric pseudo-label: {record.sample_id}")
        for path in (record.path, record.label_path):
            if not path.is_file():
                raise A3Error(f"Missing candidate asset: {path}")
        if record.metadata_path is None or not record.metadata_path.is_file():
            raise A3Error(f"Missing candidate metadata: {record.metadata_path}")
        metadata = load_json(record.metadata_path)
        if metadata.get("sample_id") != record.sample_id:
            raise A3Error(f"Candidate metadata sample_id mismatch: {record.sample_id}")
        if metadata.get("parent_scan_id") != record.parent_scan_id:
            raise A3Error(f"Candidate metadata parent mismatch: {record.sample_id}")
        if str(metadata.get("method", "")).upper() != record.method:
            raise A3Error(f"Candidate metadata method mismatch: {record.sample_id}")
        if metadata.get("parent_ply_sha256") != source_hashes[record.parent_scan_id]:
            raise A3Error(f"Candidate metadata parent hash mismatch: {record.sample_id}")
        if int(metadata.get("generation_seed", -1)) != record.generation_seed:
            raise A3Error(f"Candidate metadata seed mismatch: {record.sample_id}")
        binding = record.label_binding
        if binding is None:
            raise A3Error(f"Candidate lacks a v3 label binding: {record.sample_id}")
        validate_candidate_augmentation_metadata(metadata, record.method, record.sample_id)
        if binding.parent_source_ply_sha256 != source_hashes[record.parent_scan_id]:
            raise A3Error(f"Candidate label parent-source hash mismatch: {record.sample_id}")
        if metadata.get("fallback"):
            raise A3Error(f"Candidate metadata records forbidden fallback: {record.sample_id}")
        applied = metadata.get("applied_methods")
        if not isinstance(applied, list) or not applied:
            raise A3Error(f"Candidate contains no applied augmentation: {record.sample_id}")
        if record.method == "TRAD" and "loading_simulation" in applied:
            raise A3Error(f"TRAD candidate contains loading simulation: {record.sample_id}")
        if record.method == "LOADSIM":
            loading = metadata.get("deform_params", {}).get("loading_simulation")
            if (applied != ["loading_simulation"] or not isinstance(loading, dict) or
                    loading.get("n_operations", 0) <= 0 or loading.get("removal_ratio", 0.0) <= 0.0):
                raise A3Error(f"LOADSIM candidate lacks nonzero loading-only operation: {record.sample_id}")
        if verify_hashes:
            points, _, _ = _load_ply_snapshot(
                record.path, record.file_sha256, "Candidate PLY")
            if len(points) != record.point_count:
                raise A3Error(
                    f"Candidate point count mismatch: {record.sample_id}")
            _label_payload(
                record.label_path, record.sample_id, record.parent_scan_id,
                len(points), record.file_sha256, record.label_binding)
            if sha256_file(record.metadata_path) != raw["metadata_sha256"]:
                raise A3Error(f"Candidate metadata hash mismatch: {record.metadata_path}")
        by_parent[record.parent_scan_id].append(record)
    if len(quality_report_hashes) != 1:
        raise A3Error(
            f"Candidate manifest must bind one quality report, got {sorted(quality_report_hashes)}")
    for candidates in by_parent.values():
        candidates.sort(key=lambda item: item.sample_id)
    _assert_fingerprint(manifest_path, manifest_fingerprint, "Candidate manifest")
    return by_parent


# ---------------------------------------------------------------------------
# all-valid (all-technically-valid-v1) schema dispatch. Distinct from Q-v1:
# eligibility is EVERY technically valid candidate (288 = TRAIN-9 x 32); Q/F/P/D
# and the historical Q-v1 pass flag are carried report-only and NEVER gate. The
# receipt/closure/asset-SHA verification is delegated to a3_all_valid.verify_all_valid
# (which runs no geometric replay); this loader then stable-snapshots and parses
# the actually consumed PLY/NPZ and returns SampleRecords WITHOUT any Q gate.
# The Q-v1 load_candidates() above rejects all-valid manifests via its exact
# CANDIDATE_MANIFEST_FIELDS check; this loader rejects Q-v1 manifests symmetrically.
# ---------------------------------------------------------------------------
_ALL_VALID_NPZ_STRING_FIELDS = (
    "sample_id", "schema_version", "eligibility_rule", "source_scan_id",
    "candidate_ply_sha256", "augmentation_metadata_sha256",
    "parent_source_ply_sha256", "canonical_xyz_sha256", "canonical_normals_sha256",
    "pseudo_label_report_sha256", "config_sha256", "generator_sha256",
    "coordinate_scale_manifest_sha256",
)


def _all_valid_label_snapshot(
    path: Path, row: dict[str, str], expected_count: int,
) -> tuple[np.ndarray, tuple[int, int, int], _StatFingerprint]:
    """Read one all-valid label NPZ from a single stable snapshot and bind it to
    its manifest row. NEVER accepts the Q-v1 pseudo-label asset schema."""
    if path.suffix.lower() != ".npz":
        raise A3Error(f"all-valid labels must be NPZ files: {path}")
    payload, fingerprint = _read_stable_bytes(path, "all-valid label NPZ")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != row["label_sha256"]:
        raise A3Error(
            f"all-valid label NPZ hash differs from manifest: {path}; "
            f"declared={row['label_sha256']}, actual={actual_sha256}")
    string_bindings = {
        "sample_id": row["sample_id"],
        "schema_version": ALL_VALID_NPZ_SCHEMA,
        "eligibility_rule": ALL_VALID_ELIGIBILITY_RULE,
        "source_scan_id": row["parent_scan_id"],
        "candidate_ply_sha256": row["ply_sha256"],
        "augmentation_metadata_sha256": row["metadata_sha256"],
        "parent_source_ply_sha256": row["parent_source_ply_sha256"],
        "canonical_xyz_sha256": row["canonical_xyz_sha256"],
        "canonical_normals_sha256": row["canonical_normals_sha256"],
        "pseudo_label_report_sha256": row["pseudo_label_report_sha256"],
        "config_sha256": row["config_sha256"],
        "generator_sha256": row["generator_sha256"],
        "coordinate_scale_manifest_sha256": row["coordinate_scale_manifest_sha256"],
    }
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            actual_fields = set(archive.files)
            if "schema_version" not in actual_fields:
                raise A3Error(f"all-valid label NPZ lacks schema_version: {path}")
            actual_schema = str(_scalar(archive, "schema_version", path))
            if actual_schema != ALL_VALID_NPZ_SCHEMA:
                raise A3Error(
                    f"{path}: label schema {actual_schema!r}; all-valid requires "
                    f"{ALL_VALID_NPZ_SCHEMA} (Q-v1 asset schema is rejected here)")
            for key, expected in string_bindings.items():
                actual = str(_scalar(archive, key, path))
                if actual != expected:
                    raise A3Error(f"{path}: {key}={actual!r}, expected {expected!r}")
            labels = np.asarray(archive["labels"])
    except (OSError, ValueError) as exc:
        raise A3Error(f"Invalid all-valid label NPZ {path}: {exc}") from exc
    _assert_fingerprint(path, fingerprint, "all-valid label NPZ")
    if labels.dtype != np.dtype(np.uint8):
        raise A3Error(f"all-valid labels must use uint8 storage: {path}")
    if labels.ndim != 1 or len(labels) != expected_count:
        raise A3Error(
            f"all-valid label length mismatch for {row['sample_id']}: "
            f"{labels.shape}, expected ({expected_count},)")
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1, IGNORE_INDEX}):
        raise A3Error(f"Invalid all-valid labels {sorted(unique)} in {path}")
    if not np.any(labels != IGNORE_INDEX):
        raise A3Error(f"All points are ignored: {path}")
    counts = tuple(int(np.count_nonzero(labels == value)) for value in (0, 1, IGNORE_INDEX))
    manifest_counts = (int(row["label_0"]), int(row["label_1"]), int(row["label_255"]))
    if counts != manifest_counts or sum(counts) != expected_count:
        raise A3Error(
            f"{path}: label counts {counts} differ from manifest {manifest_counts}")
    return labels, counts, fingerprint  # type: ignore[return-value]


def load_all_valid_candidates(
    bundle_dir: Path, sources: list[SourceRecord], method: str,
    verify_hashes: bool,
) -> dict[str, list[SampleRecord]]:
    """Load an all-valid main-analysis bundle (no Q gate). Verifies the receipt,
    closures and every consumed asset SHA via a3_all_valid.verify_all_valid, then
    stable-snapshots/parses the consumed PLY and label NPZ. Returns 32 candidates
    per TRAIN source keyed by parent, exactly ALL_VALID_EXPECTED_TOTAL in total."""
    import a3_all_valid

    method = method.upper()
    if method not in ("TRAD", "LOADSIM"):
        raise A3Error(f"all-valid bundles are per-method TRAD/LOADSIM, not {method}")
    train_ids = [source.scan_id for source in sources]
    if len(train_ids) != EXPECTED_TRAIN:
        raise A3Error(f"all-valid requires {EXPECTED_TRAIN} TRAIN sources, got {len(train_ids)}")
    source_hashes = {source.scan_id: source.file_sha256 for source in sources}
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "candidate_manifest.csv"
    if not manifest_path.is_file():
        raise A3Error(f"all-valid bundle missing candidate_manifest.csv: {bundle_dir}")

    # Field/order gate FIRST: an all-valid consumer must reject a Q-v1 manifest.
    manifest_payload, manifest_fingerprint = _read_stable_bytes(
        manifest_path, "all-valid manifest")
    try:
        text = manifest_payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise A3Error(f"Invalid all-valid manifest {manifest_path}: {exc}") from exc
    if fields != ALL_VALID_MANIFEST_FIELDS:
        actual = set(fields)
        expected = set(ALL_VALID_MANIFEST_FIELDS)
        raise A3Error(
            "all-valid manifest fields/order mismatch (Q-v1 manifests are rejected); "
            f"missing={sorted(expected-actual)}, unknown={sorted(actual-expected)}")

    # Full receipt/closure/asset-SHA verification (runs NO geometric replay).
    receipt = a3_all_valid.verify_all_valid(bundle_dir)
    if receipt.get("method", "").upper() != method:
        raise A3Error(
            f"all-valid bundle is for method {receipt.get('method')!r}, not {method}")

    by_parent: dict[str, list[SampleRecord]] = {scan_id: [] for scan_id in train_ids}
    seen: set[str] = set()
    for raw in rows:
        # Every row is technically valid by construction; Q columns are report-only.
        if raw.get("schema_version") != ALL_VALID_MANIFEST_SCHEMA or \
                raw.get("manifest_schema") != ALL_VALID_MANIFEST_SCHEMA:
            raise A3Error(f"all-valid row has wrong manifest schema: {raw.get('sample_id')}")
        if raw.get("eligibility_rule") != ALL_VALID_ELIGIBILITY_RULE:
            raise A3Error(f"all-valid row eligibility_rule mismatch: {raw.get('sample_id')}")
        if _coerce_bool(raw.get("technical_valid"), "technical_valid") is not True:
            raise A3Error(f"all-valid row is not technically valid: {raw.get('sample_id')}")
        if raw.get("quality_role") != "report_only_not_eligibility":
            raise A3Error(f"all-valid row quality_role must be report-only: {raw.get('sample_id')}")
        if raw.get("method", "").upper() != method:
            raise A3Error(f"all-valid row method mismatch: {raw.get('sample_id')}")
        sample_id = raw["sample_id"]
        if sample_id in seen:
            raise A3Error(f"Duplicate all-valid sample_id: {sample_id}")
        seen.add(sample_id)
        parent = raw["parent_scan_id"]
        if parent not in by_parent:
            raise A3Error(f"all-valid candidate parent not in TRAIN: {parent}")
        if raw.get("parent_source_ply_sha256") != source_hashes[parent]:
            raise A3Error(f"all-valid parent-source hash mismatch: {sample_id}")
        variant = _parse_int(raw.get("variant_id"), f"{sample_id} variant_id", minimum=0)
        if variant >= ALL_VALID_EXPECTED_PER_PARENT:
            raise A3Error(f"all-valid variant out of 0..{ALL_VALID_EXPECTED_PER_PARENT-1}: {sample_id}")
        point_count = _parse_int(raw.get("point_count"), f"{sample_id} point_count", minimum=1)
        ply_path = resolve_under(bundle_dir, raw["ply_path"])
        label_path = resolve_under(bundle_dir, raw["label_path"])
        metadata_path = resolve_under(bundle_dir, raw["metadata_path"])
        quality_q = _parse_float(raw.get("quality_Q"), f"{sample_id} quality_Q")
        if verify_hashes:
            points, _, _ = _load_ply_snapshot(ply_path, raw["ply_sha256"], "all-valid PLY")
            if len(points) != point_count:
                raise A3Error(f"all-valid point count mismatch: {sample_id}")
            _all_valid_label_snapshot(label_path, raw, len(points))
            if sha256_file(metadata_path) != raw["metadata_sha256"]:
                raise A3Error(f"all-valid metadata hash mismatch: {metadata_path}")
        by_parent[parent].append(SampleRecord(
            sample_id=sample_id, parent_scan_id=parent, method=method,
            path=ply_path, point_count=point_count, file_sha256=raw["ply_sha256"],
            label_path=label_path, label_sha256=raw["label_sha256"],
            metadata_path=metadata_path,
            generation_seed=_parse_int(raw.get("generation_seed"), f"{sample_id} seed"),
            quality_config_sha256=raw.get("quality_config_sha256"),
            quality_report_sha256=raw.get("quality_report_sha256"),
            quality_q=quality_q, label_binding=None,
            # Carry the verified manifest row (hashable) so the dataset/preflight
            # consumers re-verify this label NPZ from disk with identical strictness
            # via _all_valid_label_snapshot (no Q-v1 LabelBinding exists here).
            all_valid_label_row=tuple(sorted(raw.items())),
        ))

    if len(seen) != ALL_VALID_EXPECTED_TOTAL:
        raise A3Error(
            f"all-valid bundle must publish exactly {ALL_VALID_EXPECTED_TOTAL} "
            f"candidates for {method}, got {len(seen)}")
    for scan_id, candidates in by_parent.items():
        if len(candidates) != ALL_VALID_EXPECTED_PER_PARENT:
            raise A3Error(
                f"all-valid source {scan_id} must have {ALL_VALID_EXPECTED_PER_PARENT} "
                f"candidates, got {len(candidates)}")
        variants = sorted(record.generation_seed for record in candidates)
        if len(set(variants)) != len(variants):
            raise A3Error(f"all-valid source {scan_id} has duplicate generation seeds")
        candidates.sort(key=lambda item: item.sample_id)
    _assert_fingerprint(manifest_path, manifest_fingerprint, "all-valid manifest")
    return by_parent


def build_source_occurrence_vector(train_ids: list[str], budget: int, seed: int) -> list[str]:
    """Balanced, deterministic parent exposure list shared by every method at (B, seed)."""
    train_ids = ensure_unique(train_ids, "TRAIN source IDs")
    if len(train_ids) != EXPECTED_TRAIN:
        raise A3Error(f"M must be {EXPECTED_TRAIN}, got {len(train_ids)}")
    if budget <= 0:
        raise A3Error("B must be positive")
    rng = np.random.default_rng(stable_seed("A3-source-occurrences-v1", seed, budget))
    result: list[str] = []
    while len(result) < budget:
        cycle = list(train_ids)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:budget]


def build_selection(
    method: str, budget: int, seed: int, sources: list[SourceRecord],
    original_labels: dict[str, LabelBinding], candidate_manifest: Path | None,
    quality_contract: dict[str, Any], verify_hashes: bool = True,
    *, canonical_scale_manifest: Path, eligibility_rule: str | None = None,
) -> tuple[list[SampleRecord], list[str]]:
    if canonical_scale_manifest is None:
        raise A3Error("The canonical A3 coordinate-scale manifest must be supplied")
    method = method.upper()
    if method not in METHODS:
        raise A3Error(f"method must be one of {sorted(METHODS)}")
    train_ids = [record.scan_id for record in sources]
    vector = build_source_occurrence_vector(train_ids, budget, seed)
    # Unambiguous candidate-loader dispatch driven solely by the validated
    # config's eligibility_rule (never file existence or CLI flags):
    #   * RAW_REPEAT reads no candidates (originals repeated).
    #   * all-valid (all-technically-valid-v1) TRAD/LOADSIM MUST use the
    #     all-valid loader (no Q gate); the Q-v1 loader must never be reached.
    #   * every other (Q-v1) TRAD/LOADSIM MUST use the Q-v1 loader.
    if method == "RAW_REPEAT":
        candidates = {item: [] for item in train_ids}
    elif eligibility_rule == ALL_VALID_ELIGIBILITY_RULE:
        if candidate_manifest is None:
            raise A3Error(f"all-valid {method} requires a candidate bundle")
        # candidate_manifest points at the all-valid bundle's manifest; the
        # loader consumes the bundle directory that contains it.
        candidates = load_all_valid_candidates(
            Path(candidate_manifest).parent, sources, method, verify_hashes)
    elif eligibility_rule is not None:
        raise A3Error(f"Unknown A3 eligibility_rule: {eligibility_rule!r}")
    else:
        candidates = load_candidates(
            candidate_manifest, sources, method, verify_hashes, quality_contract,
            canonical_scale_manifest=canonical_scale_manifest)
    source_by_id = {record.scan_id: record for record in sources}
    needed = Counter(vector)
    if method != "RAW_REPEAT":
        shortages = {parent: (needed[parent], len(candidates[parent])) for parent in train_ids
                     if len(candidates[parent]) < needed[parent]}
        if shortages:
            details = ", ".join(f"{parent}: need {need}, have {have}"
                                for parent, (need, have) in shortages.items())
            raise A3Error(f"Insufficient {method} candidates; no replacement is allowed: {details}")
    cursors = Counter()
    selection: list[SampleRecord] = []
    for parent in vector:
        if method == "RAW_REPEAT":
            source = source_by_id[parent]
            if parent not in original_labels:
                raise A3Error(f"No original pseudo-label entry for {parent}")
            binding = original_labels[parent]
            if not binding.path.is_file():
                raise A3Error(f"Missing original label file: {binding.path}")
            record = SampleRecord(
                sample_id=parent, parent_scan_id=parent, method=method, path=source.path,
                point_count=source.point_count, file_sha256=source.file_sha256,
                label_path=binding.path, label_sha256=binding.label_sha256,
                label_binding=binding,
            )
        else:
            record = candidates[parent][cursors[parent]]
            cursors[parent] += 1
        selection.append(record)
    return selection, vector


def validate_records(records: list[SampleRecord], verify_hashes: bool,
                     cache_size: int = 2) -> tuple[dict[str, int], dict[str, int]]:
    """Validate all point counts/labels and return class counts and point counts."""
    class_counts = {"0": 0, "1": 0, "255": 0}
    point_counts: dict[str, int] = {}
    multiplicities = Counter((record.sample_id, str(record.path), str(record.label_path)) for record in records)
    representative = {(record.sample_id, str(record.path), str(record.label_path)): record for record in records}
    for identity, multiplier in multiplicities.items():
        record = representative[identity]
        declared_ply_sha256 = record.file_sha256 if verify_hashes else None
        points, _, _ = _load_ply_snapshot(
            record.path, declared_ply_sha256, "Preflight PLY")
        if len(points) != record.point_count:
            raise A3Error(f"Point count mismatch for {record.sample_id}: {len(points)} != {record.point_count}")
        labels, _ = _record_label_snapshot(record, len(points))
        point_counts[record.sample_id] = len(points)
        for label in (0, 1, 255):
            class_counts[str(label)] += multiplier * int(np.count_nonzero(labels == label))
    if class_counts["0"] == 0 or class_counts["1"] == 0:
        raise A3Error(f"Both train classes must be represented, got {class_counts}")
    return class_counts, point_counts


def _single_bundle_path(
    rows: list[dict[str, str]], root: Path, field: str, where: str,
) -> Path:
    values = {row.get(field, "") for row in rows}
    if len(values) != 1 or not next(iter(values)):
        raise A3Error(f"{where} must bind exactly one non-empty {field}")
    return resolve_under(root, next(iter(values)))


def load_label_manifest(
    manifest_path: Path, expected: list[SourceRecord], verify_hashes: bool,
    *, canonical_scale_manifest: Path,
) -> dict[str, LabelBinding]:
    """Strictly load and verify an original scale-bound v3 pseudo-label manifest."""
    if canonical_scale_manifest is None:
        raise A3Error("The canonical A3 coordinate-scale manifest must be supplied")
    canonical_scale_manifest = Path(canonical_scale_manifest)
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        legacy = manifest_path.with_name("pseudo_label_manifest.csv")
        if manifest_path.name == "pseudo_label_manifest_v3.csv" and legacy.is_file():
            raise A3Error(
                f"P0-2 blocker: only legacy pseudo-label manifest exists: {legacy}; "
                f"formal A3 requires {PSEUDO_LABEL_MANIFEST_SCHEMA} at {manifest_path}")
        raise A3Error(f"Missing pseudo-label manifest: {manifest_path}")
    try:
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise A3Error(f"Invalid pseudo-label manifest {manifest_path}: {exc}") from exc
    if not rows:
        raise A3Error(f"Empty pseudo-label manifest: {manifest_path}")
    declared = rows[0].get("manifest_schema") or rows[0].get("schema_version") or "unversioned"
    if declared != PSEUDO_LABEL_MANIFEST_SCHEMA:
        legacy_labels = sorted({row.get("label_schema", "") for row in rows if row.get("label_schema")})
        detail = f"; label_schema={legacy_labels}" if legacy_labels else ""
        raise A3Error(
            f"P0-2 blocker: legacy/unsupported pseudo-label manifest {manifest_path} "
            f"declares {declared!r}{detail}; formal A3 requires "
            f"{PSEUDO_LABEL_MANIFEST_SCHEMA} and {PSEUDO_LABEL_ASSET_SCHEMA}")
    if fields != ORIGINAL_LABEL_MANIFEST_FIELDS:
        raise A3Error(
            "Pseudo-label manifest v3 fields mismatch; "
            f"missing={sorted(ORIGINAL_LABEL_FIELDS-set(fields))}, "
            f"unknown={sorted(set(fields)-ORIGINAL_LABEL_FIELDS)}")
    expected_by_id = {record.scan_id: record for record in expected}
    if len(expected_by_id) != len(expected):
        raise A3Error("Expected source identities are not unique")
    root = manifest_path.parent
    report_path = _single_bundle_path(
        rows, root, "pseudo_label_report_path", "Original pseudo-label manifest")
    config_path = _single_bundle_path(
        rows, root, "pseudo_label_config_path", "Original pseudo-label manifest")
    generator_path = _single_bundle_path(
        rows, root, "pseudo_label_generator_path", "Original pseudo-label manifest")
    scale_path = _single_bundle_path(
        rows, root, "coordinate_scale_manifest_path", "Original pseudo-label manifest")
    payload = load_json(report_path)
    labels_by_id = validate_geometric_pseudo_payload(payload, report_path)
    contract = validate_pseudo_label_report_bindings(
        payload, config_path, generator_path, scale_path)
    report_hash = sha256_file(report_path)
    if set(labels_by_id) != set(expected_by_id):
        raise A3Error(
            "Original pseudo-label report IDs must exactly equal frozen TRAIN+DEV IDs; "
            f"missing={sorted(set(expected_by_id)-set(labels_by_id))}, "
            f"extra={sorted(set(labels_by_id)-set(expected_by_id))}")
    source_hashes = {scan_id: record.file_sha256
                     for scan_id, record in expected_by_id.items()}
    scale_rows = load_coordinate_scale_manifest(scale_path, source_hashes)
    canonical_scale_rows = load_coordinate_scale_manifest(
        canonical_scale_manifest, source_hashes)
    if sha256_file(scale_path) != sha256_file(canonical_scale_manifest):
        raise A3Error(
            "Original bundle scale manifest differs from configured canonical file")
    if scale_rows != canonical_scale_rows:
        raise A3Error(
            "Original bundle scale rows differ from configured canonical manifest")
    result: dict[str, LabelBinding] = {}
    provenance_sets = {name: set() for name in (
        "pseudo_label_source_json_sha256", "config_sha256", "generator_sha256",
        "coordinate_scale_manifest_sha256",
    )}
    for line, row in enumerate(rows, start=2):
        sample_id, parent = row["sample_id"], row["parent_scan_id"]
        where = f"Pseudo-label manifest line {line} ({sample_id or '<empty>'})"
        if not sample_id or sample_id not in expected_by_id:
            raise A3Error(f"{where} is outside frozen TRAIN+DEV identities")
        if sample_id in result:
            raise A3Error(f"Duplicate original pseudo-label: {sample_id}")
        source = expected_by_id[sample_id]
        if parent != sample_id:
            raise A3Error(f"Original label parent must equal sample_id: {sample_id}")
        if row["split"] != source.split:
            raise A3Error(f"Pseudo-label split mismatch: {sample_id}")
        if _parse_int(row["point_count"], f"{where} point_count", minimum=1) != source.point_count:
            raise A3Error(f"Pseudo-label point count mismatch: {sample_id}")
        source_hash = _required_sha256(row["source_ply_sha256"],
                                       f"{where} source_ply_sha256")
        if source_hash != source.file_sha256 or row["parent_source_ply_sha256"] != source.file_sha256:
            raise A3Error(f"Pseudo-label source/parent PLY identity mismatch: {sample_id}")
        binding = _label_binding(row, manifest_path.parent, where)
        scale_row = scale_rows[sample_id]
        labels, report_provenance = validate_geometric_label_entry(
            labels_by_id.get(sample_id), sample_id, source.point_count,
            source.file_sha256, scale_row["scale_evidence_sha256"])
        manifest_bindings = {
            "pseudo_label_source_json_sha256": report_hash,
            "config_sha256": contract["config_sha256"],
            "generator_sha256": contract["generator_sha256"],
            "coordinate_scale_manifest_sha256":
                contract["coordinate_scale_manifest_sha256"],
        }
        for name, expected_value in manifest_bindings.items():
            if row[name] != expected_value:
                raise A3Error(
                    f"Original manifest/report {name} binding mismatch: {sample_id}")
        for name in (
                "front_axis_computed", "front_sign_computed",
                "front_computation_source"):
            if report_provenance.get(name) != getattr(binding, name):
                raise A3Error(
                    f"Original computed direction audit differs from manifest: "
                    f"{sample_id}/{name}")
        for name in (
                "coordinate_scale_to_m", "coordinate_scale_source", "scale_semantics",
                "scale_evidence_type", "scale_evidence_path", "scale_evidence_sha256"):
            if report_provenance.get(name) != scale_row[name] or \
                    getattr(binding, name) != scale_row[name]:
                raise A3Error(
                    f"Original scale provenance differs from evidence row: {sample_id}/{name}")
        if not np.array_equal(labels, _label_payload(
                binding.path, sample_id, parent, source.point_count,
                source.file_sha256, binding)):
            raise A3Error(f"Original label NPZ differs from bound report: {sample_id}")
        if binding.parent_source_ply_sha256 != source.file_sha256:
            raise A3Error(f"Pseudo-label parent-source provenance mismatch: {sample_id}")
        if sum(binding.label_counts) != source.point_count:
            raise A3Error(f"Pseudo-label class counts do not sum to point_count: {sample_id}")
        for name, values in provenance_sets.items():
            values.add(row[name])
        result[sample_id] = binding
    missing = sorted(set(expected_by_id) - set(result))
    if missing:
        raise A3Error(f"Pseudo-label manifest misses frozen originals: {missing}")
    for name, values in provenance_sets.items():
        if len(values) != 1:
            raise A3Error(
                f"Pseudo-label manifest rows do not bind one frozen {name}: {sorted(values)}")
    return result


def make_original_records(sources: list[SourceRecord], labels: dict[str, LabelBinding],
                          method: str = "RAW_ORIGINAL") -> list[SampleRecord]:
    result: list[SampleRecord] = []
    for source in sources:
        binding = labels.get(source.scan_id)
        if binding is None or not binding.path.is_file():
            raise A3Error(f"Missing original labels for {source.scan_id}")
        result.append(SampleRecord(
            sample_id=source.scan_id, parent_scan_id=source.scan_id, method=method,
            path=source.path, point_count=source.point_count, file_sha256=source.file_sha256,
            label_path=binding.path, label_sha256=binding.label_sha256,
            label_binding=binding,
        ))
    return result


class CircularBatchSampler:
    """Deterministic full-size batches over an immutable B-length circular vector."""

    def __init__(self, length: int, batch_size: int):
        if length <= 0 or batch_size <= 0:
            raise A3Error("CircularBatchSampler requires positive length and batch_size")
        self.length = length
        self.batch_size = batch_size
        self.batch_count = length // math.gcd(length, batch_size)

    def __iter__(self):
        for batch_index in range(self.batch_count):
            start = batch_index * self.batch_size
            yield [((start + offset) % self.length, start + offset)
                   for offset in range(self.batch_size)]

    def __len__(self) -> int:
        return self.batch_count


class PointCloudDataset:
    """Map-style dataset with stateless deterministic point sampling per exposure."""

    def __init__(self, records: list[SampleRecord], num_points: int, base_seed: int,
                 training: bool, use_normals: bool, cache_size: int = 4):
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise A3Error("PyTorch is required for A3 datasets") from exc
        if not records or num_points < 2 or cache_size < 0:
            raise A3Error("Dataset requires records, num_points >= 2 and cache_size >= 0")
        self.records = records
        self.num_points = num_points
        self.base_seed = base_seed
        self.training = training
        self.use_normals = use_normals
        self.cache_size = cache_size
        self.epoch = 0
        self._cache: OrderedDict[
            tuple[Any, ...],
            tuple[_StatFingerprint, _StatFingerprint,
                  tuple[np.ndarray, np.ndarray, np.ndarray]],
        ] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load(self, record: SampleRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        binding = record.label_binding
        # Exactly one provenance must be present; the label hash must match it.
        # (All-valid records carry all_valid_label_row instead of a v3 binding.)
        if binding is not None:
            if record.label_sha256 != binding.label_sha256:
                raise A3Error(
                    f"Record label hash differs from its v3 manifest binding: "
                    f"{record.label_path}")
            provenance: Any = binding
        elif record.all_valid_label_row is not None:
            provenance = record.all_valid_label_row
        else:
            raise A3Error(
                f"Formal A3 label lacks its v3 manifest binding: {record.label_path}")
        key = (record.sample_id, record.parent_scan_id, record.point_count,
               str(record.path), record.file_sha256,
               str(record.label_path), record.label_sha256, provenance)
        if key in self._cache:
            ply_fingerprint, label_fingerprint, value = self._cache.pop(key)
            _verify_cached_snapshot(
                record.path, record.file_sha256, ply_fingerprint, "Cached PLY")
            _verify_cached_snapshot(
                record.label_path, record.label_sha256,
                label_fingerprint, "Cached label NPZ")
            self._cache[key] = (ply_fingerprint, label_fingerprint, value)
            return value
        points, normals, ply_fingerprint = _load_ply_snapshot(
            record.path, record.file_sha256, "Runtime PLY")
        if len(points) != record.point_count:
            raise A3Error(f"Runtime point count mismatch: {record.sample_id}")
        labels, label_fingerprint = _record_label_snapshot(record, len(points))
        _assert_fingerprint(record.path, ply_fingerprint, "Runtime PLY")
        if normals is None:
            if self.use_normals:
                raise A3Error(f"Normals required but absent: {record.path}")
            normals = np.zeros_like(points)
        value = (points, normals, labels)
        if self.cache_size:
            self._cache[key] = (ply_fingerprint, label_fingerprint, value)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int | tuple[int, int]):
        import torch
        if isinstance(index, tuple):
            record_index, presentation_index = index
        else:
            record_index, presentation_index = index, index
        record = self.records[record_index]
        points, normals, labels = self._load(record)
        occurrence_seed = stable_seed("A3-point-sample-v1", self.base_seed, self.epoch,
                                      presentation_index, record.sample_id, record.parent_scan_id)
        rng = np.random.default_rng(occurrence_seed)
        replace = len(points) < self.num_points
        chosen = rng.choice(len(points), self.num_points, replace=replace)
        xyz = points[chosen].astype(np.float32, copy=True)
        nrm = normals[chosen].astype(np.float32, copy=True)
        target = labels[chosen].astype(np.int64, copy=True)
        center = xyz.mean(0)
        scale = float(np.max(np.abs(xyz - center)))
        if not np.isfinite(scale) or scale <= 0:
            raise A3Error(f"Degenerate sampled cloud: {record.sample_id}")
        xyz = (xyz - center) / scale
        nrm_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(nrm_norm, 1e-8)
        if self.training:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], np.float32)
            xyz = xyz @ rotation.T
            nrm = nrm @ rotation.T
            xyz *= np.float32(rng.uniform(0.9, 1.1))
            jitter = np.clip(rng.normal(0.0, 0.01, xyz.shape), -0.02, 0.02).astype(np.float32)
            xyz += jitter
        features = np.concatenate((xyz, nrm), 1) if self.use_normals else xyz
        return (torch.from_numpy(features.T.copy()), torch.from_numpy(target),
                record.sample_id, record.parent_scan_id)


def source_exposure_rows(vector: list[str], method: str, budget: int, seed: int,
                         max_updates: int, batch_size: int) -> list[dict[str, Any]]:
    """Report one-cycle and actual sample presentations through exactly U updates."""
    if batch_size <= 0 or max_updates <= 0:
        raise A3Error("batch_size and max_updates must be positive")
    per_cycle = Counter(vector)
    repeat_factor = batch_size // math.gcd(len(vector), batch_size)
    full_stream = vector * repeat_factor
    batches = [full_stream[start:start + batch_size]
               for start in range(0, len(full_stream), batch_size)]
    actual = Counter()
    for update in range(max_updates):
        actual.update(batches[update % len(batches)])
    return [{
        "method": method, "budget_B": budget, "seed": seed,
        "max_updates_U": max_updates, "batch_size": batch_size,
        "parent_scan_id": parent,
        "occurrences_in_B_vector": per_cycle[parent],
        "actual_sample_presentations_through_U": actual[parent],
    } for parent in sorted(per_cycle)]
