"""Canonical deterministic replay for geometry-internal pseudo-label contract v3."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from a3_io import (
    A3Error, ADAPTIVE_RELAXATION_POLICY, COORDINATE_SCALE_POLICY,
    FRONT_COMPUTATION_FIELDS, FRONT_DIRECTION_POLICY, PSEUDO_LABEL_ASSET_SCHEMA,
    PSEUDO_LABEL_CONTRACT_VERSION, PSEUDO_LABEL_MANIFEST_SCHEMA,
    PSEUDO_LABEL_REPORT_SCHEMA, SCALE_IDENTITY_FIELDS, SCALE_SEMANTICS,
    atomic_write_json, load_coordinate_scale_manifest, load_json, read_ply_xyzn,
    resolve_under, sha256_file, validate_geometric_label_entry,
    validate_geometric_pseudo_payload, validate_pseudo_label_report_bindings,
    validate_relaxation_fields, validate_scale_identity_match,
)

GENERATOR_CONTRACT = "pseudo_label_replay.py:strict_and_labels"
EXECUTING_GENERATOR_SHA256 = sha256_file(Path(__file__).resolve())
_DIRECTION_AUDIT_FIELDS = set(FRONT_COMPUTATION_FIELDS)
_SCALE_PROVENANCE_FIELDS = set(SCALE_IDENTITY_FIELDS) | {
    "effective_convex_radius_raw",
}
SOURCE_PROVENANCE_FIELDS = {
    "sample_id", "source_ply_sha256", "point_count", "strict_and_applied",
    "adaptive_relaxation_applied", "relaxation_stage", "mask_combination",
} | _DIRECTION_AUDIT_FIELDS | _SCALE_PROVENANCE_FIELDS
CANDIDATE_PROVENANCE_FIELDS = SOURCE_PROVENANCE_FIELDS | {
    "candidate_ply_sha256", "parent_scan_id", "parent_source_ply_sha256",
    "augmentation_metadata_sha256", "candidate_inventory_sha256",
}


@dataclass(frozen=True)
class ReplaySample:
    sample_id: str
    ply_path: Path
    ply_sha256: str
    point_count: int
    provenance: dict[str, Any]
    metadata_path: Path | None = None
    metadata_sha256: str | None = None
    inventory_path: Path | None = None
    inventory_sha256: str | None = None


def _verify_optional_replay_bindings(sample: ReplaySample) -> None:
    """Recheck candidate-only assets around geometry replay to close path TOCTOU."""
    bindings = (
        (sample.metadata_path, sample.metadata_sha256, "metadata"),
        (sample.inventory_path, sample.inventory_sha256, "inventory"),
    )
    for path, expected, name in bindings:
        if (path is None) != (expected is None):
            raise A3Error(f"Replay {name} binding is incomplete: {sample.sample_id}")
        if path is not None and sha256_file(path) != expected:
            raise A3Error(f"Replay {name} identity changed: {sample.sample_id}")

def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise A3Error("PyYAML is required to read the frozen pseudo-label config") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise A3Error(f"Invalid pseudo-label config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A3Error(f"Pseudo-label config must be an object: {path}")

    def exact(mapping: Any, fields: set[str], where: str) -> dict[str, Any]:
        if not isinstance(mapping, dict) or set(mapping) != fields:
            actual = set(mapping) if isinstance(mapping, dict) else set()
            raise A3Error(
                f"Frozen pseudo-label config {where} fields mismatch; "
                f"missing={sorted(fields-actual)}, unknown={sorted(actual-fields)}")
        return mapping

    top = exact(value, {
        "version", "frozen", "conditions", "combination", "coordinate_system",
        "adaptive_relaxation", "output",
    }, "top-level")
    conditions = exact(top["conditions"], {
        "front_region", "local_slope", "local_convexity",
    }, "conditions")
    front = exact(conditions["front_region"], {
        "method", "axis_policy", "sign_policy", "percentile", "note",
    }, "conditions.front_region")
    slope = exact(conditions["local_slope"], {
        "method", "orientation", "normalization", "angle_definition",
        "angle_range", "note",
    }, "conditions.local_slope")
    convex = exact(conditions["local_convexity"], {
        "method", "radius", "radius_unit", "criterion", "note",
    }, "conditions.local_convexity")
    coordinate = exact(top["coordinate_system"], {
        "up_axis", "scale_policy",
    }, "coordinate_system")
    adaptive = exact(top["adaptive_relaxation"], {
        "enabled", "ratio_min", "fallback_min", "widen_slope_degrees",
        "overcap", "trim_percentile",
    }, "adaptive_relaxation")
    output = exact(top["output"], {"format", "dtype", "labels"}, "output")
    labels = exact(output["labels"], {
        "scooping_region", "background", "ignore",
    }, "output.labels")
    checks = (
        (top["version"] == "v2", "version must be v2"),
        (top["frozen"] is True, "frozen must be true"),
        (front["method"] == "percentile_threshold", "front method mismatch"),
        (front["axis_policy"] == "derived_per_sample_from_frozen_normals",
         "front axis_policy must be derived_per_sample_from_frozen_normals"),
        (front["sign_policy"] == "positive_coordinate", "front sign_policy mismatch"),
        (front["percentile"] == 70, "front percentile must be 70"),
        (slope["method"] == "frozen_ply_normal", "slope method mismatch"),
        (slope["orientation"] == "flip_entire_normal_when_nz_negative",
         "normal orientation mismatch"),
        (slope["normalization"] == "none", "normalization must be none"),
        (slope["angle_definition"] == "90_minus_arccos_abs_nz_degrees",
         "slope angle definition mismatch"),
        (slope["angle_range"] == [20, 40], "slope range must be [20, 40]"),
        (convex["method"] == "spherical_max", "convex method mismatch"),
        (convex["radius"] == 0.5, "convex radius must be 0.5 m"),
        (convex["radius_unit"] == "meter", "convex radius unit mismatch"),
        (convex["criterion"] == "z_max", "convex criterion must be z_max"),
        (top["combination"] == "AND", "combination must be AND"),
        (adaptive["enabled"] is True, "adaptive relaxation must be enabled (plan A)"),
        (adaptive["ratio_min"] == ADAPTIVE_RELAXATION_POLICY["ratio_min"],
         "adaptive ratio_min must be 0.01"),
        (adaptive["fallback_min"] == ADAPTIVE_RELAXATION_POLICY["fallback_min"],
         "adaptive fallback_min must be 0.005"),
        (adaptive["widen_slope_degrees"] ==
         ADAPTIVE_RELAXATION_POLICY["widen_slope_degrees"],
         "adaptive widen_slope_degrees must be [10, 55]"),
        (adaptive["overcap"] == ADAPTIVE_RELAXATION_POLICY["overcap"],
         "adaptive overcap must be 0.40"),
        (adaptive["trim_percentile"] == ADAPTIVE_RELAXATION_POLICY["trim_percentile"],
         "adaptive trim_percentile must be 50"),
        (coordinate["up_axis"] == "z", "up axis must be z"),
        (coordinate["scale_policy"] == "canonical_per_sample_raw_to_meter",
         "coordinate scale policy mismatch"),
        (output["format"] == "npz", "output format must be npz"),
        (output["dtype"] == "uint8", "output dtype must be uint8"),
        (labels == {"scooping_region": 1, "background": 0, "ignore": 255},
         "output label mapping mismatch"),
    )
    failures = [message for passed, message in checks if not passed]
    if failures:
        raise A3Error(f"Frozen pseudo-label config contract mismatch: {failures}")
    return value


def canonical_algorithm_contract(
    config_path: Path, generator_path: Path, scale_manifest: Path,
) -> dict[str, Any]:
    """Return the executable contract bound only to canonical scale evidence."""
    config_before = sha256_file(config_path)
    _load_config(config_path)
    if sha256_file(config_path) != config_before:
        raise A3Error("Pseudo-label config changed while its contract was loaded")
    scale_before = sha256_file(scale_manifest)
    load_coordinate_scale_manifest(scale_manifest)
    if sha256_file(scale_manifest) != scale_before:
        raise A3Error("Canonical scale manifest changed while it was loaded")
    supplied_generator_hash = sha256_file(generator_path)
    if supplied_generator_hash != EXECUTING_GENERATOR_SHA256:
        raise A3Error(
            "Pseudo-label generator bytes are not the executing canonical replay implementation")
    return {
        "version": PSEUDO_LABEL_CONTRACT_VERSION,
        "combination": "strict_AND_with_preregistered_relaxation",
        "adaptive_relaxation": True,
        "adaptive_relaxation_policy": dict(ADAPTIVE_RELAXATION_POLICY),
        "slope_definition": "step3_abs_up_component_degrees",
        "normalize_input_normals": False,
        "front_direction_policy": FRONT_DIRECTION_POLICY,
        "coordinate_scale_policy": COORDINATE_SCALE_POLICY,
        "source_identity_policy": "ply_sha256_and_ordered_point_count",
        "convex_radius_m": 0.5, "front_percentile": 0.70,
        "slope_range_degrees": [20.0, 40.0], "k_normal": 20,
        "generator_entrypoint": GENERATOR_CONTRACT,
        "normal_source": "frozen_ply_normals_required",
        "front_test": (
            "step3_float32_normals_flip_entire_vector_if_nz_negative_then_"
            "axis_y_if_mean_abs_ny_gte_mean_abs_nx_else_x_then_"
            "positive_coordinate_gt_numpy_70th_percentile"),
        "percentile_method": "numpy_percentile_default_on_frozen_ply_float32",
        "config_slope_method_handling": "frozen_ply_normals_required_no_pca_fallback",
        "convexity_test": "z_equals_max_z_in_closed_3d_euclidean_ball",
        "convexity_tolerance": 1e-6,
        "config_sha256": config_before,
        "generator_sha256": supplied_generator_hash,
        "coordinate_scale_manifest_sha256": scale_before,
    }


def _step3_oriented_normals(normals: np.ndarray, sample_id: str) -> np.ndarray:
    """Reproduce step3's float32 copy and whole-vector upward orientation."""
    values = np.asarray(normals, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise A3Error(f"Frozen PLY normals are invalid: {sample_id}")
    values[values[:, 2] < 0.0] *= -1.0
    return values


def _closed_ball_maxima(points: np.ndarray, candidates: np.ndarray,
                         radius: float) -> np.ndarray:
    """Return exact local z maxima using a deterministic spatial hash."""
    points64 = np.asarray(points, dtype=np.float64)
    cells = np.floor(points64 / radius).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        buckets.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(index)
    radius_squared = radius * radius
    result = np.zeros(len(points64), dtype=bool)
    for index in np.flatnonzero(candidates):
        cell = cells[index]
        neighbours: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbours.extend(buckets.get(
                        (int(cell[0] + dx), int(cell[1] + dy), int(cell[2] + dz)), ()))
        neighbour_indices = np.asarray(neighbours, dtype=np.int64)
        delta = points64[neighbour_indices] - points64[index]
        inside = np.einsum("ij,ij->i", delta, delta) <= radius_squared
        local = neighbour_indices[inside]
        result[index] = points64[index, 2] >= np.max(points64[local, 2]) - 1e-6
    return result


def strict_and_labels(
    points: np.ndarray, normals: np.ndarray, coordinate_scale_to_m: float,
    relaxation: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """v3 rule with geometry-internal axis and +1 sign.

    When ``relaxation`` is ``None`` the pure strict three-condition AND is used
    (backward-compatible, no fallbacks). When the frozen configuration supplies
    the pre-registered relaxation policy, the deterministic step3 fallback fires:
    if the strict ratio < ``ratio_min`` the convex condition is dropped, and if
    ``front & slope`` is still < ``fallback_min`` the slope band is widened; if
    the resulting ratio > ``overcap`` only the highest-Z half is retained. Every
    branch is a deterministic function of the frozen PLY, normals and scale.
    """
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise A3Error("Canonical pseudo-label points must be finite N x 3")
    nrm = _step3_oriented_normals(normals, "in-memory sample")
    if len(nrm) != len(xyz):
        raise A3Error("Canonical pseudo-label point/normal count mismatch")
    if (isinstance(coordinate_scale_to_m, bool) or
            not isinstance(coordinate_scale_to_m, (int, float)) or
            not math.isfinite(float(coordinate_scale_to_m)) or
            float(coordinate_scale_to_m) <= 0.0):
        raise A3Error("Canonical coordinate_scale_to_m must be positive and finite")
    axis_index = 1 if np.abs(nrm[:, :2]).mean(0)[1] >= np.abs(
        nrm[:, :2]).mean(0)[0] else 0
    axis = "y" if axis_index == 1 else "x"
    radius_raw = 0.5 / float(coordinate_scale_to_m)
    coordinate = xyz[:, axis_index]
    threshold = float(np.percentile(coordinate, 70.0))
    front = coordinate > threshold
    cos_theta = np.clip(np.abs(nrm[:, 2]), 0, 1)
    angle = 90.0 - np.degrees(np.arccos(cos_theta))
    slope = (angle >= 20.0) & (angle <= 40.0)
    eligible = front & slope
    convex = _closed_ball_maxima(xyz, eligible, radius_raw)
    strict = eligible & convex
    n = len(xyz)
    strict_count = int(np.count_nonzero(strict))

    stage = "none"
    combination = "front_and_slope_and_convex"
    mask = strict
    if relaxation is not None:
        ratio_min = float(relaxation["ratio_min"])
        fallback_min = float(relaxation["fallback_min"])
        low, high = relaxation["widen_slope_degrees"]
        overcap = float(relaxation["overcap"])
        trim_percentile = float(relaxation["trim_percentile"])
        # step3_labels.py:234-243 — widen only when strict is structurally sparse.
        if strict_count / n < ratio_min:
            front_slope = eligible
            if int(np.count_nonzero(front_slope)) / n >= fallback_min:
                mask = front_slope
                stage = "drop_convex"
                combination = "front_and_slope"
            else:
                widened = (angle >= float(low)) & (angle <= float(high))
                mask = front & widened
                stage = "widen_slope"
                combination = "front_and_widened_slope"
        # step3_labels.py:246-250 — trim to the highest-Z half when over-large.
        labels_int = mask.astype(np.int32)
        if int(np.count_nonzero(labels_int)) / n > overcap:
            selected = np.flatnonzero(labels_int == 1)
            trim_threshold = float(np.percentile(xyz[selected, 2], trim_percentile))
            labels_int[selected[xyz[selected, 2] < trim_threshold]] = 0
            mask = labels_int.astype(bool)
            stage = "overcap_trim" if stage == "none" else f"{stage}+overcap_trim"

    labels = mask.astype(np.uint8)
    final_count = int(np.count_nonzero(labels))
    strict_applied = stage == "none"
    detail = {
        "front_axis_computed": axis,
        "front_sign_computed": 1,
        "front_computation_source": "geometry_internal",
        "front_threshold_raw": threshold,
        "effective_convex_radius_raw": radius_raw,
        "relaxation_stage": stage,
        "mask_combination": combination,
        "strict_and_applied": strict_applied,
        "adaptive_relaxation_applied": not strict_applied,
        "condition_counts": {
            "front": int(np.count_nonzero(front)),
            "slope": int(np.count_nonzero(slope)),
            "convex_among_front_and_slope": int(np.count_nonzero(convex)),
            "strict_and": strict_count,
            "final": final_count,
        },
    }
    return labels, detail


def _scale_provenance(scale: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: scale[key] for key in SCALE_IDENTITY_FIELDS},
        "effective_convex_radius_raw": 0.5 / scale["coordinate_scale_to_m"],
    }


def source_samples(
    sources: Iterable[Any], scales: dict[str, dict[str, Any]],
) -> list[ReplaySample]:
    source_list = list(sources)
    source_ids = {source.scan_id for source in source_list}
    if len(source_ids) != len(source_list) or set(scales) != source_ids:
        raise A3Error("Source replay requires one exact canonical scale row per source")
    result: list[ReplaySample] = []
    for source in source_list:
        scale = scales[source.scan_id]
        if scale["source_ply_sha256"] != source.file_sha256:
            raise A3Error(f"Source replay scale binding mismatch: {source.scan_id}")
        provenance = {
            "sample_id": source.scan_id,
            "source_ply_sha256": source.file_sha256,
            "point_count": source.point_count,
            **_scale_provenance(scale),
        }
        result.append(ReplaySample(
            source.scan_id, Path(source.path), source.file_sha256,
            source.point_count, provenance))
    return result


def candidate_samples(
    inventory_path: Path, parent_hashes: dict[str, str], scale_manifest: Path,
) -> list[ReplaySample]:
    """Load candidates while validating identity and canonical scale evidence only."""
    scales = load_coordinate_scale_manifest(
        scale_manifest, parent_hashes, allow_extra=True)
    scale_manifest_hash = sha256_file(scale_manifest)
    try:
        with inventory_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise A3Error(f"Invalid candidate inventory {inventory_path}: {exc}") from exc
    required = {"sample_id", "parent_scan_id", "ply_path", "point_count",
                "ply_sha256", "metadata_path", "metadata_sha256"}
    if not rows or not required.issubset(rows[0]):
        raise A3Error(f"Candidate inventory lacks replay fields: {inventory_path}")
    result: list[ReplaySample] = []
    inventory_hash = sha256_file(inventory_path)
    for row in rows:
        sample_id, parent = row["sample_id"], row["parent_scan_id"]
        if parent not in parent_hashes:
            raise A3Error(f"Candidate parent is outside supplied canonical sources: {sample_id}")
        ply = resolve_under(inventory_path.parent, row["ply_path"])
        metadata_path = resolve_under(inventory_path.parent, row["metadata_path"])
        if sha256_file(ply) != row["ply_sha256"]:
            raise A3Error(f"Candidate PLY hash mismatch: {sample_id}")
        if sha256_file(metadata_path) != row["metadata_sha256"]:
            raise A3Error(f"Candidate metadata hash mismatch: {sample_id}")
        metadata = load_json(metadata_path)
        from a3_candidate_bundle import validate_candidate_augmentation_metadata
        method = str(metadata.get("method", "")).upper()
        validate_candidate_augmentation_metadata(metadata, method, sample_id)
        scale = scales[parent]
        if (metadata.get("sample_id") != sample_id or
                metadata.get("parent_scan_id") != parent or
                metadata.get("parent_ply_sha256") != parent_hashes[parent]):
            raise A3Error(f"Candidate metadata identity mismatch: {sample_id}")
        if metadata.get("coordinate_scale_manifest_sha256") != scale_manifest_hash:
            raise A3Error(f"Candidate metadata canonical scale manifest mismatch: {sample_id}")
        validate_scale_identity_match(
            metadata, scale,
            f"Candidate metadata scale evidence mismatch for {sample_id}")
        try:
            count = int(row["point_count"])
        except (TypeError, ValueError) as exc:
            raise A3Error(f"Candidate point_count is invalid: {sample_id}") from exc
        if count <= 0:
            raise A3Error(f"Candidate point_count is invalid: {sample_id}")
        provenance = {
            "sample_id": sample_id, "source_ply_sha256": row["ply_sha256"],
            "candidate_ply_sha256": row["ply_sha256"], "point_count": count,
            "parent_scan_id": parent,
            "parent_source_ply_sha256": parent_hashes[parent],
            "augmentation_metadata_sha256": row["metadata_sha256"],
            "candidate_inventory_sha256": inventory_hash,
            **_scale_provenance(scale),
        }
        result.append(ReplaySample(
            sample_id, ply, row["ply_sha256"], count, provenance,
            metadata_path=metadata_path, metadata_sha256=row["metadata_sha256"],
            inventory_path=inventory_path, inventory_sha256=inventory_hash))
    if len({sample.sample_id for sample in result}) != len(result):
        raise A3Error("Candidate inventory has duplicate sample_id values")
    if sha256_file(inventory_path) != inventory_hash:
        raise A3Error("Candidate inventory changed while replay inputs were loaded")
    return result


def replay_entries(
    samples: Iterable[ReplaySample], relaxation: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    ordered = sorted(list(samples), key=lambda item: item.sample_id)
    for sample in ordered:
        if sample.sample_id in entries:
            raise A3Error(f"Duplicate replay sample: {sample.sample_id}")
        _verify_optional_replay_bindings(sample)
        if sha256_file(sample.ply_path) != sample.ply_sha256:
            raise A3Error(f"Replay PLY identity changed: {sample.sample_id}")
        points, normals = read_ply_xyzn(sample.ply_path)
        if sha256_file(sample.ply_path) != sample.ply_sha256:
            raise A3Error(f"Replay PLY changed while it was being read: {sample.sample_id}")
        _verify_optional_replay_bindings(sample)
        if len(points) != sample.point_count:
            raise A3Error(f"Replay PLY point count mismatch: {sample.sample_id}")
        if normals is None:
            raise A3Error(f"Canonical v3 replay requires real frozen PLY normals: {sample.sample_id}")
        labels, detail = strict_and_labels(
            points, normals, sample.provenance["coordinate_scale_to_m"], relaxation)
        if detail["effective_convex_radius_raw"] != sample.provenance[
                "effective_convex_radius_raw"]:
            raise A3Error(f"Replay effective radius mismatch: {sample.sample_id}")
        provenance = dict(sample.provenance)
        provenance.update({key: detail[key] for key in _DIRECTION_AUDIT_FIELDS})
        provenance.update({
            "relaxation_stage": detail["relaxation_stage"],
            "mask_combination": detail["mask_combination"],
            "strict_and_applied": detail["strict_and_applied"],
            "adaptive_relaxation_applied": detail["adaptive_relaxation_applied"],
        })
        validate_relaxation_fields(
            provenance["relaxation_stage"], provenance["mask_combination"],
            provenance["strict_and_applied"],
            provenance["adaptive_relaxation_applied"],
            f"Replay provenance for {sample.sample_id}")
        entries[sample.sample_id] = {
            "labels": labels.tolist(),
            "label_counts": {str(value): int(np.count_nonzero(labels == value))
                             for value in (0, 1, 255)},
            "condition_counts": detail["condition_counts"],
            "front_threshold_raw": detail["front_threshold_raw"],
            "provenance": provenance,
        }
    if not entries:
        raise A3Error("Canonical replay requires at least one sample")
    return entries


def _validate_samples_against_scale_manifest(
    samples: list[ReplaySample], scale_manifest: Path,
) -> None:
    """Fail closed unless every replay sample binds an exact canonical scale row."""
    scales = load_coordinate_scale_manifest(scale_manifest)
    keys = SCALE_IDENTITY_FIELDS
    for sample in samples:
        provenance = sample.provenance
        scale_id = provenance.get("parent_scan_id", sample.sample_id)
        if scale_id not in scales:
            raise A3Error(f"Replay sample lacks canonical scale row: {sample.sample_id}")
        scale = scales[scale_id]
        expected_source_hash = provenance.get(
            "parent_source_ply_sha256", provenance.get("source_ply_sha256"))
        if scale["source_ply_sha256"] != expected_source_hash:
            raise A3Error(f"Replay sample scale source binding mismatch: {sample.sample_id}")
        for key in keys:
            if provenance.get(key) != scale[key]:
                raise A3Error(
                    f"Replay sample canonical scale evidence mismatch: {sample.sample_id}/{key}")
        if provenance.get("effective_convex_radius_raw") != 0.5 / scale[
                "coordinate_scale_to_m"]:
            raise A3Error(f"Replay sample effective scale radius mismatch: {sample.sample_id}")


def _relaxation_from_config(config_path: Path) -> dict[str, Any]:
    """Extract the pre-registered relaxation policy from the frozen config."""
    adaptive = _load_config(config_path)["adaptive_relaxation"]
    return {
        "ratio_min": adaptive["ratio_min"],
        "fallback_min": adaptive["fallback_min"],
        "widen_slope_degrees": adaptive["widen_slope_degrees"],
        "overcap": adaptive["overcap"],
        "trim_percentile": adaptive["trim_percentile"],
    }


def build_report(
    samples: Iterable[ReplaySample], *, config_path: Path, generator_path: Path,
    scale_manifest: Path,
) -> dict[str, Any]:
    sample_list = list(samples)
    _validate_samples_against_scale_manifest(sample_list, scale_manifest)
    entries = replay_entries(sample_list, _relaxation_from_config(config_path))
    return {
        "schema_version": PSEUDO_LABEL_REPORT_SCHEMA,
        "summary": {
            "algorithm_contract": canonical_algorithm_contract(
                config_path, generator_path, scale_manifest),
            "sample_count": len(entries),
            "point_count": sum(entry["provenance"]["point_count"]
                               for entry in entries.values()),
            "label_counts": {str(value): sum(entry["label_counts"][str(value)]
                                             for entry in entries.values())
                             for value in (0, 1, 255)},
        },
        "labels": entries,
    }


def _exact_float(actual: Any, expected: Any, where: str) -> None:
    if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or
            float(actual) != float(expected)):
        raise A3Error(f"Replay {where} mismatch: report={actual!r}, replay={expected!r}")


def compare_report(payload: dict[str, Any], expected: dict[str, Any], path: Path) -> None:
    """Compare every replay-relevant array, count, contract and provenance field."""
    validate_geometric_pseudo_payload(payload, path)
    if set(payload) != {"schema_version", "summary", "labels"}:
        raise A3Error("Pseudo-label report top-level fields are not canonical")
    summary, expected_summary = payload["summary"], expected["summary"]
    if not isinstance(summary, dict) or set(summary) != set(expected_summary):
        raise A3Error("Pseudo-label report summary schema differs from canonical replay")
    if summary["algorithm_contract"] != expected_summary["algorithm_contract"]:
        raise A3Error("Pseudo-label report algorithm contract differs from executed replay")
    for key in ("sample_count", "point_count", "label_counts"):
        if summary.get(key) != expected_summary[key]:
            raise A3Error(f"Pseudo-label report summary {key} differs from replay")
    actual_entries, expected_entries = payload["labels"], expected["labels"]
    if set(actual_entries) != set(expected_entries):
        raise A3Error("Pseudo-label report sample IDs differ from replay inputs")
    for sample_id, replayed in expected_entries.items():
        actual = actual_entries[sample_id]
        if not isinstance(actual, dict) or set(actual) != set(replayed):
            raise A3Error(f"Pseudo-label report entry schema mismatch: {sample_id}")
        if not np.array_equal(np.asarray(actual["labels"]),
                              np.asarray(replayed["labels"], dtype=np.uint8)):
            raise A3Error(f"Pseudo-label report labels differ from deterministic replay: {sample_id}")
        for key in ("label_counts", "condition_counts"):
            if actual[key] != replayed[key]:
                raise A3Error(f"Pseudo-label report {key} differs from replay: {sample_id}")
        _exact_float(actual["front_threshold_raw"], replayed["front_threshold_raw"],
                     f"front threshold for {sample_id}")
        expected_fields = (CANDIDATE_PROVENANCE_FIELDS
                           if "candidate_ply_sha256" in replayed["provenance"]
                           else SOURCE_PROVENANCE_FIELDS)
        provenance = actual["provenance"]
        if not isinstance(provenance, dict) or set(provenance) != expected_fields:
            raise A3Error(f"Pseudo-label provenance schema mismatch: {sample_id}")
        if provenance != replayed["provenance"]:
            raise A3Error(f"Pseudo-label provenance differs from replay evidence: {sample_id}")
        validate_geometric_label_entry(
            actual, sample_id, replayed["provenance"]["point_count"],
            replayed["provenance"]["source_ply_sha256"],
            replayed["provenance"]["scale_evidence_sha256"])


def replay_report(
    report_path: Path, samples: Iterable[ReplaySample], *, config_path: Path,
    generator_path: Path, scale_manifest: Path,
) -> dict[str, Any]:
    """Recompute and exactly verify an existing canonical v3 report."""
    report_hash_before = sha256_file(report_path)
    payload = load_json(report_path)
    if sha256_file(report_path) != report_hash_before:
        raise A3Error("Pseudo-label report changed while it was loaded for replay")
    validate_geometric_pseudo_payload(payload, report_path)
    validate_pseudo_label_report_bindings(
        payload, config_path, generator_path, scale_manifest)
    expected = build_report(
        samples, config_path=config_path, generator_path=generator_path,
        scale_manifest=scale_manifest)
    compare_report(payload, expected, report_path)
    if sha256_file(report_path) != report_hash_before:
        raise A3Error("Pseudo-label report changed while it was replayed")
    return expected


def write_report(
    output_path: Path, samples: Iterable[ReplaySample], *, config_path: Path,
    generator_path: Path, scale_manifest: Path,
) -> Path:
    if output_path.exists():
        raise A3Error(f"Refusing to overwrite pseudo-label report: {output_path}")
    report = build_report(
        samples, config_path=config_path, generator_path=generator_path,
        scale_manifest=scale_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.replay-{__import__('os').getpid()}.tmp")
    try:
        if temporary.exists():
            temporary.unlink()
        atomic_write_json(temporary, report)
        replay_report(
            temporary, samples, config_path=config_path, generator_path=generator_path,
            scale_manifest=scale_manifest)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


def _npz_expected_arrays(
    entry: dict[str, Any], contract: dict[str, Any], report_hash: str,
) -> dict[str, Any]:
    provenance = entry["provenance"]
    sample_id = provenance["sample_id"]
    arrays: dict[str, Any] = {
        "labels": np.asarray(entry["labels"], dtype=np.uint8),
        "sample_id": np.array(sample_id),
        "source_scan_id": np.array(provenance.get("parent_scan_id", sample_id)),
        "source_ply_sha256": np.array(provenance["source_ply_sha256"]),
        "schema_version": np.array(PSEUDO_LABEL_ASSET_SCHEMA),
        "pseudo_label_report_sha256": np.array(report_hash),
        "config_sha256": np.array(contract["config_sha256"]),
        "generator_sha256": np.array(contract["generator_sha256"]),
        "coordinate_scale_manifest_sha256": np.array(
            contract["coordinate_scale_manifest_sha256"]),
        "front_axis_computed": np.array(provenance["front_axis_computed"]),
        "front_sign_computed": np.array(provenance["front_sign_computed"], dtype=np.int8),
        "front_computation_source": np.array(provenance["front_computation_source"]),
        "coordinate_scale_to_m": np.array(provenance["coordinate_scale_to_m"]),
        "coordinate_scale_source": np.array(provenance["coordinate_scale_source"]),
        "scale_evidence_type": np.array(provenance["scale_evidence_type"]),
        "scale_evidence_path": np.array(provenance["scale_evidence_path"]),
        "scale_evidence_sha256": np.array(provenance["scale_evidence_sha256"]),
        "scale_semantics": np.array(provenance["scale_semantics"]),
        "effective_convex_radius_raw": np.array(
            provenance["effective_convex_radius_raw"]),
    }
    if "parent_scan_id" in provenance:
        arrays.update({
            "parent_source_ply_sha256": np.array(
                provenance["parent_source_ply_sha256"]),
            "augmentation_metadata_sha256": np.array(
                provenance["augmentation_metadata_sha256"]),
        })
    return arrays


def verify_packaged_manifest(
    manifest_path: Path, samples: Iterable[ReplaySample], *, config_path: Path,
    generator_path: Path, scale_manifest: Path,
) -> None:
    """Replay the report, manifest rows, scale evidence, and NPZ arrays exactly."""
    samples_by_id = {sample.sample_id: sample for sample in samples}
    try:
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise A3Error(f"Invalid packaged pseudo-label manifest {manifest_path}: {exc}") from exc
    row_ids = [row.get("sample_id", "") for row in rows]
    if (not rows or len(row_ids) != len(set(row_ids)) or
            set(row_ids) != set(samples_by_id)):
        raise A3Error("Packaged pseudo-label manifest IDs are invalid for replay inputs")
    forbidden_columns = {
        name for name in (rows[0].keys() if rows else ())
        if (("front_axis" in name and name != "front_axis_computed") or
            ("front_sign" in name and name != "front_sign_computed") or
            "front_vector" in name or "front_direction_manifest" in name or
            "direction_evidence" in name)
    }
    if forbidden_columns:
        raise A3Error(
            f"Packaged manifest contains forbidden direction fields: {sorted(forbidden_columns)}")
    report_paths = {row["pseudo_label_report_path"] for row in rows}
    if len(report_paths) != 1:
        raise A3Error("Packaged manifest must bind exactly one report")
    report_path = resolve_under(manifest_path.parent, report_paths.pop())
    evidence_bindings = {
        "pseudo_label_config_path": config_path,
        "pseudo_label_generator_path": generator_path,
        "coordinate_scale_manifest_path": scale_manifest,
    }
    for field, supplied in evidence_bindings.items():
        values = {row[field] for row in rows}
        if (len(values) != 1 or
                resolve_under(manifest_path.parent, values.pop()) != Path(supplied).resolve()):
            raise A3Error(f"Packaged manifest {field} differs from replay input")
    expected = replay_report(
        report_path, samples_by_id.values(), config_path=config_path,
        generator_path=generator_path, scale_manifest=scale_manifest)
    report_hash = sha256_file(report_path)
    contract = expected["summary"]["algorithm_contract"]
    for row in rows:
        sample_id = row["sample_id"]
        entry = expected["labels"][sample_id]
        provenance = entry["provenance"]
        bindings = {
            "schema_version": PSEUDO_LABEL_MANIFEST_SCHEMA,
            "manifest_schema": PSEUDO_LABEL_MANIFEST_SCHEMA,
            "point_count": str(provenance["point_count"]),
            "parent_scan_id": provenance.get("parent_scan_id", sample_id),
            "parent_source_ply_sha256": provenance.get(
                "parent_source_ply_sha256", provenance["source_ply_sha256"]),
            "pseudo_label_report_schema": PSEUDO_LABEL_REPORT_SCHEMA,
            "pseudo_label_source_json_sha256": report_hash,
            "config_sha256": contract["config_sha256"],
            "generator_sha256": contract["generator_sha256"],
            "coordinate_scale_manifest_sha256": contract[
                "coordinate_scale_manifest_sha256"],
            "front_axis_computed": provenance["front_axis_computed"],
            "front_sign_computed": str(provenance["front_sign_computed"]),
            "front_computation_source": provenance["front_computation_source"],
            "coordinate_scale_to_m": str(provenance["coordinate_scale_to_m"]),
            "coordinate_scale_source": provenance["coordinate_scale_source"],
            "scale_evidence_type": provenance["scale_evidence_type"],
            "scale_evidence_path": provenance["scale_evidence_path"],
            "scale_evidence_sha256": provenance["scale_evidence_sha256"],
            "scale_semantics": provenance["scale_semantics"],
            "effective_convex_radius_raw": str(
                provenance["effective_convex_radius_raw"]),
            "strict_and_applied": "true" if provenance["strict_and_applied"] else "false",
            "adaptive_relaxation_applied":
                "true" if provenance["adaptive_relaxation_applied"] else "false",
            "label_0": str(entry["label_counts"]["0"]),
            "label_1": str(entry["label_counts"]["1"]),
            "label_255": str(entry["label_counts"]["255"]),
        }
        if "source_ply_sha256" in row:
            bindings["source_ply_sha256"] = provenance["source_ply_sha256"]
        if "ply_sha256" in row:
            bindings["ply_sha256"] = provenance["source_ply_sha256"]
            bindings["metadata_sha256"] = provenance["augmentation_metadata_sha256"]
            packaged_ply = resolve_under(manifest_path.parent, row["ply_path"])
            packaged_metadata = resolve_under(manifest_path.parent, row["metadata_path"])
            if (sha256_file(packaged_ply) != provenance["source_ply_sha256"] or
                    sha256_file(packaged_metadata) !=
                    provenance["augmentation_metadata_sha256"]):
                raise A3Error(f"Packaged candidate PLY/metadata hash mismatch: {sample_id}")
            packaged_points, _ = read_ply_xyzn(packaged_ply)
            if len(packaged_points) != provenance["point_count"]:
                raise A3Error(f"Packaged candidate PLY point-count mismatch: {sample_id}")
        for key, value in bindings.items():
            if row.get(key) != value:
                raise A3Error(f"Packaged manifest replay mismatch: {sample_id}/{key}")
        label_path = resolve_under(manifest_path.parent, row["label_path"])
        if sha256_file(label_path) != row["label_sha256"]:
            raise A3Error(f"Packaged label hash mismatch: {sample_id}")
        with np.load(label_path, allow_pickle=False) as archive:
            expected_arrays = _npz_expected_arrays(entry, contract, report_hash)
            if set(archive.files) != set(expected_arrays):
                raise A3Error(f"Packaged label NPZ replay fields differ: {sample_id}")
            for key, value in expected_arrays.items():
                if not np.array_equal(np.asarray(archive[key]), np.asarray(value)):
                    raise A3Error(f"Packaged label NPZ replay mismatch: {sample_id}/{key}")


def _parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source", "candidate"), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-report", type=Path)
    action.add_argument("--verify-report", type=Path)
    parser.add_argument("--verify-packaged-manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=here.parent)
    parser.add_argument("--splits", type=Path, default=here / "manifest/splits_v2.json")
    parser.add_argument("--source-manifest", type=Path,
                        default=here / "manifest/source_scans_v2.csv")
    parser.add_argument("--candidate-inventory", type=Path)
    parser.add_argument("--pseudo-label-config", type=Path, required=True)
    parser.add_argument("--canonical-scale-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from a3_data import load_frozen_sources
    project = args.project_root.resolve()
    train, dev = load_frozen_sources(
        project, args.splits.resolve(), args.source_manifest.resolve())
    sources = train + dev
    source_hashes = {source.scan_id: source.file_sha256 for source in sources}
    scale_manifest = args.canonical_scale_manifest.resolve()
    scales = load_coordinate_scale_manifest(scale_manifest, source_hashes)
    if args.mode == "source":
        samples = source_samples(sources, scales)
    else:
        if args.candidate_inventory is None:
            raise A3Error("Candidate replay requires --candidate-inventory")
        samples = candidate_samples(
            args.candidate_inventory.resolve(), source_hashes, scale_manifest)
    common = {
        "config_path": args.pseudo_label_config.resolve(),
        "generator_path": Path(__file__).resolve(),
        "scale_manifest": scale_manifest,
    }
    report = args.write_report or args.verify_report
    if args.write_report:
        write_report(args.write_report.resolve(), samples, **common)
    else:
        replay_report(args.verify_report.resolve(), samples, **common)
    if args.verify_packaged_manifest:
        verify_packaged_manifest(
            args.verify_packaged_manifest.resolve(), samples, **common)
    print(f"canonical pseudo-label replay PASS: mode={args.mode}, samples={len(samples)}, "
          f"report={report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A3Error as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
