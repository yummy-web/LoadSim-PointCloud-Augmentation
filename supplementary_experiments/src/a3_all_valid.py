"""all-valid (all-technically-valid-v1) main-analysis bundle: technical contract,
NPZ materialization, receipt issuance, durable publish and fast verification.

Distinct from the Q-v1 pipeline in a3_candidate_bundle.py. Eligibility here is
"every technically valid candidate" (288 = TRAIN-9 x 32); Q/F/P/D and the
historical Q-v1 pass flag are carried report-only and NEVER decide eligibility.

This module does NOT modify the frozen historical producers
(a3_candidate_bundle.py / pseudo_label_replay.py / step1_augmentation.py); it
reuses their public replay functions so their bytes (and pinned SHAs) are
unchanged and can serve as the historical-producer evidence closure.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from a3_io import (
    A3Error, ALL_VALID_ACTIVATION_SCHEMA, ALL_VALID_CATALOG_SCHEMA,
    ALL_VALID_CONTRACT_SCHEMA, ALL_VALID_ELIGIBILITY_RULE,
    ALL_VALID_EVIDENCE_MODES, ALL_VALID_EXPECTED_PER_PARENT,
    ALL_VALID_EXPECTED_TOTAL, ALL_VALID_MANIFEST_FIELDS,
    ALL_VALID_MANIFEST_SCHEMA, ALL_VALID_NPZ_SCHEMA, ALL_VALID_RECEIPT_SCHEMA,
    SCALE_IDENTITY_FIELDS, atomic_save_npz, durable_atomic_write_bytes,
    durable_publish_dir, read_stable_bytes, resolve_under, sha256_file,
    stable_digest_parse, stable_path_fingerprint,
)

_ALLOWED_LABELS = (0, 1, 255)


def _canonical_json_bytes(value: Any) -> bytes:
    """Deterministic, sorted, compact JSON bytes for digesting (no NaN)."""
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise A3Error(f"all-valid payload is not deterministic JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_payload_digest(value: Any) -> str:
    """SHA-256 of the canonical JSON payload (used to avoid receipt hash cycles)."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _float32_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f4").tobytes(order="C")


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n").encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A3Error(message)


def _expected_sample_ids(train_ids: list[str], method: str) -> list[str]:
    method_l = method.lower()
    return [f"{scan}_{method_l}_{variant:03d}"
            for scan in train_ids for variant in range(ALL_VALID_EXPECTED_PER_PARENT)]


def _read_candidate_geometry(ply_path: Path, sample_id: str):
    """Stable-read a candidate PLY and return (xyz, normals, xyz_sha, nrm_sha, count)."""
    from a3_io import read_ply_xyzn  # local import: leaf helper
    stable_path_fingerprint(ply_path, f"candidate PLY {sample_id}")
    before = sha256_file(ply_path)
    xyz, normals = read_ply_xyzn(ply_path)
    if sha256_file(ply_path) != before:
        raise A3Error(f"Candidate PLY changed while read: {sample_id}")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise A3Error(f"Candidate XYZ shape/finite invalid: {sample_id}")
    if normals is None or normals.shape != xyz.shape or not np.isfinite(normals).all():
        raise A3Error(f"Candidate normals shape/finite invalid: {sample_id}")
    xyz_sha = hashlib.sha256(_float32_bytes(xyz)).hexdigest()
    nrm_sha = hashlib.sha256(_float32_bytes(normals)).hexdigest()
    return xyz, normals, xyz_sha, nrm_sha, len(xyz)


def technical_valid_entry(
    *, sample_id: str, method: str, inventory_row: dict[str, str],
    report_entry: dict[str, Any], manifest_root: Path, inventory_root: Path,
    parent_hashes: dict[str, str], inventory_sha256: str, report_sha256: str,
    scales: dict[str, dict[str, Any]], seen_seeds: set[int],
    seen_ply_bytes: set[str], seen_xyz: set[str],
) -> dict[str, Any]:
    """Fail-closed technical-valid contract for ONE candidate; returns manifest-ready fields."""
    provenance = report_entry["provenance"]
    _require("candidate_ply_sha256" in provenance,
             f"all-valid entry must be a candidate (not source): {sample_id}")
    method = method.upper()
    parent = inventory_row["parent_scan_id"]
    _require(inventory_row["method"].upper() == method,
             f"Inventory method mismatch: {sample_id}")
    _require(provenance["parent_scan_id"] == parent,
             f"Provenance/inventory parent mismatch: {sample_id}")
    # identity: sample_id binds parent/method/variant; seed unique
    parts = sample_id.rsplit("_", 2)
    _require(len(parts) == 3 and parts[0] == parent and parts[1] == method.lower(),
             f"sample_id identity malformed: {sample_id}")
    variant_id = int(parts[2])
    _require(0 <= variant_id < ALL_VALID_EXPECTED_PER_PARENT,
             f"variant_id out of range: {sample_id}")
    seed = int(inventory_row["generation_seed"])
    _require(seed not in seen_seeds, f"Duplicate generation seed: {sample_id}")
    seen_seeds.add(seed)
    # resolve assets under roots, reject symlink/escape
    ply = resolve_under(inventory_root, inventory_row["ply_path"])
    metadata = resolve_under(inventory_root, inventory_row["metadata_path"])
    stable_path_fingerprint(ply, f"candidate PLY {sample_id}")
    stable_path_fingerprint(metadata, f"candidate metadata {sample_id}")
    # actual SHAs vs inventory declarations
    actual_ply = sha256_file(ply)
    actual_meta = sha256_file(metadata)
    _require(actual_ply == inventory_row["ply_sha256"],
             f"Candidate PLY SHA differs from inventory: {sample_id}")
    _require(actual_meta == inventory_row["metadata_sha256"],
             f"Candidate metadata SHA differs from inventory: {sample_id}")
    _require(provenance["candidate_ply_sha256"] == actual_ply,
             f"Report candidate_ply_sha256 mismatch: {sample_id}")
    _require(provenance["parent_source_ply_sha256"] == parent_hashes[parent],
             f"Report parent PLY SHA mismatch: {sample_id}")
    # geometry
    xyz, normals, xyz_sha, nrm_sha, count = _read_candidate_geometry(ply, sample_id)
    _require(count == int(inventory_row["point_count"]),
             f"Inventory point_count differs from PLY: {sample_id}")
    _require(count == provenance["point_count"],
             f"Report point_count differs from PLY: {sample_id}")
    # cross-candidate dedup (bytes + canonical geometry)
    _require(actual_ply not in seen_ply_bytes, f"Duplicate candidate PLY bytes: {sample_id}")
    seen_ply_bytes.add(actual_ply)
    _require(xyz_sha not in seen_xyz, f"Duplicate canonical XYZ geometry: {sample_id}")
    seen_xyz.add(xyz_sha)
    # scale identity
    scale_row = scales[parent]
    for key in SCALE_IDENTITY_FIELDS:
        _require(provenance[key] == scale_row[key],
                 f"Scale identity mismatch: {sample_id}/{key}")
    # labels: length/allowed/counts == report entry
    labels = np.asarray(report_entry["labels"], dtype=np.int64)
    _require(labels.ndim == 1 and len(labels) == count,
             f"Label length mismatch: {sample_id}")
    _require(set(np.unique(labels)).issubset(set(_ALLOWED_LABELS)),
             f"Label values outside allowed set: {sample_id}")
    counts = {value: int(np.count_nonzero(labels == value)) for value in _ALLOWED_LABELS}
    for value in _ALLOWED_LABELS:
        _require(counts[value] == report_entry["label_counts"][str(value)],
                 f"Label count differs from report: {sample_id}/{value}")
    return {
        "variant_id": variant_id, "seed": seed, "ply": ply, "metadata": metadata,
        "ply_sha256": actual_ply, "metadata_sha256": actual_meta,
        "canonical_xyz_sha256": xyz_sha, "canonical_normals_sha256": nrm_sha,
        "point_count": count, "labels": labels.astype(np.uint8),
        "label_counts": counts, "provenance": provenance, "scale_row": scale_row,
    }


def _load_inventory(inventory: Path) -> tuple[list[dict[str, str]], str]:
    """Stable-read the merged candidate inventory CSV; return rows + its SHA."""
    import csv
    import io
    from a3_candidate_bundle import INVENTORY_FIELD_ORDER

    def parse(payload: bytes, path: Path) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
        if reader.fieldnames != list(INVENTORY_FIELD_ORDER):
            raise A3Error(f"Candidate inventory field mismatch: {path}")
        rows = list(reader)
        if not rows or any(None in row for row in rows):
            raise A3Error(f"Malformed candidate inventory: {path}")
        return rows

    rows, sha, _ = stable_digest_parse(inventory, None, "candidate inventory", parse)
    return rows, sha


def _load_report(report_path: Path) -> tuple[dict[str, Any], str]:
    """Stable-read a geometric pseudo-label report; return payload + its SHA."""
    def parse(payload: bytes, path: Path) -> dict[str, Any]:
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict) or "labels" not in value or "summary" not in value:
            raise A3Error(f"Not a pseudo-label report: {path}")
        return value

    payload, sha, _ = stable_digest_parse(report_path, None, "pseudo-label report", parse)
    return payload, sha


def _validate_coverage(sample_ids: list[str], train_ids: list[str], method: str) -> None:
    expected = set(_expected_sample_ids(train_ids, method))
    actual = set(sample_ids)
    if actual != expected:
        raise A3Error(
            f"all-valid set must be exactly TRAIN-9x{ALL_VALID_EXPECTED_PER_PARENT}="
            f"{ALL_VALID_EXPECTED_TOTAL}; missing={sorted(expected-actual)[:6]}, "
            f"extra={sorted(actual-expected)[:6]}")
    if len(sample_ids) != ALL_VALID_EXPECTED_TOTAL:
        raise A3Error(
            f"all-valid requires exactly {ALL_VALID_EXPECTED_TOTAL} candidates; "
            f"got {len(sample_ids)}")
    per_parent: dict[str, int] = {scan: 0 for scan in train_ids}
    for sample_id in sample_ids:
        per_parent[sample_id.rsplit("_", 2)[0]] += 1
    short = {scan: n for scan, n in per_parent.items()
             if n != ALL_VALID_EXPECTED_PER_PARENT}
    if short:
        raise A3Error(f"Each parent needs exactly {ALL_VALID_EXPECTED_PER_PARENT}: {short}")


# ---------------------------------------------------------------------------
# Evidence auto-detection (§3.1 migrated vs §3.2 fresh). Never user-forced.
# ---------------------------------------------------------------------------
HISTORICAL_PRODUCER_CLOSURE = {
    "build_a3_candidates.py":
        "88af0902e3aa7996350cbcf7a27b8d1ea0f6c01b0955b32b63aa3dd10d5c57cf",
    "a3_candidate_bundle.py":
        "4dd6e2234147fd35b170ca42128849268e7422d7bc6dd0f2f7a92ef3009e69bd",
    "pseudo_label_replay.py":
        "6b6d941f813c8875965f83c57f5a91f2993bd1ea43f465a965a2aa20f752a6dd",
    "step1_augmentation.py":
        "866dc177ff0c847a68195ee40530de947622d1d860d490c76abe7bf621287d51",
}
_MERGE_EVIDENCE_REQUIRED = (
    "argv", "cwd", "utc_start", "utc_end", "stdout_sha256", "stderr_sha256",
    "exit_code_file_sha256", "producer_closure", "inventory_sha256",
)
_PSEUDO_EVIDENCE_REQUIRED = (
    "argv", "cwd", "utc_start", "utc_end", "stdout_sha256", "stderr_sha256",
    "exit_code_file_sha256", "producer_closure", "report_sha256",
)


def _closure_matches(evidence_dir: Path, declared: dict[str, Any]) -> bool:
    """True only if every historical-producer byte is present externally and matches."""
    if not isinstance(declared, dict) or set(declared) != set(HISTORICAL_PRODUCER_CLOSURE):
        return False
    for name, expected in HISTORICAL_PRODUCER_CLOSURE.items():
        stored = evidence_dir / "producer" / name
        if declared.get(name) != expected or not stored.is_file():
            return False
        if sha256_file(stored) != expected:
            return False
    return True


def detect_evidence_mode(
    evidence_dir: Path | None, inventory_sha256: str, report_sha256: str,
) -> str:
    """Auto-select migrated_atomic_success_v1 iff full structured evidence is present."""
    if evidence_dir is None or not Path(evidence_dir).is_dir():
        return "fresh_deep_replay_v1"
    evidence_dir = Path(evidence_dir)
    merge_desc = evidence_dir / "merge_success.json"
    pseudo_desc = evidence_dir / "pseudo_success.json"
    if not merge_desc.is_file() or not pseudo_desc.is_file():
        return "fresh_deep_replay_v1"
    try:
        merge = json.loads(read_stable_bytes(merge_desc, "merge evidence")[0])
        pseudo = json.loads(read_stable_bytes(pseudo_desc, "pseudo evidence")[0])
    except (A3Error, ValueError):
        return "fresh_deep_replay_v1"
    if any(k not in merge for k in _MERGE_EVIDENCE_REQUIRED):
        return "fresh_deep_replay_v1"
    if any(k not in pseudo for k in _PSEUDO_EVIDENCE_REQUIRED):
        return "fresh_deep_replay_v1"
    if merge.get("exit_code") != 0 or pseudo.get("exit_code") != 0:
        return "fresh_deep_replay_v1"
    if merge.get("inventory_sha256") != inventory_sha256:
        return "fresh_deep_replay_v1"
    if pseudo.get("report_sha256") != report_sha256:
        return "fresh_deep_replay_v1"
    if not _closure_matches(evidence_dir, merge.get("producer_closure", {})):
        return "fresh_deep_replay_v1"
    if not _closure_matches(evidence_dir, pseudo.get("producer_closure", {})):
        return "fresh_deep_replay_v1"
    return "migrated_atomic_success_v1"


# ---------------------------------------------------------------------------
# NPZ materialization (from report entry only; NO geometry recomputation).
# ---------------------------------------------------------------------------
_MATERIALIZER_IMPL_TAG = "a3-all-valid-materializer-v1"


def _entry_digest(sample_id: str, checked: dict[str, Any]) -> str:
    """Stable digest of the per-entry evidence bound into the NPZ and manifest."""
    return canonical_payload_digest({
        "sample_id": sample_id, "point_count": checked["point_count"],
        "canonical_xyz_sha256": checked["canonical_xyz_sha256"],
        "canonical_normals_sha256": checked["canonical_normals_sha256"],
        "ply_sha256": checked["ply_sha256"],
        "metadata_sha256": checked["metadata_sha256"],
        "label_counts": {str(k): checked["label_counts"][k] for k in _ALLOWED_LABELS},
    })


def materialize_label_npz(
    npz_path: Path, *, sample_id: str, checked: dict[str, Any],
    report_sha256: str, entry_digest: str, contract: dict[str, Any],
) -> str:
    """Write one all-valid label NPZ from the validated report entry; return its SHA."""
    provenance = checked["provenance"]
    atomic_save_npz(
        npz_path,
        labels=checked["labels"],
        sample_id=np.array(sample_id),
        schema_version=np.array(ALL_VALID_NPZ_SCHEMA),
        materializer_impl=np.array(_MATERIALIZER_IMPL_TAG),
        eligibility_rule=np.array(ALL_VALID_ELIGIBILITY_RULE),
        source_scan_id=np.array(provenance["parent_scan_id"]),
        candidate_ply_sha256=np.array(checked["ply_sha256"]),
        augmentation_metadata_sha256=np.array(checked["metadata_sha256"]),
        parent_source_ply_sha256=np.array(provenance["parent_source_ply_sha256"]),
        canonical_xyz_sha256=np.array(checked["canonical_xyz_sha256"]),
        canonical_normals_sha256=np.array(checked["canonical_normals_sha256"]),
        pseudo_label_report_sha256=np.array(report_sha256),
        report_entry_digest=np.array(entry_digest),
        config_sha256=np.array(contract["config_sha256"]),
        generator_sha256=np.array(contract["generator_sha256"]),
        coordinate_scale_manifest_sha256=np.array(
            contract["coordinate_scale_manifest_sha256"]),
        coordinate_scale_to_m=np.array(provenance["coordinate_scale_to_m"]),
        scale_evidence_sha256=np.array(provenance["scale_evidence_sha256"]),
        scale_semantics=np.array(provenance["scale_semantics"]),
    )
    return sha256_file(npz_path)


def _metadata_attempt(metadata_path: Path) -> int:
    """Read the frozen candidate metadata JSON and return its generation_attempt."""
    payload, _sha, _fp = stable_digest_parse(
        metadata_path, None, "candidate metadata",
        lambda data, path: json.loads(data.decode("utf-8")))
    value = payload.get("generation_attempt")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise A3Error(f"Candidate metadata generation_attempt invalid: {metadata_path}")
    return value


def _manifest_row(
    *, sample_id: str, method: str, checked: dict[str, Any], staging_root: Path,
    ply_rel: str, label_rel: str, metadata_rel: str, report_rel: str,
    config_rel: str, generator_rel: str, scale_rel: str, quality_rel: str,
    report_sha256: str, inventory_sha256: str, contract: dict[str, Any],
    quality: dict[str, Any], quality_provenance: dict[str, Any],
    quality_report_sha256: str, label_sha256: str,
) -> dict[str, Any]:
    provenance = checked["provenance"]
    scale = checked["scale_row"]
    counts = checked["label_counts"]
    return {
        "schema_version": ALL_VALID_MANIFEST_SCHEMA,
        "manifest_schema": ALL_VALID_MANIFEST_SCHEMA,
        "eligibility_rule": ALL_VALID_ELIGIBILITY_RULE,
        "technical_valid": "true",
        "quality_role": "report_only_not_eligibility",
        "sample_id": sample_id, "parent_scan_id": provenance["parent_scan_id"],
        "method": method, "variant_id": checked["variant_id"],
        "ply_path": ply_rel, "point_count": checked["point_count"],
        "ply_sha256": checked["ply_sha256"],
        "canonical_xyz_sha256": checked["canonical_xyz_sha256"],
        "canonical_normals_sha256": checked["canonical_normals_sha256"],
        "parent_source_ply_sha256": provenance["parent_source_ply_sha256"],
        "label_path": label_rel, "label_sha256": label_sha256,
        "label_schema": ALL_VALID_NPZ_SCHEMA, "metadata_path": metadata_rel,
        "metadata_sha256": checked["metadata_sha256"], "generation_seed": checked["seed"],
        "generation_attempt": _metadata_attempt(checked["metadata"]),
        "label_type": "geometric_pseudo_label",
        "pseudo_label_report_schema": "geometric-pseudo-label-report-v3",
        "pseudo_label_report_path": report_rel,
        "pseudo_label_report_sha256": report_sha256,
        "pseudo_label_config_path": config_rel,
        "pseudo_label_generator_path": generator_rel,
        "coordinate_scale_manifest_path": scale_rel,
        "config_sha256": contract["config_sha256"],
        "generator_sha256": contract["generator_sha256"],
        "coordinate_scale_manifest_sha256": contract["coordinate_scale_manifest_sha256"],
        "candidate_inventory_sha256": inventory_sha256,
        "front_axis_computed": provenance["front_axis_computed"],
        "front_sign_computed": provenance["front_sign_computed"],
        "front_computation_source": provenance["front_computation_source"],
        "coordinate_scale_to_m": scale["coordinate_scale_to_m"],
        "coordinate_scale_source": scale["coordinate_scale_source"],
        "scale_evidence_type": scale["scale_evidence_type"],
        "scale_evidence_path": scale["scale_evidence_path"],
        "scale_evidence_sha256": scale["scale_evidence_sha256"],
        "scale_semantics": scale["scale_semantics"],
        "effective_convex_radius_raw": provenance["effective_convex_radius_raw"],
        "strict_and_applied":
            "true" if provenance["strict_and_applied"] else "false",
        "adaptive_relaxation_applied":
            "true" if provenance["adaptive_relaxation_applied"] else "false",
        "quality_rule": quality_provenance["quality_rule"],
        "quality_threshold_Q": quality_provenance["quality_threshold_Q"],
        "quality_config_sha256": quality_provenance["quality_config_sha256"],
        "quality_report_path": quality_rel,
        "quality_report_sha256": quality_report_sha256,
        "quality_F": quality["F"], "quality_P": quality["P"],
        "quality_D": quality["D"], "quality_Q": quality["Q"],
        "historical_q_v1_passed": "true" if quality["passed"] else "false",
        "label_0": counts[0], "label_1": counts[1], "label_255": counts[255],
    }


# ---------------------------------------------------------------------------
# Deep-replay fallback (§3.2): exactly one geometric replay per ID, call-counted.
# ---------------------------------------------------------------------------
class _CallCounter:
    """Wrap a module function to count invocations for exactly-once proofs."""

    def __init__(self, module: Any, name: str) -> None:
        self.module = module
        self.name = name
        self.original = getattr(module, name)
        self.count = 0

    def __enter__(self) -> "_CallCounter":
        counter = self

        def wrapped(*args: Any, **kwargs: Any):
            counter.count += 1
            return counter.original(*args, **kwargs)

        setattr(self.module, self.name, wrapped)
        return self

    def __exit__(self, *exc: Any) -> None:
        setattr(self.module, self.name, self.original)


def deep_replay_report(
    report_path: Path, inventory: Path, parent_hashes: dict[str, str],
    *, config_path: Path, generator_path: Path, scale_manifest: Path,
) -> tuple[dict[str, Any], int]:
    """Do exactly one geometric deep replay per candidate; return (report, call_count)."""
    import pseudo_label_replay as plr
    samples = plr.candidate_samples(inventory, parent_hashes, scale_manifest)
    with _CallCounter(plr, "strict_and_labels") as counter:
        verified = plr.replay_report(
            report_path, samples, config_path=config_path,
            generator_path=generator_path, scale_manifest=scale_manifest)
    if counter.count != len(samples):
        raise A3Error(
            f"Deep replay must call the geometric generator exactly once per ID; "
            f"expected {len(samples)}, got {counter.count}")
    return verified, counter.count


# ---------------------------------------------------------------------------
# finalize-all-valid orchestrator.
# ---------------------------------------------------------------------------
def finalize_all_valid(
    *, inventory: Path, candidate_report: Path, quality_report: Path,
    quality_config: Path, pseudo_label_config: Path, pseudo_label_generator: Path,
    coordinate_scale_manifest: Path, output_dir: Path, method: str,
    project_root: Path, splits: Path, source_manifest: Path,
    evidence_dir: Path | None = None,
) -> Path:
    """Build, validate and durably publish one all-valid bundle (auto migrated/fresh)."""
    from a3_candidate_bundle import INVENTORY_FIELD_ORDER  # noqa: F401 (contract)
    from a3_data import load_frozen_sources
    from a3_io import load_coordinate_scale_manifest
    from quality_contract import load_quality_contract, load_quality_report
    import pseudo_label_replay as plr

    method = method.upper()
    _require(method in {"TRAD", "LOADSIM"}, "method must be TRAD or LOADSIM")
    output_dir = Path(output_dir)
    _require(not output_dir.exists(),
             f"Refusing to write into existing all-valid output: {output_dir}")
    train_sources, _ = load_frozen_sources(project_root, splits, source_manifest)
    train_ids = [s.scan_id for s in train_sources]
    parent_hashes = {s.scan_id: s.file_sha256 for s in train_sources}
    scales = load_coordinate_scale_manifest(
        coordinate_scale_manifest, parent_hashes, allow_extra=True)

    inventory_rows, inventory_sha = _load_inventory(inventory)
    report_payload, report_sha = _load_report(candidate_report)

    # Evidence auto-detection then the corresponding one-time deep verification.
    mode = detect_evidence_mode(evidence_dir, inventory_sha, report_sha)
    if mode == "fresh_deep_replay_v1":
        verified, _calls = deep_replay_report(
            candidate_report, inventory, parent_hashes,
            config_path=pseudo_label_config, generator_path=pseudo_label_generator,
            scale_manifest=coordinate_scale_manifest)
    else:  # migrated_atomic_success_v1: trust prior atomic success; no fresh geometry.
        plr.validate_geometric_pseudo_payload(report_payload, candidate_report)
        verified = report_payload
    contract = verified["summary"]["algorithm_contract"]

    entries = verified["labels"]
    _validate_coverage(list(entries), train_ids, method)
    inventory_by_id = {row["sample_id"]: row for row in inventory_rows}
    _require(set(inventory_by_id) == set(entries),
             "Inventory IDs must exactly equal report IDs")
    quality_contract = load_quality_contract(quality_config)
    quality_by_id, quality_provenance = load_quality_report(
        quality_report, quality_contract)
    _require(set(quality_by_id) == set(entries),
             "Quality report IDs must exactly equal report IDs")
    quality_report_sha = sha256_file(quality_report)

    return _stage_and_publish_all_valid(
        entries=entries, inventory_by_id=inventory_by_id, method=method,
        train_ids=train_ids, parent_hashes=parent_hashes, scales=scales,
        inventory=inventory, inventory_sha=inventory_sha,
        candidate_report=candidate_report, report_sha=report_sha, contract=contract,
        quality_by_id=quality_by_id, quality_provenance=quality_provenance,
        quality_report=quality_report, quality_report_sha=quality_report_sha,
        quality_config=quality_config, pseudo_label_config=pseudo_label_config,
        pseudo_label_generator=pseudo_label_generator,
        coordinate_scale_manifest=coordinate_scale_manifest,
        output_dir=output_dir, evidence_mode=mode)


def _stage_and_publish_all_valid(
    *, entries: dict[str, Any], inventory_by_id: dict[str, dict[str, str]],
    method: str, train_ids: list[str], parent_hashes: dict[str, str],
    scales: dict[str, dict[str, Any]], inventory: Path, inventory_sha: str,
    candidate_report: Path, report_sha: str, contract: dict[str, Any],
    quality_by_id: dict[str, Any], quality_provenance: dict[str, Any],
    quality_report: Path, quality_report_sha: str, quality_config: Path,
    pseudo_label_config: Path, pseudo_label_generator: Path,
    coordinate_scale_manifest: Path, output_dir: Path, evidence_mode: str,
) -> Path:
    import csv
    import io
    import shutil

    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    if staging.exists():
        raise A3Error(f"all-valid staging already exists: {staging}")
    try:
        (staging / "labels").mkdir(parents=True)
        evidence = staging / "evidence"
        (evidence / "scale").mkdir(parents=True)
        # Freeze the consumed reports/config/generator/scale as evidence copies.
        shutil.copy2(candidate_report, staging / "pseudo_label_report.json")
        shutil.copy2(quality_report, staging / "quality_report.json")
        shutil.copy2(pseudo_label_config, evidence / "pseudo_label_config")
        shutil.copy2(pseudo_label_generator, evidence / "pseudo_label_generator.py")
        shutil.copy2(coordinate_scale_manifest, evidence / "scale" / "coordinate_scales_v2.csv")
        shutil.copy2(inventory, staging / "candidate_inventory.csv")
        if sha256_file(staging / "pseudo_label_report.json") != report_sha:
            raise A3Error("Candidate report changed while frozen")
        if sha256_file(staging / "candidate_inventory.csv") != inventory_sha:
            raise A3Error("Candidate inventory changed while frozen")

        seen_seeds: set[int] = set()
        seen_ply: set[str] = set()
        seen_xyz: set[str] = set()
        rows: list[dict[str, Any]] = []
        leaves: list[dict[str, Any]] = []
        for sample_id in sorted(entries):
            checked = technical_valid_entry(
                sample_id=sample_id, method=method,
                inventory_row=inventory_by_id[sample_id],
                report_entry=entries[sample_id], manifest_root=staging,
                inventory_root=inventory.parent, parent_hashes=parent_hashes,
                inventory_sha256=inventory_sha, report_sha256=report_sha,
                scales=scales, seen_seeds=seen_seeds, seen_ply_bytes=seen_ply,
                seen_xyz=seen_xyz)
            entry_digest = _entry_digest(sample_id, checked)
            # Copy the consumed candidate PLY + metadata into the bundle so it is
            # fully self-contained; verify resolves these under the bundle root.
            ply_rel = inventory_by_id[sample_id]["ply_path"]
            metadata_rel = inventory_by_id[sample_id]["metadata_path"]
            staged_ply = resolve_under(staging, ply_rel)
            staged_meta = resolve_under(staging, metadata_rel)
            staged_ply.parent.mkdir(parents=True, exist_ok=True)
            staged_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checked["ply"], staged_ply)
            shutil.copy2(checked["metadata"], staged_meta)
            if sha256_file(staged_ply) != checked["ply_sha256"]:
                raise A3Error(f"Candidate PLY changed while frozen: {sample_id}")
            if sha256_file(staged_meta) != checked["metadata_sha256"]:
                raise A3Error(f"Candidate metadata changed while frozen: {sample_id}")
            label_path = staging / "labels" / f"{sample_id}.npz"
            label_sha = materialize_label_npz(
                label_path, sample_id=sample_id, checked=checked,
                report_sha256=report_sha, entry_digest=entry_digest, contract=contract)
            row = _manifest_row(
                sample_id=sample_id, method=method, checked=checked,
                staging_root=staging,
                ply_rel=ply_rel,
                label_rel=f"labels/{sample_id}.npz",
                metadata_rel=metadata_rel,
                report_rel="pseudo_label_report.json",
                config_rel="evidence/pseudo_label_config",
                generator_rel="evidence/pseudo_label_generator.py",
                scale_rel="evidence/scale/coordinate_scales_v2.csv",
                quality_rel="quality_report.json", report_sha256=report_sha,
                inventory_sha256=inventory_sha, contract=contract,
                quality=quality_by_id[sample_id],
                quality_provenance=quality_provenance,
                quality_report_sha256=quality_report_sha, label_sha256=label_sha)
            rows.append(row)
            leaves.append({
                "sample_id": sample_id, "generation_seed": checked["seed"],
                "ply_sha256": checked["ply_sha256"],
                "canonical_xyz_sha256": checked["canonical_xyz_sha256"],
                "canonical_normals_sha256": checked["canonical_normals_sha256"],
                "metadata_sha256": checked["metadata_sha256"],
                "label_sha256": label_sha, "report_entry_digest": entry_digest,
                "point_count": checked["point_count"],
                "label_counts": {str(k): checked["label_counts"][k]
                                 for k in _ALLOWED_LABELS}})

        # Write manifest with exact field order.
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(ALL_VALID_MANIFEST_FIELDS),
                                lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        manifest_bytes = stream.getvalue().encode("utf-8")
        manifest_sha = durable_atomic_write_bytes(
            staging / "candidate_manifest.csv", manifest_bytes)
        # Digest the exact CSV string cells (order-preserving) so the receipt binding
        # is stable across a CSV round-trip and free of float-repr ambiguity.
        manifest_payload_digest = canonical_payload_digest(
            [[str(row[field]) for field in ALL_VALID_MANIFEST_FIELDS] for row in rows])

        receipt = _build_receipt(
            method=method, ordered_ids=sorted(entries), train_ids=train_ids,
            leaves=leaves, inventory_sha=inventory_sha, report_sha=report_sha,
            quality_report_sha=quality_report_sha, contract=contract,
            manifest_payload_digest=manifest_payload_digest,
            evidence_mode=evidence_mode, staging=staging,
            source_manifest_sha=None, parent_hashes=parent_hashes)
        durable_atomic_write_bytes(
            staging / "all_valid_receipt.json", _pretty_json_bytes(receipt))
        return durable_publish_dir(staging, output_dir)
    except Exception:
        import shutil as _sh
        _sh.rmtree(staging, ignore_errors=True)
        raise


_ISSUER_VERIFIER_CLOSURE = ("a3_all_valid.py", "a3_io.py")


def _current_closure(staging: Path) -> dict[str, str]:
    """SHA of the current issuer/verifier implementation (distinct from historical)."""
    here = Path(__file__).resolve().parent
    return {name: sha256_file(here / name) for name in _ISSUER_VERIFIER_CLOSURE}


def _build_receipt(
    *, method: str, ordered_ids: list[str], train_ids: list[str],
    leaves: list[dict[str, Any]], inventory_sha: str, report_sha: str,
    quality_report_sha: str, contract: dict[str, Any],
    manifest_payload_digest: str, evidence_mode: str, staging: Path,
    source_manifest_sha: str | None, parent_hashes: dict[str, str],
) -> dict[str, Any]:
    if evidence_mode not in ALL_VALID_EVIDENCE_MODES:
        raise A3Error(f"Unknown evidence mode: {evidence_mode}")
    per_parent = {scan: sum(1 for i in ordered_ids if i.rsplit('_', 2)[0] == scan)
                  for scan in train_ids}
    return {
        "schema_version": ALL_VALID_RECEIPT_SCHEMA,
        "eligibility_rule": ALL_VALID_ELIGIBILITY_RULE,
        "contract_schema": ALL_VALID_CONTRACT_SCHEMA,
        "method": method, "evidence_mode": evidence_mode,
        "expected_total": ALL_VALID_EXPECTED_TOTAL,
        "expected_per_parent": ALL_VALID_EXPECTED_PER_PARENT,
        "ordered_sample_ids": ordered_ids,
        "per_parent_coverage": per_parent,
        "candidate_inventory_sha256": inventory_sha,
        "pseudo_label_report_sha256": report_sha,
        "quality_report_sha256": quality_report_sha,
        "algorithm_contract": contract,
        "parent_source_ply_sha256": parent_hashes,
        "manifest_payload_digest": manifest_payload_digest,
        "historical_producer_closure": dict(HISTORICAL_PRODUCER_CLOSURE),
        "issuer_verifier_closure": _current_closure(staging),
        "quality_role": "report_only_not_eligibility",
        "leaves": leaves,
    }


def verify_all_valid(bundle_dir: Path) -> dict[str, Any]:
    """Fast verify a published bundle: receipt+manifest+asset SHAs; NO geometry replay."""
    import csv
    import io
    import pseudo_label_replay as plr

    bundle_dir = Path(bundle_dir)
    manifest = bundle_dir / "candidate_manifest.csv"
    receipt_path = bundle_dir / "all_valid_receipt.json"
    for path in (manifest, receipt_path):
        if not path.is_file():
            raise A3Error(f"all-valid bundle missing {path.name}: {bundle_dir}")
    receipt = json.loads(read_stable_bytes(receipt_path, "receipt")[0])
    if receipt.get("schema_version") != ALL_VALID_RECEIPT_SCHEMA:
        raise A3Error("Receipt schema is not all-valid v1")
    if receipt.get("eligibility_rule") != ALL_VALID_ELIGIBILITY_RULE:
        raise A3Error("Receipt eligibility rule mismatch")
    if receipt.get("evidence_mode") not in ALL_VALID_EVIDENCE_MODES:
        raise A3Error("Receipt evidence mode invalid")
    if receipt.get("historical_producer_closure") != dict(HISTORICAL_PRODUCER_CLOSURE):
        raise A3Error("Receipt historical-producer closure drifted")
    if receipt.get("issuer_verifier_closure") != _current_closure(bundle_dir):
        raise A3Error("Receipt issuer/verifier closure differs from current implementation")

    def parse(payload: bytes, path: Path) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
        if tuple(reader.fieldnames or ()) != ALL_VALID_MANIFEST_FIELDS:
            raise A3Error(f"all-valid manifest field/order mismatch: {path}")
        return list(reader)

    rows, _msha, _ = stable_digest_parse(manifest, None, "all-valid manifest", parse)
    if len(rows) != ALL_VALID_EXPECTED_TOTAL:
        raise A3Error(f"all-valid manifest must have {ALL_VALID_EXPECTED_TOTAL} rows")
    if [r["sample_id"] for r in rows] != receipt["ordered_sample_ids"]:
        raise A3Error("Manifest IDs/order differ from receipt")
    # Recompute the receipt's manifest_payload_digest from the manifest rows
    # using the exact same representation the issuer hashed: ordered lists of the
    # raw CSV string cells (order-preserving, free of float-repr ambiguity). This
    # proves the manifest⇔receipt binding without the receipt referencing its own
    # or the manifest file's final SHA.
    payload = [[str(r[field]) for field in ALL_VALID_MANIFEST_FIELDS] for r in rows]
    if canonical_payload_digest(payload) != receipt["manifest_payload_digest"]:
        raise A3Error("Manifest payload digest differs from receipt binding")
    leaf_by_id = {leaf["sample_id"]: leaf for leaf in receipt["leaves"]}
    with _CallCounter(plr, "strict_and_labels") as counter:
        _verify_all_valid_leaves(bundle_dir, rows, leaf_by_id)
    if counter.count != 0:
        raise A3Error("Fast verify must not run any geometric replay")
    return receipt


def _verify_all_valid_leaves(
    bundle_dir: Path, rows: list[dict[str, str]], leaf_by_id: dict[str, Any],
) -> None:
    """Recompute actual SHAs of every consumed asset and match manifest + receipt."""
    if set(leaf_by_id) != {r["sample_id"] for r in rows}:
        raise A3Error("Receipt leaves do not cover the manifest IDs exactly")
    seen_xyz: set[str] = set()
    seen_ply: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        leaf = leaf_by_id[sample_id]
        ply = _resolve_leaf_no_symlink(bundle_dir, row["ply_path"], f"PLY {sample_id}")
        label = _resolve_leaf_no_symlink(bundle_dir, row["label_path"], f"label {sample_id}")
        metadata = _resolve_leaf_no_symlink(
            bundle_dir, row["metadata_path"], f"metadata {sample_id}")
        actual_ply = _stable_sha(ply, f"PLY {sample_id}")
        actual_label = _stable_sha(label, f"label {sample_id}")
        actual_meta = _stable_sha(metadata, f"metadata {sample_id}")
        if actual_ply != row["ply_sha256"] or actual_ply != leaf["ply_sha256"]:
            raise A3Error(f"PLY SHA drift on verify: {sample_id}")
        if actual_label != row["label_sha256"] or actual_label != leaf["label_sha256"]:
            raise A3Error(f"Label NPZ SHA drift on verify: {sample_id}")
        if actual_meta != row["metadata_sha256"] or actual_meta != leaf["metadata_sha256"]:
            raise A3Error(f"Metadata SHA drift on verify: {sample_id}")
        if actual_ply in seen_ply:
            raise A3Error(f"Duplicate PLY bytes on verify: {sample_id}")
        seen_ply.add(actual_ply)
        if row["canonical_xyz_sha256"] in seen_xyz:
            raise A3Error(f"Duplicate canonical XYZ on verify: {sample_id}")
        seen_xyz.add(row["canonical_xyz_sha256"])


def _resolve_leaf_no_symlink(bundle_dir: Path, rel: str, asset: str) -> Path:
    """Root-constrained resolve that rejects any symlink component along the path,
    so a leaf cannot be redirected to other bytes after issuance (resolve_under
    would silently follow an in-root symlink)."""
    bundle_dir = Path(bundle_dir).resolve()
    resolved = resolve_under(bundle_dir, rel)  # root-escape / '..' / backslash gate
    current = bundle_dir
    for part in Path(rel).parts:
        current = current / part
        if current.is_symlink():
            raise A3Error(f"{asset} path component must not be a symlink: {rel}")
    return resolved


def _stable_sha(path: Path, asset: str) -> str:
    payload, _fp = read_stable_bytes(path, asset)
    return hashlib.sha256(payload).hexdigest()
