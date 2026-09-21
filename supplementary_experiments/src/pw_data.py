"""Isolated PointWOLF candidate bank loader + dataset (reviewer supplement R2-3).

This module is the PointWOLF analogue of the TRAD/LOADSIM candidate path, kept
DELIBERATELY SEPARATE from the frozen a3_data.py Q-v1 / all-valid schemas (which
carry a 54-field pseudo-label provenance contract PointWOLF cannot satisfy
without re-running the frozen label generator).  Instead PointWOLF uses its own
lightweight, self-describing schema:

  * A frozen candidate bank produced by build_pw_candidate_bank.py:
        pw_assets/pw_bank_v1/
            candidate_manifest.csv        (PW_MANIFEST_FIELDS)
            bank_receipt.json             (build provenance + inventory SHA)
            <parent>/<sample_id>.ply      (warped XYZ + recomputed normals)
            <parent>/<sample_id>.npz      (labels, copied verbatim from parent)
  * Each candidate's labels are BYTE-IDENTICAL to its parent's original
    pseudo-labels (PointWOLF preserves point count and order => Y' = Y).  The
    bank records the parent label SHA so the loader can re-verify equality.

The dataset reuses A3's exact PLY snapshot reader (_load_ply_snapshot) and copies
A3's per-crop transform (normalise -> online rot/scale/jitter) VERBATIM so a
PointWOLF crop is presented to the network identically to a TRAD/LOADSIM/
RAW_REPEAT crop.  The only difference from A3 is the offline geometry source.
"""
from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from a3_data import (
    IGNORE_INDEX, EXPECTED_TRAIN, SampleRecord, SourceRecord,
    _load_ply_snapshot, _read_stable_bytes, _assert_fingerprint,
    build_source_occurrence_vector,
)
from a3_io import A3Error, sha256_file, stable_seed

PW_METHOD = "POINTWOLF"
PW_MANIFEST_SCHEMA = "pw-candidate-manifest-v1"
PW_LABEL_SCHEMA = "pw-label-v1"
PW_EXPECTED_PER_PARENT = 32
PW_EXPECTED_TOTAL = EXPECTED_TRAIN * PW_EXPECTED_PER_PARENT  # 9 x 32 = 288

PW_MANIFEST_FIELDS = (
    "schema_version", "sample_id", "parent_scan_id", "method", "variant_id",
    "ply_path", "point_count", "ply_sha256",
    "label_path", "label_sha256", "label_schema",
    "parent_source_ply_sha256", "parent_label_sha256",
    "generation_seed", "adapter_version", "normal_policy",
    "metadata_path", "metadata_sha256",
    "label_0", "label_1", "label_255",
)

# Fields stored inside each PW label NPZ (string scalars unless noted).
PW_LABEL_NPZ_FIELDS = frozenset({
    "schema_version", "labels", "sample_id", "parent_scan_id",
    "parent_source_ply_sha256", "parent_label_sha256", "candidate_ply_sha256",
})


@dataclass(frozen=True)
class PWCandidate:
    sample_id: str
    parent_scan_id: str
    variant_id: int
    ply_path: Path
    point_count: int
    ply_sha256: str
    label_path: Path
    label_sha256: str
    parent_label_sha256: str
    generation_seed: int


def _scalar_str(archive: Any, key: str, path: Path) -> str:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise A3Error(f"{path}: {key} must be a scalar")
    return str(value.item())


def _pw_label_snapshot(path: Path, row: dict[str, str], expected_count: int) -> np.ndarray:
    """Read + verify one PW label NPZ from a stable snapshot, bound to its row."""
    if path.suffix.lower() != ".npz":
        raise A3Error(f"PW labels must be NPZ files: {path}")
    payload, fingerprint = _read_stable_bytes(path, "PW label NPZ")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != row["label_sha256"]:
        raise A3Error(
            f"PW label NPZ hash differs from manifest: {path}; "
            f"declared={row['label_sha256']}, actual={actual_sha256}")
    string_bindings = {
        "schema_version": PW_LABEL_SCHEMA,
        "sample_id": row["sample_id"],
        "parent_scan_id": row["parent_scan_id"],
        "parent_source_ply_sha256": row["parent_source_ply_sha256"],
        "parent_label_sha256": row["parent_label_sha256"],
        "candidate_ply_sha256": row["ply_sha256"],
    }
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != PW_LABEL_NPZ_FIELDS:
                raise A3Error(
                    f"PW label NPZ field mismatch for {path}: "
                    f"missing={sorted(PW_LABEL_NPZ_FIELDS - set(archive.files))}, "
                    f"unknown={sorted(set(archive.files) - PW_LABEL_NPZ_FIELDS)}")
            if _scalar_str(archive, "schema_version", path) != PW_LABEL_SCHEMA:
                raise A3Error(f"{path}: PW label schema mismatch")
            for key, expected in string_bindings.items():
                if _scalar_str(archive, key, path) != expected:
                    raise A3Error(f"{path}: {key} differs from manifest")
            labels = np.asarray(archive["labels"])
    except (OSError, ValueError) as exc:
        raise A3Error(f"Invalid PW label NPZ {path}: {exc}") from exc
    _assert_fingerprint(path, fingerprint, "PW label NPZ")
    if labels.dtype != np.dtype(np.uint8):
        raise A3Error(f"PW labels must be uint8: {path}")
    if labels.ndim != 1 or len(labels) != expected_count:
        raise A3Error(f"PW label length mismatch for {row['sample_id']}: {labels.shape}")
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1, IGNORE_INDEX}):
        raise A3Error(f"Invalid PW labels {sorted(unique)} in {path}")
    if not np.any(labels != IGNORE_INDEX):
        raise A3Error(f"All PW points are ignored: {path}")
    counts = tuple(int(np.count_nonzero(labels == v)) for v in (0, 1, IGNORE_INDEX))
    manifest_counts = (int(row["label_0"]), int(row["label_1"]), int(row["label_255"]))
    if counts != manifest_counts or sum(counts) != expected_count:
        raise A3Error(f"{path}: PW label counts {counts} differ from manifest {manifest_counts}")
    return labels


def load_pw_candidates(
    bundle_dir: Path, sources: list[SourceRecord], verify_hashes: bool,
) -> dict[str, list[SampleRecord]]:
    """Load the frozen PointWOLF candidate bank; return 32 candidates per TRAIN
    parent keyed by parent (PW_EXPECTED_TOTAL total).  Verifies bank receipt
    inventory SHA and every consumed asset SHA, and re-checks that each
    candidate's labels are byte-identical to its parent's original labels."""
    import json

    train_ids = [s.scan_id for s in sources]
    if len(train_ids) != EXPECTED_TRAIN:
        raise A3Error(f"PW bank requires {EXPECTED_TRAIN} TRAIN sources, got {len(train_ids)}")
    source_hashes = {s.scan_id: s.file_sha256 for s in sources}
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "candidate_manifest.csv"
    receipt_path = bundle_dir / "bank_receipt.json"
    if not manifest_path.is_file():
        raise A3Error(f"PW bank missing candidate_manifest.csv: {bundle_dir}")
    if not receipt_path.is_file():
        raise A3Error(f"PW bank missing bank_receipt.json: {bundle_dir}")

    manifest_payload, manifest_fingerprint = _read_stable_bytes(manifest_path, "PW manifest")
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "pw-bank-receipt-v1":
        raise A3Error("PW bank receipt has wrong schema_version")
    if receipt.get("candidate_manifest_sha256") != manifest_sha:
        raise A3Error("PW bank receipt inventory SHA does not match candidate_manifest.csv")
    if receipt.get("expected_total") != PW_EXPECTED_TOTAL:
        raise A3Error("PW bank receipt expected_total mismatch")

    try:
        text = manifest_payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise A3Error(f"Invalid PW manifest {manifest_path}: {exc}") from exc
    if fields != PW_MANIFEST_FIELDS:
        raise A3Error(
            "PW manifest fields/order mismatch; "
            f"missing={sorted(set(PW_MANIFEST_FIELDS) - set(fields))}, "
            f"unknown={sorted(set(fields) - set(PW_MANIFEST_FIELDS))}")

    by_parent: dict[str, list[SampleRecord]] = {scan_id: [] for scan_id in train_ids}
    seen: set[str] = set()
    for raw in rows:
        if raw.get("schema_version") != PW_MANIFEST_SCHEMA:
            raise A3Error(f"PW row wrong schema: {raw.get('sample_id')}")
        if raw.get("method", "").upper() != PW_METHOD:
            raise A3Error(f"PW row method mismatch: {raw.get('sample_id')}")
        sample_id = raw["sample_id"]
        if sample_id in seen:
            raise A3Error(f"Duplicate PW sample_id: {sample_id}")
        seen.add(sample_id)
        parent = raw["parent_scan_id"]
        if parent not in by_parent:
            raise A3Error(f"PW candidate parent not in TRAIN: {parent}")
        if raw.get("parent_source_ply_sha256") != source_hashes[parent]:
            raise A3Error(f"PW parent-source hash mismatch: {sample_id}")
        variant = int(raw["variant_id"])
        if not 0 <= variant < PW_EXPECTED_PER_PARENT:
            raise A3Error(f"PW variant out of range: {sample_id}")
        point_count = int(raw["point_count"])
        if point_count < 1:
            raise A3Error(f"PW point_count invalid: {sample_id}")
        ply_path = (bundle_dir / raw["ply_path"]).resolve()
        label_path = (bundle_dir / raw["label_path"]).resolve()
        metadata_path = (bundle_dir / raw["metadata_path"]).resolve()
        if verify_hashes:
            points, _, _ = _load_ply_snapshot(ply_path, raw["ply_sha256"], "PW candidate PLY")
            if len(points) != point_count:
                raise A3Error(f"PW point count mismatch: {sample_id}")
            _pw_label_snapshot(label_path, raw, len(points))
            if sha256_file(metadata_path) != raw["metadata_sha256"]:
                raise A3Error(f"PW metadata hash mismatch: {metadata_path}")
        by_parent[parent].append(SampleRecord(
            sample_id=sample_id, parent_scan_id=parent, method=PW_METHOD,
            path=ply_path, point_count=point_count, file_sha256=raw["ply_sha256"],
            label_path=label_path, label_sha256=raw["label_sha256"],
            metadata_path=metadata_path,
            generation_seed=int(raw["generation_seed"]),
            label_binding=None, all_valid_label_row=None,
        ))

    if len(seen) != PW_EXPECTED_TOTAL:
        raise A3Error(f"PW bank must publish {PW_EXPECTED_TOTAL} candidates, got {len(seen)}")
    for scan_id, cands in by_parent.items():
        if len(cands) != PW_EXPECTED_PER_PARENT:
            raise A3Error(
                f"PW source {scan_id} must have {PW_EXPECTED_PER_PARENT}, got {len(cands)}")
        cands.sort(key=lambda item: item.sample_id)
    _assert_fingerprint(manifest_path, manifest_fingerprint, "PW manifest")
    return by_parent


def build_pw_selection(
    budget: int, seed: int, sources: list[SourceRecord], bundle_dir: Path,
    verify_hashes: bool = True,
) -> tuple[list[SampleRecord], list[str]]:
    """PointWOLF matched-budget selection.  Uses the SAME balanced source-
    occurrence vector A3 uses (build_source_occurrence_vector), so per-source
    exposure at (B, seed) is IDENTICAL to A3's RAW_REPEAT/TRAD/LOADSIM."""
    train_ids = [s.scan_id for s in sources]
    vector = build_source_occurrence_vector(train_ids, budget, seed)
    candidates = load_pw_candidates(Path(bundle_dir), sources, verify_hashes)
    needed = Counter(vector)
    shortages = {p: (needed[p], len(candidates[p])) for p in train_ids
                 if len(candidates[p]) < needed[p]}
    if shortages:
        details = ", ".join(f"{p}: need {n}, have {h}" for p, (n, h) in shortages.items())
        raise A3Error(f"Insufficient PW candidates; no replacement allowed: {details}")
    cursors: Counter = Counter()
    selection: list[SampleRecord] = []
    for parent in vector:
        selection.append(candidates[parent][cursors[parent]])
        cursors[parent] += 1
    return selection, vector


# ---------------------------------------------------------------------------
# Dataset: reuses A3's PLY reader + copies A3's per-crop transform VERBATIM.
# ---------------------------------------------------------------------------
_StatFingerprint = tuple[int, int, int, int]


class PWPointCloudDataset:
    """Map-style dataset for PointWOLF candidates.

    The point sampling + per-crop transform (centre/unit-scale normalise, then
    online rotation/anisotropic-scale/jitter in training) is a VERBATIM copy of
    a3_data.PointCloudDataset.__getitem__ so a PointWOLF crop is presented to the
    network byte-for-byte identically to any A3 crop.  Only the label read path
    differs (PW label NPZ instead of the Q-v1/all-valid schemas)."""

    def __init__(self, records: list[SampleRecord], num_points: int, base_seed: int,
                 training: bool, use_normals: bool, cache_size: int = 4,
                 pw_rows: dict[str, dict[str, str]] | None = None):
        import torch  # noqa: F401  (parity with A3 import-time guard)
        if not records or num_points < 2 or cache_size < 0:
            raise A3Error("Dataset requires records, num_points >= 2 and cache_size >= 0")
        self.records = records
        self.num_points = num_points
        self.base_seed = base_seed
        self.training = training
        self.use_normals = use_normals
        self.cache_size = cache_size
        self.epoch = 0
        self.pw_rows = pw_rows or {}
        self._cache: OrderedDict[tuple[Any, ...], tuple[Any, Any, tuple[np.ndarray, np.ndarray, np.ndarray]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load(self, record: SampleRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (record.sample_id, record.parent_scan_id, record.point_count,
               str(record.path), record.file_sha256,
               str(record.label_path), record.label_sha256)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][2]
        points, normals, ply_fingerprint = _load_ply_snapshot(
            record.path, record.file_sha256, "PW runtime PLY")
        if len(points) != record.point_count:
            raise A3Error(f"PW runtime point count mismatch: {record.sample_id}")
        row = self.pw_rows.get(record.sample_id)
        if row is None:
            raise A3Error(f"PW dataset missing manifest row for {record.sample_id}")
        labels = _pw_label_snapshot(record.label_path, row, len(points))
        _assert_fingerprint(record.path, ply_fingerprint, "PW runtime PLY")
        if normals is None:
            if self.use_normals:
                raise A3Error(f"Normals required but absent: {record.path}")
            normals = np.zeros_like(points)
        value = (points, normals, labels)
        if self.cache_size:
            self._cache[key] = (ply_fingerprint, None, value)
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
        # --- identical to a3_data.PointCloudDataset.__getitem__ from here ------
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


def load_pw_manifest_rows(bundle_dir: Path) -> dict[str, dict[str, str]]:
    """Return {sample_id: manifest_row} for the dataset's label re-verification."""
    manifest_path = Path(bundle_dir) / "candidate_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["sample_id"]: row for row in reader}
