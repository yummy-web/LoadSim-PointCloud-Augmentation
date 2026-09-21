"""V3D budget-curve trainer (Reviewer-3 cross-domain): ISPRS Vaihingen 3D.

Derived from a4b_budget_train.py (which itself descends from a4_train.py). This
is the ADDITIONAL cross-domain proxy experiment: a SECOND proxy domain (ALS
urban scene, ground/non-ground binary, same task shape as KITTI A4/A4B) that
sweeps the real training-block budget B while holding the optimizer update
budget U fixed. It tests whether "augmentation saturates under a matched
budget" reproduces on a domain that is not the main bulk-material scene and not
KITTI driving LiDAR.

The scientific core (sampling, transforms, update-driven training loop, fp32,
matched M/B/U, val-mIoU checkpoint selection, metrics) is reused VERBATIM from
a4b_budget_train.py. Only the data layer differs:
  * "frames" are prepared XY-tiled blocks (.npz) instead of KITTI raw scans;
    I/O goes through v3d_io.py (load_bin/load_label/map_to_binary).
  * the split is manifest-driven: v3d_prepare.py writes manifest.json with
    train/val/test block lists (contiguous X-bands). FORMAL_SPLIT / TRAIN_POOL
    are populated at runtime from that manifest, not hardcoded.
  * train split = first B blocks of the train pool (deterministic prefix,
    so smaller B is a strict subset of larger B); val/test fixed across runs.

Kept from A4B: UPDATE-driven loop until U, update-space CosineAnnealingLR,
forced fp32, no two-factor authorization gate (waived-ceremony extra run).
Scientific integrity (matched M/B/U, honest framing, fixed val/test) is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
KITTI_CODE_DIR = PROJECT_DIR / "kitti_experiment"
if str(KITTI_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(KITTI_CODE_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

METHODS = ("BASELINE", "TRADITIONAL", "LOADSIM", "LOADSIM_GEOM")
# LOADSIM_GEOM reuses the LOADSIM config block but applies a component gate that
# disables directional material removal and lateral collapse (see loadsim
# transform); the yaml records the gate under methods.LOADSIM_GEOM.
LOADSIM_METHODS = ("LOADSIM", "LOADSIM_GEOM")
# V3D split pools are DYNAMIC: populated from the prepared manifest.json at
# runtime (see discover_frames). They start as None and are filled by
# _set_split_globals() before any split selection / manifest hashing runs.
# Layout mirrors the A4B contract: three contiguous half-open index ranges over
# the manifest-ordered block list (train first, then val, then test), so the
# "first B blocks" budget prefix stays deterministic and val/test are byte-fixed.
FORMAL_SPLIT: dict[str, tuple[int, int]] | None = None
TRAIN_POOL: tuple[int, int] | None = None
# Default log-spaced budget grid; the actual max is the manifest train count and
# is validated at runtime (2 <= B <= n_train). Configurable per run.
V3D_BUDGET_GRID = (25, 50, 100, 200, 400)
V3D_EXPERIMENT = "V3D"
CHECKPOINT_SCHEMA = "a4-complete-checkpoint-v1"
CACHE_SCHEMA = "v3d-block-cache-v1-binary-label-mapping"
LABEL_MAPPING_VERSION = "v3d-impervious-surface-v2"
LABEL_MAPPING_SOURCE = "ISPRS Vaihingen 3D 9-class -> binary impervious-surface/other"
GROUND_SEMANTIC_IDS = [2]
IGNORE_SEMANTIC_IDS: list[int] = []
VALID_RAW_SEMANTIC_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
MANIFEST_SCHEMA = "v3d-blocks-v1"
# Method set: the three shared methods plus a component-gated LoadSim variant
# (LOADSIM_GEOM) that DISABLES the components without physical meaning on a
# flat rigid surface (directional material removal + lateral collapse), keeping
# only slope/surface reshaping. This is the R3-6 "disable non-semantic
# components and do a component contrast" requirement, run as its own method.


def _set_split_globals(n_train: int, n_val: int, n_test: int) -> None:
    """Fill FORMAL_SPLIT / TRAIN_POOL from manifest split counts (block order:
    all train blocks, then val, then test -- as written by v3d_prepare.py)."""
    global FORMAL_SPLIT, TRAIN_POOL
    t_end = n_train
    v_end = n_train + n_val
    x_end = n_train + n_val + n_test
    FORMAL_SPLIT = {"train": (0, t_end), "val": (t_end, v_end), "test": (v_end, x_end)}
    TRAIN_POOL = (0, t_end)


class A4Error(RuntimeError):
    """Fail-closed contract violation (name kept for reused helper bodies)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=False,
    ).encode("utf-8") + b"\n")


def atomic_torch_save(path: Path, value: Any) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, tmp_name)
        with open(tmp_name, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def import_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        import yaml
    except ImportError as exc:
        raise A4Error(f"required dependency is unavailable: {exc.name}") from exc
    try:
        from v3d_io import (
            V3D_GROUND_IDS as runtime_ground_ids,
            V3D_IGNORE_IDS as runtime_ignore_ids,
            LABEL_MAPPING_SOURCE as runtime_mapping_source,
            LABEL_MAPPING_VERSION as runtime_mapping_version,
            load_bin, load_label, map_to_binary, traditional_augment,
        )
        # Backbone is domain-agnostic PointNet++ (xyz only); reuse verbatim.
        from kitti_train import PointNetPPKITTI
    except (ImportError, AttributeError) as exc:
        raise A4Error(f"cannot import V3D runtime implementation: {exc}") from exc
    runtime_contract = (
        sorted(runtime_ground_ids), sorted(runtime_ignore_ids),
        runtime_mapping_version, runtime_mapping_source,
    )
    expected_contract = (
        GROUND_SEMANTIC_IDS, IGNORE_SEMANTIC_IDS,
        LABEL_MAPPING_VERSION, LABEL_MAPPING_SOURCE,
    )
    if runtime_contract != expected_contract:
        raise A4Error(
            "runtime V3D label mapping does not match the frozen V3D contract")
    # Probe: ground ids {1,2}->1, every other valid id 0..8 -> 0.
    # Probe: only impervious surfaces (id 2) -> ground(1); every other valid id
    # (incl. low vegetation id 1) -> 0.
    probe_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    expected_probe = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    if not np.array_equal(map_to_binary(probe_ids), expected_probe):
        raise A4Error("runtime V3D map_to_binary violates the frozen V3D contract")
    return torch, nn, DataLoader, Dataset, yaml, (
        load_bin, load_label, map_to_binary, traditional_augment,
        PointNetPPKITTI,
    )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and all Torch devices; request deterministic CUDA."""
    torch, *_ = import_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(deterministic)


def capture_rng_state(torch: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
    if set(state) != expected:
        raise A4Error("checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def require_cuda(torch: Any) -> Any:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise A4Error("CUDA is unavailable; CPU fallback is forbidden")
    try:
        probe = torch.tensor([2.0], device="cuda") * 3.0
        if probe.item() != 6.0:
            raise A4Error("CUDA computation probe returned an invalid result")
    except Exception as exc:
        raise A4Error(f"CUDA computation probe failed: {exc}") from exc
    return torch.device("cuda")


def resolve_relative(base: Path, value: str, field: str) -> Path:
    rel = PurePosixPath(value)
    if rel.is_absolute() or "\\" in value or ".." in rel.parts:
        # Parent paths are part of the frozen run schema and are resolved below,
        # but absolute and Windows-spelled paths remain forbidden.
        if rel.is_absolute() or "\\" in value:
            raise A4Error(f"{field} must be a relative POSIX path")
    return base.joinpath(*rel.parts).resolve()


def validate_a4b_config(cfg: Any, budget_arg: int) -> dict[str, Any]:
    """Self-contained V3D run-config schema (no A3/A4 contract lock).

    val/test block counts are dynamic (they come from the prepared manifest),
    so only the lower bound of budget_B is checked here; the upper bound
    (B <= n_train) is enforced at runtime once the manifest is read.
    """
    if not isinstance(cfg, dict):
        raise A4Error("run config must be a JSON object")
    if cfg.get("experiment") != V3D_EXPERIMENT:
        raise A4Error("v3d_budget_train accepts only experiment=V3D")
    if cfg.get("entrypoint") != "v3d_budget_train.py":
        raise A4Error("entrypoint must be v3d_budget_train.py")
    if cfg.get("method") not in METHODS:
        raise A4Error(f"method must be one of {METHODS}")
    proto = cfg.get("protocol", {})
    budget_B = int(proto.get("budget_B", -1))
    if budget_B < 2:
        raise A4Error(f"protocol.budget_B={budget_B} must be >= 2")
    if budget_arg != budget_B:
        raise A4Error("CLI --budget does not match config.protocol.budget_B")
    for key in ("max_updates_U", "num_points"):
        if key not in proto:
            raise A4Error(f"protocol.{key} is required")
    _expect(proto["num_points"], 4096, "protocol.num_points")
    expected_run_id = f"V3D_{cfg['method']}_B{budget_B}_{int(cfg['seed'])}"
    if cfg.get("run_id") != expected_run_id:
        raise A4Error(f"run_id must be {expected_run_id!r}, got {cfg.get('run_id')!r}")
    return cfg


def load_contract(config_path: Path, method_arg: str, seed_arg: int,
                  run_dir: Path, budget_arg: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            run_cfg = validate_a4b_config(json.load(handle), budget_arg)
    except A4Error:
        raise
    except Exception as exc:
        raise A4Error(f"invalid run config: {exc}") from exc
    if method_arg != run_cfg["method"] or seed_arg != run_cfg["seed"]:
        raise A4Error("CLI method/seed do not match the run config")
    if run_dir.name != run_cfg["run_id"]:
        raise A4Error("run_dir leaf must exactly equal config.run_id")

    _, _, _, _, yaml, _ = import_runtime()
    yaml_path = resolve_relative(BASE_DIR, run_cfg["paths"]["v3d_config"],
                                 "paths.v3d_config")
    if not yaml_path.is_file():
        raise A4Error(f"V3D semantic config is missing: {yaml_path}")
    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            semantic_cfg = yaml.safe_load(handle)
    except Exception as exc:
        raise A4Error(f"cannot parse V3D semantic config: {exc}") from exc
    validate_semantic_contract(run_cfg, semantic_cfg)
    return run_cfg, semantic_cfg, yaml_path


def _expect(value: Any, expected: Any, where: str) -> None:
    if value != expected:
        raise A4Error(f"{where} must be {expected!r}, got {value!r}")


def validate_semantic_contract(run_cfg: dict[str, Any], cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise A4Error("V3D semantic config must be an object")
    _expect(cfg.get("version"), "v3d-cross-domain-v1", "version")
    ds = cfg.get("dataset", {})
    _expect(ds.get("name"), "Vaihingen3D", "dataset.name")
    _expect(ds.get("task"), "ground_nonground_binary", "dataset.task")
    split = ds.get("split", {})
    _expect(split.get("mode"), "spatial_band_x", "dataset.split.mode")
    # V3D split block counts are dynamic (from manifest.json); there are no
    # fixed index bounds to cross-check here. The train budget B is applied at
    # block-selection time and bounded by the runtime train count.
    _expect(ds.get("num_points"), run_cfg["protocol"]["num_points"],
            "dataset.num_points")
    _expect(ds.get("features"), ["x", "y", "z"], "dataset.features")
    _expect(ds.get("label_mapping"), {
        "version": LABEL_MAPPING_VERSION,
        "source": LABEL_MAPPING_SOURCE,
        "output_values": [0, 1, 255],
    }, "dataset.label_mapping")
    _expect(ds.get("ground_semantic_ids"), GROUND_SEMANTIC_IDS,
            "dataset.ground_semantic_ids")
    _expect(ds.get("ignore_semantic_ids"), IGNORE_SEMANTIC_IDS,
            "dataset.ignore_semantic_ids")
    _expect(ds.get("valid_raw_semantic_ids"), VALID_RAW_SEMANTIC_IDS,
            "dataset.valid_raw_semantic_ids")
    _expect(ds.get("non_ground_rule"), "all_other_valid_raw_semantic_ids",
            "dataset.non_ground_rule")
    preprocessing = cfg.get("preprocessing", {})
    _expect(preprocessing.get("reject_zero_xyz"), True,
            "preprocessing.reject_zero_xyz")
    _expect(preprocessing.get("ignore_label_value"), 255,
            "preprocessing.ignore_label_value")
    _expect(preprocessing.get("exclude_ignore_before_sampling"), True,
            "preprocessing.exclude_ignore_before_sampling")
    _expect(preprocessing.get("training_label_values"), [0, 1],
            "preprocessing.training_label_values")
    _expect(run_cfg["model"]["use_normals"], False, "model.use_normals")

    methods = cfg.get("methods", {})
    for name in METHODS:
        if name not in methods:
            raise A4Error(f"methods.{name} is missing")
        _expect(methods[name].get("applies_to"), "train_only",
                f"methods.{name}.applies_to")
        _expect(methods[name].get("online_augmentation"), False,
                f"methods.{name}.online_augmentation")
    _expect(methods["BASELINE"].get("train_transform"), "none",
            "methods.BASELINE.train_transform")
    _expect(methods["TRADITIONAL"].get("train_transform"), "traditional",
            "methods.TRADITIONAL.train_transform")
    loadsim = methods["LOADSIM"]
    _expect(loadsim.get("train_transform"), "loadsim",
            "methods.LOADSIM.train_transform")
    _expect(loadsim.get("class_mask"),
            {"source": "binary_ground_label", "value": 1, "strict": True},
            "methods.LOADSIM.class_mask")
    _expect(loadsim.get("smoothing", {}).get("enabled"), False,
            "methods.LOADSIM.smoothing.enabled")
    # LOADSIM (full) must run every physical component: the on/off contrast is
    # meaningless if the "on" arm silently disabled one. An absent gate == all on.
    ls_gate = loadsim.get("component_gate")
    if ls_gate is not None:
        _expect(ls_gate.get("directional_removal"), True,
                "methods.LOADSIM.component_gate.directional_removal")
        _expect(ls_gate.get("lateral_collapse"), True,
                "methods.LOADSIM.component_gate.lateral_collapse")
        _expect(ls_gate.get("surface_noise"), True,
                "methods.LOADSIM.component_gate.surface_noise")
    # LOADSIM_GEOM = out-of-domain geometric stress test (R3-6): keep only the
    # non-semantic component (surface_noise); disable directional_removal and
    # lateral_collapse, which have no physical meaning on a flat rigid surface.
    geom = methods["LOADSIM_GEOM"]
    _expect(geom.get("train_transform"), "loadsim",
            "methods.LOADSIM_GEOM.train_transform")
    _expect(geom.get("class_mask"),
            {"source": "binary_ground_label", "value": 1, "strict": True},
            "methods.LOADSIM_GEOM.class_mask")
    _expect(geom.get("smoothing", {}).get("enabled"), False,
            "methods.LOADSIM_GEOM.smoothing.enabled")
    geom_gate = geom.get("component_gate")
    if not isinstance(geom_gate, dict):
        raise A4Error("methods.LOADSIM_GEOM.component_gate must be an object")
    _expect(geom_gate.get("directional_removal"), False,
            "methods.LOADSIM_GEOM.component_gate.directional_removal")
    _expect(geom_gate.get("lateral_collapse"), False,
            "methods.LOADSIM_GEOM.component_gate.lateral_collapse")
    _expect(geom_gate.get("surface_noise"), True,
            "methods.LOADSIM_GEOM.component_gate.surface_noise")
    evaluation = cfg.get("evaluation", {})
    _expect(evaluation.get("validation_source"), "raw_frames",
            "evaluation.validation_source")
    _expect(evaluation.get("test_source"), "raw_frames",
            "evaluation.test_source")
    _expect(evaluation.get("augmentation"), "none", "evaluation.augmentation")


@dataclass(frozen=True)
class FrameRef:
    index: int
    stem: str
    bin_path: Path
    label_path: Path


def locate_sequence_dir(v3d_root: Path) -> Path:
    """Return the prepared-blocks root (must hold manifest.json + blocks/)."""
    candidates = [v3d_root, v3d_root / "blocks_root"]
    for candidate in candidates:
        if (candidate / "manifest.json").is_file() and (candidate / "blocks").is_dir():
            return candidate.resolve()
    raise A4Error(
        "prepared V3D blocks (manifest.json + blocks/) are missing under "
        f"{v3d_root}; run v3d_prepare.py first"
    )


def _read_manifest(seq_dir: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((seq_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise A4Error(f"cannot read V3D manifest.json: {exc}") from exc
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise A4Error(f"V3D manifest schema mismatch: {manifest.get('schema')!r}")
    return manifest


def discover_frames(seq_dir: Path) -> list[FrameRef]:
    """Build the manifest-ordered block list and set FORMAL_SPLIT/TRAIN_POOL.

    v3d_prepare.py sorts blocks (train, then val, then test) then by frame_id,
    so the on-disk order defines three contiguous index ranges. We honor that
    order exactly and derive the split bounds from the per-split counts.
    """
    manifest = _read_manifest(seq_dir)
    blocks = manifest.get("blocks", [])
    if not blocks:
        raise A4Error("V3D manifest contains no blocks")
    order = {"train": 0, "val": 1, "test": 2}
    ordered = sorted(blocks, key=lambda m: (order[m["split"]], m["frame_id"]))
    counts = {"train": 0, "val": 0, "test": 0}
    frames: list[FrameRef] = []
    for index, blk in enumerate(ordered):
        split = blk["split"]
        if split not in counts:
            raise A4Error(f"unknown split in manifest block: {split!r}")
        counts[split] += 1
        npz_path = (seq_dir / blk["npz"]).resolve()
        if not npz_path.is_file():
            raise A4Error(f"missing V3D block file: {npz_path}")
        if npz_path.stat().st_size == 0:
            raise A4Error(f"empty V3D block file: {npz_path}")
        # bin and label are the same .npz (xyz + sem read separately).
        frames.append(FrameRef(index, blk["frame_id"], npz_path, npz_path))
    if counts["train"] < 2 or counts["val"] < 1 or counts["test"] < 1:
        raise A4Error(f"insufficient blocks per split: {counts}")
    # Enforce that manifest order is grouped (train block indices are a prefix).
    seen_val = seen_test = False
    for index, blk in enumerate(ordered):
        if blk["split"] == "val":
            seen_val = True
        elif blk["split"] == "test":
            seen_test = True
        elif blk["split"] == "train" and (seen_val or seen_test):
            raise A4Error("manifest block order is not grouped train|val|test")
    _set_split_globals(counts["train"], counts["val"], counts["test"])
    return frames


def derive_seed(seed: int, purpose: str, frame_index: int) -> int:
    payload = f"{seed}|{purpose}|{frame_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def sample_raw_frame(frame: FrameRef, n_points: int, sample_seed: int,
                     load_bin: Any, load_label: Any,
                     map_to_binary: Any) -> tuple[np.ndarray, np.ndarray]:
    """Load a raw frame, discard mapped ignore=255 rows, then sample 0/1 labels."""
    points4 = load_bin(frame.bin_path)
    semantic = load_label(frame.label_path)
    if points4.ndim != 2 or points4.shape[1] != 4:
        raise A4Error(f"invalid point shape in frame {frame.stem}: {points4.shape}")
    if len(points4) != len(semantic):
        raise A4Error(f"point/label count mismatch in frame {frame.stem}")
    unknown = sorted(set(np.unique(semantic).tolist()) - set(VALID_RAW_SEMANTIC_IDS))
    if unknown:
        raise A4Error(f"frame {frame.stem} contains unknown raw semantic IDs: {unknown}")
    valid = np.isfinite(points4[:, :3]).all(axis=1)
    valid &= np.any(points4[:, :3] != 0.0, axis=1)
    mapped = map_to_binary(semantic).astype(np.int64, copy=False)
    if mapped.shape != semantic.shape or not np.isin(mapped, [0, 1, 255]).all():
        raise A4Error(f"frame {frame.stem} produced an invalid raw-ID mapping")
    valid &= mapped != 255
    xyz = points4[valid, :3].astype(np.float32, copy=True)
    labels = mapped[valid]
    if len(xyz) < 100:
        raise A4Error(
            f"frame {frame.stem} has fewer than 100 valid non-ignore points")
    if not np.isin(labels, [0, 1]).all():
        raise A4Error(f"frame {frame.stem} produced non-binary training labels")

    rng = np.random.default_rng(sample_seed)
    replace = len(xyz) < n_points
    indices = rng.choice(len(xyz), n_points, replace=replace)
    min_ground = min(int(n_points * 0.10), int((labels == 1).sum()))
    selected_ground = int((labels[indices] == 1).sum())
    if not replace and selected_ground < min_ground:
        selected_ground_positions = labels[indices] == 1
        used = np.zeros(len(xyz), dtype=bool)
        used[indices] = True
        available_ground = np.flatnonzero((labels == 1) & ~used)
        swap_positions = np.flatnonzero(~selected_ground_positions)
        need = min(min_ground - selected_ground, len(available_ground),
                   len(swap_positions))
        if need:
            indices[swap_positions[:need]] = rng.choice(
                available_ground, need, replace=False)
    sampled_xyz = xyz[indices]
    sampled_labels = labels[indices]
    if not np.isin(sampled_labels, [0, 1]).all():
        raise A4Error(f"frame {frame.stem} sampled labels outside {{0, 1}}")
    sampled_xyz -= sampled_xyz.mean(axis=0, keepdims=True)
    return sampled_xyz.astype(np.float32), sampled_labels.astype(np.int64)


def traditional_transform(xyz: np.ndarray, labels: np.ndarray, cfg: dict[str, Any],
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    out = xyz.copy()
    out *= rng.uniform(*map(float, cfg["scale_range"]))
    theta = np.deg2rad(rng.uniform(*map(float, cfg["rotate_z_degrees"])))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float32)
    out = out @ rotation.T
    noise = rng.normal(0.0, float(cfg["jitter_sigma"]), out.shape)
    out += np.clip(noise, -float(cfg["jitter_clip"]),
                   float(cfg["jitter_clip"])).astype(np.float32)
    flipped = bool(rng.random() < float(cfg["flip_x_probability"]))
    if flipped:
        out[:, 0] *= -1.0
    return out.astype(np.float32), labels.copy(), {
        "transform": "traditional", "flipped_x": flipped,
    }


def loadsim_transform(xyz: np.ndarray, labels: np.ndarray, cfg: dict[str, Any],
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the enabled physical components under labels == 1, then restore N.

    cfg["component_gate"] (optional) toggles individual components:
      directional_removal, lateral_collapse, surface_noise (default all True).
    On a flat rigid surface (V3D impervious surfaces) directional material
    removal and lateral collapse have no physical meaning, so the LOADSIM_GEOM
    contrast method disables them and keeps only surface_noise (R3-6). The RNG
    draw sequence is preserved regardless of the gate so a disabled component
    does not change the stream consumed by the enabled ones.
    """
    if not np.isin(labels, [0, 1]).all():
        raise A4Error("LoadSim received non-binary labels")
    gate = cfg.get("component_gate", {}) or {}
    do_removal = bool(gate.get("directional_removal", True))
    do_collapse = bool(gate.get("lateral_collapse", True))
    points = xyz.copy()
    current_labels = labels.copy()
    original_non_ground = points[current_labels == 0].copy()
    removed_ground = 0
    collapsed_ground = 0
    n_ops = int(rng.integers(int(cfg["n_ops"][0]), int(cfg["n_ops"][1]) + 1))
    for _ in range(n_ops):
        ground_idx = np.flatnonzero(current_labels == 1)
        if len(ground_idx) < 10:
            break
        ground = points[ground_idx]
        centre = ground.mean(axis=0)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        side = np.array([-np.sin(angle), np.cos(angle), 0.0])
        projection = (ground - centre) @ direction
        span = float(projection.max() - projection.min())
        if span <= 0.0:
            continue
        alpha = rng.uniform(*map(float, cfg["front_alpha"]))
        threshold = float(projection.max() - alpha * span)
        front = ground_idx[projection > threshold]
        if len(front) < 5:
            continue
        core = points[int(rng.choice(front))]
        width = rng.uniform(*map(float, cfg["bucket_width_m"]))
        projection_all = (points - centre) @ direction
        side_distance = np.abs((points - core) @ side)
        ground_mask = current_labels == 1
        excavate = ground_mask & (side_distance < width / 2.0) & (
            projection_all > threshold)
        take_idx = np.flatnonzero(excavate)
        collapse = ground_mask & (side_distance >= width / 2.0) & (
            side_distance < width * 1.2) & (projection_all > threshold)
        collapse_idx = np.flatnonzero(collapse)
        if len(collapse_idx):
            # Draw the collapse magnitudes regardless of the gate to keep the
            # RNG stream identical; only apply the displacement when enabled.
            dz = rng.uniform(*map(float, cfg["collapse_dz_m"]))
            dx = rng.uniform(*map(float, cfg["collapse_dx_m"]))
            if do_collapse:
                distance = side_distance[collapse_idx] - width / 2.0
                factor = np.clip(1.0 - distance / (width * 0.2 + 1e-8), 0.0, 1.0)
                points[collapse_idx, 2] -= dz * factor
                toward = core - points[collapse_idx]
                toward[:, 2] = 0.0
                toward /= np.linalg.norm(toward, axis=1, keepdims=True) + 1e-8
                points[collapse_idx] += toward * (dx * factor)[:, None]
                collapsed_ground += len(collapse_idx)
        if len(take_idx) and do_removal:
            removed_ground += len(take_idx)
            keep = np.ones(len(points), dtype=bool)
            keep[take_idx] = False
            points = points[keep]
            current_labels = current_labels[keep]

    noise_cfg = cfg["surface_noise"]
    do_noise = bool(gate.get("surface_noise", True))
    applied_noise = False
    if noise_cfg.get("enabled", False) and do_noise:
        ground_idx = np.flatnonzero(current_labels == 1)
        sigma = float(noise_cfg["sigma_m"])
        clip = float(noise_cfg["clip_m"])
        points[ground_idx, 2] += np.clip(
            rng.normal(0.0, sigma, len(ground_idx)), -clip, clip)
        applied_noise = len(ground_idx) > 0
    if not np.array_equal(points[current_labels == 0], original_non_ground):
        raise A4Error("LoadSim ground-only invariant failed: non-ground was changed")
    if not np.isfinite(points).all() or len(points) == 0:
        raise A4Error("LoadSim produced invalid points")

    # Preserve every non-ground row exactly once. Restore only the ground
    # quota by sampling transformed ground points; fixed-size resampling must
    # not become an unmasked operation on the non-ground class.
    non_ground = points[current_labels == 0]
    ground = points[current_labels == 1]
    target_ground = len(xyz) - len(non_ground)
    if target_ground < 0 or (target_ground > 0 and len(ground) == 0):
        raise A4Error("LoadSim cannot restore the fixed ground-only block")
    if target_ground:
        ground_choice = rng.choice(
            len(ground), target_ground, replace=(len(ground) < target_ground))
        ground_out = ground[ground_choice]
    else:
        ground_out = np.empty((0, 3), dtype=np.float32)
    out_points = np.concatenate([non_ground, ground_out], axis=0)
    out_labels = np.concatenate([
        np.zeros(len(non_ground), dtype=labels.dtype),
        np.ones(len(ground_out), dtype=labels.dtype),
    ])
    if not np.array_equal(out_points[:len(non_ground)], original_non_ground):
        raise A4Error("LoadSim output changed a non-ground row")
    return out_points.astype(np.float32), out_labels, {
        "transform": "loadsim", "ground_only_mask": True,
        "n_ops": n_ops, "removed_ground": int(removed_ground),
        "collapsed_ground": int(collapsed_ground),
        "surface_noise_applied": bool(applied_noise),
        "component_gate": {
            "directional_removal": do_removal,
            "lateral_collapse": do_collapse,
            "surface_noise": do_noise,
        },
        "non_ground_unchanged": True,
        "non_ground_preserved_once": True,
    }


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def cache_one_frame(frame: FrameRef, split: str, method: str, seed: int,
                    n_points: int, semantic_cfg: dict[str, Any],
                    cache_fingerprint: str, runtime_symbols: tuple[Any, ...],
                    path: Path) -> dict[str, Any]:
    load_bin, load_label, map_to_binary, _, _ = runtime_symbols
    if split == "train":
        sample_seed = derive_seed(seed, "raw-train-sampling", frame.index)
    else:
        eval_seed = int(semantic_cfg["preprocessing"]["eval_sampling_seed"])
        sample_seed = derive_seed(eval_seed, f"raw-{split}-sampling", frame.index)
    xyz, labels = sample_raw_frame(
        frame, n_points, sample_seed, load_bin, load_label, map_to_binary)
    audit: dict[str, Any] = {"transform": "none", "raw_original": True}
    if split == "train" and method == "TRADITIONAL":
        aug_seed = derive_seed(seed, "traditional", frame.index)
        xyz, labels, audit = traditional_transform(
            xyz, labels, semantic_cfg["methods"][method],
            np.random.default_rng(aug_seed))
        audit["raw_original"] = False
    elif split == "train" and method in LOADSIM_METHODS:
        # LOADSIM and LOADSIM_GEOM share the transform; the per-method config
        # block carries its own component_gate (LOADSIM = all on, LOADSIM_GEOM =
        # only surface_noise). Derive the seed from the method name so the two
        # arms do not share an identical RNG stream by accident.
        aug_seed = derive_seed(seed, method.lower(), frame.index)
        xyz, labels, audit = loadsim_transform(
            xyz, labels, semantic_cfg["methods"][method],
            np.random.default_rng(aug_seed))
        audit["raw_original"] = False
    elif split == "train" and method != "BASELINE":
        raise A4Error(f"unsupported method: {method}")
    if split != "train" and audit["transform"] != "none":
        raise A4Error(f"{split} must remain unaugmented")
    array_sha256 = hashlib.sha256(
        xyz.astype(np.float32).tobytes(order="C") +
        labels.astype(np.int64).tobytes(order="C")
    ).hexdigest()
    metadata = {
        "schema": CACHE_SCHEMA,
        "fingerprint": cache_fingerprint,
        "split": split,
        "method": method if split == "train" else "SHARED_RAW_EVAL",
        "frame_index": frame.index,
        "frame_stem": frame.stem,
        "sample_seed": sample_seed,
        "n_points": n_points,
        "label_mapping": {
            "version": LABEL_MAPPING_VERSION,
            "source": LABEL_MAPPING_SOURCE,
            "ground_semantic_ids": GROUND_SEMANTIC_IDS,
            "ignore_semantic_ids": IGNORE_SEMANTIC_IDS,
        },
        "array_sha256": array_sha256,
        "audit": audit,
    }
    atomic_save_npz(path, xyz=xyz.astype(np.float32),
                    labels=labels.astype(np.int64),
                    metadata=np.array(json.dumps(metadata, sort_keys=True)))
    return metadata


def validate_cached_frame(path: Path, expected_fingerprint: str,
                          n_points: int, frame: FrameRef) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as data:
            xyz = data["xyz"]
            labels = data["labels"]
            metadata = json.loads(str(data["metadata"].item()))
    except Exception as exc:
        raise A4Error(f"invalid cache file {path}: {exc}") from exc
    if metadata.get("schema") != CACHE_SCHEMA:
        raise A4Error(f"cache schema mismatch: {path}")
    if metadata.get("fingerprint") != expected_fingerprint:
        raise A4Error(f"stale/cross-run cache rejected: {path}")
    expected_mapping = {
        "version": LABEL_MAPPING_VERSION,
        "source": LABEL_MAPPING_SOURCE,
        "ground_semantic_ids": GROUND_SEMANTIC_IDS,
        "ignore_semantic_ids": IGNORE_SEMANTIC_IDS,
    }
    if metadata.get("label_mapping") != expected_mapping:
        raise A4Error(f"cache label mapping mismatch: {path}")
    if metadata.get("frame_index") != frame.index or metadata.get("frame_stem") != frame.stem:
        raise A4Error(f"cache frame identity mismatch: {path}")
    if xyz.shape != (n_points, 3) or labels.shape != (n_points,):
        raise A4Error(f"cache tensor shape mismatch: {path}")
    if xyz.dtype != np.float32 or not np.isfinite(xyz).all():
        raise A4Error(f"invalid cache XYZ: {path}")
    if not np.isin(labels, [0, 1]).all():
        raise A4Error(f"invalid cache labels: {path}")
    expected_array_hash = hashlib.sha256(
        xyz.astype(np.float32).tobytes(order="C") +
        labels.astype(np.int64).tobytes(order="C")
    ).hexdigest()
    if metadata.get("array_sha256") != expected_array_hash:
        raise A4Error(f"cache content hash mismatch: {path}")
    return metadata


def select_split_frames(frames: list[FrameRef], debug: bool,
                        debug_cfg: dict[str, Any],
                        budget_B: int) -> dict[str, list[FrameRef]]:
    """Train = first budget_B frames of TRAIN_POOL; val/test fixed and raw.

    Selecting the first B frames (chronological prefix) keeps the same frozen
    ordering as the A4 formal split, so smaller budgets are strict subsets of
    larger ones and the val/test blocks are byte-identical across every run.
    """
    pool_start, pool_end = TRAIN_POOL
    pool_size = pool_end - pool_start
    if not (2 <= budget_B <= pool_size):
        raise A4Error(
            f"budget_B={budget_B} out of range; expected 2..{pool_size}")
    selected: dict[str, list[FrameRef]] = {}
    for split, (start, end) in FORMAL_SPLIT.items():
        subset = frames[start:end]
        if split == "train":
            subset = subset[:budget_B]
        if debug:
            subset = subset[:int(debug_cfg[f"{split}_frames"])]
        if not subset:
            raise A4Error(f"selected {split} split is empty")
        selected[split] = subset
    return selected


def build_caches(selected: dict[str, list[FrameRef]], run_dir: Path,
                 method: str, seed: int, n_points: int,
                 semantic_cfg: dict[str, Any], fingerprint: str,
                 runtime_symbols: tuple[Any, ...]) -> tuple[dict[str, list[Path]], dict]:
    cache_root = run_dir / "cache"
    paths: dict[str, list[Path]] = {}
    content_hashes: dict[str, list[str]] = {}
    audit_totals = {"removed_ground": 0, "collapsed_ground": 0,
                    "surface_noise_frames": 0,
                    "non_ground_invariant_checks": 0}
    for split, frames in selected.items():
        split_dir = cache_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        paths[split] = []
        content_hashes[split] = []
        for offset, frame in enumerate(frames, start=1):
            path = split_dir / f"{frame.stem}.npz"
            if path.exists():
                metadata = validate_cached_frame(path, fingerprint, n_points, frame)
            else:
                metadata = cache_one_frame(
                    frame, split, method, seed, n_points, semantic_cfg,
                    fingerprint, runtime_symbols, path)
                validate_cached_frame(path, fingerprint, n_points, frame)
            audit = metadata.get("audit", {})
            if split != "train":
                if audit.get("transform") != "none" or not audit.get("raw_original"):
                    raise A4Error(f"{split} cache is not raw/unaugmented: {path}")
            if split == "train" and method == "BASELINE":
                if audit.get("transform") != "none" or not audit.get("raw_original"):
                    raise A4Error("Baseline cache contains augmentation")
            if split == "train" and method in LOADSIM_METHODS:
                if not audit.get("ground_only_mask") or not audit.get("non_ground_unchanged"):
                    raise A4Error("LoadSim cache lacks ground-only audit evidence")
                audit_totals["removed_ground"] += int(audit.get("removed_ground", 0))
                audit_totals["collapsed_ground"] += int(audit.get("collapsed_ground", 0))
                audit_totals["surface_noise_frames"] += int(bool(audit.get("surface_noise_applied")))
                audit_totals["non_ground_invariant_checks"] += 1
            paths[split].append(path)
            content_hashes[split].append(str(metadata["array_sha256"]))
            if offset == 1 or offset % 200 == 0 or offset == len(frames):
                print(f"cache {split}: {offset}/{len(frames)}", flush=True)
    if method in LOADSIM_METHODS:
        if audit_totals["non_ground_invariant_checks"] != len(selected["train"]):
            raise A4Error("LoadSim cache has incomplete non-ground invariance checks")
        if method == "LOADSIM":
            # Full LoadSim must show real material removal / lateral collapse.
            effective = audit_totals["removed_ground"] + audit_totals["collapsed_ground"]
            if effective <= 0:
                raise A4Error("LoadSim cache has no aggregate effective ground deformation")
        else:  # LOADSIM_GEOM: removal/collapse disabled by design; effect is surface noise
            if audit_totals["removed_ground"] or audit_totals["collapsed_ground"]:
                raise A4Error(
                    "LOADSIM_GEOM must disable directional removal and lateral "
                    "collapse, but the audit recorded some")
            if audit_totals["surface_noise_frames"] <= 0:
                raise A4Error(
                    "LOADSIM_GEOM cache shows no surface-noise deformation on any "
                    "train frame (the only enabled component produced nothing)")
    manifest = {
        "schema": CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "run_id": run_dir.name,
        "method": method,
        "seed": seed,
        "cache_root": str(cache_root),
        "splits": {name: {
            "count": len(split_paths),
            "files": [path.name for path in split_paths],
        } for name, split_paths in paths.items()},
        "evaluation_cache": {
            "source": "raw_frames", "augmentation": "none",
            "sample_seed_depends_on_method_or_run_seed": False,
            "val_content_sha256": hashlib.sha256(
                canonical_json(content_hashes["val"])).hexdigest(),
            "test_content_sha256": hashlib.sha256(
                canonical_json(content_hashes["test"])).hexdigest(),
        },
        "loadsim_ground_only_audit": audit_totals,
        "completed_at": utc_now(),
    }
    atomic_write_json(cache_root / "manifest.json", manifest)
    return paths, manifest


def load_cache_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["xyz"].astype(np.float32), data["labels"].astype(np.int64)


def batch_from_paths(torch: Any, paths: list[Path], indices: list[int],
                     device: Any) -> tuple[Any, Any]:
    arrays = [load_cache_arrays(paths[index]) for index in indices]
    xyz = np.stack([item[0] for item in arrays], axis=0)
    labels = np.stack([item[1] for item in arrays], axis=0)
    x = torch.from_numpy(xyz).permute(0, 2, 1).contiguous().to(
        device, non_blocking=True)
    y = torch.from_numpy(labels).to(device, non_blocking=True)
    return x, y


class MetricAccumulator:
    def __init__(self) -> None:
        self.confusion = np.zeros((2, 2), dtype=np.int64)
        self.loss_sum = 0.0
        self.loss_weight = 0

    def update(self, pred: np.ndarray, label: np.ndarray,
               loss: float | None = None) -> None:
        pred = np.asarray(pred).reshape(-1)
        label = np.asarray(label).reshape(-1)
        if pred.shape != label.shape or not np.isin(pred, [0, 1]).all() or not np.isin(label, [0, 1]).all():
            raise A4Error("invalid binary predictions/labels for metrics")
        self.confusion += np.bincount(
            label * 2 + pred, minlength=4).reshape(2, 2)
        if loss is not None:
            self.loss_sum += float(loss) * len(label)
            self.loss_weight += len(label)

    def state_dict(self) -> dict[str, Any]:
        return {"confusion": self.confusion.copy(),
                "loss_sum": self.loss_sum, "loss_weight": self.loss_weight}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        confusion = np.asarray(state["confusion"], dtype=np.int64)
        if confusion.shape != (2, 2):
            raise A4Error("checkpoint metric accumulator is invalid")
        self.confusion = confusion.copy()
        self.loss_sum = float(state["loss_sum"])
        self.loss_weight = int(state["loss_weight"])

    def compute(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "confusion_matrix": self.confusion.tolist(),
            "n_points": int(self.confusion.sum()),
        }
        ious, f1s = [], []
        for cls, name in ((0, "non_ground"), (1, "ground")):
            tp = int(self.confusion[cls, cls])
            fp = int(self.confusion[:, cls].sum() - tp)
            fn = int(self.confusion[cls, :].sum() - tp)
            denom_iou = tp + fp + fn
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            iou = tp / denom_iou if denom_iou else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            result[f"IoU_{name}"] = float(iou)
            result[f"F1_{name}"] = float(f1)
            result[f"precision_{name}"] = float(precision)
            result[f"recall_{name}"] = float(recall)
            ious.append(iou)
            f1s.append(f1)
        result["mIoU"] = float(np.mean(ious))
        result["F1_mean"] = float(np.mean(f1s))
        result["accuracy"] = float(np.trace(self.confusion) / max(self.confusion.sum(), 1))
        result["pred_ground_ratio"] = float(
            self.confusion[:, 1].sum() / max(self.confusion.sum(), 1))
        result["label_ground_ratio"] = float(
            self.confusion[1, :].sum() / max(self.confusion.sum(), 1))
        if self.loss_weight:
            result["loss"] = float(self.loss_sum / self.loss_weight)
        return result


def epoch_permutation(n_items: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng(
        derive_seed(seed, "train-permutation", epoch)).permutation(n_items)


def make_grad_scaler(torch: Any, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(torch: Any, enabled: bool) -> Any:
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def checkpoint_payload(torch: Any, fingerprint: str, run_id: str, method: str,
                       seed: int, model: Any, optimizer: Any, scheduler: Any,
                       scaler: Any, epoch: int, next_batch: int,
                       global_updates: int, best_miou: float, best_epoch: int,
                       history: list[dict[str, Any]], accumulator: MetricAccumulator,
                       completed: bool = False) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "fingerprint": fingerprint,
        "run_id": run_id,
        "method": method,
        "seed": seed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "global_updates": global_updates,
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "history": history,
        "epoch_accumulator": accumulator.state_dict(),
        "rng_state": capture_rng_state(torch),
        "completed": completed,
        "saved_at": utc_now(),
    }


def validate_checkpoint(ckpt: dict[str, Any], fingerprint: str,
                        run_id: str, method: str, seed: int) -> None:
    required = {
        "schema", "fingerprint", "run_id", "method", "seed", "model_state",
        "optimizer_state", "scheduler_state", "scaler_state", "epoch",
        "next_batch", "global_updates", "best_miou", "best_epoch", "history",
        "epoch_accumulator", "rng_state", "completed", "saved_at",
    }
    missing = required - set(ckpt)
    if missing:
        raise A4Error(f"checkpoint is incomplete; missing={sorted(missing)}")
    if ckpt["schema"] != CHECKPOINT_SCHEMA:
        raise A4Error("unsupported checkpoint schema")
    expected = (fingerprint, run_id, method, seed)
    actual = (ckpt["fingerprint"], ckpt["run_id"], ckpt["method"], ckpt["seed"])
    if actual != expected:
        raise A4Error("checkpoint belongs to a different config/run/method/seed")


def evaluate_split(torch: Any, model: Any, paths: list[Path], batch_size: int,
                   device: Any, criterion: Any, split: str,
                   frame_refs: list[FrameRef]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(paths) != len(frame_refs):
        raise A4Error(f"{split} cache/frame count mismatch")
    model.eval()
    total = MetricAccumulator()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            indices = list(range(start, min(start + batch_size, len(paths))))
            x, y = batch_from_paths(torch, paths, indices, device)
            logits = model(x)
            loss = criterion(logits.permute(0, 2, 1).reshape(-1, logits.shape[1]), y.reshape(-1))
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
            for local, index in enumerate(indices):
                one = MetricAccumulator()
                one.update(pred[local], labels[local], float(loss.item()))
                metrics = one.compute()
                total.update(pred[local], labels[local], float(loss.item()))
                rows.append({
                    "split": split,
                    "frame_index": frame_refs[index].index,
                    "frame_stem": frame_refs[index].stem,
                    "block_index": 0,
                    "n_blocks_in_frame": 1,
                    **metrics,
                })
    if not rows:
        raise A4Error(f"evaluation produced no {split} rows")
    return total.compute(), rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)


def train_model(torch: Any, nn: Any, PointNetPPKITTI: Any,
                run_cfg: dict[str, Any], semantic_cfg: dict[str, Any],
                cache_paths: dict[str, list[Path]],
                selected: dict[str, list[FrameRef]], run_dir: Path,
                fingerprint: str, resume: bool, debug: bool,
                device: Any) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    training = dict(run_cfg["training"])
    if debug:
        dbg = semantic_cfg["debug"]
        training["epochs"] = int(dbg["epochs"])
        training["batch_size"] = int(dbg["batch_size"])
        max_updates = int(dbg["max_updates"])
        checkpoint_interval = int(dbg["checkpoint_every_updates"])
    else:
        max_updates = int(run_cfg["protocol"]["max_updates_U"])
        checkpoint_interval = int(
            semantic_cfg["training_runtime"]["checkpoint_every_updates"])
    batch_size = int(training["batch_size"])
    if batch_size < 1 or max_updates < 1 or checkpoint_interval < 1:
        raise A4Error("invalid positive training runtime value")
    # A4B is UPDATE-driven: the training loop below is `while global_updates <
    # max_updates`, iterating as many passes over the B-frame train set as
    # needed to reach exactly U updates. Small B => more passes. This is what
    # keeps U matched across the whole budget grid (no fixed epoch count).

    model = PointNetPPKITTI(in_ch=3, n_cls=2).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]))
    # A4B uses an UPDATE-space schedule (cosine over max_updates), stepped once
    # per optimizer update. This makes the LR trajectory identical across every
    # budget B (it depends only on U, not on epoch count). An epoch-space StepLR
    # would decay to ~0 almost immediately at small B, breaking matched
    # optimization. Mirrors the proven a3_train_engine.py design.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(max_updates))
    # A4B forces fp32 regardless of the yaml amp flag: the A4 formal runs failed
    # from fp16 gradient overflow tripping the fixed-U step guard. fp32 removes
    # that failure mode entirely (guard becomes moot) at a modest speed cost.
    amp_enabled = False
    scaler = make_grad_scaler(torch, amp_enabled)
    class_weights = torch.tensor(
        semantic_cfg["training_runtime"]["class_weights"],
        dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_dir = run_dir / "checkpoints"
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    start_batch = 0
    global_updates = 0
    best_miou = -1.0
    best_epoch = -1
    accumulator = MetricAccumulator()

    if resume:
        if not latest_path.is_file():
            raise A4Error("--resume requested but checkpoints/latest.pt is missing")
        try:
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(latest_path, map_location=device)
        except Exception as exc:
            raise A4Error(f"cannot load resume checkpoint: {exc}") from exc
        validate_checkpoint(ckpt, fingerprint, run_cfg["run_id"],
                            run_cfg["method"], run_cfg["seed"])
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["epoch"])
        start_batch = int(ckpt["next_batch"])
        global_updates = int(ckpt["global_updates"])
        best_miou = float(ckpt["best_miou"])
        best_epoch = int(ckpt["best_epoch"])
        history = list(ckpt["history"])
        accumulator.load_state_dict(ckpt["epoch_accumulator"])
        restore_rng_state(torch, ckpt["rng_state"])
        print(f"resumed epoch={start_epoch} batch={start_batch} "
              f"updates={global_updates}", flush=True)
    elif latest_path.exists() or best_path.exists():
        raise A4Error("existing checkpoint found; use --resume or a clean run_dir")

    n_train = len(cache_paths["train"])
    n_batches = (n_train + batch_size - 1) // batch_size
    # Update-space evaluation cadence, anchored to one full-train-pool pass
    # (ceil(n_train_pool/batch) updates), so val frequency is identical at every
    # budget. TRAIN_POOL is set from the manifest at discovery time.
    full_pool_batches = (TRAIN_POOL[1] - TRAIN_POOL[0] + batch_size - 1) // batch_size
    eval_interval = max(1, full_pool_batches)
    # Update-driven loop: iterate passes over the B-frame train set until U is
    # reached. `epoch` here is just a pass counter used for the frame shuffle.
    epoch = start_epoch
    while global_updates < max_updates:
        permutation = epoch_permutation(n_train, run_cfg["seed"], epoch)
        batch_begin = start_batch if epoch == start_epoch else 0
        if batch_begin < 0 or batch_begin > n_batches:
            raise A4Error("checkpoint next_batch is outside the epoch")
        model.train()
        made_progress = False
        for batch_index in range(batch_begin, n_batches):
            if global_updates >= max_updates:
                break
            indices = permutation[
                batch_index * batch_size:min((batch_index + 1) * batch_size, n_train)
            ].tolist()
            if len(indices) < 2:
                # Last odd batch at tiny B: skip rather than train on <2 samples.
                continue
            made_progress = True
            x, y = batch_from_paths(torch, cache_paths["train"], indices, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(torch, amp_enabled):
                logits = model(x)
                loss = criterion(logits.permute(0, 2, 1).reshape(-1, logits.shape[1]), y.reshape(-1))
            if not torch.isfinite(loss):
                raise A4Error(f"non-finite loss at pass={epoch} batch={batch_index}")
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if amp_enabled and scaler.get_scale() < scale_before:
                raise A4Error("AMP skipped an optimizer step (should not happen in fp32)")
            scheduler.step()
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
            accumulator.update(pred, labels, float(loss.item()))
            global_updates += 1
            next_batch = batch_index + 1

            should_evaluate = (global_updates % eval_interval == 0
                               or global_updates >= max_updates)
            if should_evaluate:
                train_metrics = accumulator.compute()
                val_metrics, _ = evaluate_split(
                    torch, model, cache_paths["val"], batch_size, device,
                    criterion, "val", selected["val"])
                history.append({
                    "global_updates": global_updates,
                    "pass": epoch + 1,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train": train_metrics,
                    "val": val_metrics,
                })
                accumulator = MetricAccumulator()
                improved = float(val_metrics["mIoU"]) > best_miou
                if improved:
                    best_miou = float(val_metrics["mIoU"])
                    best_epoch = global_updates
                model.train()
                if improved:
                    atomic_torch_save(best_path, checkpoint_payload(
                        torch, fingerprint, run_cfg["run_id"], run_cfg["method"],
                        run_cfg["seed"], model, optimizer, scheduler, scaler,
                        epoch, next_batch, global_updates, best_miou, best_epoch,
                        history, accumulator))
                atomic_write_json(run_dir / "history.json", {
                    "run_id": run_cfg["run_id"], "method": run_cfg["method"],
                    "seed": run_cfg["seed"], "history": history,
                    "best_val_mIoU": best_miou, "best_update": best_epoch,
                })
                print(f"pass {epoch + 1} updates={global_updates}/{max_updates} "
                      f"val_mIoU={val_metrics['mIoU']:.6f}", flush=True)

            if global_updates % checkpoint_interval == 0 or should_evaluate:
                atomic_torch_save(latest_path, checkpoint_payload(
                    torch, fingerprint, run_cfg["run_id"], run_cfg["method"],
                    run_cfg["seed"], model, optimizer, scheduler, scaler,
                    epoch, next_batch, global_updates, best_miou, best_epoch,
                    history, accumulator))
        if not made_progress and global_updates < max_updates:
            raise A4Error("training pass made no progress (budget too small for batch)")
        epoch += 1
        start_batch = 0

    if not history or not best_path.is_file() or not latest_path.is_file():
        raise A4Error("training finished without complete latest/best checkpoints")
    # Mark the coherent final training state complete before switching model
    # weights to the validation-selected checkpoint for evaluation.
    try:
        final_training = torch.load(
            latest_path, map_location=device, weights_only=False)
    except TypeError:
        final_training = torch.load(latest_path, map_location=device)
    validate_checkpoint(final_training, fingerprint, run_cfg["run_id"],
                        run_cfg["method"], run_cfg["seed"])
    final_training["completed"] = True
    final_training["saved_at"] = utc_now()
    atomic_torch_save(latest_path, final_training)
    try:
        best = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        best = torch.load(best_path, map_location=device)
    validate_checkpoint(best, fingerprint, run_cfg["run_id"],
                        run_cfg["method"], run_cfg["seed"])
    model.load_state_dict(best["model_state"], strict=True)
    val_metrics, val_rows = evaluate_split(
        torch, model, cache_paths["val"], batch_size, device, criterion,
        "val", selected["val"])
    test_metrics, test_rows = evaluate_split(
        torch, model, cache_paths["test"], batch_size, device, criterion,
        "test", selected["test"])
    return history, {
        "selection": {"metric": "val.mIoU", "best_update": best_epoch,
                      "best_value": best_miou},
        "val": val_metrics,
        "test": test_metrics,
        "training": {"epochs_completed": len(history),
                     "global_updates": global_updates,
                     "max_updates": max_updates},
    }, val_rows + test_rows


def source_revision(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
    try:
        top = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", top, "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", top, "status", "--porcelain", "--", str(path)],
            check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        result.update({"git_root": top, "git_commit": commit,
                       "git_path_dirty": bool(dirty)})
    except (OSError, subprocess.SubprocessError):
        result["git"] = "unavailable"
    return result


def make_split_manifest(selected: dict[str, list[FrameRef]], seq_dir: Path,
                        debug: bool) -> tuple[dict[str, Any], str]:
    """Hash each selected raw file once and return its auditable manifest/hash."""
    unknown_splits = set(selected) - set(FORMAL_SPLIT)
    if unknown_splits:
        raise A4Error(f"unknown selected splits: {sorted(unknown_splits)}")
    split_order = [name for name in FORMAL_SPLIT if name in selected]
    split_records: dict[str, Any] = {}
    content_owners: dict[tuple[str, str], tuple[str, str]] = {}
    index_owners: dict[int, tuple[str, str]] = {}
    for split_position, name in enumerate(split_order):
        frames = selected[name]
        start, end = FORMAL_SPLIT[name]
        records = []
        for order, frame in enumerate(frames):
            if not start <= frame.index < end:
                raise A4Error(f"frame {frame.stem} lies outside frozen {name} bounds")
            if frame.index in index_owners:
                raise A4Error(
                    f"frame index overlap: {index_owners[frame.index]} and {(name, frame.stem)}")
            index_owners[frame.index] = (name, frame.stem)
            bin_size = frame.bin_path.stat().st_size
            label_size = frame.label_path.stat().st_size
            bin_sha = sha256_file(frame.bin_path)
            label_sha = sha256_file(frame.label_path)
            content_identity = (bin_sha, label_sha)
            if content_identity in content_owners:
                raise A4Error(
                    f"raw content duplicated across selected frames: "
                    f"{content_owners[content_identity]} and {(name, frame.stem)}")
            content_owners[content_identity] = (name, frame.stem)
            records.append({
                "order": order,
                "index": frame.index,
                "stem": frame.stem,
                "bin": {
                    "relative_path": frame.bin_path.relative_to(seq_dir).as_posix(),
                    "size": bin_size,
                    "sha256": bin_sha,
                },
                "label": {
                    "relative_path": frame.label_path.relative_to(seq_dir).as_posix(),
                    "size": label_size,
                    "sha256": label_sha,
                },
            })
        split_records[name] = {
            "order": split_position,
            "formal_bounds_half_open": [start, end],
            "formal_count": end - start,
            "selected_count": len(records),
            "records": records,
            "records_sha256": hashlib.sha256(canonical_json(records)).hexdigest(),
        }
    fingerprint_payload = {
        "schema": "v3d-selected-block-content-v1",
        "dataset": "Vaihingen3D",
        "sequence": "spatial_band_x",
        "split_order": split_order,
        "splits": split_records,
    }
    content_fingerprint = hashlib.sha256(
        canonical_json(fingerprint_payload)).hexdigest()
    manifest = {
        **fingerprint_payload,
        "mode": "spatial_band_x",
        "debug_subset": debug,
        "disjoint": {
            "verified": True,
            "criteria": ["block_index", "block_npz_sha256"],
            "selected_frame_count": len(index_owners),
        },
        "content_hash_algorithm": "sha256",
        "selected_content_fingerprint": content_fingerprint,
    }
    return manifest, content_fingerprint


def make_provenance(torch: Any, run_cfg: dict[str, Any],
                    semantic_cfg: dict[str, Any], config_path: Path,
                    yaml_path: Path, seq_dir: Path, frames: list[FrameRef],
                    selected_content_fingerprint: str, fingerprint: str,
                    cache_manifest: dict[str, Any], debug: bool) -> dict[str, Any]:
    source_paths = [
        Path(__file__).resolve(), BASE_DIR / "v3d_io.py",
        BASE_DIR / "v3d_prepare.py", KITTI_CODE_DIR / "kitti_train.py",
    ]
    gpu = torch.cuda.get_device_properties(0)
    return {
        "schema": "v3d-provenance-v1",
        "run_id": run_cfg["run_id"], "method": run_cfg["method"],
        "seed": run_cfg["seed"], "debug": debug,
        "run_fingerprint": fingerprint,
        "created_at": utc_now(),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path),
                   "resolved": run_cfg},
        "semantic_config": {"path": str(yaml_path), "sha256": sha256_file(yaml_path),
                            "resolved": semantic_cfg},
        "dataset": {
            "sequence_dir": str(seq_dir), "sequence": "spatial_band_x",
            "frame_count": len(frames),
            "selected_content_fingerprint": selected_content_fingerprint,
            "selected_content_hash_algorithm": "sha256",
        },
        "method_semantics": semantic_cfg["methods"][run_cfg["method"]],
        "evaluation_semantics": semantic_cfg["evaluation"],
        "cache_manifest_sha256": hashlib.sha256(
            canonical_json(cache_manifest)).hexdigest(),
        "sources": [source_revision(path) for path in source_paths],
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": gpu.name, "gpu_total_memory": gpu.total_memory,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "simulation_or_fixed_model_reuse": False,
    }


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    method = args.method.upper()
    seed = int(args.seed)
    if method not in METHODS:
        raise A4Error(f"method must be one of {METHODS}")
    run_dir.mkdir(parents=True, exist_ok=True)

    budget_B = int(args.budget)
    run_cfg, semantic_cfg, yaml_path = load_contract(
        config_path, method, seed, run_dir, budget_B)
    torch, nn, _, _, _, runtime_symbols = import_runtime()
    device = require_cuda(torch)
    seed_everything(seed, bool(
        semantic_cfg["training_runtime"]["deterministic_algorithms"]))

    v3d_root = resolve_relative(BASE_DIR, run_cfg["paths"]["v3d_root"],
                                "paths.v3d_root")
    seq_dir = locate_sequence_dir(v3d_root)
    frames = discover_frames(seq_dir)  # sets FORMAL_SPLIT / TRAIN_POOL
    # Runtime upper-bound check on the budget now that the pool size is known.
    pool_size = TRAIN_POOL[1] - TRAIN_POOL[0]
    if budget_B > pool_size:
        raise A4Error(
            f"budget_B={budget_B} exceeds prepared train pool ({pool_size}); "
            f"regenerate blocks or lower the budget")
    selected = select_split_frames(
        frames, args.debug, semantic_cfg["debug"], budget_B)
    source_hashes = {
        "v3d_budget_train.py": sha256_file(Path(__file__).resolve()),
        "v3d_io.py": sha256_file(BASE_DIR / "v3d_io.py"),
        "v3d_prepare.py": sha256_file(BASE_DIR / "v3d_prepare.py"),
        "kitti_train.py": sha256_file(KITTI_CODE_DIR / "kitti_train.py"),
    }
    split_manifest, selected_content_fingerprint = make_split_manifest(
        selected, seq_dir, args.debug)
    fingerprint_payload = {
        "run_config": run_cfg,
        "semantic_config": semantic_cfg,
        "selected_dataset": {
            "schema": split_manifest["schema"],
            "dataset": split_manifest["dataset"],
            "sequence": split_manifest["sequence"],
            "split_order": split_manifest["split_order"],
            "splits": split_manifest["splits"],
            "selected_content_fingerprint": selected_content_fingerprint,
        },
        "source_hashes": source_hashes,
        "debug": bool(args.debug),
    }
    fingerprint = hashlib.sha256(canonical_json(fingerprint_payload)).hexdigest()
    split_manifest["run_fingerprint"] = fingerprint
    atomic_write_json(run_dir / "split_manifest.json", split_manifest)
    resolved = {
        "run_id": run_cfg["run_id"], "method": method, "seed": seed,
        "debug": bool(args.debug), "run_dir": str(run_dir),
        "run_fingerprint": fingerprint,
        "selected_dataset_content": {
            "manifest": "split_manifest.json",
            "schema": split_manifest["schema"],
            "hash_algorithm": split_manifest["content_hash_algorithm"],
            "fingerprint": selected_content_fingerprint,
            "split_order": split_manifest["split_order"],
        },
        "run_config": run_cfg, "semantic_config": semantic_cfg,
        "resolved_paths": {"config": str(config_path),
                           "v3d_config": str(yaml_path),
                           "v3d_root": str(v3d_root),
                           "sequence_dir": str(seq_dir)},
        "effective_split": {name: [values[0].index, values[-1].index + 1]
                            for name, values in selected.items()},
        "effective_counts": {name: len(values)
                             for name, values in selected.items()},
        "source_hashes": source_hashes,
    }
    atomic_write_json(run_dir / "resolved_config.json", resolved)
    atomic_write_json(run_dir / "status.json", {
        "status": "running", "run_id": run_cfg["run_id"],
        "method": method, "seed": seed, "debug": bool(args.debug),
        "run_fingerprint": fingerprint, "started_at": utc_now(),
    })

    cache_paths, cache_manifest = build_caches(
        selected, run_dir, method, seed, int(run_cfg["protocol"]["num_points"]),
        semantic_cfg, fingerprint, runtime_symbols)
    provenance = make_provenance(
        torch, run_cfg, semantic_cfg, config_path, yaml_path, seq_dir, frames,
        selected_content_fingerprint, fingerprint, cache_manifest, args.debug)
    atomic_write_json(run_dir / "provenance.json", provenance)

    PointNetPPKITTI = runtime_symbols[-1]
    history, metrics, frame_rows = train_model(
        torch, nn, PointNetPPKITTI, run_cfg, semantic_cfg, cache_paths,
        selected, run_dir, fingerprint, args.resume, args.debug, device)
    metrics.update({
        "schema": "v3d-budget-metrics-v1", "status": "completed",
        "run_id": run_cfg["run_id"],
        "method": method, "seed": seed, "budget_B": budget_B,
        "debug": bool(args.debug),
        "run_fingerprint": fingerprint,
        "n_frames": {name: len(values) for name, values in selected.items()},
        "per_frame_block_metrics": "per_frame_block_metrics.jsonl",
        "completed_at": utc_now(),
    })
    atomic_write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "per_frame_block_metrics.jsonl", frame_rows)
    # Re-write history only to add immutable identity around the epoch records.
    atomic_write_json(run_dir / "history.json", {
        "schema": "v3d-history-v1", "run_id": run_cfg["run_id"],
        "method": method, "seed": seed, "run_fingerprint": fingerprint,
        "history": history,
        "best_val_mIoU": metrics["selection"]["best_value"],
        "best_update": metrics["selection"]["best_update"],
    })
    atomic_write_json(run_dir / "status.json", {
        "status": "completed", "run_id": run_cfg["run_id"],
        "method": method, "seed": seed, "debug": bool(args.debug),
        "run_fingerprint": fingerprint, "completed_at": utc_now(),
        "metrics": "metrics.json", "best_checkpoint": "checkpoints/best.pt",
    })
    print(json.dumps({"run_id": run_cfg["run_id"], "status": "completed",
                      "test_mIoU": metrics["test"]["mIoU"]}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget", required=True, type=int,
                        help="training-block budget B (2..n_train from manifest)")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="real-frame CUDA smoke: 2/1/1 frames, one update")
    return parser


def main(argv: list[str] | None = None) -> int:
    # A4B is an additional exploratory proxy-domain experiment; the two-factor
    # formal-run authorization gate is intentionally not enforced here.
    args = build_parser().parse_args(argv)
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        message = "interrupted by user"
        status = "interrupted"
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        status = "failed"
        traceback.print_exc()
    try:
        run_dir = args.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        prior: dict[str, Any] = {}
        status_path = run_dir / "status.json"
        if status_path.is_file():
            try:
                prior = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior = {}
        atomic_write_json(status_path, {
            **prior, "status": status, "method": args.method,
            "seed": args.seed, "failed_at": utc_now(), "error": message,
        })
    except Exception:
        traceback.print_exc()
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    return 130 if status == "interrupted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
