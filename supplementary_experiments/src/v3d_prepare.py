"""V3D (ISPRS Vaihingen 3D) prep for the cross-domain budget-curve experiment.

Reads the raw V3D point files (whitespace-delimited .pts/.txt), tiles the single
large ALS scene into fixed-size XY blocks, samples a fixed point count per block,
and writes each block as an .npz "frame" (xyz float32 Nx3 + sem int32 Nx1 raw
9-class IDs). A manifest.json records every block, its split assignment, and
diagnostics. The training script (v3d_budget_train.py) consumes these blocks
through v3d_io.py, reusing the frozen A4B transforms / budget loop unchanged.

Task = binary ground/non-ground (same proxy task as KITTI A4/A4B). The 9->2
class mapping is a documented, VERIFIABLE constant (see CLASS_MAP); after the
real download, run with --inspect first and confirm the printed label histogram
and class-ID set match the assumption before trusting the mapping.

Split = spatial bands along X (no block-adjacency leakage across splits):
default train:val:test = 60:20:20 of the X-range. All splits come from the
public labeled training file, because the official V3D test labels are withheld
by the benchmark. This is an internal controlled experiment, stated as such.

Usage:
  # 1) inspect raw file (format autodetect, histogram, ranges) BEFORE trusting map
  python v3d_prepare.py --inspect --train-file Vaihingen3D_Traininig.pts
  # 2) build blocks
  python v3d_prepare.py --train-file Vaihingen3D_Traininig.pts \
      --out-root v3d_experiment/blocks --block-size 30 --points-per-block 4096
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import os
from pathlib import Path
from typing import Any

import numpy as np

# --- V3D 9-class catalog (ISPRS Vaihingen 3D semantic labeling benchmark) ---
# class_id : name  (VERIFY these IDs against --inspect output after download)
V3D_CLASS_NAMES = {
    0: "Powerline",
    1: "Low vegetation",
    2: "Impervious surfaces",
    3: "Car",
    4: "Fence/Hedge",
    5: "Roof",
    6: "Facade",
    7: "Shrub",
    8: "Tree",
}
# Binary "load-bearing surface" / other mapping (R3-10 out-of-domain geometric
# stress test; R3-6: pick the class whose geometry best matches a flat
# load-bearing surface the operator would act on).
# ground(1)  = Impervious surfaces ONLY (roads/pavement: flat, rigid, the closest
#              real analogue of a load-bearing operating surface). Low vegetation
#              is deliberately EXCLUDED: it is neither load-bearing nor
#              geometrically flat, and lumping it in was a semantic mismatch.
# non-ground(0) = everything else (incl. low vegetation, structures, objects).
GROUND_CLASS_IDS = (2,)              # Impervious surfaces only
NONGROUND_CLASS_IDS = (0, 1, 3, 4, 5, 6, 7, 8)
IGNORE_CLASS_IDS: tuple[int, ...] = ()   # V3D has no explicit ignore class

# Split fractions along X (contiguous spatial bands, no cross-split adjacency).
SPLIT_FRACTIONS = {"train": 0.60, "val": 0.20, "test": 0.20}
MANIFEST_SCHEMA = "v3d-blocks-v1"


class PrepError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Atomic writers
# ---------------------------------------------------------------------------
def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))


def _save_npz(path: Path, xyz: np.ndarray, sem: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez(tmp, xyz=xyz.astype(np.float32), sem=sem.astype(np.int32))
        # np.savez appends .npz to the name we gave; normalize.
        if os.path.exists(tmp + ".npz"):
            os.replace(tmp + ".npz", path)
        else:
            os.replace(tmp, path)
    finally:
        for leftover in (tmp, tmp + ".npz"):
            if os.path.exists(leftover):
                os.remove(leftover)


# ---------------------------------------------------------------------------
# Raw reader (whitespace-delimited .pts/.txt with column autodetect)
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a V3D ascii point file. Returns (xyz float32 Nx3, labels int64 N).

    V3D training file rows are typically: X Y Z [refl] [ret] [num] ... label
    where the LAST integer column is the class label. Columns 0..2 are XYZ.
    We autodetect: first 3 numeric columns -> XYZ, last column -> label. This
    is verified interactively via --inspect before any block is written.
    """
    rows = np.loadtxt(path, dtype=np.float64, comments="//")
    if rows.ndim != 2 or rows.shape[1] < 4:
        raise PrepError(
            f"expected >=4 columns (XYZ + label), got shape {rows.shape} from {path}"
        )
    # Keep XYZ in float64: V3D coordinates are UTM (X~4.97e5, Y~5.42e6), where
    # float32 has ~0.5 m spacing and would quantize the geometry. The origin is
    # subtracted (per-block save time) before the final float32 cast.
    xyz = rows[:, 0:3].astype(np.float64)
    labels = np.rint(rows[:, -1]).astype(np.int64)
    return xyz, labels


# ---------------------------------------------------------------------------
# Inspect mode
# ---------------------------------------------------------------------------
def do_inspect(train_file: Path) -> int:
    if not train_file.is_file():
        print(f"[inspect] file not found: {train_file}", file=sys.stderr)
        return 2
    xyz, labels = read_raw(train_file)
    n = len(labels)
    print(f"[inspect] file          : {train_file}")
    print(f"[inspect] sha256        : {_sha256(train_file)}")
    print(f"[inspect] points        : {n}")
    print(f"[inspect] X range       : [{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}]"
          f"  span={xyz[:,0].max()-xyz[:,0].min():.3f}")
    print(f"[inspect] Y range       : [{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}]"
          f"  span={xyz[:,1].max()-xyz[:,1].min():.3f}")
    print(f"[inspect] Z range       : [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]"
          f"  span={xyz[:,2].max()-xyz[:,2].min():.3f}")
    uniq, counts = np.unique(labels, return_counts=True)
    print("[inspect] label histogram (raw id -> count, pct, assumed name):")
    for cid, cnt in zip(uniq.tolist(), counts.tolist()):
        name = V3D_CLASS_NAMES.get(cid, "!! UNKNOWN / UNMAPPED !!")
        tag = ""
        if cid in GROUND_CLASS_IDS:
            tag = "-> ground(1)"
        elif cid in NONGROUND_CLASS_IDS:
            tag = "-> non-ground(0)"
        elif cid in IGNORE_CLASS_IDS:
            tag = "-> ignore(255)"
        else:
            tag = "-> UNHANDLED (will become ignore/255!)"
        print(f"    {cid:>4d} : {cnt:>10d}  {100.0*cnt/n:6.2f}%  {name:<22s} {tag}")
    unmapped = [int(c) for c in uniq
                if c not in GROUND_CLASS_IDS
                and c not in NONGROUND_CLASS_IDS
                and c not in IGNORE_CLASS_IDS]
    if unmapped:
        print(f"[inspect] WARNING: unmapped class ids present: {unmapped} "
              f"-- FIX the *_CLASS_IDS constants before building blocks.",
              file=sys.stderr)
    else:
        print("[inspect] OK: every observed class id is covered by the mapping.")
    return 0


# ---------------------------------------------------------------------------
# Binary mapping + split assignment
# ---------------------------------------------------------------------------
def map_to_binary(labels: np.ndarray) -> np.ndarray:
    out = np.full(labels.shape, 255, dtype=np.int32)
    out[np.isin(labels, GROUND_CLASS_IDS)] = 1
    out[np.isin(labels, NONGROUND_CLASS_IDS)] = 0
    return out


def assign_split_by_x(cx: float, x_min: float, x_span: float) -> str:
    """Contiguous bands along X: [0,0.6)=train, [0.6,0.8)=val, [0.8,1.0]=test."""
    frac = 0.0 if x_span <= 0 else (cx - x_min) / x_span
    if frac < SPLIT_FRACTIONS["train"]:
        return "train"
    if frac < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------
def build_blocks(
    train_file: Path,
    out_root: Path,
    block_size: float,
    points_per_block: int,
    min_points: int,
    seed: int,
) -> int:
    if not train_file.is_file():
        print(f"[build] file not found: {train_file}", file=sys.stderr)
        return 2
    xyz, labels = read_raw(train_file)

    uniq = set(int(c) for c in np.unique(labels))
    known = set(V3D_CLASS_NAMES.keys())
    unknown = sorted(uniq - known)
    if unknown:
        raise PrepError(
            f"unknown class ids {unknown} not in V3D_CLASS_NAMES -- run --inspect "
            f"and fix the class constants before building."
        )

    x_min, y_min = float(xyz[:, 0].min()), float(xyz[:, 1].min())
    x_max = float(xyz[:, 0].max())
    x_span = x_max - x_min
    # Global coordinate origin subtracted before the float32 cast so stored
    # blocks keep sub-mm precision (UTM values are ~5e5/5.4e6). Training centers
    # each block on its own mean anyway, so the absolute origin is irrelevant to
    # the model; this only protects numerical precision at write time.
    origin = np.array([xyz[:, 0].min(), xyz[:, 1].min(), xyz[:, 2].min()],
                      dtype=np.float64)

    # Grid cell index per point.
    gx = np.floor((xyz[:, 0] - x_min) / block_size).astype(np.int64)
    gy = np.floor((xyz[:, 1] - y_min) / block_size).astype(np.int64)
    cell_key = gx.astype(np.int64) * 1_000_000 + gy.astype(np.int64)

    rng = np.random.default_rng(seed)
    block_dir = out_root / "blocks"
    block_dir.mkdir(parents=True, exist_ok=True)

    binary_all = map_to_binary(labels)
    blocks_meta: list[dict[str, Any]] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for key in np.unique(cell_key):
        idx = np.where(cell_key == key)[0]
        if len(idx) < min_points:
            skipped += 1
            continue
        if len(idx) >= points_per_block:
            sel = rng.choice(idx, size=points_per_block, replace=False)
        else:
            sel = rng.choice(idx, size=points_per_block, replace=True)

        block_xyz = xyz[sel]
        block_sem = labels[sel].astype(np.int32)
        cx = float(block_xyz[:, 0].mean())
        cy = float(block_xyz[:, 1].mean())
        split = assign_split_by_x(cx, x_min, x_span)

        frame_id = f"v3d_{int(key):d}"
        rel = f"blocks/{frame_id}.npz"
        # Subtract the global origin in float64, THEN cast to float32 inside
        # _save_npz (which keeps sub-mm precision for the ~400 m local extent).
        _save_npz(out_root / rel, block_xyz - origin, block_sem)

        b = binary_all[sel]
        blocks_meta.append({
            "frame_id": frame_id,
            "npz": rel,
            "split": split,
            "num_points": int(points_per_block),
            "centroid_xy": [cx, cy],
            "ground_frac": float(np.mean(b == 1)),
            "nonground_frac": float(np.mean(b == 0)),
            "ignore_frac": float(np.mean(b == 255)),
        })
        split_counts[split] += 1

    if not blocks_meta:
        raise PrepError("no blocks produced -- check block_size / min_points.")

    # Deterministic ordering: split then frame_id, so the trainer's chronological
    # "first B" budget prefix is stable across machines.
    order = {"train": 0, "val": 1, "test": 2}
    blocks_meta.sort(key=lambda m: (order[m["split"]], m["frame_id"]))

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_file": str(train_file),
        "source_sha256": _sha256(train_file),
        "block_size": block_size,
        "points_per_block": points_per_block,
        "min_points": min_points,
        "sampling_seed": seed,
        "split_fractions": SPLIT_FRACTIONS,
        "split_axis": "X",
        "coord_origin_xyz": origin.tolist(),
        "coord_origin_note": "stored block xyz = true_utm_xyz - coord_origin_xyz (float64 subtract, then float32); training re-centers per block",
        "class_names": V3D_CLASS_NAMES,
        "ground_class_ids": list(GROUND_CLASS_IDS),
        "nonground_class_ids": list(NONGROUND_CLASS_IDS),
        "ignore_class_ids": list(IGNORE_CLASS_IDS),
        "label_mapping": "v3d-impervious-surface-v2",
        "num_blocks": len(blocks_meta),
        "split_counts": split_counts,
        "skipped_sparse_cells": skipped,
        "total_points_raw": int(len(labels)),
        "blocks": blocks_meta,
    }
    _atomic_write_json(out_root / "manifest.json", manifest)

    print(f"[build] wrote {len(blocks_meta)} blocks to {block_dir}")
    print(f"[build] split_counts   : {split_counts}  (skipped sparse cells: {skipped})")
    print(f"[build] manifest       : {out_root / 'manifest.json'}")
    print("[build] NOTE: train budget B selects the first B *train* blocks "
          "(chronological/deterministic prefix); val/test are fixed across all runs.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare ISPRS Vaihingen 3D blocks.")
    p.add_argument("--train-file", type=Path, required=True,
                   help="raw V3D labeled training file (.pts/.txt).")
    p.add_argument("--out-root", type=Path, default=Path("v3d_experiment"),
                   help="output root (blocks/ + manifest.json written here).")
    p.add_argument("--inspect", action="store_true",
                   help="print format/histogram/ranges and exit (no blocks written).")
    p.add_argument("--block-size", type=float, default=30.0,
                   help="XY tile edge length in dataset units (meters for V3D).")
    p.add_argument("--points-per-block", type=int, default=4096,
                   help="fixed points sampled per block (matches KITTI num_points).")
    p.add_argument("--min-points", type=int, default=512,
                   help="drop grid cells sparser than this before sampling.")
    p.add_argument("--seed", type=int, default=20260915,
                   help="sampling RNG seed (block point selection).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.inspect:
            return do_inspect(args.train_file)
        return build_blocks(
            train_file=args.train_file,
            out_root=args.out_root,
            block_size=args.block_size,
            points_per_block=args.points_per_block,
            min_points=args.min_points,
            seed=args.seed,
        )
    except PrepError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
