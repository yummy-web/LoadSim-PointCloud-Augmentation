"""V3D I/O adapter: makes prepared V3D .npz blocks look like KITTI frames.

The V3D budget trainer reuses the frozen A4B sampling/transform/training core
verbatim. That core calls three I/O primitives per frame:
  load_bin(path)      -> (N,4) float32  (xyz + a dummy 0 intensity column)
  load_label(path)    -> (N,) int       (raw 9-class V3D IDs)
  map_to_binary(ids)  -> (N,) int in {0,1,255}
Here a "frame" is one prepared block .npz (keys: xyz (N,3) float32, sem (N,) int).
Both bin_path and label_path point at the same .npz, so load_bin/load_label
read the two arrays from one file.

Binary ground/non-ground mapping mirrors v3d_prepare.py and is the scientific
choice for the proxy task (VERIFY class IDs after download via
`python v3d_prepare.py --inspect`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Must match v3d_prepare.py. Keep in one place conceptually; duplicated as
# module constants so the trainer can import them for contract checks.
V3D_VALID_RAW_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
V3D_GROUND_IDS = (2,)            # Impervious surfaces ONLY (load-bearing surface)
V3D_NONGROUND_IDS = (0, 1, 3, 4, 5, 6, 7, 8)
V3D_IGNORE_IDS: tuple[int, ...] = ()
LABEL_MAPPING_VERSION = "v3d-impervious-surface-v2"
LABEL_MAPPING_SOURCE = "ISPRS Vaihingen 3D 9-class -> binary impervious-surface/other"


def load_bin(path: Path) -> np.ndarray:
    """Return (N,4) float32: xyz + a zero 4th column (KITTI-shape compatible)."""
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"v3d block xyz must be (N,3): {path} has {xyz.shape}")
    pad = np.zeros((len(xyz), 1), dtype=np.float32)
    return np.concatenate([xyz, pad], axis=1)


def load_label(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        sem = np.asarray(data["sem"]).astype(np.int64, copy=False)
    if sem.ndim != 1:
        raise ValueError(f"v3d block sem must be (N,): {path} has {sem.shape}")
    return sem


def map_to_binary(raw_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(raw_ids)
    out = np.full(ids.shape, 255, dtype=ids.dtype)
    out[np.isin(ids, V3D_GROUND_IDS)] = 1
    out[np.isin(ids, V3D_NONGROUND_IDS)] = 0
    return out


def traditional_augment(*_args, **_kwargs):
    """Placeholder to satisfy the runtime-symbols tuple shape. The V3D trainer
    uses the inline traditional_transform (pure numpy), not this symbol."""
    raise NotImplementedError("v3d uses inline traditional_transform")
