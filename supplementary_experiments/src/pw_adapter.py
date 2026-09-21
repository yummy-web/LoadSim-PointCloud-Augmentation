"""PointWOLF scene-segmentation adapter (reviewer supplement R2-3).

WHY THIS FILE EXISTS
--------------------
Reviewer 2 (R2-3) noted the baseline set contained only conventional geometric
augmentation (rotation / scale / noise = TRAD) and no *recent point-cloud-specific*
augmentation method.  This module is a fixed-parameter, auditable adapter of
PointWOLF (Kim et al., 2021, "Point Cloud Augmentation with Weighted Local
Transformations") converted from its original *classification* setting into a
*point-wise scene-segmentation* operator, so it can be compared under the SAME
matched-budget protocol A3 already uses (M=9 / B / U=1330 / PointNet++ / 3 seeds).

ALGORITHM (faithful to PointWOLF, no learned components)
--------------------------------------------------------
PointWOLF applies W smoothly-blended local rigid-ish transformations anchored at
W control ("anchor") points sampled from the cloud:

  1. Sample W anchor points a_j from the input cloud (farthest-point sampling for
     spread, seeded/deterministic).
  2. For each anchor j, draw ONE local transformation T_j = (R_j, S_j, t_j):
        R_j : rotation with per-axis angle in [-rot, rot]
        S_j : anisotropic scaling with per-axis factor in [1/(1+sca), 1+sca]
        t_j : translation in [-trs, trs] (in the SAME normalised frame)
     Each T_j is applied about its own anchor a_j:
        f_j(x) = R_j @ (S_j * (x - a_j)) + a_j + t_j
  3. Blend the W transformed copies per point with distance-based weights
        w_j(x) = exp(-||x - a_j||^2 / sigma^2),  normalised over j,
     with a kernel bandwidth sigma set from the inter-anchor spacing, so the
     result is a single smooth non-rigid warp:
        x' = sum_j  w_j(x) * f_j(x).

KEY PROPERTIES THAT MAKE IT A VALID SEGMENTATION OPERATOR
---------------------------------------------------------
* It is a pure per-point coordinate map: input point i maps to output point i.
  => The point COUNT and point ORDER are preserved exactly (no sampling, no
     re-ordering, no add/drop).  Therefore point-wise labels propagate by
     identity:  Y' = Y.  This is asserted, not assumed (see pw_tests).
* It never touches the labels, so it cannot leak or invent supervision.
* It is fully deterministic given (points, seed, params): reproducible bank.

NORMALS
-------
Because the warp is non-rigid, the parent's photogrammetric normals are no longer
consistent with the deformed surface.  We therefore RECOMPUTE normals on the
warped cloud with a deterministic kNN-PCA estimator (recompute_normals), and
orient each recomputed normal to agree in sign with the parent's original normal
(stable, avoids arbitrary PCA sign flips).  This keeps the 6-channel
(XYZ+normal) input protocol identical to TRAD/LOADSIM/RAW_REPEAT so the completed
A3 runs remain the matched comparators.  The policy is recorded in every
candidate's metadata.

FROZEN DEFAULT PARAMETERS (declared, not tuned on any label/metric)
-------------------------------------------------------------------
The PointWOLF paper's released defaults for its strongest setting are used as-is:
  num_anchor W = 4, sample_type = 'fps', rot = 10 deg, sca = 3.0, trs = 0.25,
  kernel bandwidth from mean nearest-anchor spacing.
These are fixed in PW_DEFAULT_PARAMS and never selected against a result.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Frozen default parameters (PointWOLF released defaults; NOT tuned here).
# ---------------------------------------------------------------------------
PW_DEFAULT_PARAMS: dict[str, Any] = {
    "algorithm": "PointWOLF",
    "reference": "Kim et al. 2021 (Point Cloud Augmentation with Weighted Local Transformations)",
    "num_anchor": 4,
    "sample_type": "fps",           # farthest-point-sampled anchors (deterministic)
    "rotation_range_deg": 10.0,     # per-axis rotation amplitude
    "scale_range": 3.0,             # anisotropic scale amplitude (paper 'sca')
    "translation_range": 0.25,      # per-axis translation amplitude (normalised frame)
    "kernel": "gaussian_inverse_spacing",
    "normalize_frame": "unit_sphere",  # warp computed in a centred unit-sphere frame
    "adapter_version": "pw-adapter-v1",
}

PW_ADAPTER_VERSION = "pw-adapter-v1"


@dataclass(frozen=True)
class PWParams:
    """Resolved, frozen PointWOLF parameters for one bank build."""
    num_anchor: int = 4
    sample_type: str = "fps"
    rotation_range_deg: float = 10.0
    scale_range: float = 3.0
    translation_range: float = 0.25

    def as_metadata(self) -> dict[str, Any]:
        meta = dict(PW_DEFAULT_PARAMS)
        meta.update(asdict(self))
        return meta


def _farthest_point_sample(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic farthest-point sampling of k anchor indices.

    The first seed index is drawn from ``rng`` so different variants pick
    different anchor sets while staying fully reproducible for a fixed seed.
    """
    n = len(points)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    selected = np.empty(k, dtype=np.int64)
    selected[0] = int(rng.integers(0, n))
    distances = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, k):
        last = points[selected[i - 1]]
        delta = points - last
        d = np.einsum("ij,ij->i", delta, delta)
        distances = np.minimum(distances, d)
        selected[i] = int(np.argmax(distances))
    return selected


def _rotation_matrix(angles_rad: np.ndarray) -> np.ndarray:
    """Compose a 3x3 rotation from per-axis Euler angles (x, y, z)."""
    cx, cy, cz = np.cos(angles_rad)
    sx, sy, sz = np.sin(angles_rad)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def pointwolf_warp(
    points: np.ndarray, params: PWParams, seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the PointWOLF weighted-local-transformation warp to ``points``.

    Returns (warped_xyz float32 [N,3], audit dict).  Point count and order are
    preserved: output row i corresponds to input row i.  Deterministic in
    (points, params, seed).
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"PointWOLF expects an [N,3] cloud, got {pts.shape}")
    n = len(pts)
    if n < 2:
        raise ValueError("PointWOLF needs at least 2 points")
    rng = np.random.default_rng(np.uint64(seed) if seed >= 0 else seed)

    # --- normalise into a centred unit-sphere frame (warp is scale-free here) --
    center = pts.mean(0)
    centred = pts - center
    radius = float(np.max(np.linalg.norm(centred, axis=1)))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Degenerate cloud: zero radius")
    frame = centred / radius  # unit sphere

    # --- anchors ---------------------------------------------------------------
    w = int(params.num_anchor)
    if params.sample_type == "fps":
        anchor_idx = _farthest_point_sample(frame, w, rng)
    else:
        anchor_idx = rng.choice(n, size=min(w, n), replace=False)
    anchors = frame[anchor_idx]
    w = len(anchors)

    # --- per-anchor local transforms ------------------------------------------
    rot = np.deg2rad(params.rotation_range_deg)
    sca = float(params.scale_range)
    trs = float(params.translation_range)
    transformed = np.empty((w, n, 3), dtype=np.float64)
    per_anchor = []
    for j in range(w):
        angles = rng.uniform(-rot, rot, size=3)
        rmat = _rotation_matrix(angles)
        # anisotropic scale in [1/(1+sca), 1+sca]
        s_hi = 1.0 + sca
        s_lo = 1.0 / s_hi
        scale = rng.uniform(s_lo, s_hi, size=3)
        trans = rng.uniform(-trs, trs, size=3)
        local = (frame - anchors[j]) * scale
        local = local @ rmat.T
        transformed[j] = local + anchors[j] + trans
        per_anchor.append({
            "anchor_index": int(anchor_idx[j]),
            "euler_deg": [float(np.rad2deg(a)) for a in angles],
            "scale": [float(v) for v in scale],
            "translation": [float(v) for v in trans],
        })

    # --- distance-based blending weights --------------------------------------
    # sigma from the mean nearest-anchor spacing (kernel = gaussian_inverse_spacing)
    if w > 1:
        aa = anchors[:, None, :] - anchors[None, :, :]
        adist = np.sqrt(np.einsum("ijk,ijk->ij", aa, aa))
        np.fill_diagonal(adist, np.inf)
        sigma = float(np.mean(np.min(adist, axis=1)))
    else:
        sigma = 1.0
    sigma = max(sigma, 1e-6)
    diff = frame[None, :, :] - anchors[:, None, :]      # [w, n, 3]
    sq = np.einsum("wnk,wnk->wn", diff, diff)            # [w, n]
    logits = -sq / (sigma * sigma)
    logits -= logits.max(axis=0, keepdims=True)          # stabilise
    weights = np.exp(logits)
    weights /= weights.sum(axis=0, keepdims=True)         # [w, n], columns sum to 1

    warped_frame = np.einsum("wn,wnk->nk", weights, transformed)  # [n, 3]

    # --- back to original scale/position --------------------------------------
    warped = warped_frame * radius + center
    warped = warped.astype(np.float32)
    if not np.isfinite(warped).all():
        raise ValueError("PointWOLF produced non-finite coordinates")

    audit = {
        "adapter_version": PW_ADAPTER_VERSION,
        "seed": int(seed),
        "num_anchor_effective": int(w),
        "kernel_sigma": sigma,
        "frame_center": [float(v) for v in center],
        "frame_radius": radius,
        "per_anchor_transforms": per_anchor,
        "params": params.as_metadata(),
    }
    return warped, audit


def recompute_normals(
    points: np.ndarray, k: int = 16, orient_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministic kNN-PCA normal estimation on ``points`` ([N,3]).

    For each point, the normal is the eigenvector of the local covariance
    (k nearest neighbours, including self) with the smallest eigenvalue.  If
    ``orient_reference`` (the parent's original per-point normals, same order) is
    given, each recomputed normal's sign is flipped to agree with it (dot >= 0),
    giving a stable, reproducible orientation instead of arbitrary PCA signs.

    Uses scipy.cKDTree when available; otherwise a chunked exact-numpy fallback.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    n = len(pts)
    k = int(min(max(k, 3), n))
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=k)
        if idx.ndim == 1:
            idx = idx[:, None]
    except Exception:
        idx = _knn_indices_numpy(pts, k)

    neighbours = pts[idx]                                  # [n, k, 3]
    mean = neighbours.mean(axis=1, keepdims=True)
    centred = neighbours - mean
    cov = np.einsum("nki,nkj->nij", centred, centred) / float(k)
    eigvals, eigvecs = np.linalg.eigh(cov)                # ascending eigenvalues
    normals = eigvecs[:, :, 0]                            # smallest-eigenvalue vector
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-12)
    if orient_reference is not None:
        ref = np.ascontiguousarray(orient_reference, dtype=np.float64)
        if ref.shape == normals.shape:
            sign = np.sign(np.einsum("ij,ij->i", normals, ref))
            sign[sign == 0] = 1.0
            normals = normals * sign[:, None]
    return normals.astype(np.float32)


def _knn_indices_numpy(pts: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    """Exact kNN indices without scipy (chunked to bound memory)."""
    n = len(pts)
    out = np.empty((n, k), dtype=np.int64)
    sq_norm = np.einsum("ij,ij->i", pts, pts)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = pts[start:stop]
        d = (sq_norm[start:stop, None] - 2.0 * block @ pts.T + sq_norm[None, :])
        out[start:stop] = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    return out
