#!/usr/bin/env python3
"""A5b (route B): geometric redundancy of augmentation variants from frozen
shape descriptors (reviewer comment 2, augmentation-saturation mechanism).

This is the DEGRADED / route-B analysis chosen by the user: instead of running
a controlled REMOVAL/SLOPE/COLLAPSE ablation (whose variants do not exist), it
quantifies how geometrically REDUNDANT the frozen augmentation variants are,
per method, directly from the 7-D per-variant `shape_descriptor` already stored
in each method's quality_report.json (288 variants x {trad, loadsim}).

Mechanism link to "augmentation saturation": if additional variants are
geometrically near-duplicates of ones already present (small pairwise descriptor
distances, low effective dimensionality), then adding more variants injects
little new geometric information -> diminishing returns == saturation.

NON-CIRCULARITY: the pipeline's diversity_contribution D is itself the mean
absolute z-score of the descriptor, so this tool NEVER uses D to "predict"
redundancy. It uses only the raw 7-D descriptors and computes independent
redundancy statistics (pairwise standardized distances, nearest-neighbour
redundancy, participation-ratio effective dimensionality). Read-only over the
frozen quality_report.json; no training; no authorization; frozen assets
untouched. Pure stdlib (no numpy/scipy) for auditability.

USAGE:
  python a5_descriptor_redundancy_v1.py --quality-report trad_quality_report.json \
      --method TRAD --json-output out.json
  python a5_descriptor_redundancy_v1.py --synthetic --json-output out.json
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "a5-descriptor-redundancy-v1"
DESCRIPTOR_DIM = 7
DESCRIPTOR_SEMANTICS = (
    "xz_aspect(dx/dz)", "yz_aspect(dy/dz)", "z_span", "eigen_anisotropy(e0/e2)",
    "radial_cov(std/mean)", "z_skewness", "centroid_z",
)


class DescriptorRedundancyError(RuntimeError):
    pass


def _finite(x: Any, label: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise DescriptorRedundancyError(f"{label}: not a float: {x!r}") from exc
    if not math.isfinite(v):
        raise DescriptorRedundancyError(f"{label}: non-finite {x!r}")
    return v
def load_variants(path: Path, method: str) -> list[dict[str, Any]]:
    """Read quality_report.json -> list of {variant_id, source, descriptor[7]}."""
    d = json.loads(path.read_text(encoding="utf-8"))
    results = d.get("results")
    if not isinstance(results, list) or not results:
        raise DescriptorRedundancyError(f"{path}: no results array")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, r in enumerate(results):
        vid = str(r.get("variant_id") or "").strip()
        if not vid:
            raise DescriptorRedundancyError(f"{path}[{i}]: empty variant_id")
        if vid in seen:
            raise DescriptorRedundancyError(f"{path}: duplicate variant_id {vid}")
        seen.add(vid)
        desc = r.get("shape_descriptor")
        if not isinstance(desc, list) or len(desc) != DESCRIPTOR_DIM:
            raise DescriptorRedundancyError(
                f"{path}:{vid}: shape_descriptor must be length {DESCRIPTOR_DIM}")
        vec = [_finite(x, f"{vid}[{j}]") for j, x in enumerate(desc)]
        out.append({"variant_id": vid, "source": str(r.get("source") or ""),
                    "descriptor": vec})
    return out


def _standardize(vectors: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Per-dimension z-score standardization; degenerate dims (sd=0) -> sd=1."""
    n = len(vectors)
    mu = [math.fsum(v[j] for v in vectors) / n for j in range(DESCRIPTOR_DIM)]
    sd = []
    for j in range(DESCRIPTOR_DIM):
        var = math.fsum((v[j] - mu[j]) ** 2 for v in vectors) / (n - 1) if n > 1 else 0.0
        s = math.sqrt(var)
        sd.append(s if s >= 1e-12 else 1.0)
    z = [[(v[j] - mu[j]) / sd[j] for j in range(DESCRIPTOR_DIM)] for v in vectors]
    return z, mu, sd


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(math.fsum((a[j] - b[j]) ** 2 for j in range(len(a))))


def _pairwise_stats(z: list[list[float]]) -> dict[str, Any]:
    """Mean/median/min pairwise standardized Euclidean distance + nearest-neighbour."""
    n = len(z)
    if n < 2:
        return {"n": n, "note": "need >=2 variants"}
    dists = []
    nn = [math.inf] * n
    for i in range(n):
        for k in range(i + 1, n):
            dd = _euclidean(z[i], z[k])
            dists.append(dd)
            if dd < nn[i]:
                nn[i] = dd
            if dd < nn[k]:
                nn[k] = dd
    dists.sort()
    m = len(dists)
    median = dists[m // 2] if m % 2 else (dists[m // 2 - 1] + dists[m // 2]) / 2
    nn_sorted = sorted(nn)
    nn_med = nn_sorted[n // 2] if n % 2 else (nn_sorted[n // 2 - 1] + nn_sorted[n // 2]) / 2
    return {
        "n_variants": n,
        "pairwise_mean": math.fsum(dists) / m,
        "pairwise_median": median,
        "pairwise_min": dists[0],
        "pairwise_p05": dists[max(0, int(0.05 * m))],
        "nearest_neighbour_mean": math.fsum(nn) / n,
        "nearest_neighbour_median": nn_med,
        "nearest_neighbour_min": min(nn),
    }


def _participation_ratio(z: list[list[float]]) -> dict[str, Any]:
    """Effective dimensionality via covariance eigen participation ratio.
    PR = (sum eig)^2 / sum(eig^2) in [1, 7]; low PR => variants vary along few
    axes => geometrically redundant. Eigenvalues via Jacobi on 7x7 covariance."""
    n = len(z)
    if n < 2:
        return {"note": "need >=2 variants"}
    cov = [[math.fsum(z[r][i] * z[r][k] for r in range(n)) / (n - 1)
            for k in range(DESCRIPTOR_DIM)] for i in range(DESCRIPTOR_DIM)]
    eig = _jacobi_eigenvalues(cov)
    eig = [max(0.0, e) for e in eig]
    s1 = math.fsum(eig)
    s2 = math.fsum(e * e for e in eig)
    pr = (s1 * s1 / s2) if s2 > 0 else float("nan")
    total = s1 if s1 > 0 else 1.0
    return {
        "eigenvalues_desc": sorted(eig, reverse=True),
        "participation_ratio": pr,
        "effective_dim_fraction": pr / DESCRIPTOR_DIM,
        "top1_variance_fraction": max(eig) / total,
        "top2_variance_fraction": sum(sorted(eig, reverse=True)[:2]) / total,
    }


def _jacobi_eigenvalues(a: list[list[float]], iters: int = 100) -> list[float]:
    """Symmetric Jacobi eigenvalue iteration (small fixed 7x7)."""
    n = len(a)
    m = [row[:] for row in a]
    for _ in range(iters):
        off = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(m[i][j]) > off:
                    off = abs(m[i][j]); p, q = i, j
        if off < 1e-12:
            break
        app, aqq, apq = m[p][p], m[q][q], m[p][q]
        phi = 0.5 * math.atan2(2 * apq, aqq - app) if aqq != app else math.pi / 4
        c, s = math.cos(phi), math.sin(phi)
        for k in range(n):
            mkp, mkq = m[k][p], m[k][q]
            m[k][p] = c * mkp - s * mkq
            m[k][q] = s * mkp + c * mkq
        for k in range(n):
            mpk, mqk = m[p][k], m[q][k]
            m[p][k] = c * mpk - s * mqk
            m[q][k] = s * mpk + c * mqk
    return [m[i][i] for i in range(n)]


def analyze(variants: list[dict[str, Any]], method: str) -> dict[str, Any]:
    vectors = [v["descriptor"] for v in variants]
    z, mu, sd = _standardize(vectors)
    overall = {**_pairwise_stats(z), **_participation_ratio(z)}
    # per source-scan stratum (redundancy WITHIN a parent scan's variants)
    by_source: dict[str, list[list[float]]] = {}
    for v, zz in zip(variants, z):
        by_source.setdefault(v["source"], []).append(zz)
    per_source = []
    for src in sorted(by_source):
        zz = by_source[src]
        per_source.append({"source": src, "n": len(zz),
                           **_pairwise_stats(zz)})
    return {
        "method": method,
        "n_variants": len(variants),
        "descriptor_semantics": list(DESCRIPTOR_SEMANTICS),
        "standardization": {"mean": mu, "sd": sd},
        "overall_redundancy": overall,
        "per_source_redundancy": per_source,
        "interpretation": (
            "Lower pairwise/nearest-neighbour standardized distance and lower "
            "participation ratio (effective dim << 7) => variants are geometrically "
            "redundant, so additional variants add little new shape information "
            "-> mechanism consistent with augmentation saturation. Compare TRAD vs "
            "LOADSIM: the method whose variants are MORE redundant saturates sooner."),
    }


def _synthetic() -> list[dict[str, Any]]:
    """Placeholder ONLY: two clusters, verifies pipeline (clearly flagged)."""
    import random
    rng = random.Random(20260914)
    out = []
    for i in range(20):
        base = [1.0, 1.0, 400.0, 20.0, 0.4, 0.1, 0.5]
        jit = [b + rng.gauss(0, 0.01 * (abs(b) + 1)) for b in base]
        out.append({"variant_id": f"syn_{i:03d}", "source": f"s{i % 4}", "descriptor": jit})
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.synthetic:
        variants = _synthetic()
        method = "SYNTHETIC"
        source = "SYNTHETIC_PLACEHOLDER"
        input_sha = None
    else:
        if args.quality_report is None or args.method is None:
            raise DescriptorRedundancyError("need --quality-report and --method (or --synthetic)")
        import hashlib
        source = str(args.quality_report)
        input_sha = hashlib.sha256(args.quality_report.read_bytes()).hexdigest()
        variants = load_variants(args.quality_report, args.method)
        method = args.method
    report = analyze(variants, method)
    report.update({
        "schema_version": SCHEMA_VERSION,
        "data_source": source,
        "input_sha256": input_sha,
        "is_synthetic_placeholder": bool(args.synthetic),
        "reviewer_comment": "R2 augmentation-saturation mechanism via variant geometric redundancy",
        "non_circularity_note": (
            "Uses only raw 7-D shape_descriptor; never uses diversity_contribution D "
            "(which is itself derived from the descriptor) to predict redundancy."),
    })
    return report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quality-report", type=Path)
    p.add_argument("--method", type=str, help="e.g. TRAD / LOADSIM")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--json-output", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    except (DescriptorRedundancyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    o = report["overall_redundancy"]
    print(f"Wrote {args.json_output}  (method={report['method']} n={report['n_variants']})")
    print(f"  pairwise_mean={o.get('pairwise_mean'):.4f} nn_mean={o.get('nearest_neighbour_mean'):.4f} "
          f"PR={o.get('participation_ratio'):.3f}/{DESCRIPTOR_DIM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
