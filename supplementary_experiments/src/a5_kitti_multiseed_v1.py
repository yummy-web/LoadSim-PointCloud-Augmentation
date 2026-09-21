#!/usr/bin/env python3
"""A4 SemanticKITTI multi-seed statistics (reviewer comment 3).

Reports per-method mean +/- SD and small-sample (Student-t) CI over the 3 frozen
seeds, plus paired BASELINE-vs-LOADSIM and BASELINE-vs-TRADITIONAL comparisons.
Honest framing: 3 seeds is a tiny sample; CIs are wide and this is a proxy-domain
robustness study, NOT a negative-transfer-free claim. Read-only; no training; no
authorization; never touches frozen assets.

INPUT: long CSV with columns method,seed,mIoU (extra columns kept, ignored).
Methods expected in {BASELINE,TRADITIONAL,LOADSIM}; 3 seeds each = 9 rows.

USAGE:
  python a5_kitti_multiseed_v1.py --results a4_results_long.csv --json-output out.json
  python a5_kitti_multiseed_v1.py --synthetic --json-output out.json

STATUS: scaffold verified on synthetic data; swap --results for the real A4
aggregate once the 9-run completes. Synthetic mode is clearly flagged.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "a5-kitti-multiseed-v1"
METHODS = ("BASELINE", "TRADITIONAL", "LOADSIM")
REQUIRED_COLUMNS = ("method", "seed", "mIoU")
# Two-sided t critical values by degrees of freedom (df=n-1), 95% CI.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


class KittiStatError(RuntimeError):
    pass
def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise KittiStatError(f"Empty CSV: {path}")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise KittiStatError(f"Missing required column(s) {missing}: {path}")
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for i, raw in enumerate(reader, 2):
            method = (raw.get("method") or "").strip()
            if not method:
                raise KittiStatError(f"{path}:{i} empty method")
            seed = (raw.get("seed") or "").strip()
            if not seed:
                raise KittiStatError(f"{path}:{i} empty seed")
            try:
                miou = float(raw.get("mIoU"))
            except (TypeError, ValueError):
                raise KittiStatError(f"{path}:{i} mIoU not a float: {raw.get('mIoU')!r}")
            if not math.isfinite(miou) or not 0.0 <= miou <= 1.0:
                raise KittiStatError(f"{path}:{i} mIoU out of [0,1] or non-finite: {miou}")
            identity = (method, seed)
            if identity in seen:
                raise KittiStatError(f"{path}:{i} duplicate (method,seed)={identity}")
            seen.add(identity)
            rows.append({"method": method, "seed": seed, "mIoU": miou})
    if not rows:
        raise KittiStatError(f"No data rows: {path}")
    return rows


def _mean_sd(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    mu = math.fsum(xs) / n
    if n < 2:
        return mu, 0.0
    var = math.fsum((x - mu) ** 2 for x in xs) / (n - 1)  # sample SD
    return mu, math.sqrt(var)


def _t_ci(xs: Sequence[float]) -> dict[str, Any]:
    """Student-t 95% CI of the mean. n<2 -> undefined half-width."""
    n = len(xs)
    mu, sd = _mean_sd(xs)
    if n < 2:
        return {"mean": mu, "sd": sd, "ci95_low": None, "ci95_high": None,
                "ci95_halfwidth": None, "note": "n<2: CI undefined"}
    df = n - 1
    tcrit = T95.get(df)
    if tcrit is None:
        return {"mean": mu, "sd": sd, "ci95_low": None, "ci95_high": None,
                "ci95_halfwidth": None, "note": f"df={df} outside t-table"}
    half = tcrit * sd / math.sqrt(n)
    return {"mean": mu, "sd": sd, "ci95_low": mu - half, "ci95_high": mu + half,
            "ci95_halfwidth": half, "t_crit": tcrit, "df": df}
def _paired_compare(by_seed_a: dict[str, float], by_seed_b: dict[str, float],
                    label_a: str, label_b: str) -> dict[str, Any]:
    """Paired comparison across shared seeds (a - b). Paired t on differences.
    Honest: with 3 seeds this is exploratory; report the raw per-seed diffs too."""
    shared = sorted(set(by_seed_a) & set(by_seed_b))
    if len(shared) < 2:
        return {"comparison": f"{label_a}_minus_{label_b}", "shared_seeds": shared,
                "verdict": "insufficient_shared_seeds"}
    diffs = [by_seed_a[s] - by_seed_b[s] for s in shared]
    mu, sd = _mean_sd(diffs)
    ci = _t_ci(diffs)
    # sign consistency: does the difference keep the same sign across all seeds?
    all_pos = all(d > 0 for d in diffs)
    all_neg = all(d < 0 for d in diffs)
    if ci.get("ci95_low") is not None and (ci["ci95_low"] > 0 or ci["ci95_high"] < 0):
        verdict = "difference_excludes_zero_at_95pct"
    elif all_pos or all_neg:
        verdict = "consistent_sign_but_ci_includes_zero"
    else:
        verdict = "inconsistent_sign"
    return {
        "comparison": f"{label_a}_minus_{label_b}",
        "shared_seeds": shared,
        "per_seed_diff": {s: by_seed_a[s] - by_seed_b[s] for s in shared},
        "mean_diff": mu, "sd_diff": sd,
        "ci95_low": ci.get("ci95_low"), "ci95_high": ci.get("ci95_high"),
        "verdict": verdict,
        "interpretation": (
            "positive mean_diff => first method higher mIoU. CI excluding 0 is the only "
            "statistically supported directional claim; with 3 seeds CIs are wide, so "
            "report as proxy-domain robustness, not negative-transfer-free."),
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, dict[str, float]] = {}
    for r in rows:
        by_method.setdefault(r["method"], {})[r["seed"]] = r["mIoU"]
    per_method = []
    for m in sorted(by_method):
        vals = list(by_method[m].values())
        stat = _t_ci(vals)
        stat.update({"method": m, "n_seeds": len(vals),
                     "seeds": sorted(by_method[m]),
                     "min": min(vals), "max": max(vals)})
        per_method.append(stat)
    comparisons = []
    if "BASELINE" in by_method:
        for other in ("LOADSIM", "TRADITIONAL"):
            if other in by_method:
                comparisons.append(_paired_compare(
                    by_method[other], by_method["BASELINE"], other, "BASELINE"))
    return {"per_method": per_method, "comparisons": comparisons,
            "methods_present": sorted(by_method)}
def _synthetic_rows() -> list[dict[str, Any]]:
    """Placeholder ONLY. Mirrors the manuscript's concern: BASELINE slightly ABOVE
    LOADSIM (0.7863 vs 0.7713), TRADITIONAL between. Deterministic, no RNG."""
    data = {
        "BASELINE": {"s1": 0.7863, "s2": 0.7841, "s3": 0.7885},
        "TRADITIONAL": {"s1": 0.7802, "s2": 0.7788, "s3": 0.7815},
        "LOADSIM": {"s1": 0.7713, "s2": 0.7699, "s3": 0.7728},
    }
    return [{"method": m, "seed": s, "mIoU": v}
            for m, seeds in data.items() for s, v in seeds.items()]


def run(results_path: Path | None, synthetic: bool) -> dict[str, Any]:
    if synthetic:
        rows = _synthetic_rows()
        source = "SYNTHETIC_PLACEHOLDER"
    else:
        if results_path is None:
            raise KittiStatError("either --results or --synthetic is required")
        rows = load_results(results_path)
        source = str(results_path)
    analysis = analyze(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "data_source": source,
        "is_synthetic_placeholder": synthetic,
        "n_rows": len(rows),
        "reviewer_comment": "R3 SemanticKITTI multi-seed statistics / negative-transfer claim boundary",
        "framing": ("Proxy-domain robustness study with limitations; 3 seeds -> wide CIs; "
                    "do NOT claim negative-transfer-free."),
        **analysis,
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, help="Long CSV method,seed,mIoU")
    p.add_argument("--synthetic", action="store_true",
                   help="Use flagged synthetic placeholder data to verify the scaffold")
    p.add_argument("--json-output", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args.results, args.synthetic)
        out = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        args.json_output.write_text(out, encoding="utf-8")
    except (KittiStatError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.json_output}  (source={report['data_source']})")
    for c in report.get("comparisons", []):
        print(f"  {c['comparison']}: mean_diff="
              f"{c.get('mean_diff')!r} verdict={c['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

