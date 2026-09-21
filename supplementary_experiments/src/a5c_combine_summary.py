#!/usr/bin/env python3
"""Combine the 7 per-group redundancy JSONs into one summary + print table."""
import json, sys
from pathlib import Path
d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "a5c_per_group_redundancy_v1"
order = ["ablation_full","ablation_no_removal","ablation_no_slope","ablation_no_collapse",
         "ablation_only_removal","ablation_only_slope","ablation_only_collapse"]
groups = {}
vps_vals, nsrc_vals = set(), set()
for g in order:
    p = d / f"redundancy_{g}.json"
    if not p.exists():
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    o = r["overall_redundancy"]
    if "variants_per_source" in r:
        vps_vals.add(r["variants_per_source"])
    if "n_sources" in r:
        nsrc_vals.add(r["n_sources"])
    groups[g] = {
        "label_en": r["label_en"], "components": r["components"], "n_variants": r["n_variants"],
        "participation_ratio": o.get("participation_ratio"),
        "effective_dim_fraction": o.get("effective_dim_fraction"),
        "pairwise_mean": o.get("pairwise_mean"), "pairwise_p05": o.get("pairwise_p05"),
        "nearest_neighbour_mean": o.get("nearest_neighbour_mean"),
        "top2_variance_fraction": o.get("top2_variance_fraction"),
    }
vps = (vps_vals.pop() if len(vps_vals) == 1 else sorted(vps_vals)) if vps_vals else None
nsrc = (nsrc_vals.pop() if len(nsrc_vals) == 1 else sorted(nsrc_vals)) if nsrc_vals else None
out = {"schema_version": "a5c-per-group-redundancy-v1", "n_groups_present": len(groups),
       "variants_per_source": vps, "n_sources": nsrc, "groups": groups}
(d / "summary_all_groups.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"{'group':24}{'rem/slp/col':>13} {'n':>4} {'PR':>6} {'effdim':>7} {'nn_mean':>8} {'pw_mean':>8} {'top2':>6}")
for g, v in groups.items():
    c = v["components"]
    flags = f"{int(c['directional_removal'])}/{int(c['slope_reshaping'])}/{int(c['lateral_collapse'])}"
    print(f"{g:24}{flags:>13} {v['n_variants']:>4} {v['participation_ratio']:>6.3f} "
          f"{v['effective_dim_fraction']:>7.3f} {v['nearest_neighbour_mean']:>8.4f} "
          f"{v['pairwise_mean']:>8.4f} {v['top2_variance_fraction']:>6.3f}")
