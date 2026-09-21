#!/usr/bin/env python3
"""A5c: per-ablation-operator-group descriptor redundancy (reviewer R3-1).

Regenerates the 7 component-ablation variant groups (FULL / no_removal /
no_slope / no_collapse / only_removal / only_slope / only_collapse) with FIXED
seeds, extracts the same frozen 7-D shape_descriptor used everywhere else
(step2_quality.DiversityEvaluator.descriptor), and computes redundancy PER
GROUP using the EXACT frozen a5b algorithm (a5_descriptor_redundancy_v1.analyze
-> participation ratio, pairwise/nearest-neighbour standardized distances).

WHY (R3-1): the reviewer objects that component ablation only shows "combining
!= better" and does not establish the causal mechanism (correlated deformations
-> super-additive distortion, sub-additive diversity). This measures variant
geometric redundancy DIRECTLY, per operator group, so redundancy can be linked
to the ablation. This run does NOT yet correlate with mIoU (deferred by user).

HONESTY: the original per-variant descriptors (lsda_final/.../quality_report.json)
were DELETED, so variants are REGENERATED with the same protocol and fixed seeds.
These are same-protocol resamples, not the exact deleted training batch. No
training; no GPU; frozen submission assets untouched. Descriptor & redundancy
math are byte-identical to the frozen tools (imported, not reimplemented).
"""
from __future__ import annotations
import sys, json, time, random, argparse
from pathlib import Path
from datetime import datetime
import numpy as np

HERE = Path(__file__).resolve().parent
LSDA = HERE.parent
sys.path.insert(0, str(LSDA))
sys.path.insert(0, str(HERE))

# frozen augmentation + loaders (guard win32 stdout wrapper as in step1_ablation_aug)
_p = sys.platform
try:
    sys.platform = "linux_import_guard"
    from step1_augmentation import _load_ply_numpy, estimate_normals_numpy, preprocess, get_data_config
finally:
    sys.platform = _p
from step1_ablation_aug import augment_one_ablation
from config import ABLATION_CONFIGS, DATASET
from step2_quality import DiversityEvaluator
# EXACT frozen redundancy algorithm (route B), reused verbatim:
from a5_descriptor_redundancy_v1 import analyze as a5b_analyze

SCHEMA_VERSION = "a5c-per-group-redundancy-v1"
BASE_SEED = 20260917


def _seed(group_idx: int, src_idx: int, var_id: int) -> int:
    return (BASE_SEED + group_idx * 100003 + src_idx * 997 + var_id) & 0x7FFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(LSDA / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "a5c_per_group_redundancy_v1"))
    ap.add_argument("--variants", type=int, default=16, help="variants per source per group")
    ap.add_argument("--groups", type=str, default="", help="comma-sep group indices 0-6; empty=all")
    ap.add_argument("--sources", type=str, default="", help="comma-sep source stems (e.g. 001,DJI_1); empty=all train_pool")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pool = set(DATASET["train_pool"])  # 11 files; excludes test_set 002/DJI_3/DJI_7
    ply_files = [f for f in sorted(data_dir.glob("*.ply")) if f.stem in train_pool]
    assert ply_files, f"no PLY in {data_dir}"
    missing = train_pool - {f.stem for f in ply_files}
    assert not missing, f"train_pool files missing on disk: {sorted(missing)}"
    print(f"[split] using {len(ply_files)} train_pool files; excluding test_set "
          f"{sorted(set(DATASET['test_set']))}", flush=True)

    all_stems = [f.stem for f in ply_files]  # canonical full source set (11)
    want_src = set(s for s in args.sources.split(",") if s.strip()) if args.sources else set(all_stems)
    if args.sources:
        ply_files = [f for f in ply_files if f.stem in want_src]
    cache_dir = out_dir / "_desc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dv = DiversityEvaluator()
    t_start = time.time()

    # preload each (selected) source once (load + preprocess + ensure normals)
    print(f"[preload] {len(ply_files)} sources", flush=True)
    sources = []  # (name, pts, normals)
    for f in ply_files:
        pts, nrm = _load_ply_numpy(f)
        if pts is None or len(pts) == 0:
            print(f"  skip {f.name}: empty", flush=True)
            continue
        pts, nrm, _ = preprocess(pts, nrm)
        if nrm is None:
            nrm = estimate_normals_numpy(pts, k=20)
        sources.append((f.name, pts, nrm))
        print(f"  {f.name}: {len(pts):,} pts", flush=True)

    group_names = list(ABLATION_CONFIGS.keys())
    sel = set(int(x) for x in args.groups.split(",") if x.strip() != "") if args.groups else set(range(len(group_names)))
    summary = {}
    for gi, gname in enumerate(group_names):
        if gi not in sel:
            continue
        en_rem, en_sl, en_col, label_cn, label_en = ABLATION_CONFIGS[gname]
        g0 = time.time()
        # PHASE 1 (generate): compute+cache per-source descriptors for this group's selected sources
        for (sname, pts, nrm) in sources:
            src_stem = Path(sname).stem
            cache_f = cache_dir / f"{gname}__{src_stem}__v{args.variants}.json"
            if cache_f.exists():
                print(f"  cache-hit {gname}/{src_stem}", flush=True)
                continue
            si = all_stems.index(src_stem)  # stable seed index tied to canonical order
            recs = []
            for v in range(args.variants):
                s = _seed(gi, si, v)
                random.seed(s)
                np.random.seed(s)
                d_pts, meta = augment_one_ablation(
                    pts, nrm, sname, v, get_data_config(sname),
                    enable_directional_removal=en_rem,
                    enable_slope_reshaping=en_sl,
                    enable_lateral_collapse=en_col,
                )
                desc = dv.descriptor(d_pts)
                recs.append({
                    "variant_id": f"{src_stem}_{gname}_v{v:03d}",
                    "source": src_stem,
                    "descriptor": [float(x) for x in desc],
                })
            cache_f.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
            print(f"  wrote {gname}/{src_stem} ({len(recs)} variants, {time.time()-g0:.1f}s cum)", flush=True)

        # PHASE 2 (analyze): only if ALL canonical sources cached for this group
        have = {p.stem.split("__")[1] for p in cache_dir.glob(f"{gname}__*__v{args.variants}.json")}
        if not set(all_stems).issubset(have):
            missing = sorted(set(all_stems) - have)
            print(f"[{gi+1}/7] {gname}: PARTIAL, cached {len(have)}/{len(all_stems)}; missing {missing}", flush=True)
            continue
        variants = []
        for st in all_stems:
            variants.extend(json.loads((cache_dir / f"{gname}__{st}__v{args.variants}.json").read_text(encoding="utf-8")))
        rep = a5b_analyze(variants, gname)
        rep.update({
            "schema_version": SCHEMA_VERSION,
            "ablation_group": gname,
            "label_en": label_en,
            "label_cn": label_cn,
            "components": {"directional_removal": en_rem,
                          "slope_reshaping": en_sl,
                          "lateral_collapse": en_col},
            "variants_per_source": args.variants,
            "n_sources": len(all_stems),
            "base_seed": BASE_SEED,
            "is_synthetic_placeholder": False,
            "regeneration_note": ("Variants REGENERATED (same protocol, fixed seeds) "
                                  "because original per-variant descriptors were deleted; "
                                  "same-protocol resample, not the exact deleted batch."),
        })
        (out_dir / f"redundancy_{gname}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        o = rep["overall_redundancy"]
        summary[gname] = {
            "label_en": label_en, "n_variants": rep["n_variants"],
            "participation_ratio": o.get("participation_ratio"),
            "effective_dim_fraction": o.get("effective_dim_fraction"),
            "pairwise_mean": o.get("pairwise_mean"),
            "pairwise_p05": o.get("pairwise_p05"),
            "nearest_neighbour_mean": o.get("nearest_neighbour_mean"),
            "top2_variance_fraction": o.get("top2_variance_fraction"),
        }
        print(f"[{gi+1}/7] {gname}: n={rep['n_variants']} PR={o.get('participation_ratio'):.3f} "
              f"nn={o.get('nearest_neighbour_mean'):.4f} pw={o.get('pairwise_mean'):.4f} "
              f"({time.time()-g0:.1f}s)", flush=True)

    if summary:
        (out_dir / "summary_all_groups.json").write_text(
            json.dumps({"schema_version": SCHEMA_VERSION,
                        "generated_at": str(datetime.now()),
                        "elapsed_seconds": round(time.time() - t_start, 1),
                        "groups": summary}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    print(f"\nDONE in {time.time()-t_start:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
