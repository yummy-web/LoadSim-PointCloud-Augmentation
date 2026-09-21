# Supplementary Experiments (Major Revision)

This directory holds the **experiment code for the supplementary studies** added
during the major revision of our paper:

> **Augmentation Saturation in Physics-Driven UAV Photogrammetric Shovel Segmentation:
> Mechanisms and Cross-Domain Behavior**

> **Where is the main pipeline?** The complete original experiment pipeline
> (data preparation, the base augmentation/quality steps, and the primary training
> scripts) lives in the **root of this same repository** (`../src/`, `../scripts/`,
> `../docs/`). This `supplementary_experiments/` subdirectory contains **only** the
> additional studies introduced in the revision, together with the shared core modules
> they depend on (kept here so the subdirectory runs self-contained).

The code here reproduces the *revision* experiments: the matched-budget comparison and
its extensions (multiple backbones, a point-cloud-specific augmentation baseline), the
cross-domain studies (SemanticKITTI proxy, its budget curve, and the ISPRS Vaihingen 3D
out-of-domain stress test), and the operator-level redundancy analysis.

> **Data policy.** No datasets, pseudo-labels, trained checkpoints, or intermediate
> artifacts are distributed. This repository contains **source code and run
> configurations only.** Paths in the configs (e.g. `manifest/…`, `a3_assets/…`,
> `runs/…`) point to local data/outputs that you provide or generate; they are
> intentionally excluded (see `.gitignore`).

## Repository layout

```
src/         all Python modules (kept flat — the scripts import each other as siblings)
configs/     run configurations, grouped by experiment
  A3MB_five_backbones/   30 configs: 5 backbones x {RAW_REPEAT, LoadSim} x 3 seeds
  PW_pointwolf/          3 configs:  PointWOLF @ B=150 x 3 seeds
  A4_kitti_multiseed/    9 configs:  SemanticKITTI seq-00 proxy, 3 methods x 3 seeds
  A4B_budget_curve/      KITTI budget curve (+ _a4b_holdout/ = budget points not run)
  V3D_vaihingen/         ISPRS Vaihingen 3D (+ _v3d_holdout/ = budget points not run)
  yaml/                  shared protocol/quality/pseudo-label configs
requirements.txt
```

## What maps to which paper section

| Paper section / table | Experiment | Entry point |
|---|---|---|
| §5 Matched-budget (shared engine) | core training engine | `src/a3_train_engine.py`, `src/a3_data.py`, `src/a3_model.py` |
| §5.3 PointWOLF baseline (Table 4) | PW | `src/pw_train.py` (+ `pw_train_engine`, `pw_data`, `pw_adapter`) |
| §8 Five backbones (Table 8) | A3MB | `src/a3mb_train.py` (+ `a3mb_train_engine`, `a3mb_backbones`) |
| §9.2 SemanticKITTI proxy (Table 9) | A4 + multi-seed agg | `src/a4_train.py`, `src/a5_kitti_multiseed_v1.py` |
| §9.3 KITTI budget curve (Table 10) | A4B | `src/a4b_budget_train.py` |
| §9.4 ISPRS Vaihingen 3D (Table 11) | V3D | `src/v3d_budget_train.py` (+ `v3d_io`, `v3d_prepare`) |
| §6.3 Operator redundancy (Table 6) | A5c | `src/a5c_per_group_redundancy_driver.py` |

Config generators (`src/gen_*_configs.py`, `src/generate_configs.py`) regenerate the JSON
files under `configs/` deterministically.

## Environment

- Python 3.10, PyTorch >= 2.0 (CUDA), NumPy, SciPy, Open3D, PyYAML, Matplotlib.
- Install: `pip install -r requirements.txt` (see the file for GPU/CUDA notes).
- Experiments were run on a single NVIDIA GPU.

## How to run

The scripts read a JSON run-config and expect the referenced data manifests/assets to
exist locally (not shipped). General pattern:

```bash
cd src
# example: one five-backbone run
python a3mb_train.py --config ../configs/A3MB_five_backbones/A3MB_pointnet2_LOADSIM_B150_s1.json
# example: PointWOLF baseline
python pw_train.py   --config ../configs/PW_pointwolf/PW_POINTWOLF_B150_s1.json
# example: SemanticKITTI proxy + aggregate
python a4_train.py               --config ../configs/A4_kitti_multiseed/A4_LOADSIM_s1.json
python a5_kitti_multiseed_v1.py  # aggregates the per-seed results into the reported CIs
# example: KITTI budget curve / Vaihingen stress test
python a4b_budget_train.py --config ../configs/A4B_budget_curve/A4B_LOADSIM_B800_s1.json
python v3d_budget_train.py --config ../configs/V3D_vaihingen/V3D_LOADSIM_B40_s1.json
```

Config fields such as `splits`, `source_manifest`, `pseudo_label_manifest`,
`candidate_manifest`, and `output_dir` are relative to `project_root` and must be
populated with your own data layout.

## Reproducibility notes

- Runs are seeded (`seed` field) and hash-verified (`verify_hashes`) against the input
  manifests to guard against silent data drift.
- `_a4b_holdout/` and `_v3d_holdout/` hold budget points that were *defined but not run*
  in the reported experiments; they are kept for transparency, not required to reproduce
  the paper's tables.
- This release corresponds to the revision experiments only. Deprecated/superseded
  variants (e.g. the earlier quality-capacity-limited A3 configuration) are **not**
  included.

## Citation

If you use this code, please cite the paper (and its conference precursor). BibTeX will
be added upon publication.

## License

Released under the MIT License (see `LICENSE`).

