# Pipeline Documentation

This document describes the LoadSim experiment pipeline in detail. All stages read
configuration from `src/config.py`.

## Experiment design

**Phase 1 — original experiment**
7 training branches × 4-fold cross-validation = 28 models:
`aug_loadsim`, `aug_lsda`, `aug_trad`, `loadsim_11/150`, `lsda_11/150`,
`trad_11/150`, plus a non-learning computer-vision baseline.

**Phase 2 — journal extension**
- LoadSim component ablation: 7 on/off configurations × 4 folds.
- Multi-backbone comparison: PointNet++ / PointNeXt / KPConv / PTv3 / RandLA-Net
  × data branches × 4 folds.

## Dataset interface

The code expects point clouds in PLY format with per-point `xyz` coordinates and
surface normals (estimated via PCA over k=20 neighbors). Coordinates are in
centimeters. The dataset split is defined in `config.py`:

- `test_set`  — permanently held-out files.
- `train_pool` — files used for 4-fold macro cross-validation.

> The actual scans used in the paper are confidential and not distributed. Place
> your own scans under `data/` and update `config.py` accordingly.

## Stage reference

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| 1 | `step1_augmentation.py` | raw PLY | augmented variants |
| 1b | `step1_ablation_aug.py` | raw PLY | 7 ablation configs |
| 2 | `step2_quality.py` | variants | quality-filtered set + scores |
| 3 | `step3_labels.py` | clouds | pseudo-label JSON |
| 4 | `step4_train.py` / `step4_train_backbones.py` | labeled clouds | model checkpoints + history |
| 5 | `step5_evaluate_fixed.py` | checkpoints | test metrics (mIoU/F1/...) |
| 6 | `step6_ablation_analysis.py` | results | aggregated ablation report |

## Quality framework

```
Q = 0.40·F + 0.35·P + 0.25·D
```

- **F (Fidelity):** Chamfer-distance ratio, Hausdorff distance, volume-change rate, feature similarity.
- **P (Physical plausibility):** angle-of-repose conformance, height-to-width ratio, surface continuity.
- **D (Diversity):** marginal contribution to ensemble diversity.

Acceptance: `Q ≥ 0.55 AND F ≥ 0.45 AND P ≥ 0.50`.

## SemanticKITTI experiment

See `src/kitti/`. Sequence 00 is split chronologically (train 0000–3199,
val 3200–3799, test 3800–4540). Augmented variants inherit their source frame's
time index to prevent temporal leakage. Configure the dataset path in
`config_kitti.py`.
