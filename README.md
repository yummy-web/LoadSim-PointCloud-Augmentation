# LoadSim: Physics-Driven Loading Simulation Augmentation for LiDAR Point Cloud Segmentation

Official implementation of **"Physics-Driven Loading Simulation Augmentation for
LiDAR-Based Shovel Region Segmentation: Interaction Mechanisms and Cross-Domain
Generalization"** (submitted to *Pattern Recognition*).

LoadSim is a physics-driven data-augmentation method for 3D point cloud semantic
segmentation in data-scarce industrial scenarios. Instead of injecting random
geometric noise, it simulates the mechanics of bucket-loader excavation —
**directional material removal**, **slope reshaping**, and **lateral collapse** —
to synthesize physically plausible bulk-material pile geometries. It is paired
with a multi-dimensional augmentation quality framework and a geometry-based
pseudo-labeling algorithm, eliminating the need for manual annotation.

## Highlights

- **Physics-driven augmentation (LoadSim):** 100% quality pass rate and +20.8%
  fidelity over noise-based augmentation.
- **Augmentation quality framework:** composite score `Q = 0.40·F + 0.35·P + 0.25·D`
  (fidelity / physical plausibility / diversity) filters implausible variants
  before training.
- **Annotation-free pseudo-labels:** a three-condition geometric rule (frontier,
  slope band, local prominence) derived from the granular angle of repose.
- **Augmentation saturation effect:** a systematic component ablation showing the
  full pipeline can underperform its single-component variants.
- **Architecture–augmentation interaction:** validated across five backbones
  (PointNet++, PointNeXt, KPConv, PTv3, RandLA-Net); augmentation benefit is
  negatively correlated with a backbone's geometric inductive bias.
- **Cross-domain study:** SemanticKITTI experiment delineating the application
  boundary of physics-driven augmentation.

## Data availability

> **Important.** The bulk-material pile point cloud data used in the paper are
> subject to confidentiality restrictions and are **NOT** included in this
> repository. This release contains **source code only**.
>
> To reproduce results you must supply your own point cloud scans (PLY format with
> `xyz` + surface normals) under a local `data/` directory, or adapt the data
> loaders in `src/`. The **SemanticKITTI** experiment uses the publicly available
> [SemanticKITTI dataset](http://www.semantic-kitti.org/).

### Expected data format

Place input scans under `data/` (or pass `--data-dir`). Each scan is a **PLY file**
(ASCII or binary) with the following per-point fields:

| Field        | Type  | Meaning                                   |
|--------------|-------|-------------------------------------------|
| `x, y, z`    | float | Point coordinates, in **centimetres**     |
| `nx, ny, nz` | float | Unit surface normals (optional)           |

If normals are absent they are estimated automatically by PCA over the `k = 20`
nearest neighbours at load time. The default split in `src/config.py` expects five
sparse scans (`001`–`005`) and nine dense scans (`DJI_1`–`DJI_9`), with
`002, DJI_3, DJI_7` held out as the fixed test set and the remaining eleven used for
4-fold cross-validation; edit `DATASET` / `CV_SPLITS` in `src/config.py` to match
your own file names.

For the SemanticKITTI experiment, set the dataset location via the `KITTI_ROOT`
environment variable (defaults to `./data/kitti/sequences`), or pass the path to
`src/kitti/kitti_train.py`:

```bash
export KITTI_ROOT=/path/to/semantic-kitti/dataset/sequences
```

## Repository structure

```
.
├── src/                          # Core pipeline (run in numbered order)
│   ├── config.py                 # Global configuration: dataset split, CV folds, hyperparameters
│   ├── step1_augmentation.py     # LoadSim / LSDA / conventional augmentation generation
│   ├── step1_ablation_aug.py     # Component-ablation augmentation (7 on/off configurations)
│   ├── step2_quality.py          # Multi-dimensional quality framework (Q = 0.40F+0.35P+0.25D)
│   ├── step3_labels.py           # Geometry-based pseudo-label generation
│   ├── step3_labels_annotated.py # Annotated/enhanced pseudo-label pipeline
│   ├── step4_train.py            # PointNet++ reference training
│   ├── step4_train_backbones.py  # Multi-backbone training (PointNeXt/KPConv/PTv3/RandLA-Net)
│   ├── step5_evaluate.py         # Evaluation metrics (mIoU/F1/recall/precision)
│   ├── step5_evaluate_fixed.py   # Fixed-test-set evaluation
│   ├── step5_visualize_labels.py # Qualitative prediction visualization
│   ├── step6_ablation_analysis.py# Ablation result aggregation/analysis
│   ├── plot_journal_figures.py   # Original figure generation
│   ├── benchmark_inference.py    # Inference-speed benchmarking
│   └── kitti/                    # SemanticKITTI cross-domain experiment
│       ├── config_kitti.py
│       ├── kitti_dataset.py
│       ├── kitti_train.py
│       └── kitti_visualize.py
├── scripts/                      # End-to-end orchestration
│   ├── run_all.py                # Original experiment pipeline
│   ├── run_all_v2.py             # Pipeline v2
│   └── run_journal.py            # Journal-extension experiments (ablation + multi-backbone)
├── figures/
│   └── make_pr_figures.py        # Publication figures (vector PDF + 300 dpi JPG, no titles)
├── docs/
│   └── PIPELINE.md               # Detailed pipeline / stage documentation
├── requirements.txt
├── LICENSE                       # MIT
└── README.md
```

## Installation

```bash
git clone https://github.com/yummy-web/LoadSim-PointCloud-Augmentation.git
cd LoadSim-PointCloud-Augmentation
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training. Install the appropriate PyTorch
build for your CUDA version (see the comment in `requirements.txt`).

## Quick start

The pipeline runs in numbered stages. Provide your own point clouds under `data/`
(PLY with `xyz` + normals), then:

```bash
# 1. Generate augmented variants (LoadSim / LSDA / conventional)
python src/step1_augmentation.py

# 2. Score and filter variants by the quality framework
python src/step2_quality.py

# 3. Generate geometry-based pseudo-labels
python src/step3_labels.py

# 4a. Train the PointNet++ reference model
python src/step4_train.py

# 4b. (Optional) Multi-backbone validation
python src/step4_train_backbones.py --backbone ptv3

# 5. Evaluate on the held-out test set
python src/step5_evaluate_fixed.py

# 6. Aggregate component-ablation results
python src/step6_ablation_analysis.py
```

Or drive the whole study through the orchestration scripts:

```bash
python scripts/run_all.py        # original experiment
python scripts/run_journal.py    # ablation + multi-backbone extension
```

Generate the publication figures (writes vector PDF + 300 dpi JPG):

```bash
python figures/make_pr_figures.py --outputs-root ./outputs4 \
    --viz-root ./outputs4/visualization --out-dir ./figures_out
```

## Method overview

LoadSim simulates `K ∈ {2,3,4}` independent loading operations per variant, with
approach directions sampled from 16 uniformly spaced angles over `[0, 2π)` plus 8
random angles. Each operation proceeds through five physically interpretable
stages:

1. **Frontier localization** — identify the pile face the bucket first contacts.
2. **Scoop-center sampling** — normal-weighted sampling favoring outward, near-vertical surfaces.
3. **Directional material removal** — remove points within a `[0.8, 1.5] m` bucket-width band.
4. **Slope reshaping** — re-profile the cut so local inclination respects the angle of repose.
5. **Lateral collapse** — adjacent material slumps into the void, followed by KNN smoothing.

No random noise is injected, so all variation arises from physically meaningful
operations.

### Pseudo-label rule (annotation-free)

A point is labeled *shovel region* only if it satisfies all three conditions:

- **Frontier:** top 30th-percentile projection onto the approach axis.
- **Slope:** local slope `θ ∈ [20°, 40°]` (granular angle of repose).
- **Local prominence:** local Z-maximum within an adaptive-radius (~0.5 m) sphere.

## Reproducibility notes

- The fixed test set (`002`, `DJI_3`, `DJI_7`) is permanently held out from all
  training and model selection; the remaining files use 4-fold macro
  cross-validation. See `src/config.py` for exact splits and hyperparameters.
- All backbones share the same data loaders, training loop, and evaluation code to
  ensure a fair comparison.
- Random seeds and per-branch settings are defined in `src/config.py`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{yin2026loadsim,
  title   = {Physics-Driven Loading Simulation Augmentation for LiDAR-Based
             Shovel Region Segmentation: Interaction Mechanisms and
             Cross-Domain Generalization},
  author  = {Yin, Xinyu and He, Guanghui},
  journal = {Pattern Recognition (submitted)},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

This work builds on the open-source point cloud community, including PointNet++,
PointNeXt, KPConv, Point Transformer V3, RandLA-Net, and the SemanticKITTI
benchmark. We thank their authors for releasing code and data.
