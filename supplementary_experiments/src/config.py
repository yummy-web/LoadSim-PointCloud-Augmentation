"""
config.py - LSDA Project Global Configuration
==============================================
Version: v5 (journal extension)

Experiment design:
  Phase 1 (original, stored in outputs4/):
    7 training branches x 4-fold CV = 28 models (already completed)

  Phase 2 (journal extension, stored in outputs_journal/):
    - LoadSim component ablation: 7 configurations x 4 folds
    - Multi-backbone comparison: PointNeXt / KPConv / PTv3 / RandLA-Net
      x all 7 data branches x 4 folds
"""
from pathlib import Path

# ── Base paths ────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
OUTPUTS_DIR     = BASE_DIR / 'outputs4'        # original experiment output
OUTPUTS_JOURNAL = BASE_DIR / 'outputs_journal' # journal extension output

# ── Dataset configuration ─────────────────────────────────────────────────
DATASET = {
    'sparse':     ['001', '002', '003', '004', '005'],
    'dense':      ['DJI_1', 'DJI_2', 'DJI_3', 'DJI_4', 'DJI_5',
                   'DJI_6', 'DJI_7', 'DJI_8', 'DJI_9'],
    'test_set':   ['002', 'DJI_3', 'DJI_7'],
    'train_pool': ['001', '003', '004', '005',
                   'DJI_1', 'DJI_2', 'DJI_4', 'DJI_5',
                   'DJI_6', 'DJI_8', 'DJI_9'],
}

# ── 4-fold macro cross-validation splits ─────────────────────────────────
CV_SPLITS = [
    (['001', 'DJI_1', 'DJI_4'],
     ['003', '004', '005', 'DJI_2', 'DJI_5', 'DJI_6', 'DJI_8', 'DJI_9']),
    (['003', 'DJI_2', 'DJI_5'],
     ['001', '004', '005', 'DJI_1', 'DJI_4', 'DJI_6', 'DJI_8', 'DJI_9']),
    (['004', 'DJI_6', 'DJI_8'],
     ['001', '003', '005', 'DJI_1', 'DJI_2', 'DJI_4', 'DJI_5', 'DJI_9']),
    (['005', 'DJI_9', 'DJI_1'],
     ['001', '003', '004', 'DJI_2', 'DJI_4', 'DJI_5', 'DJI_6', 'DJI_8']),
]
N_CV_FOLDS = len(CV_SPLITS)

# ── Augmentation parameters ───────────────────────────────────────────────
AUGMENTATION = {
    'n_variants':        40,    # variants per file per mode
    'n_workers':         1,     # single-process (Windows compatible)
    'quality_threshold': 0.55,  # quality filter threshold
    'n_sample_150':      150,   # sample count for large-N branches
    'n_sample_11':       11,    # sample count for small-N branches
}

# ── Pseudo-label generation parameters ───────────────────────────────────
PSEUDO_LABEL = {
    'front_percentile': 0.70,   # frontier region: top 30% by projection
    'slope_min':        20.0,   # minimum slope angle (degrees)
    'slope_max':        40.0,   # maximum slope angle (degrees)
    'convex_radius_cm': 50.0,   # local prominence radius (~50 cm)
    'k_normal':         20,     # neighbours for PCA normal estimation
}

# ── Model parameters ──────────────────────────────────────────────────────
MODEL = {
    'n_points':  4096,  # points sampled per cloud
    'n_classes': 2,     # background / shovel region
}

# ── Training hyperparameters (shared baseline) ────────────────────────────
TRAIN = {
    'epochs':       70,
    'batch_size':   24,
    'lr':           5e-4,
    'weight_decay': 1e-4,
    'early_stop':   20,
    'amp':          True,           # mixed-precision training
    'class_weight': [0.2, 0.8],    # bg / shovel (handles class imbalance)
    'device':       'cuda',
    'n_workers':    0,              # DataLoader workers (0 = main process)
}

# ── 7 training branches (original experiment) ─────────────────────────────
# Format: (aug_source, n_sample, use_orig, label_cn, group)
TRAIN_BRANCHES = {
    'baseline_cv':  ('none',    0,   True,  'Baseline (11 originals, CV)',    'A'),
    'lsda_150':     ('lsda',    150, False, 'LSDA variants (N=150)',          'A'),
    'trad_150':     ('trad',    150, False, 'Conventional variants (N=150)',  'A'),
    'loadsim_150':  ('loadsim', 150, False, 'LoadSim-only variants (N=150)', 'A'),
    'lsda_11':      ('lsda',    11,  False, 'LSDA variants (N=11, CV)',       'B'),
    'trad_11':      ('trad',    11,  False, 'Conventional variants (N=11)',   'B'),
    'loadsim_11':   ('loadsim', 11,  False, 'LoadSim-only variants (N=11)',  'B'),
}

# ── LoadSim ablation configurations ──────────────────────────────────────
# Format: (enable_removal, enable_slope, enable_collapse, label_cn, label_en)
ABLATION_CONFIGS = {
    'ablation_full':          (True,  True,  True,  'Full LoadSim',            'Full LoadSim'),
    'ablation_no_removal':    (False, True,  True,  'w/o directional removal', 'w/o directional_removal'),
    'ablation_no_slope':      (True,  False, True,  'w/o slope reshaping',     'w/o slope_reshaping'),
    'ablation_no_collapse':   (True,  True,  False, 'w/o lateral collapse',    'w/o lateral_collapse'),
    'ablation_only_removal':  (True,  False, False, 'Only directional removal','Only directional_removal'),
    'ablation_only_slope':    (False, True,  False, 'Only slope reshaping',    'Only slope_reshaping'),
    'ablation_only_collapse': (False, False, True,  'Only lateral collapse',   'Only lateral_collapse'),
}

# ── Ablation training parameters ──────────────────────────────────────────
# n_folds=4 matches the original experiment for reliable comparison
ABLATION_TRAIN = {
    'n_sample':   100,   # variant count per fold (memory-balanced)
    'epochs':     70,
    'batch_size': 16,    # reduced to avoid OOM on ablation runs
    'n_folds':    4,
}

# ── Backbone comparison configuration ────────────────────────────────────
# All 4 backbones tested on all 7 data branches x 4 folds
# pointnet2 results come from the original experiment (outputs4/)
BACKBONE_COMPARISON = {
    'pointnet2':  'PointNet++ (NeurIPS 2017)',
    'pointnext':  'PointNeXt (NeurIPS 2022)',
    'kpconv':     'KPConv (ICCV 2019)',
    'ptv3':       'Point Transformer V3 (CVPR 2024)',
    'randlanet':  'RandLA-Net (CVPR 2020)',
}

# All 7 branches - identical to original experiment for fair comparison
BACKBONE_DATA_BRANCHES = list(TRAIN_BRANCHES.keys())

# Per-backbone training hyperparameters
# Small-data architectures (PointNeXt, PTv3) use lower lr and more patience
BACKBONE_TRAIN = {
    'pointnet2': {'lr': 5e-4,  'epochs': 70, 'early_stop': 20},
    'pointnext': {'lr': 2e-4,  'epochs': 80, 'early_stop': 25},
    'kpconv':    {'lr': 3e-4,  'epochs': 70, 'early_stop': 20},
    'ptv3':      {'lr': 6e-4,  'epochs': 80, 'early_stop': 25},
    'randlanet': {'lr': 1e-3,  'epochs': 70, 'early_stop': 20},
}

# Backbone comparison fold count (matches original experiment)
BACKBONE_N_FOLDS = 4

# ── Debug / quick-test mode ───────────────────────────────────────────────
DEBUG = {
    'n_variants':      2,
    'n_sample_150':    6,
    'n_sample_11':     4,
    'n_points':        512,
    'epochs':          3,
    'batch_size':      2,
    'early_stop':      2,
    'n_cv_folds':      1,
    'ablation_folds':  1,
    'backbone_folds':  1,
}
