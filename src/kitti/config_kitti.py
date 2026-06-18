"""
config_kitti.py  ——  SemanticKITTI 泛化性实验全局配置 [v3]
===========================================================

实验目的:
  验证 LoadSim 增强方法在 SemanticKITTI 数据集上的泛化性。

任务:
  Ground / Non-ground 逐点二分类
    Ground (label=1): road(30) / parking(31) / sidewalk(32) /
                      other-ground(33) / terrain(52)
    Non-ground (label=0): 其他所有类别

实验设计（v3，等量公平对比）:
  数据划分（严格按时间顺序，无交叉）:
    训练集: 帧 0000-3199  (3200 帧, ~70%)
    验证集: 帧 3200-3799  (600  帧, ~13%)
    测试集: 帧 3800-4540  (741  帧, ~17%, 固定不变)

  四个分支（训练/验证集数量完全相同）:
    baseline:    原始帧直接训练（无增强）
    traditional: 每帧生成 1 个传统几何增强变体
    lsda:        每帧生成 1 个 LoadSim+传统组合变体
    loadsim:     每帧生成 1 个纯 LoadSim 物理增强变体

  关键设计:
    每个增强分支只用增强后的变体，不混入原始帧
    各分支训练/验证各 3200/600 个样本，完全等量
    变体继承原始帧时间戳：variant_i <- frame_i
    按变体索引划分 [0-3199]/[3200-3799]，无时间泄露
    测试集统一使用原始帧（未经任何增强，固定不变）
"""
import os
from pathlib import Path

# SemanticKITTI sequences root. Override with the KITTI_ROOT environment variable,
# e.g. export KITTI_ROOT=/path/to/dataset/sequences
KITTI_ROOT = Path(os.environ.get("KITTI_ROOT", "./data/kitti/sequences"))
SEQ        = "00"
DATA_DIR   = KITTI_ROOT / SEQ

OUT_ROOT   = Path("./kitti_outputs")
AUG_DIR    = OUT_ROOT / "augmented"
MODEL_DIR  = OUT_ROOT / "models"
LOG_DIR    = OUT_ROOT / "logs"
RESULT_DIR = OUT_ROOT / "results"

SPLIT = {
    "train": (0,    3200),
    "val":   (3200, 3800),
    "test":  (3800, 4541),
}

GROUND_CLASSES = {30, 31, 32, 33, 52}

N_POINTS = 4096

AUG = {
    "n_variants_per_frame": 1,   # v3: 每帧仅 1 个变体（等量设计）
    "loadsim": {
        "n_ops":         (2, 4),
        "front_alpha":   (0.15, 0.25),
        "bucket_width":  (0.8, 1.5),
        "collapse_dz":   (0.03, 0.10),
        "collapse_dx":   (0.02, 0.06),
        "smooth_k":      15,
        "smooth_factor": 0.10,
    },
    "traditional": {
        "scale_range":  (0.85, 1.15),
        "rot_z_range":  (-180, 180),
        "jitter_sigma": 0.02,
        "jitter_clip":  0.05,
        "flip_prob":    0.5,
    },
}

TRAIN = {
    "epochs":       60,
    "batch_size":   16,
    "lr":           1e-4,
    "weight_decay": 1e-4,
    "early_stop":   15,
    "amp":          True,
    "class_weight": [0.25, 0.75],
    "device":       "cuda",
    "n_workers":    0,
}

MODEL = {
    "in_ch":    6,
    "n_cls":    2,
    "n_points": N_POINTS,
}

DEBUG = {
    "n_train_frames": 100,
    "n_val_frames":   20,
    "epochs":         5,
    "batch_size":     4,
    "early_stop":     2,
}
