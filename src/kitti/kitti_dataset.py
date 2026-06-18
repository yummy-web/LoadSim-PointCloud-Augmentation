"""
kitti_dataset.py  ——  SemanticKITTI 数据加载与增强 [v3]
========================================================

核心设计（v3 等量公平实验）:
  - 每帧原始数据生成且仅生成 1 个增强变体
  - 变体按与原始帧相同的时间索引保存（variant_i ← frame_i）
  - train/val/test 划分按变体索引顺序，与原始帧划分完全对齐
  - 测试集使用原始帧，所有增强分支共享同一测试集

Bug 修复（相比上一版）:
  [FIX-1] loadsim_augment 同步维护标签（材料去除同步 keep_mask）
  [FIX-2] _resample 断言 len(pts6)==len(labels)
  [FIX-3] estimate_normals_fast 向量化（einsum 批量协方差，速度 ~7x）
  [FIX-4] preprocess_frame 分层采样（保证地面点下限）
  [FIX-5] 增强函数统一接口 (pts6, labels, cfg) → (pts6, labels)
  [FIX-6] npz 缓存改用三维数组（float32/int32，非 object）

命名规范变更:
  "conventional" → "traditional"（与原始散料堆实验保持一致）
"""

import sys, random, warnings, time
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from config_kitti import (
    DATA_DIR, SPLIT, GROUND_CLASSES, N_POINTS, AUG, MODEL, TRAIN, DEBUG
)


# ══════════════════════════════════════════════════════════════════════════
# I/O 工具
# ══════════════════════════════════════════════════════════════════════════

def load_bin(path) -> np.ndarray:
    """读取 KITTI .bin，返回 (N, 4) float32 [x, y, z, intensity]。"""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


def load_label(path) -> np.ndarray:
    """
    读取 SemanticKITTI .label，返回 (N,) int32 语义 ID。
    格式: uint32，低 16 位 = 语义 ID，高 16 位 = 实例 ID。
    """
    raw = np.fromfile(str(path), dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.int32)


def map_to_binary(semantic: np.ndarray) -> np.ndarray:
    """
    语义 ID → 二分类标签:
      Ground=1: road(30) / parking(31) / sidewalk(32) /
                other-ground(33) / terrain(52)
      Non-ground=0: 其他所有类别

    学术依据: 这 5 类在局部法向量分布、坡度分布上与散料堆铲料区域高度相似
    （均为近水平/低坡面与竖直结构对比），是验证几何特征迁移能力的合理代理任务。
    """
    binary = np.zeros(len(semantic), dtype=np.int32)
    for cls in GROUND_CLASSES:
        binary[semantic == cls] = 1
    return binary


# ══════════════════════════════════════════════════════════════════════════
# 法向量估计（向量化）[FIX-3]
# ══════════════════════════════════════════════════════════════════════════

def estimate_normals_fast(pts: np.ndarray, k: int = 20) -> np.ndarray:
    """
    向量化 KNN-PCA 法向量估计（比逐点循环快约 7×）。

    使用 einsum 批量计算协方差矩阵，再一次性求最小特征向量。
    所有法向量均修正为指向上半球（nz >= 0）。

    Args:
        pts: (N, 3) float32
        k:   近邻数（默认 20）
    Returns:
        normals: (N, 3) float32
    """
    n = len(pts)
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0  # 默认朝上

    if not HAS_SCIPY or n < k + 2:
        return normals

    k_eff = min(k, n - 1)
    tree  = cKDTree(pts)
    _, idxs = tree.query(pts, k=k_eff + 1)   # (N, k+1)，含自身

    nb  = pts[idxs[:, 1:]] - pts[:, None, :]          # (N, k, 3)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k_eff   # (N, 3, 3)

    try:
        _, vecs = np.linalg.eigh(cov)          # vecs: (N, 3, 3) 升序
        normals = vecs[:, :, 0].astype(np.float32)  # 最小特征向量
    except np.linalg.LinAlgError:
        pass

    normals[normals[:, 2] < 0] *= -1  # 朝上修正
    return normals


# ══════════════════════════════════════════════════════════════════════════
# 点云预处理（分层采样）[FIX-4]
# ══════════════════════════════════════════════════════════════════════════

def preprocess_frame(pts4: np.ndarray, labels: np.ndarray,
                     n_pts: int = N_POINTS,
                     seed: int = None,
                     min_ground_ratio: float = 0.10):
    """
    预处理单帧 KITTI 点云:
      1. 过滤无效点 (x=y=z=0)
      2. 全量标签二值化（采样前，保证比例统计准确）
      3. 分层采样 [FIX-4]: 若地面点不足 n_pts*min_ground_ratio，
         优先全取地面点，再补充背景点
      4. 坐标中心化
      5. 向量化法向量估计

    Returns:
        pts6:       (n_pts, 6) float32  [x, y, z, nx, ny, nz]
        labels_bin: (n_pts,)   int32    [0/1]，与 pts6 一一对应
        若帧无效返回 (None, None)
    """
    valid  = ~((pts4[:, 0] == 0) & (pts4[:, 1] == 0) & (pts4[:, 2] == 0))
    pts4   = pts4[valid]
    labels = labels[valid]

    if len(pts4) < 100:
        return None, None

    # 全量二值化
    labels_bin_all = map_to_binary(labels)

    # 分层采样
    rng = (np.random.RandomState(seed) if seed is not None
           else np.random.RandomState())

    ground_idx = np.where(labels_bin_all == 1)[0]
    bg_idx     = np.where(labels_bin_all == 0)[0]
    n_g_min    = int(n_pts * min_ground_ratio)

    n = len(pts4)
    if len(ground_idx) == 0:
        idx = rng.choice(n, n_pts, replace=(n < n_pts))
    elif len(ground_idx) >= n_g_min:
        idx = rng.choice(n, n_pts, replace=(n < n_pts))
    else:
        n_bg = n_pts - len(ground_idx)
        bg_s = rng.choice(bg_idx, n_bg, replace=(len(bg_idx) < n_bg))
        idx  = np.concatenate([ground_idx, bg_s])
        rng.shuffle(idx)

    pts3       = pts4[idx, :3].copy()
    labels_out = labels_bin_all[idx]

    pts3 -= pts3.mean(axis=0)  # 中心化

    normals = estimate_normals_fast(pts3, k=20)
    pts6    = np.concatenate([pts3, normals], axis=1).astype(np.float32)

    return pts6, labels_out


# ══════════════════════════════════════════════════════════════════════════
# 帧路径获取
# ══════════════════════════════════════════════════════════════════════════

def get_frame_paths(seq_dir, split_name: str, debug: bool = False):
    """
    按时间顺序返回指定 split 的 (bin_path, lbl_path) 列表。

    增强分支的 variant_i 与 frame_i 一一对应，因此
    增强分支的 train/val/test 按相同的帧索引范围划分，
    完全对齐，无时间泄露。
    """
    vel_dir  = Path(seq_dir) / "velodyne"
    lbl_dir  = Path(seq_dir) / "labels"
    all_bins = sorted(vel_dir.glob("*.bin"))
    total    = len(all_bins)

    start, end = SPLIT[split_name]
    end = min(end, total)

    if debug:
        n   = (DEBUG["n_train_frames"] if split_name == "train"
               else DEBUG["n_val_frames"])
        end = min(start + n, end)

    pairs = []
    for bp in all_bins[start:end]:
        lp = lbl_dir / f"{bp.stem}.label"
        if lp.exists():
            pairs.append((bp, lp))
    return pairs


# ══════════════════════════════════════════════════════════════════════════
# 增强函数（均接收并返回 (pts6, labels) 对）[FIX-1, FIX-5]
# ══════════════════════════════════════════════════════════════════════════

def loadsim_augment(pts6: np.ndarray, labels: np.ndarray,
                    cfg: dict) -> tuple:
    """
    LoadSim 物理装载模拟增强（KITTI 米制坐标系）[FIX-1]

    核心修复: 材料去除时用同一 keep_mask 同步删除对应标签，
              保证返回的 (pts6_aug, labels_aug) 始终等长。

    Args:
        pts6:   (N, 6) float32  已中心化的 [x,y,z,nx,ny,nz]
        labels: (N,)   int32    与 pts6 一一对应的 [0/1] 标签
        cfg:    AUG["loadsim"] 超参

    Returns:
        (pts6_aug, labels_aug): 等长，M <= N
    """
    pts  = pts6[:, :3].copy()
    nrms = pts6[:, 3:].copy()
    lbls = labels.copy()
    cent = pts.mean(axis=0)

    for _ in range(random.randint(*cfg["n_ops"])):
        if len(pts) < 200:
            break

        angle = random.uniform(0, 2 * np.pi)
        d_vec = np.array([np.cos(angle), np.sin(angle), 0.0])
        s_vec = np.array([-np.sin(angle), np.cos(angle), 0.0])

        proj  = (pts - cent) @ d_vec
        p_min, p_max = proj.min(), proj.max()
        if p_max <= p_min:
            continue

        thresh = p_max - random.uniform(*cfg["front_alpha"]) * (p_max - p_min)
        front  = np.where(proj > thresh)[0]
        if len(front) < 50:
            continue

        # 坡度加权采样挖掘中心
        fn    = nrms[front]
        theta = np.arcsin(np.clip(np.abs(fn[:, 2]), 0, 1))
        w     = (np.pi / 2 - theta) ** 2 + 1e-8
        w    /= w.sum()
        core  = pts[front[np.random.choice(len(front), p=w)]]

        bw       = random.uniform(*cfg["bucket_width"])
        dist_s   = np.abs((pts - core) @ s_vec)
        exc_mask = (dist_s < bw / 2) & (proj > thresh)
        take_idx = np.where(exc_mask)[0]
        if len(take_idx) == 0:
            continue

        # 侧向坍塌（形变，不删点）
        col_idx = np.where(
            (dist_s >= bw / 2) & (dist_s < bw * 1.2) & (proj > thresh))[0]
        if len(col_idx) > 0:
            cd = dist_s[col_idx] - bw / 2
            cf = np.clip(1.0 - cd / (bw * 0.2 + 1e-8), 0, 1)
            pts[col_idx, 2] -= random.uniform(*cfg["collapse_dz"]) * cf
            sv = core - pts[col_idx]
            sv[:, 2] = 0
            sv /= (np.linalg.norm(sv, axis=1, keepdims=True) + 1e-8)
            pts[col_idx] += sv * (random.uniform(*cfg["collapse_dx"]) * cf)[:, None]

        # 方向性材料去除（同步删除标签）[FIX-1]
        keep = np.ones(len(pts), dtype=bool)
        keep[take_idx] = False
        pts  = pts[keep]
        nrms = nrms[keep]
        lbls = lbls[keep]   # ← 关键：同步删除

    # KNN 平滑
    if HAS_SCIPY and len(pts) > cfg["smooth_k"] + 1:
        k_s  = min(cfg["smooth_k"], len(pts) - 1)
        tree = cKDTree(pts)
        _, idxs = tree.query(pts, k=k_s + 1)
        nb_mean = pts[idxs[:, 1:]].mean(axis=1)
        pts = pts * (1.0 - cfg["smooth_factor"]) + nb_mean * cfg["smooth_factor"]

    nrms_new = estimate_normals_fast(pts, k=20)
    pts6_out = np.concatenate([pts, nrms_new], axis=1).astype(np.float32)

    assert len(pts6_out) == len(lbls), (
        f"loadsim_augment: pts6({len(pts6_out)}) != labels({len(lbls)})")
    return pts6_out, lbls


def traditional_augment(pts6: np.ndarray, labels: np.ndarray,
                         cfg: dict) -> tuple:
    """
    传统几何增强: 随机缩放 + Z 轴旋转 + 高斯抖动 + 随机 X 翻转。

    不删点，不改变标签语义。
    Returns: (pts6_aug, labels) 对（labels 直接透传）。
    """
    pts = pts6[:, :3].copy()
    nrm = pts6[:, 3:].copy()

    pts *= random.uniform(*cfg["scale_range"])

    theta = np.radians(random.uniform(*cfg["rot_z_range"]))
    c, s  = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = pts @ R.T
    nrm = nrm @ R.T

    noise = np.clip(
        np.random.randn(*pts.shape).astype(np.float32) * cfg["jitter_sigma"],
        -cfg["jitter_clip"], cfg["jitter_clip"])
    pts += noise

    if random.random() < cfg["flip_prob"]:
        pts[:, 0] *= -1
        nrm[:, 0] *= -1

    return np.concatenate([pts, nrm], axis=1).astype(np.float32), labels.copy()


def _resample(pts6: np.ndarray, labels: np.ndarray,
              n_pts: int, seed: int = None) -> tuple:
    """
    重采样 (pts6, labels) 到 n_pts 点。[FIX-2]

    前提: len(pts6) == len(labels)（由增强函数保证）。
    点数不足时使用带替换采样。
    """
    n = len(pts6)
    if n != len(labels):
        raise ValueError(
            f"_resample: pts6 长度({n}) 与 labels 长度({len(labels)}) 不一致！"
            f"请检查增强函数是否正确同步维护了标签。")

    rng = (np.random.RandomState(seed) if seed is not None
           else np.random.RandomState())
    idx = rng.choice(n, n_pts, replace=(n < n_pts))
    return pts6[idx].astype(np.float32), labels[idx]


# ══════════════════════════════════════════════════════════════════════════
# 增强缓存生成（支持断点续传）[FIX-5, FIX-6, v3 等量设计]
# ══════════════════════════════════════════════════════════════════════════

def build_aug_cache(pairs, strategy: str,
                    cfg_aug: dict, n_pts: int,
                    debug: bool = False,
                    cache_dir=None):
    """
    预生成增强变体缓存（v3 等量设计：每帧 1 个变体）。

    v3 设计说明:
      - baseline:    直接用原始帧预处理结果（无增强变体）
      - traditional / lsda / loadsim:
          每帧生成 1 个变体，仅保存变体（不混入原始帧）
      - 所有分支结果数量相同（等于输入帧数）

    断点续传支持:
      - 逐帧将结果追加到 cache_dir/{strategy}_partial.npz
      - 已处理的帧数记录在 cache_dir/{strategy}_progress.txt
      - 完成后重命名为 {strategy}.npz，删除临时文件

    Args:
        pairs:     [(bin_path, lbl_path), ...]  按时间顺序排列
        strategy:  'baseline' | 'traditional' | 'lsda' | 'loadsim'
        cfg_aug:   AUG 字典
        n_pts:     每个样本点数
        debug:     调试模式
        cache_dir: 缓存保存目录（None 则不保存）

    Returns:
        list of (pts6: ndarray(n_pts,6), labels: ndarray(n_pts,))
        按与输入 pairs 相同的时间顺序排列
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    # ── 读取已完成的缓存 ─────────────────────────────────────────────────
    if cache_dir is not None:
        final_file = cache_dir / f"{strategy}.npz"
        if final_file.exists():
            print(f"  [CACHE] 读取已有缓存: {final_file}", flush=True)
            data       = np.load(str(final_file))
            pts6_arr   = data["pts6_list"].astype(np.float32)   # (T, N, 6)
            labels_arr = data["labels_list"].astype(np.int32)   # (T, N)
            result = [(pts6_arr[i], labels_arr[i])
                      for i in range(len(pts6_arr))]
            gr = float(np.mean([lbl.mean() for _, lbl in result]))
            print(f"    已加载 {len(result)} 个样本，"
                  f"平均地面点比例={gr:.4f}", flush=True)
            return result

    # ── 断点续传：读取已处理进度 ─────────────────────────────────────────
    start_idx  = 0
    done_pts6  = []
    done_lbls  = []

    if cache_dir is not None:
        progress_file = cache_dir / f"{strategy}_progress.txt"
        partial_file  = cache_dir / f"{strategy}_partial.npz"

        if progress_file.exists() and partial_file.exists():
            try:
                start_idx = int(progress_file.read_text().strip())
                pdata     = np.load(str(partial_file))
                done_pts6 = list(pdata["pts6_list"].astype(np.float32))
                done_lbls = list(pdata["labels_list"].astype(np.int32))
                print(f"  [RESUME] {strategy}: 从第 {start_idx} 帧恢复"
                      f"（已有 {len(done_pts6)} 个样本）", flush=True)
            except Exception as e:
                print(f"  [WARN] 读取续传文件失败，从头开始: {e}", flush=True)
                start_idx = 0
                done_pts6 = []
                done_lbls = []

    print(f"\n  生成缓存: strategy={strategy}  "
          f"frames={len(pairs)}  start={start_idx}", flush=True)

    t0     = time.time()
    result = list(zip(done_pts6, done_lbls))  # 已完成部分

    # ── 逐帧处理 ──────────────────────────────────────────────────────────
    save_interval = 200  # 每 200 帧保存一次进度

    for fi in range(start_idx, len(pairs)):
        bin_path, lbl_path = pairs[fi]

        if (fi + 1) % 500 == 0 or fi == start_idx:
            elapsed = time.time() - t0
            print(f"    [{fi+1}/{len(pairs)}] {bin_path.name}  "
                  f"cache={len(result)}  "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

        try:
            pts4 = load_bin(bin_path)
            lbls = load_label(lbl_path)
            pts6, labels = preprocess_frame(pts4, lbls, n_pts, seed=fi)
            if pts6 is None:
                continue

            if strategy == "baseline":
                # 直接使用原始预处理帧，不生成变体
                result.append((pts6.copy(), labels.copy()))

            elif strategy == "traditional":
                aug6, aug_lbl = traditional_augment(
                    pts6, labels, cfg_aug["traditional"])
                aug6, aug_lbl = _resample(aug6, aug_lbl, n_pts,
                                           seed=fi * 1000 + 1)
                result.append((aug6, aug_lbl))

            elif strategy == "loadsim":
                aug6, aug_lbl = loadsim_augment(
                    pts6, labels, cfg_aug["loadsim"])
                aug6, aug_lbl = _resample(aug6, aug_lbl, n_pts,
                                           seed=fi * 1000 + 1)
                result.append((aug6, aug_lbl))

            elif strategy == "lsda":
                aug6, aug_lbl = loadsim_augment(
                    pts6, labels, cfg_aug["loadsim"])
                aug6, aug_lbl = traditional_augment(
                    aug6, aug_lbl, cfg_aug["traditional"])
                aug6, aug_lbl = _resample(aug6, aug_lbl, n_pts,
                                           seed=fi * 1000 + 1)
                result.append((aug6, aug_lbl))

        except Exception as e:
            print(f"    [WARN] {bin_path.name}: {e}", flush=True)
            continue

        # 断点续传：定期保存进度
        if (cache_dir is not None
                and (fi + 1) % save_interval == 0
                and result):
            _save_partial(cache_dir, strategy, result, fi + 1)

    elapsed = time.time() - t0

    # ── 统计并打印 ────────────────────────────────────────────────────────
    if result:
        gr = float(np.mean([lbl.mean() for _, lbl in result]))
        print(f"  ✅ 缓存完成: {len(result)} 个样本  "
              f"平均地面比例={gr:.4f}  耗时={elapsed/60:.1f}min", flush=True)

    # ── 保存最终缓存，清理临时文件 ───────────────────────────────────────
    if cache_dir is not None and result:
        cache_dir.mkdir(parents=True, exist_ok=True)
        final_file = cache_dir / f"{strategy}.npz"
        pts6_arr   = np.stack([r[0] for r in result]).astype(np.float32)
        labels_arr = np.stack([r[1] for r in result]).astype(np.int32)
        np.savez_compressed(str(final_file),
                            pts6_list=pts6_arr,
                            labels_list=labels_arr)
        print(f"  [CACHE] 已保存: {final_file}  "
              f"shape={pts6_arr.shape}", flush=True)

        # 清理临时文件
        for tmp in [cache_dir / f"{strategy}_partial.npz",
                    cache_dir / f"{strategy}_progress.txt"]:
            if tmp.exists():
                tmp.unlink()

    return result


def _save_partial(cache_dir, strategy: str, result: list, fi: int):
    """保存断点续传的临时进度文件。"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        pts6_arr   = np.stack([r[0] for r in result]).astype(np.float32)
        labels_arr = np.stack([r[1] for r in result]).astype(np.int32)
        np.savez_compressed(
            str(cache_dir / f"{strategy}_partial.npz"),
            pts6_list=pts6_arr,
            labels_list=labels_arr)
        (cache_dir / f"{strategy}_progress.txt").write_text(str(fi))
    except Exception as e:
        pass  # 续传保存失败不中断主流程


# ══════════════════════════════════════════════════════════════════════════
# 在线轻量增强（不改变点数和标签）
# ══════════════════════════════════════════════════════════════════════════

def _online_aug(pts6: np.ndarray) -> np.ndarray:
    """
    训练时在线轻量增强: 随机 Z 旋转 + 随机 X 翻转。
    不删点，不改变标签语义。
    """
    theta = np.random.uniform(0, 2 * np.pi)
    c, s  = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts6[:, :3] = pts6[:, :3] @ R.T
    pts6[:, 3:] = pts6[:, 3:] @ R.T
    if np.random.rand() < 0.5:
        pts6[:, 0] *= -1
        pts6[:, 3] *= -1
    return pts6


# ══════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class KITTIGroundDataset(Dataset):
        """
        SemanticKITTI Ground Segmentation Dataset

        两种模式:
          cache ── 从预生成增强缓存加载（训练/验证集推荐）
          real  ── 直接从 .bin/.label 加载（测试集使用）

        注: 测试集始终使用 real 模式 + 原始帧，不经过任何增强缓存。
        """

        def __init__(self, pairs, aug_cache=None,
                     augment: bool = False,
                     n_pts: int = N_POINTS,
                     split_seed: int = 42):
            self.pairs      = pairs
            self.augment    = augment
            self.n_pts      = n_pts
            self.split_seed = split_seed

            if aug_cache is not None:
                self._data = aug_cache
                self._mode = "cache"
            else:
                self._data = pairs
                self._mode = "real"

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            if self._mode == "cache":
                pts6, labels = self._data[idx]
                pts6   = pts6.copy()
                labels = labels.copy()
            else:
                bin_path, lbl_path = self._data[idx]
                pts4   = load_bin(bin_path)
                lbls   = load_label(lbl_path)
                pts6, labels = preprocess_frame(
                    pts4, lbls, self.n_pts,
                    seed=idx + self.split_seed)
                if pts6 is None:
                    pts6   = np.zeros((self.n_pts, 6), dtype=np.float32)
                    labels = np.zeros(self.n_pts, dtype=np.int32)

            if self.augment:
                pts6 = _online_aug(pts6)

            x = torch.from_numpy(pts6).float()    # (N, 6)
            y = torch.from_numpy(labels).long()   # (N,)
            return x.permute(1, 0), y             # (6, N), (N,)

else:
    class KITTIGroundDataset:
        """Placeholder when PyTorch is not installed."""
        def __init__(self, *a, **kw):
            pass
        def __len__(self):
            return 0
