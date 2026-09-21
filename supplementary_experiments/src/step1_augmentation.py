"""
step1_augmentation.py  (v5 — Loading Simulation Core + Literature-based Augmentation)
=======================================================================================
散料堆点云数据增强 — LSDA (Loading Simulation-Driven Augmentation)

【核心创新与文献依据 v5】
每个变体强制经历装载模拟 (100%)，体现 LSDA 的核心创新。
传统方法按文献赋权叠加:

  1. 装载模拟 (Loading Simulation, P=1.00) — 核心创新 (本文)
     原理: 模拟铲斗装载引起的作业前沿识别、散料去除与坡面重塑。

  2. 缩放与旋转 (Scale + Z-Rotation, P=0.80)
     依据: Qi et al. (CVPR 2017) PointNet: 标准点云增强使用随机旋转与缩放;
           Zhu et al. (Pattern Recognit. 2024) 综述证实其为点云增强基线。
     物理意义: 模拟不同传感器安装位置与拍摄距离变化。

  3. 高斯表面噪声 (Gaussian Surface Noise, P=0.50)
     依据: Qi et al. (CVPR 2017) PointNet 训练中使用高斯抖动;
           Zhu et al. (Pattern Recognit. 2024) 综述将噪声列为基础增强方法。
     物理意义: 模拟户外环境LiDAR/无人机扫描的固有测量误差。

  4. RBF局部形变 (RBF Non-rigid Deformation, P=0.30)
     依据: Zhu et al. (Pattern Recognit. 2024) 将非刚性形变归类为高级增强;
           权重较低以防破坏散料堆全局物理约束（安息角等）。
     物理意义: 模拟局部地形微扰与散料局部塌陷。

参考文献:
  [1] Qi CR, Su H, Mo K, Guibas LJ. PointNet. CVPR 2017.
  [2] Zhu Q, Fan L, Weng N. Point Cloud Data Augmentation Survey. Pattern Recognit. 2024;153:110532.
  [3] Qi CR, Yi L, Su H, Guibas LJ. PointNet++. NeurIPS 2017.
  [9] Wu X et al. Point Transformer V3. CVPR 2024:4840-4851.
  [11] Xu M et al. Physical simulation constraints for excavation. Autom Constr. 2024;162:105386.
  [12] Liao PC et al. Shovel tip detection for autonomous excavation. Autom Constr. 2024;165:105546.

用法:
    python step1_augmentation.py \\
        --data-dir "./data" \\
        --output-dir "./outputs/augmented" \\
        --variants 40 --workers 1
"""

import os, sys, io, json, time, argparse, warnings, hashlib
import numpy as np
from pathlib import Path

# Windows控制台UTF-8输出修复
if sys.platform == 'win32' and not os.environ.get('DISABLE_STDOUT_WRAP'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings('ignore')

try:
    from scipy.interpolate import Rbf
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy 未安装，RBF形变和KNN平滑将退化")

# 强制禁用 Open3D，全程 numpy (避免Windows多进程冲突)
HAS_O3D = False

# ══════════════════════════════════════════════════════════
# PLY 工具函数
# ══════════════════════════════════════════════════════════

def _load_ply_numpy(path):
    """纯numpy PLY读取器 (支持ASCII和Binary little endian)"""
    pts, nrms = [], []
    has_nx, n_vert, is_binary = False, 0, False
    with open(path, 'rb') as f:
        while True:
            raw = f.readline()
            line = raw.decode('ascii', errors='ignore').strip()
            if line.startswith('element vertex'):
                try:
                    n_vert = int(line.split()[-1])
                except ValueError:
                    pass
            elif line.lower().startswith('property') and ' nx' in line.lower():
                has_nx = True
            elif 'binary_little_endian' in line:
                is_binary = True
            elif line == 'end_header':
                break
        if is_binary:
            dtype_fields = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            if has_nx:
                dtype_fields += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
            dt = np.dtype(dtype_fields)
            data = np.frombuffer(f.read(dt.itemsize * n_vert), dtype=dt)
            pts_arr = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
            nrm_arr = np.column_stack([data['nx'], data['ny'], data['nz']]).astype(np.float32) if has_nx else None
            return pts_arr, nrm_arr
        else:
            for _ in range(n_vert):
                raw_line = f.readline()
                if not raw_line:
                    break
                vals_str = raw_line.decode('ascii', errors='ignore').split()
                if len(vals_str) < 3:
                    continue
                try:
                    vals = [float(v) for v in vals_str]
                    pts.append(vals[:3])
                    if has_nx and len(vals) >= 6:
                        nrms.append(vals[3:6])
                except ValueError:
                    continue
    pts_arr = np.array(pts, dtype=np.float32)
    nrm_arr = np.array(nrms, dtype=np.float32) if (has_nx and nrms) else None
    return pts_arr, nrm_arr


def _save_ply_numpy(path, pts, normals=None):
    """纯numpy PLY ASCII写入器 (强制utf-8，修复Windows乱码)"""
    has_n = normals is not None and len(normals) == len(pts)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_n:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        for i in range(len(pts)):
            row = f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f}"
            if has_n:
                row += f" {normals[i,0]:.6f} {normals[i,1]:.6f} {normals[i,2]:.6f}"
            f.write(row + "\n")


# ══════════════════════════════════════════════════════════
# 预处理工具
# ══════════════════════════════════════════════════════════

def estimate_normals_numpy(pts, k=20):
    """纯numpy法向量估计 (无Open3D依赖)"""
    if not HAS_SCIPY:
        return np.tile([0, 0, 1], (len(pts), 1)).astype(np.float32)
    try:
        tree = cKDTree(pts)
        nrm = np.zeros((len(pts), 3), dtype=np.float32)
        k_actual = min(k, len(pts) - 1)
        for i in range(len(pts)):
            _, idxs = tree.query(pts[i], k=k_actual)
            nb = pts[idxs] - pts[i]
            cov = nb.T @ nb
            try:
                _, vecs = np.linalg.eigh(cov)
                nrm[i] = vecs[:, 0]
            except np.linalg.LinAlgError:
                nrm[i] = [0, 0, 1]
    except Exception:
        return np.tile([0, 0, 1], (len(pts), 1)).astype(np.float32)
    flip = nrm[:, 2] < 0
    nrm[flip] *= -1
    return nrm


def preprocess(pts, normals=None):
    """中心化 + 离群点去除"""
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    if HAS_SCIPY and len(pts) > 500:
        tree = cKDTree(pts)
        k = min(20, len(pts) - 1)
        dists, _ = tree.query(pts, k=k + 1)
        mean_d = dists[:, 1:].mean(axis=1)
        mu, sigma = mean_d.mean(), mean_d.std()
        mask = mean_d < (mu + 2.5 * sigma)
        pts = pts[mask]
        if normals is not None:
            normals = normals[mask]
    if normals is not None:
        flip = normals[:, 2] < 0
        normals[flip] *= -1
    return pts, normals, centroid


def knn_smooth(pts, k=12, factor=0.10):
    """KNN加权平滑，用于装载模拟后的边界过渡"""
    if not HAS_SCIPY or len(pts) < k + 1:
        return pts
    tree = cKDTree(pts)
    smoothed = pts.copy()
    k_actual = min(k, len(pts) - 1)
    batch = 1000
    for i in range(0, len(pts), batch):
        end = min(i + batch, len(pts))
        _, idxs = tree.query(pts[i:end], k=k_actual)
        nb_mean = pts[idxs].mean(axis=1)
        smoothed[i:end] = (1 - factor) * pts[i:end] + factor * nb_mean
    return smoothed


def get_data_config(filename):
    """
    根据文件名自动识别扫描类型并返回对应参数配置

    分类规则（基于文件命名约定）:
      - 文件名含 'DJI' → 无人机航测扫描 (Aerial, 9个: DJI_01~DJI_09)
        参数: 较大噪声范围和装载深度，反映无人机测距误差更大的特性

      - 文件名为纯数字(如001~009)或含 'NEAR' → 近景地面扫描 (Near-field, 5个: 001~005)
        参数: 较小噪声范围和装载深度，反映近景地面扫描精度更高的特性
        说明: 近景扫描文件命名为 001.ply~005.ply，全部为纯数字编号。
              判断条件: 文件名主干为纯数字（不含字母），即 isdigit() == True。

    数据集构成（14个PLY文件）:
      - 近景地面扫描: 001.ply, 002.ply, 003.ply, 004.ply, 005.ply  (5个, ~6-12万点)
      - 无人机航测:   DJI_01.ply ~ DJI_09.ply                       (9个, ~20-50万点)
      - 合计: 14个文件，均参与增强和质量筛选
      - 基线N=14: 使用全部14个原始文件作为基线（5近景+9无人机）

    注意: 两种类型均使用相同的装载模拟核心参数（前沿比例、移除比例等），
    仅噪声强度和装载深度范围根据采集方式自适应调整。
    """
    fname = str(filename).upper()
    fname_stem = str(Path(filename).stem)
    if 'DJI' in fname:
        return {'type': 'aerial',
                'scale_range': (0.7, 1.4),
                'rbf_ctrl_range': (12, 22),
                'rbf_intensity': (0.20, 0.35),
                'noise_int': (0.06, 0.14)}
    elif fname_stem.isdigit() or fname_stem.upper().startswith('NEAR'):
        # 纯数字文件名 (001~005等) 或含NEAR前缀 → 近景地面扫描
        return {'type': 'near_field',
                'scale_range': (0.85, 1.2),
                'rbf_ctrl_range': (8, 15),
                'rbf_intensity': (0.15, 0.25),
                'noise_int': (0.04, 0.10)}
    else:
        # 其他未识别文件名: 使用近景保守参数，并记录警告
        import warnings
        warnings.warn(f"文件 '{filename}' 未能识别为已知类型(DJI系列或纯数字编号)，"
                      f"使用近景保守参数。如需精确分类，请更新get_data_config()中的判断逻辑。")
        return {'type': 'unknown_use_near_field_params',
                'scale_range': (0.85, 1.2),
                'rbf_ctrl_range': (8, 15),
                'rbf_intensity': (0.15, 0.25),
                'noise_int': (0.04, 0.10)}


# ══════════════════════════════════════════════════════════
# 核心创新: 装载模拟 (100%触发)
# ══════════════════════════════════════════════════════════

def nonlinear_loading_simulation(pts, normals, meta, cfg, rng):
    """Apply physical-unit LoadSim with one seeded random direction per operation."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError('loading simulation requires a numpy.random.Generator')
    if normals is None or len(normals) != len(pts):
        raise ValueError('loading simulation requires one real normal per point')

    scale_to_m = cfg.get('coordinate_scale_to_m')
    if (isinstance(scale_to_m, bool) or not isinstance(scale_to_m, (int, float)) or
            not np.isfinite(scale_to_m) or scale_to_m <= 0):
        raise ValueError('loading simulation requires positive finite coordinate_scale_to_m')

    deformed_pts = pts.copy()
    deformed_normals = normals.copy()
    centroid = deformed_pts.mean(axis=0)
    n_ops = int(rng.integers(2, 5))
    operations = []

    for _ in range(n_ops):
        if len(deformed_pts) < 500:
            break

        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        direction_vec = np.array([np.cos(angle), np.sin(angle), 0.0])
        side_vec = np.array([-np.sin(angle), np.cos(angle), 0.0])
        proj = (deformed_pts - centroid) @ direction_vec
        p_min, p_max = proj.min(), proj.max()
        if p_max <= p_min:
            continue

        front_ratio = float(rng.uniform(0.15, 0.25))
        front_thresh = p_max - front_ratio * (p_max - p_min)
        front_indices = np.where(proj > front_thresh)[0]
        if len(front_indices) < 100:
            continue

        front_normals = deformed_normals[front_indices]
        slope_angles = np.arcsin(np.abs(front_normals[:, 2]))
        steepness_weights = (np.pi / 2 - slope_angles) ** 2 + 1e-6
        steepness_weights /= steepness_weights.sum()
        core_local_idx = int(rng.choice(len(front_indices), p=steepness_weights))
        excavation_center = deformed_pts[front_indices[core_local_idx]]

        bucket_width_m = float(rng.uniform(0.8, 1.5))
        bucket_width = bucket_width_m / float(scale_to_m)
        dist_from_center_line = np.abs((deformed_pts - excavation_center) @ side_vec)
        excavation_mask = (dist_from_center_line < bucket_width / 2) & (proj > front_thresh)
        take_indices = np.where(excavation_mask)[0]
        if len(take_indices) == 0:
            continue

        collapse_zone_mask = (dist_from_center_line >= bucket_width / 2) & \
                             (dist_from_center_line < bucket_width * 1.2) & \
                             (proj > front_thresh)
        collapse_indices = np.where(collapse_zone_mask)[0]
        if len(collapse_indices) > 0:
            collapse_dist = dist_from_center_line[collapse_indices] - bucket_width / 2
            max_collapse_dist = bucket_width * 1.2 - bucket_width / 2
            collapse_factor = 1 - collapse_dist / max_collapse_dist
            sink_depth = float(rng.uniform(0.1, 0.3)) * collapse_factor / float(scale_to_m)
            deformed_pts[collapse_indices, 2] -= sink_depth
            inward_slide = (float(rng.uniform(0.05, 0.15)) * collapse_factor /
                            float(scale_to_m))
            slide_vectors = excavation_center - deformed_pts[collapse_indices]
            slide_vectors[:, 2] = 0
            slide_vectors /= np.linalg.norm(slide_vectors, axis=1, keepdims=True) + 1e-8
            deformed_pts[collapse_indices] += slide_vectors * inward_slide[:, np.newaxis]

        keep_mask = np.ones(len(deformed_pts), dtype=bool)
        keep_mask[take_indices] = False
        deformed_pts = deformed_pts[keep_mask]
        deformed_normals = deformed_normals[keep_mask]
        operations.append({
            'type': 'advanced',
            'angle_deg': float(np.degrees(angle)),
            'bucket_width_m': bucket_width_m,
            'bucket_width_raw': float(bucket_width),
            'removed_points': int(len(take_indices)),
        })

    if operations and len(deformed_pts) > 100 and HAS_SCIPY:
        deformed_pts = knn_smooth(deformed_pts, k=15, factor=0.15)

    removal = 1.0 - len(deformed_pts) / max(len(pts), 1)
    meta['deform_params']['loading_simulation'] = {
        'version': 'v3_seeded_random_direction_physical_units',
        'direction_policy': 'seeded_random_per_operation_v3',
        'n_unique_operation_directions': len(operations),
        'coordinate_scale_to_m': float(scale_to_m),
        'n_operations': len(operations),
        'operations': operations,
        'removal_ratio': float(removal),
    }
    return deformed_pts


# ══════════════════════════════════════════════════════════
# 传统增强方法 (文献赋权)
# ══════════════════════════════════════════════════════════

def linear_scale_rot(pts, meta, cfg, rng):
    """
    缩放 + Z轴旋转 (P=0.80)
    依据: Qi et al. CVPR 2017 (PointNet) — 训练中随机旋转+缩放;
          Zhu et al. Pattern Recognit. 2024 (Survey) — 基础线性增强;
          Zhang et al. Sci. Rep. 2024 (PCAlign) — PCA对齐增强框架。
    物理意义: 模拟传感器安装位置和扫描距离的刚性变化。
    """
    deformed = pts.copy()
    # 各向同性缩放: 模拟不同扫描距离
    s = float(rng.uniform(*cfg['scale_range']))
    deformed = deformed * s
    meta['deform_params']['linear_scale'] = float(s)
    meta['deform_params']['scale_semantics'] = \
        'physical_geometry_scaling_same_coordinate_units'

    # Z轴旋转: 模拟传感器方位角变化
    angle = np.radians(float(rng.uniform(0, 360)))
    c, s_rad = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s_rad, 0], [s_rad, c, 0], [0, 0, 1]], dtype=np.float32)
    deformed = deformed @ R.T
    meta['deform_params']['rotation_z_deg'] = float(np.degrees(angle))
    return deformed


def nonlinear_surface_noise(pts, meta, cfg, rng):
    """
    多尺度高斯表面噪声 (P=0.50)
    依据: Qi et al. CVPR 2017 (PointNet) — 训练时使用高斯抖动 σ=0.01;
          Zhu et al. Pattern Recognit. 2024 (Survey) — 噪声增强综述;
          Shi et al. CVPR 2019 (PointRCNN) — 3D检测训练噪声策略。
    物理意义: 模拟户外LiDAR/无人机扫描的固有测量误差与大气扰动。
    """
    intensity = float(rng.uniform(*cfg['noise_int']))
    result = pts.copy()
    # 多尺度叠加: 大尺度低频 + 小尺度高频
    for scale in [1.0, 0.5]:
        result[:, 2] += rng.normal(0, intensity * scale, len(pts)).astype(np.float32)
    meta['deform_params']['surface_noise'] = {
        'intensity': float(intensity), 'axis_policy': 'z_only'
    }
    return result


def nonlinear_rbf(pts, meta, cfg, rng):
    """
    RBF径向基函数局部非刚性形变 (P=0.30)
    依据: Zhu et al. Pattern Recognit. 2024 (Survey) — 非刚性形变为高级增强;
          权重较低以防破坏散料堆安息角等物理约束。
    物理意义: 模拟局部地形微扰与散料颗粒局部塌陷。
    """
    if not HAS_SCIPY:
        return pts
    n_ctrl = int(rng.integers(cfg['rbf_ctrl_range'][0], cfg['rbf_ctrl_range'][1] + 1))
    n_ctrl = min(n_ctrl, len(pts))
    ctrl_idx = rng.choice(len(pts), n_ctrl, replace=False)
    ctrl_pts = pts[ctrl_idx]
    intensity = float(rng.uniform(*cfg['rbf_intensity']))
    disps = rng.normal(0, intensity, (n_ctrl, 3)).astype(np.float32)
    # Z方向重力约束: 向下变形概率较高
    disps[:, 2] = np.abs(disps[:, 2]) * rng.choice([-1, 1], n_ctrl, p=[0.7, 0.3])

    try:
        coords = (ctrl_pts[:, 0], ctrl_pts[:, 1], ctrl_pts[:, 2])
        rbf_funcs = [Rbf(*coords, disps[:, i], function='thin_plate') for i in range(3)]
        delta = np.column_stack([f(pts[:, 0], pts[:, 1], pts[:, 2]) for f in rbf_funcs])
        deformed = pts + delta.astype(np.float32)
        meta['deform_params']['rbf'] = {'n_ctrl': n_ctrl, 'intensity': float(intensity)}
        return deformed
    except Exception:
        return pts


# ══════════════════════════════════════════════════════════
# 单变体生成逻辑 (v6: LSDA模式100%装载 + 传统方法; traditional模式仅传统方法)
# 参数 mode: 'lsda'(默认) 或 'traditional_only'
# 当 mode='traditional_only' 时跳过装载模拟，用于生成纯传统方法对比基线变体
# ══════════════════════════════════════════════════════════

def augment_one(pts, normals, filename, var_id, cfg, mode, rng):
    """
    变体生成管道。所有随机性仅来自调用方提供的 Generator，不修改全局 RNG。
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError('augment_one requires a numpy.random.Generator')
    meta = {
        'id': f"{Path(filename).stem}_var{var_id:03d}",
        'source': str(filename),
        'data_type': cfg['type'],
        'variant_id': var_id,
        'deform_params': {},
        'applied_methods': [],
        'scheme': mode,  # 记录实际增强模式: lsda / loading_only / traditional_only
        'original_count': int(len(pts))
    }

    deformed_pts = pts.copy()
    deformed_normals = normals.copy() if normals is not None else None

    # ── 核心: 装载模拟 ────────────────────────────────
    # lsda 和 loading_only 模式执行装载模拟
    # traditional_only 模式跳过装载模拟（用于生成纯传统方法对比基线）
    if mode in ('lsda', 'loading_only'):
        deformed_pts = nonlinear_loading_simulation(
            deformed_pts, deformed_normals, meta, cfg, rng)
        meta['applied_methods'].append('loading_simulation')
        # 不做前沿方向对齐；每个 operation 使用种子驱动的独立随机方向。

    # ── 传统增强 (文献赋权, 在 lsda 和 traditional_only 模式下执行) ──
    # loading_only 模式跳过所有传统方法
    if mode == 'loading_only':
        pass  # 仅装载模拟，不叠加任何传统方法
    else:
        pass  # 下方 if 语句会正常执行

    # Scale + Rotation (P=0.80) — 仅 lsda 和 traditional_only 模式
    if mode != 'loading_only' and rng.random() < 0.80:
        try:
            deformed_pts = linear_scale_rot(deformed_pts, meta, cfg, rng)
            meta['applied_methods'].append('scale_and_rotation')
        except Exception:
            pass

    # Surface Noise (P=0.50) — 仅 lsda 和 traditional_only 模式
    if mode != 'loading_only' and rng.random() < 0.50:
        try:
            deformed_pts = nonlinear_surface_noise(deformed_pts, meta, cfg, rng)
            meta['applied_methods'].append('surface_noise')
        except Exception:
            pass

    # RBF Deformation (P=0.30) — 仅 lsda 和 traditional_only 模式
    if mode != 'loading_only' and rng.random() < 0.30:
        try:
            deformed_pts = nonlinear_rbf(deformed_pts, meta, cfg, rng)
            meta['applied_methods'].append('rbf_deformation')
        except Exception:
            pass

    # 安全检查: 点云过小时回退
    if len(deformed_pts) < 50:
        deformed_pts = pts.copy()
        meta['fallback'] = True

    meta['final_count'] = int(len(deformed_pts))
    meta['retention_ratio'] = float(len(deformed_pts) / max(len(pts), 1))
    return deformed_pts, meta


# ══════════════════════════════════════════════════════════
# 单文件处理函数 (顺序执行，避免Windows多进程问题)
# ══════════════════════════════════════════════════════════

def _validate_coordinate_scale_to_m(mode, value):
    """Validate standalone CLI scale before any file is processed."""
    if mode in ('lsda', 'loading_only'):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not np.isfinite(value) or value <= 0):
            raise ValueError(
                f"mode={mode} requires --coordinate-scale-to-m as a positive finite value")
        return float(value)
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not np.isfinite(value) or value <= 0):
        raise ValueError("--coordinate-scale-to-m must be positive and finite when supplied")
    return float(value)


def _standalone_variant_seed(parent_scan_id, method, variant_id, attempt=0):
    """Derive a stable local seed for non-A3 CLI generation identities."""
    identity = "\x1f".join((
        "step1-augmentation-v3", str(parent_scan_id), str(method),
        str(variant_id), str(attempt),
    ))
    return int.from_bytes(hashlib.sha256(identity.encode('utf-8')).digest()[:4], 'big')


def _process_one_file(args):
    # args: (ply_file_path, n_variants, output_dir, mode, coordinate_scale_to_m)
    ply_file_path, n_variants, output_dir, mode, coordinate_scale_to_m = args
    coordinate_scale_to_m = _validate_coordinate_scale_to_m(
        mode, coordinate_scale_to_m)
    output_dir = Path(output_dir)
    ply_file = Path(ply_file_path)
    results = []

    try:
        pts, normals = _load_ply_numpy(ply_file)
        if pts is None or len(pts) == 0:
            raise ValueError(f"点云为空: {ply_file}")
        pts, normals, _ = preprocess(pts, normals)
        try:
            print(f"  [{ply_file.name}] {len(pts):,}点 → 生成{n_variants}变体 (mode={mode})", flush=True)
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"  [ERROR] 加载 {ply_file.name}: {e}", flush=True)
        except Exception:
            pass
        return results

    cfg = get_data_config(ply_file.name)
    if coordinate_scale_to_m is not None:
        cfg['coordinate_scale_to_m'] = coordinate_scale_to_m

    for var_id in range(n_variants):
        try:
            seed = _standalone_variant_seed(ply_file.stem, mode, var_id)
            rng = np.random.default_rng(seed)
            d_pts, meta = augment_one(
                pts, normals, ply_file.name, var_id, cfg, mode=mode, rng=rng)
            d_nrm = estimate_normals_numpy(d_pts, k=20)
            out_ply = output_dir / f"{meta['id']}.ply"
            out_json = output_dir / f"{meta['id']}_meta.json"

            _save_ply_numpy(out_ply, d_pts, d_nrm)
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            results.append(meta)
            try:
                print(f"    [生成] {meta['id']}", flush=True)
            except Exception:
                pass
        except Exception as e:
            try:
                print(f"  [WARN] {ply_file.name} 变体{var_id} 失败: {e}", flush=True)
            except Exception:
                pass

    return results


# ══════════════════════════════════════════════════════════
# 主类
# ══════════════════════════════════════════════════════════

class BulkMaterialAugmentor:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, data_dir, n_variants=40, max_workers=1, mode='lsda',
            coordinate_scale_to_m=None):
        self.mode = mode  # lsda / loading_only / traditional_only
        coordinate_scale_to_m = _validate_coordinate_scale_to_m(
            mode, coordinate_scale_to_m)
        data_dir = Path(data_dir)
        ply_files = sorted(data_dir.glob('*.ply'))

        if not ply_files:
            print(f"[ERROR] 在 {data_dir} 中未找到 .ply 文件")
            return []

        print(f"\n{'='*62}")
        print(f"  LSDA 数据增强 v5 — Loading Simulation Core")
        print(f"  文件数: {len(ply_files)}")
        print(f"  每文件变体数: {n_variants}")
        strategy_map = {
            'lsda': '装载模拟(100%) + Scale/Rot(80%) + Noise(50%) + RBF(30%)',
            'loading_only': '仅装载模拟(100%)，不叠加传统方法',
            'traditional_only': '仅传统方法: Scale/Rot(80%) + Noise(50%) + RBF(30%)',
        }
        strategy_msg = strategy_map.get(mode, mode)
        print(f"  增强策略: {strategy_msg}")
        print(f"{'='*62}\n")

        args_list = [
            (str(f), n_variants, str(self.output_dir), self.mode,
             coordinate_scale_to_m)
            for f in ply_files
        ]
        all_meta = []
        t0 = time.time()
        done = 0

        for args in args_list:
            try:
                res = _process_one_file(args)
                if res is not None:
                    all_meta.extend(res)
                done += 1
                try:
                    print(f"  进度: {done}/{len(ply_files)} — {Path(args[0]).name} → {len(res) if res else 0} 变体", flush=True)
                except Exception:
                    pass
            except Exception as e:
                try:
                    import traceback
                    print(f"  [ERROR] {Path(args[0]).name}: {e}", flush=True)
                    traceback.print_exc()
                except Exception:
                    pass
                done += 1

        elapsed = time.time() - t0

        method_counts = {}
        for m in all_meta:
            for method in m.get('applied_methods', []):
                method_counts[method] = method_counts.get(method, 0) + 1

        summary = {
            'augmentation_mode': mode,  # lsda / loading_only / traditional_only
            'total_files': len(ply_files),
            'variants_per_file': n_variants,
            'total_variants': len(all_meta),
            'method_distribution': method_counts,
            'elapsed_seconds': round(elapsed, 1),
            'output_dir': str(self.output_dir),
            'augmentation_probs': {
                'loading_simulation': 1.00,
                'scale_and_rotation': 0.80,
                'surface_noise': 0.50,
                'rbf_deformation': 0.30,
            },
            'references': [
                '[1] Qi et al. CVPR 2017. PointNet.',
                '[2] Zhu et al. Pattern Recognit. 2024;153:110532.',
                '[3] Qi et al. NeurIPS 2017. PointNet++.',
                '[9] Wu et al. CVPR 2024. Point Transformer V3.',
                '[11] Xu et al. Autom Constr. 2024;162:105386.',
                '[12] Liao et al. Autom Constr. 2024;165:105546.',
            ],
            'all_metadata': all_meta
        }

        try:
            with open(self.output_dir / 'augmentation_summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [WARN] 保存summary失败: {e}")

        print(f"\n{'='*62}")
        print(f"  增强完成! 耗时: {elapsed:.1f}s")
        print(f"  总生成: {len(all_meta)} 个变体")
        print(f"  各增强方法触发统计:")
        for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"    - {m}: {c} 次 ({100*c/max(len(all_meta),1):.1f}%)")
        print(f"  输出目录: {self.output_dir}")
        print(f"{'='*62}\n")

        return all_meta


# ══════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════

def _parser():
    parser = argparse.ArgumentParser(
        description='散料堆点云数据增强 — LSDA v5 框架',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='./data', help='原始PLY文件目录')
    parser.add_argument('--output-dir', default='./outputs/augmented',
                        help='增强结果输出目录 (相对于运行目录)')
    parser.add_argument('--variants', type=int, default=40,
                        help='每文件生成变体数 (建议40)')
    parser.add_argument('--workers', type=int, default=1,
                        help='进程数 (Windows建议固定1)')
    parser.add_argument('--mode', default='lsda',
                        choices=['lsda', 'loading_only', 'traditional_only'],
                        help="增强模式: 'lsda'(默认,装载模拟+传统) | 'loading_only'(仅装载模拟) | 'traditional_only'(仅传统方法)")
    parser.add_argument(
        '--coordinate-scale-to-m', type=float, default=None,
        help='每个原始坐标单位对应的米数；lsda/loading_only 必填且必须为正有限值')
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        coordinate_scale_to_m = _validate_coordinate_scale_to_m(
            args.mode, args.coordinate_scale_to_m)
    except ValueError as exc:
        parser.error(str(exc))

    # Windows强制单进程
    args.workers = 1

    # 解析相对路径 (基于脚本所在目录)
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    aug = BulkMaterialAugmentor(str(output_dir))
    aug.run(args.data_dir, args.variants, args.workers, mode=args.mode,
            coordinate_scale_to_m=coordinate_scale_to_m)

    print(f"\n下一步: python step2_quality.py --aug-dir \"{output_dir}\"")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
