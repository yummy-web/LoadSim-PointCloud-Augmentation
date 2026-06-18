"""
step3_labels_annotated.py — 铲料区域几何伪标签生成器 (理论注释增强版)
======================================================================
任务: 点云逐点分割 → 铲料区域(1) vs 非铲料区域(0)

【伪标签生成理论基础】
═══════════════════════════════════════════════════════════════════
本模块基于三个物理驱动的几何约束生成伪标签，无需人工标注。
核心假设: 铲料区域具有三个可计算的几何特征:
  (A) 位于装载机朝向的前沿方向
  (B) 具有适合铲装作业的局部坡度 (20°~40°)
  (C) 在局部邻域中具有突出的高度 (凸起特征)

三个约束的数学形式化
───────────────────────────────────────────────────────────────────
设点云 P = {p_i ∈ R^3 | i=1,...,N}，各点法向量 n_i ∈ R^3。

[A] 前沿区域约束 (Frontier Constraint)
    ─────────────────────────────────────
    Step 1: 估计装载机主朝向轴
      ĥ = argmax_{j∈{X,Y}} E_i[|n_i^j|]
      即: 法向量水平分量绝对值均值最大的轴方向
      物理意义: 法向量水平分量主要集中于垂直土堆面的方向，
               即装载机面向土堆的方向

    Step 2: 前沿阈值
      F = {p_i | p_i^ĥ > Q_{α}({p_j^ĥ})}
      Q_α 为 α 分位数, α = 0.70
      物理意义: 沿主朝向轴坐标靠前的30%点为前沿候选

[B] 局部坡度约束 (Slope Constraint)
    ─────────────────────────────────────
    Step 1: k近邻PCA法向量估计
      对点 p_i 的 k=20 近邻集 N_k(p_i) = {p_{j1},...,p_{jk}}:
        Σ_i = (1/k) Σ_{j∈N_k(i)} (p_j - p_i)(p_j - p_i)^T
        n_i = argmin_{v: ||v||=1} v^T Σ_i v   (最小特征向量)
        若 n_i^z < 0, 则 n_i ← -n_i  (法向量朝上修正)

    Step 2: 坡度角计算
      cos(φ_i) = |n_i^z| / ||n_i||  (法向量与Z轴夹角余弦)
      θ_i = 90° - arccos(|n_i^z|)   (坡度角 = 90°减与Z轴夹角)
      注: θ=0° 表示水平面, θ=90° 表示垂直面

    Step 3: 坡度范围滤波
      S = {p_i | θ_min ≤ θ_i ≤ θ_max}
      θ_min = 20°, θ_max = 40°
      物理依据: 散料堆铲料区域的自然安息角通常在20°~45°之间,
               铲斗作业坡面角集中在20°~40°范围

[C] 局部显著性约束 (Local Prominence Constraint)
    ─────────────────────────────────────────────
    定义球形邻域:
      B_r(p_i) = {p_j ∈ P | ||p_j - p_i|| ≤ r}
      r = 50cm (或根据点云尺度自适应调整)

    显著性判定:
      C = {p_i | p_i^z ≥ max_{p_j ∈ B_r(p_i)} p_j^z - ε}
      ε = 1e-6 (数值容差)
      物理意义: p_i 是其球形邻域内Z值最高的点，
               即局部"山顶"，铲料作业通常从突出位置开始

[综合判定]
    Label(p_i) = 1  ⟺  p_i ∈ F ∩ S ∩ C
    Label(p_i) = 0  否则

    后处理:
    - 若铲料比例 < 1%: 逐步放宽约束(去凸起条件→放宽坡度)
    - 若铲料比例 > 40%: 保留Z值最高的50%点

参考文献:
  [L1] Liao PC et al. Shovel tip detection for autonomous excavation.
       Autom Constr. 2024;165:105546.
  [L2] Xu M et al. Physical simulation constraints for excavation.
       Autom Constr. 2024;162:105386.
  [L3] Rusu RB, Blodow N, Beetz M. Fast Point Feature Histograms.
       ICRA 2009.
  [L4] 安息角标准: GB/T 11270.2-2008 散料堆积角测定方法

═══════════════════════════════════════════════════════════════════

数据格式:
  输出 JSON: { "summary":{...}, "labels": { "stem": {...}, ... } }
"""

# ── 说明: 本文件是step3_labels.py的注释增强版本 ────────────────────────────
# 核心逻辑与step3_labels.py完全相同，增加了详细的理论注释
# 可直接替换step3_labels.py使用

import sys, io, json, argparse, warnings
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


def load_ply_with_normals(path):
    """返回 (pts: N×3, normals: N×3 or None)"""
    path = Path(path)
    if HAS_O3D:
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(pcd.points, dtype=np.float32)
            if len(pts) == 0:
                return None, None
            nrm = np.asarray(pcd.normals, dtype=np.float32)
            return pts, (nrm if len(nrm) == len(pts) else None)
        except Exception:
            pass
    pts, nrms = [], []
    has_nx = False
    n_vert = 0
    is_binary = False
    try:
        with open(path, 'rb') as f:
            while True:
                raw = f.readline()
                ln = raw.decode('ascii', errors='ignore').strip()
                if ln.startswith('element vertex'):
                    n_vert = int(ln.split()[-1])
                elif 'property' in ln.lower() and ' nx' in ln.lower():
                    has_nx = True
                elif 'binary_little_endian' in ln:
                    is_binary = True
                elif ln == 'end_header':
                    break
            if is_binary:
                dtype_f = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                if has_nx:
                    dtype_f += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
                dt = np.dtype(dtype_f)
                data = np.frombuffer(f.read(dt.itemsize * n_vert), dtype=dt)
                p = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
                n = np.column_stack([data['nx'], data['ny'], data['nz']]).astype(np.float32) if has_nx else None
                return p, n
            else:
                for _ in range(n_vert):
                    vals = f.readline().decode('ascii', errors='ignore').split()
                    if len(vals) >= 3:
                        try:
                            pts.append([float(v) for v in vals[:3]])
                            if has_nx and len(vals) >= 6:
                                nrms.append([float(v) for v in vals[3:6]])
                        except ValueError:
                            pass
    except Exception as e:
        print(f"  [WARN] {path.name}: {e}")
        return None, None
    p = np.array(pts, dtype=np.float32) if pts else None
    n = np.array(nrms, dtype=np.float32) if (has_nx and nrms and len(nrms) == len(pts)) else None
    return p, n


class ShovelRegionLabeler:
    """
    基于物理规则的铲料区域逐点分割伪标签生成器

    【算法核心思想】
    铲料区域是散料堆被铲斗作用的界面区域，具有三个可从点云几何中
    直接提取的物理特征，形成三重约束逐步收紧标签范围。

    参数说明:
      front_percentile:  前沿区域分位数阈值 α，默认0.70 (即前30%为前沿)
      slope_min/max:     有效坡度角范围 [θ_min, θ_max]，单位度，默认[20°,40°]
      convex_radius_abs: 局部显著性球形邻域半径，默认50cm
      k_normal:          法向量估计的近邻数，默认20
    """

    def __init__(self,
                 front_percentile: float = 0.70,
                 slope_min: float = 20.0,
                 slope_max: float = 40.0,
                 convex_radius_abs: float = 50.0,
                 convex_radius_rel: float = 0.05,
                 k_normal: int = 20):
        self.front_percentile  = front_percentile
        self.slope_min         = slope_min
        self.slope_max         = slope_max
        self.convex_radius_abs = convex_radius_abs
        self.convex_radius_rel = convex_radius_rel
        self.k_normal          = k_normal

    def _pca_normals(self, pts: np.ndarray) -> np.ndarray:
        """
        K近邻PCA法向量估计 [约束B Step1]

        数学原理:
          对点 p_i 的 k 近邻集合构建协方差矩阵:
            Σ_i = Σ_{j∈N_k(i)} (p_j - p_i)(p_j - p_i)^T / k
          法向量 n_i 是 Σ_i 的最小特征向量 (对应最小特征值)
          几何意义: 最小变化方向即局部切平面法向量
        """
        n = len(pts)
        normals = np.tile([0, 0, 1.0], (n, 1)).astype(np.float32)
        if not HAS_SCIPY or n < self.k_normal + 1:
            return normals
        tree = cKDTree(pts)
        k = min(self.k_normal, n - 1)
        _, idxs = tree.query(pts, k=k + 1)
        for i in range(n):
            nb = pts[idxs[i, 1:]] - pts[i]
            cov = (nb.T @ nb) / k
            try:
                _, vecs = np.linalg.eigh(cov)
                normals[i] = vecs[:, 0]  # 最小特征值对应的特征向量
            except np.linalg.LinAlgError:
                pass
        # 法向量朝上修正: 保证 n_z ≥ 0 (地面点云向上朝向约定)
        normals[normals[:, 2] < 0] *= -1
        return normals

    def _front_direction(self, normals: np.ndarray) -> int:
        """
        估计装载机主朝向轴 [约束A Step1]

        方法: 比较法向量水平分量 (x, y) 的绝对值均值
          ĥ = argmax_{j∈{X,Y}} E_i[|n_i^j|]
        返回: 0=X轴主方向, 1=Y轴主方向
        """
        h = np.abs(normals[:, :2]).mean(0)  # [mean|nx|, mean|ny|]
        return 1 if h[1] >= h[0] else 0

    def generate(self, pts: np.ndarray, normals: np.ndarray = None) -> np.ndarray:
        """
        生成铲料区域逐点二值标签

        Returns:
            labels: shape(N,) int32, 1=铲料区域, 0=背景
        """
        n = len(pts)
        labels = np.zeros(n, dtype=np.int32)
        if n < 100:
            return labels

        # 点云坐标尺度自动检测
        bbox = pts.max(0) - pts.min(0)
        bbox_diag = float(np.linalg.norm(bbox))
        z_range = float(bbox[2])
        xy_range = float(max(bbox[0], bbox[1]))

        # 自适应邻域半径: 米级坐标 vs 厘米级坐标
        if z_range < 50.0 and xy_range < 200.0:
            radius = self.convex_radius_abs / 100.0  # cm → m
        else:
            radius = min(self.convex_radius_abs,
                         self.convex_radius_rel * bbox_diag * 2)
            radius = max(radius, bbox_diag * 0.02)

        # 法向量: 使用提供的法向量或PCA估计
        if normals is not None and len(normals) == n:
            nrm = normals.astype(np.float32).copy()
            nrm[nrm[:, 2] < 0] *= -1
        else:
            nrm = self._pca_normals(pts)

        # ── [约束A] 前沿区域 ─────────────────────────────────────────────
        # 公式: F = {p_i | p_i^ĥ > Q_{α}({p_j^ĥ})}
        axis = self._front_direction(nrm)
        thr = np.percentile(pts[:, axis], self.front_percentile * 100)
        mask_front = pts[:, axis] > thr

        # ── [约束B] 局部坡度 ─────────────────────────────────────────────
        # 公式: θ_i = 90° - arccos(|n_i^z|), 范围: [θ_min, θ_max]
        cos_theta = np.clip(np.abs(nrm[:, 2]), 0, 1)
        slope_deg = 90.0 - np.degrees(np.arccos(cos_theta))
        mask_slope = (slope_deg >= self.slope_min) & (slope_deg <= self.slope_max)

        # ── [约束C] 局部显著性 ────────────────────────────────────────────
        # 公式: C = {p_i | p_i^z ≥ max_{p_j ∈ B_r(p_i)} p_j^z - ε}
        mask_convex = np.zeros(n, dtype=bool)
        cand_idx = np.where(mask_front & mask_slope)[0]

        if len(cand_idx) > 0:
            if HAS_SCIPY:
                tree = cKDTree(pts)
                for ci in cand_idx:
                    nb_idx = tree.query_ball_point(pts[ci], r=radius)
                    if nb_idx:
                        max_z_nb = pts[nb_idx, 2].max()
                        mask_convex[ci] = (pts[ci, 2] >= max_z_nb - 1e-6)
            else:
                # 无scipy: 用Z分位数近似凸起条件
                z_thr = np.percentile(pts[cand_idx, 2], 80)
                mask_convex[cand_idx] = pts[cand_idx, 2] >= z_thr

        # ── 综合判定: F ∩ S ∩ C ────────────────────────────────────────
        labels = (mask_front & mask_slope & mask_convex).astype(np.int32)
        ratio = labels.sum() / n

        # 后处理: 约束渐进放宽
        if ratio < 0.01:
            labels_r1 = (mask_front & mask_slope).astype(np.int32)
            if labels_r1.sum() / n >= 0.005:
                labels = labels_r1
            else:
                mask_slope2 = (slope_deg >= 10.0) & (slope_deg <= 55.0)
                labels = (mask_front & mask_slope2).astype(np.int32)
            ratio = labels.sum() / n

        if ratio > 0.40:
            si = np.where(labels == 1)[0]
            thr = np.percentile(pts[si, 2], 50)
            labels[si[pts[si, 2] < thr]] = 0

        return labels


def generate_labels(input_dirs, output_path: str,
                    labeler: ShovelRegionLabeler = None,
                    tag: str = ""):
    """批量生成铲料区域伪标签"""
    if labeler is None:
        labeler = ShovelRegionLabeler()

    if isinstance(input_dirs, (str, Path)):
        input_dirs = [input_dirs]

    ply_files = []
    for d in input_dirs:
        d = Path(d)
        if d.is_dir():
            ply_files.extend(sorted(d.glob("*.ply")))
        elif d.is_file() and d.suffix == '.ply':
            ply_files.append(d)

    seen, unique = set(), []
    for p in ply_files:
        if p.stem not in seen:
            seen.add(p.stem)
            unique.append(p)
    ply_files = unique

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  铲料区域伪标签生成 [{tag}]")
    print(f"  PLY文件: {len(ply_files)} 个")
    print(f"  算法参数: front={labeler.front_percentile:.0%} "
          f"slope=[{labeler.slope_min}°,{labeler.slope_max}°] "
          f"radius={labeler.convex_radius_abs}cm")
    print(f"{'='*55}")

    all_labels = {}
    ratios = []
    ok = fail = 0

    for i, ply in enumerate(ply_files):
        if (i + 1) % 100 == 0 or (i + 1) == len(ply_files):
            print(f"  进度: {i+1}/{len(ply_files)}", flush=True)

        pts, nrm = load_ply_with_normals(ply)
        if pts is None or len(pts) < 100:
            fail += 1
            continue

        seg = labeler.generate(pts, nrm)
        ratio = float(seg.sum() / len(seg))
        ratios.append(ratio)

        all_labels[ply.stem] = {
            'labels':   seg.tolist(),
            'n_pts':    int(len(pts)),
            'n_shovel': int(seg.sum()),
            'ratio':    ratio,
        }
        ok += 1

    summary = {
        'total': len(ply_files), 'ok': ok, 'fail': fail,
        'avg_ratio': float(np.mean(ratios)) if ratios else 0.0,
        'params': {
            'front_percentile':  labeler.front_percentile,
            'slope_range':       [labeler.slope_min, labeler.slope_max],
            'convex_radius_cm':  labeler.convex_radius_abs,
            'k_normal':          labeler.k_normal,
        },
        'theory': {
            'constraint_A': 'Frontier: p_i in F iff p_i^h > Q_alpha',
            'constraint_B': 'Slope: theta_i = 90 - arccos(|n_i^z|) in [theta_min, theta_max]',
            'constraint_C': 'Prominence: p_i^z >= max_{B_r} p_j^z - epsilon',
            'decision_rule': 'Label=1 iff p_i in F AND S AND C',
        },
        'generated_at': str(datetime.now()),
        'tag': tag,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'labels': all_labels},
                  f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ 完成: {ok}成功 / {fail}失败")
    print(f"  平均铲料比例: {summary['avg_ratio']:.2%}")
    print(f"  输出: {out_path}")
    return all_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='铲料区域逐点分割伪标签生成 (理论注释版)')
    parser.add_argument('--input-dirs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--front-percentile', type=float, default=0.70)
    parser.add_argument('--slope-min',        type=float, default=20.0)
    parser.add_argument('--slope-max',        type=float, default=40.0)
    parser.add_argument('--convex-radius',    type=float, default=50.0)
    parser.add_argument('--k-normal',         type=int,   default=20)
    parser.add_argument('--tag',              default='')
    a = parser.parse_args()

    _base = Path(__file__).parent
    def res(p): return str(Path(p) if Path(p).is_absolute() else _base / p)

    labeler = ShovelRegionLabeler(
        front_percentile=a.front_percentile,
        slope_min=a.slope_min, slope_max=a.slope_max,
        convex_radius_abs=a.convex_radius, k_normal=a.k_normal)
    generate_labels([res(d) for d in a.input_dirs], res(a.output), labeler, a.tag)
