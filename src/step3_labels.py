"""
step3_labels.py — 铲料区域几何伪标签生成器
==========================================
任务: 点云逐点分割 → 铲料区域(1) vs 非铲料区域(0)

算法 (基于物理规则，所有分支完全相同以保证公平性):
  Step 1: 前沿区域  — 利用法向量水平分量确定装载机朝向
                     Y轴坐标 > 70%分位数的点标记为前沿候选
  Step 2: 局部坡度  — 20近邻PCA法向量与Z轴夹角 ∈ [20°, 40°]
  Step 3: 局部凸起  — 0.5m(=50cm)球形邻域内Z值最大的点
  Step 4: 综合判定  — 三条件AND → 铲料区域

数据格式:
  输出 JSON: { "summary":{...}, "labels": { "stem": [0,1,0,...], ... } }
"""
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


# ── PLY 加载 (支持法向量) ──────────────────────────────────────────────────

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
    # numpy fallback
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
                dtype_f = [('x','f4'),('y','f4'),('z','f4')]
                if has_nx:
                    dtype_f += [('nx','f4'),('ny','f4'),('nz','f4')]
                dt = np.dtype(dtype_f)
                data = np.frombuffer(f.read(dt.itemsize * n_vert), dtype=dt)
                p = np.column_stack([data['x'],data['y'],data['z']]).astype(np.float32)
                n = np.column_stack([data['nx'],data['ny'],data['nz']]).astype(np.float32) if has_nx else None
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
    n = np.array(nrms, dtype=np.float32) if (has_nx and nrms and len(nrms)==len(pts)) else None
    return p, n


# ── 铲料区域伪标签生成器 ──────────────────────────────────────────────────

class ShovelRegionLabeler:
    """
    基于物理规则的铲料区域逐点分割伪标签生成器

    所有分支 (baseline / traditional / lsda) 使用完全相同的参数，保证公平性。

    输入点云单位假设: 与原始扫描一致 (近景cm级，无人机可能m级)
    自适应处理: 根据点云包围盒尺寸自动调整 convex_radius
    """
    def __init__(self,
                 front_percentile: float = 0.70,
                 slope_min: float = 20.0,
                 slope_max: float = 40.0,
                 convex_radius_abs: float = 50.0,   # 绝对半径(cm)
                 convex_radius_rel: float = 0.05,   # 相对半径(包围盒5%)
                 k_normal: int = 20):
        self.front_percentile  = front_percentile
        self.slope_min         = slope_min
        self.slope_max         = slope_max
        self.convex_radius_abs = convex_radius_abs
        self.convex_radius_rel = convex_radius_rel
        self.k_normal          = k_normal

    def _pca_normals(self, pts: np.ndarray) -> np.ndarray:
        """使用K近邻PCA估计法向量 (当输入不包含法向量时)"""
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
                normals[i] = vecs[:, 0]
            except np.linalg.LinAlgError:
                pass
        # 法向量朝上修正
        normals[normals[:, 2] < 0] *= -1
        return normals

    def _front_direction(self, normals: np.ndarray) -> int:
        """
        利用法向量水平分量确定装载机朝向轴 (0=X, 1=Y)
        装载机面向土堆 → 法向量水平分量的主方向即为装载机方向
        """
        h = np.abs(normals[:, :2]).mean(0)   # [mean|nx|, mean|ny|]
        return 1 if h[1] >= h[0] else 0      # Y轴(=1) or X轴(=0)

    def generate(self, pts: np.ndarray, normals: np.ndarray = None) -> np.ndarray:
        """
        生成铲料区域逐点标签

        重要: 点云坐标单位自适应处理
          - 算法自动检测坐标尺度 (米/厘米均可)
          - convex_radius_abs 在 config.py 中单位为cm,
            但内部会按实际点云尺度换算

        Returns:
            labels: np.ndarray shape(N,) dtype=int32, 1=铲料, 0=非铲料
        """
        n = len(pts)
        labels = np.zeros(n, dtype=np.int32)
        if n < 100:
            return labels

        # ── 坐标尺度自动检测 ─────────────────────────────────────────
        # 通过包围盒Z轴高度判断单位
        # 近景扫描: Z范围通常 50-500 (cm) 或 0.5-5 (m)
        # 无人机扫描: 相似
        # 策略: 若Z范围 < 20，认为单位是米; 否则认为是cm或任意单位
        bbox       = pts.max(0) - pts.min(0)
        bbox_diag  = float(np.linalg.norm(bbox))
        z_range    = float(bbox[2])
        xy_range   = float(max(bbox[0], bbox[1]))

        # 相邻点典型间距估计 (用于设定有意义的convex半径)
        # 若 z_range < 50: 可能是米级坐标，将50cm转换为0.5
        if z_range < 50.0 and xy_range < 200.0:
            # 米级坐标系: 凸起半径 = convex_radius_abs / 100 (cm→m)
            radius = self.convex_radius_abs / 100.0
            unit_label = 'm'
        else:
            # cm级或其他: 直接用绝对半径，与包围盒5%取较小值防止过大
            radius = min(self.convex_radius_abs,
                         self.convex_radius_rel * bbox_diag * 2)
            # 保证半径至少覆盖一定点数：与点间距相关
            radius = max(radius, bbox_diag * 0.02)
            unit_label = 'cm'

        # ── 使用提供的法向量，否则用PCA估计 ──────────────────────────
        if normals is not None and len(normals) == n:
            nrm = normals.astype(np.float32).copy()
            nrm[nrm[:, 2] < 0] *= -1   # 保证朝上
        else:
            nrm = self._pca_normals(pts)

        # ── Step 1: 前沿区域 ──────────────────────────────────────────
        axis = self._front_direction(nrm)   # 0=X, 1=Y
        thr  = np.percentile(pts[:, axis], self.front_percentile * 100)
        mask_front = pts[:, axis] > thr

        # ── Step 2: 局部坡度 ──────────────────────────────────────────
        cos_theta  = np.clip(np.abs(nrm[:, 2]), 0, 1)
        slope_deg  = 90.0 - np.degrees(np.arccos(cos_theta))
        mask_slope = (slope_deg >= self.slope_min) & (slope_deg <= self.slope_max)

        # ── Step 3: 局部凸起 ──────────────────────────────────────────
        mask_convex = np.zeros(n, dtype=bool)
        cand_idx    = np.where(mask_front & mask_slope)[0]

        if len(cand_idx) > 0:
            if HAS_SCIPY:
                tree = cKDTree(pts)
                for ci in cand_idx:
                    nb_idx = tree.query_ball_point(pts[ci], r=radius)
                    if nb_idx:
                        max_z_nb = pts[nb_idx, 2].max()
                        mask_convex[ci] = (pts[ci, 2] >= max_z_nb - 1e-6)
            else:
                z_thr = np.percentile(pts[cand_idx, 2], 80)
                mask_convex[cand_idx] = pts[cand_idx, 2] >= z_thr

        # ── Step 4: 综合判定 ──────────────────────────────────────────
        labels = (mask_front & mask_slope & mask_convex).astype(np.int32)
        ratio  = labels.sum() / n

        # 后处理: 若比例 < 1% 则逐步放宽约束
        if ratio < 0.01:
            # 放宽1: 去掉凸起条件
            labels_r1 = (mask_front & mask_slope).astype(np.int32)
            if labels_r1.sum() / n >= 0.005:
                labels = labels_r1
            else:
                # 放宽2: 同时放宽坡度
                mask_slope2 = (slope_deg >= 10.0) & (slope_deg <= 55.0)
                labels = (mask_front & mask_slope2).astype(np.int32)
            ratio = labels.sum() / n

        if ratio > 0.40:
            # 过多 → 只保留Z值最高的前半
            si  = np.where(labels == 1)[0]
            thr = np.percentile(pts[si, 2], 50)
            labels[si[pts[si, 2] < thr]] = 0

        return labels


# ── 批量生成接口 ───────────────────────────────────────────────────────────

def generate_labels(input_dirs, output_path: str,
                     labeler: ShovelRegionLabeler = None,
                     tag: str = ""):
    """
    对多个目录中的PLY文件批量生成铲料区域伪标签

    Args:
        input_dirs: str / Path / list[str|Path]
        output_path: 输出JSON路径
        labeler: ShovelRegionLabeler实例
        tag: 日志标识符
    """
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

    # 去重
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
    ratios     = []
    ok = fail  = 0

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
            'labels':  seg.tolist(),
            'n_pts':   int(len(pts)),
            'n_shovel': int(seg.sum()),
            'ratio':   ratio,
        }
        ok += 1

    summary = {
        'total': len(ply_files), 'ok': ok, 'fail': fail,
        'avg_ratio': float(np.mean(ratios)) if ratios else 0.0,
        'params': {
            'front_percentile': labeler.front_percentile,
            'slope_range': [labeler.slope_min, labeler.slope_max],
            'convex_radius_cm': labeler.convex_radius_abs,
            'k_normal': labeler.k_normal,
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


# ── 主程序 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='铲料区域逐点分割伪标签生成')
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
