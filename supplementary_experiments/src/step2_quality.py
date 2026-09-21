"""
step2_quality.py
================
散料堆点云质量评估与筛选

评估框架: Q = 0.40·F + 0.35·P + 0.25·D
  F: 保真度 (Fidelity)   — CD距离、HD距离、体积变化、特征相似度
  P: 物理可信度 (Physics) — 安息角、高宽比、表面连续性
  D: 多样性贡献 (Diversity) — 形状描述符偏差

筛选条件: Q >= 0.55；F/P/D 仅为评分分量，不独立作为硬门槛。

用法:
    python step2_quality.py \\
        --original-dir "D:/...data" \\
        --aug-dir "./outputs/augmented" \\
        --quality-config "./revision_v2/configs/quality_v1.yaml"
"""

import json, argparse, shutil, warnings, sys, io, hashlib, os, csv
import numpy as np
from pathlib import Path
from datetime import datetime
try:
    from scipy.spatial import cKDTree, ConvexHull
    HAS_SCIPY = True
except ImportError:
    cKDTree = ConvexHull = None
    HAS_SCIPY = False

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


# ══════════════════════════════════════════════════════════
# 点云加载工具
# ══════════════════════════════════════════════════════════

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


INVENTORY_FIELDS = {
    'sample_id', 'parent_scan_id', 'method', 'ply_path', 'point_count',
    'ply_sha256', 'metadata_path', 'metadata_sha256', 'generation_seed',
}


def _resolve_inventory_asset(root, relative, label):
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError(f'{label} must be a safe relative path: {relative!r}')
    root = Path(root).resolve()
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'{label} escapes inventory directory: {relative!r}') from exc
    if not resolved.is_file():
        raise FileNotFoundError(f'missing inventory {label}: {resolved}')
    return resolved


def load_candidate_inventory(path):
    """Load the generated nested candidate layout and verify every byte identity."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'missing candidate inventory: {path}')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    if not rows or not INVENTORY_FIELDS.issubset(fields):
        raise ValueError('invalid candidate inventory schema')
    ids = [row.get('sample_id', '') for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError('candidate inventory sample IDs must be non-empty and unique')

    metadata_rows = []
    methods = set()
    for row in rows:
        sample_id = row['sample_id']
        method = row['method'].upper()
        methods.add(method)
        if method not in {'TRAD', 'LOADSIM'}:
            raise ValueError(f'invalid inventory method for {sample_id}: {method}')
        try:
            point_count = int(row['point_count'])
            generation_seed = int(row['generation_seed'])
        except ValueError as exc:
            raise ValueError(f'invalid numeric inventory row: {sample_id}') from exc
        if point_count <= 0:
            raise ValueError(f'invalid point_count for {sample_id}')
        ply = _resolve_inventory_asset(path.parent, row['ply_path'], 'candidate PLY')
        metadata_path = _resolve_inventory_asset(
            path.parent, row['metadata_path'], 'candidate metadata')
        if sha256_file(ply) != row['ply_sha256']:
            raise ValueError(f'candidate PLY hash mismatch: {sample_id}')
        if sha256_file(metadata_path) != row['metadata_sha256']:
            raise ValueError(f'candidate metadata hash mismatch: {sample_id}')
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'invalid candidate metadata: {sample_id}') from exc
        if not isinstance(metadata, dict):
            raise ValueError(f'candidate metadata is not an object: {sample_id}')
        expected = {
            'sample_id': sample_id,
            'parent_scan_id': row['parent_scan_id'],
            'method': method,
            'generation_seed': generation_seed,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f'candidate metadata {key} mismatch: {sample_id}')
        if metadata.get('id') not in (None, sample_id):
            raise ValueError(f'candidate metadata id mismatch: {sample_id}')
        source = metadata.get('source')
        if not isinstance(source, str) or Path(source).name != source:
            raise ValueError(f'candidate metadata source is not an exact basename: {sample_id}')
        metadata = dict(metadata)
        metadata['id'] = sample_id
        metadata['_candidate_ply_path'] = str(ply)
        metadata['_candidate_ply_sha256'] = row['ply_sha256']
        metadata_rows.append(metadata)
    if len(methods) != 1:
        raise ValueError('one quality inventory must contain exactly one method')
    binding = {
        'schema_version': 'a3-candidate-inventory-v1',
        'sha256': sha256_file(path),
        'row_count': len(rows),
        'file_name': path.name,
        'method': next(iter(methods)),
    }
    return metadata_rows, binding


def load_pts(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        if HAS_O3D:
            pcd = o3d.io.read_point_cloud(str(path))
            return np.asarray(pcd.points, dtype=np.float32)
        else:
            pts = []
            with open(path, 'rb') as f:
                reading = False
                while True:
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    if line == 'end_header':
                        reading = True
                        break
                while reading:
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    if not line:
                        break
                    vals = line.split()
                    if len(vals) >= 3:
                        pts.append(vals[:3])
            return np.array(pts, dtype=np.float32) if pts else None
    except:
        return None


def sample(pts, n, *, seed):
    """Deterministic, call-order-independent point subsampling."""
    if len(pts) <= n:
        return pts
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.choice(len(pts), n, replace=False)
    idx.sort()
    return pts[idx]


# ══════════════════════════════════════════════════════════
# 保真度评估
# ══════════════════════════════════════════════════════════

class FidelityEvaluator:
    SAMPLE = 2000
    
    def chamfer_dist(self, p1, p2):
        """归一化倒角距离"""
        s1 = sample(p1, self.SAMPLE, seed=1101)
        s2 = sample(p2, self.SAMPLE, seed=1102)
        t1, t2 = cKDTree(s1), cKDTree(s2)
        d12, _ = t2.query(s1)
        d21, _ = t1.query(s2)
        cd = float(d12.mean() + d21.mean())
        scale = np.mean(p1.max(0) - p1.min(0))
        return cd / (scale + 1e-8)
    
    def hausdorff_dist(self, p1, p2):
        """归一化豪斯多夫距离"""
        s1 = sample(p1, self.SAMPLE, seed=1201)
        s2 = sample(p2, self.SAMPLE, seed=1202)
        t1, t2 = cKDTree(s1), cKDTree(s2)
        d12, _ = t2.query(s1)
        d21, _ = t1.query(s2)
        hd = float(max(d12.max(), d21.max()))
        scale = np.mean(p1.max(0) - p1.min(0))
        return hd / (scale + 1e-8)
    
    def volume_change(self, p1, p2):
        """凸包体积变化率"""
        def hull_vol(pts, seed):
            s = sample(pts, 1000, seed=seed)
            try:
                return ConvexHull(s).volume
            except Exception:
                dims = s.max(0) - s.min(0)
                return float(np.prod(np.maximum(dims, 1e-6)))

        v1, v2 = hull_vol(p1, 1301), hull_vol(p2, 1302)
        if v1 < 1e-10:
            return float('inf')
        return float(abs(v2 - v1) / v1)
    
    def feature_sim(self, p1, p2):
        """几何特征相似度"""
        def features(pts):
            c = pts.mean(0)
            dims = pts.max(0) - pts.min(0)
            cov = np.cov((pts - c).T)
            eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
            return np.array([
                dims[2], np.mean(dims[:2]),
                dims[2] / (np.mean(dims[:2]) + 1e-8),
                np.sqrt(1 - eigs[2] / (eigs[0] + 1e-8)),
                c[2]
            ])
        
        f1, f2 = features(p1), features(p2)
        diff = np.abs(f2 - f1) / (np.abs(f1) + 1e-8)
        return float(1 / (1 + diff.mean()))
    
    def compute(self, orig, deformed):
        """综合保真度评分 [0, 1]"""
        cd = self.chamfer_dist(orig, deformed)
        hd = self.hausdorff_dist(orig, deformed)
        vcr = self.volume_change(orig, deformed)
        fs = self.feature_sim(orig, deformed)
        
        cd_s = 1 / (1 + 5 * cd)
        hd_s = 1 / (1 + 3 * hd)
        vcr_s = 1 / (1 + 2 * vcr) if not np.isinf(vcr) else 0
        
        score = 0.30 * cd_s + 0.20 * hd_s + 0.25 * vcr_s + 0.25 * fs
        return float(score), {
            'estimated': False,
            'chamfer_dist': float(cd), 'hausdorff_dist': float(hd),
            'volume_change_ratio': float(vcr) if not np.isinf(vcr) else 999,
            'feature_similarity': float(fs),
            'cd_score': float(cd_s), 'hd_score': float(hd_s),
            'vcr_score': float(vcr_s), 'fs_score': float(fs)
        }


# ══════════════════════════════════════════════════════════
# 物理可信性评估
# ══════════════════════════════════════════════════════════

class PhysicsChecker:
    """细粒土物理参数约束验证"""
    REPOSE_MIN = 25.0   # 度，安息角下限
    REPOSE_MAX = 45.0   # 度，安息角上限
    HB_MIN = 0.10       # 高宽比下限
    HB_MAX = 1.50       # 高宽比上限
    
    def slope_score(self, pts):
        """坡面角度检验"""
        c = pts.mean(0)
        centered = pts - c
        r = np.linalg.norm(centered[:, :2], axis=1)
        r_max = r.max()
        
        slopes = []
        for i in range(8):
            r_lo, r_hi = i * r_max / 8, (i + 1) * r_max / 8
            mask = (r >= r_lo) & (r < r_hi)
            if mask.sum() < 8:
                continue
            local = centered[mask]
            cov = np.cov(local.T)
            _, vecs = np.linalg.eigh(cov)
            normal = vecs[:, 0]
            angle = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1))))
            slopes.append(angle)
        
        if not slopes:
            return 0.5
        
        mean_slope = np.mean(slopes)
        if self.REPOSE_MIN <= mean_slope <= self.REPOSE_MAX:
            return 1.0
        elif mean_slope < self.REPOSE_MIN:
            return max(0, 1 - (self.REPOSE_MIN - mean_slope) / 20)
        else:
            return max(0, 1 - (mean_slope - self.REPOSE_MAX) / 20)
    
    def hb_ratio_score(self, pts):
        """高宽比检验"""
        dims = pts.max(0) - pts.min(0)
        base = np.sqrt(dims[0] * dims[1])
        if base < 1e-6:
            return 0.0
        ratio = dims[2] / base
        if self.HB_MIN <= ratio <= self.HB_MAX:
            return 1.0
        elif ratio < self.HB_MIN:
            return max(0, ratio / self.HB_MIN)
        else:
            return max(0, 1 - (ratio - self.HB_MAX) / self.HB_MAX)
    
    def continuity_score(self, pts):
        """表面连续性检验 (KNN距离变异系数)"""
        s = sample(pts, 2000, seed=1401)
        tree = cKDTree(s)
        dists, _ = tree.query(s, k=6)
        nn = dists[:, 1:]
        cv = nn.std() / (nn.mean() + 1e-8)
        return float(max(0, 1 - cv * 0.5))
    
    def compute(self, pts):
        """综合物理可信度 [0, 1]"""
        if len(pts) < 100:
            return 0.0, {}
        
        try:
            ss = self.slope_score(pts)
            hs = self.hb_ratio_score(pts)
            cs = self.continuity_score(pts)
            total = float((ss + hs + cs) / 3)
            return total, {'slope_score': float(ss), 'hb_score': float(hs), 'continuity_score': float(cs)}
        except Exception as e:
            return 0.3, {'error': str(e)}


# ══════════════════════════════════════════════════════════
# 多样性评估
# ══════════════════════════════════════════════════════════

class DiversityEvaluator:
    def descriptor(self, pts):
        """提取7维形状描述符"""
        c = pts.mean(0)
        centered = pts - c
        dims = pts.max(0) - pts.min(0)
        cov = np.cov(centered.T)
        eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
        r = np.linalg.norm(centered[:, :2], axis=1)
        z = centered[:, 2]
        
        return np.array([
            dims[0] / (dims[2] + 1e-8),
            dims[1] / (dims[2] + 1e-8),
            dims[2],
            eigs[0] / (eigs[2] + 1e-8),
            r.std() / (r.mean() + 1e-8),
            (np.mean((z - z.mean())**3) / (z.std()**3 + 1e-8)),  # skewness
            c[2]
        ])
    
    def diversity_contribution(self, desc, all_descs):
        """计算该描述符相对于整体的多样性贡献"""
        if len(all_descs) < 2:
            return 0.5
        D = np.array(all_descs)
        mu = D.mean(0)
        std = D.std(0)
        std[std < 1e-8] = 1.0
        norm = (desc - mu) / std
        deviation = np.abs(norm).mean()
        return float(min(deviation / 2, 1.0))


# ══════════════════════════════════════════════════════════
# 综合质量分析器
# ══════════════════════════════════════════════════════════

class QualityAnalyzer:
    WEIGHTS = {'fidelity': 0.40, 'physics': 0.35, 'diversity': 0.25}
    THRESHOLDS = {
        'total': 0.55, 'excellent': 0.72, 'good': 0.62, 'fair': 0.55
    }
    FILTER_RULE = {'name': 'Q-only', 'threshold_Q': 0.55,
                   'component_hard_gates': []}

    def __init__(self, original_dir, aug_dir, weights=None, threshold=0.55):
        self.original_dir = Path(original_dir)
        self.aug_dir = Path(aug_dir)
        self.WEIGHTS = dict(weights or self.WEIGHTS)
        self.THRESHOLDS = dict(self.THRESHOLDS)
        self.THRESHOLDS['total'] = float(threshold)
        self.FILTER_RULE = {'name': 'Q-only', 'threshold_Q': float(threshold),
                            'component_hard_gates': []}
        self._candidate_paths = {}
        self.fid_eval = FidelityEvaluator()
        self.phys_eval = PhysicsChecker()
        self.div_eval = DiversityEvaluator()
    
    def load_all_meta(self):
        """加载所有变体元数据"""
        summary = self.aug_dir / 'augmentation_summary.json'
        if summary.exists():
            with open(summary, encoding='utf-8') as f:
                return json.load(f)['all_metadata']
        meta_files = sorted(self.aug_dir.glob('*_meta.json'))
        result = []
        for mf in meta_files:
            with open(mf, encoding='utf-8') as f:
                result.append(json.load(f))
        return result
    
    def load_original_cache(self, all_meta):
        """按元数据中的精确 basename 预加载原始点云；任何缺失均失败。"""
        cache = {}
        for meta in all_meta:
            source = meta.get('source')
            if not isinstance(source, str) or not source:
                raise ValueError("candidate metadata lacks a non-empty source")
            source_path = Path(source)
            src = source_path.name
            if source_path.is_absolute() or src != source:
                raise ValueError(f"source must be an exact basename, got: {source!r}")
            if src in cache:
                continue
            original = self.original_dir / src
            if not original.is_file():
                raise FileNotFoundError(f"missing exact original point cloud: {original}")
            pts = load_pts(original)
            if pts is None or len(pts) == 0:
                raise ValueError(f"unreadable/empty original point cloud: {original}")
            cache[src] = {'points': pts, 'sha256': sha256_file(original)}
        print(f"  缓存原始点云: {len(cache)} 个")
        return cache
    
    def analyze(self, all_meta=None):
        """执行批量质量分析"""
        if not HAS_SCIPY:
            raise RuntimeError("SciPy is required for quality metric computation")
        if all_meta is None:
            all_meta = self.load_all_meta()
        if not isinstance(all_meta, list) or not all_meta:
            raise ValueError("quality analysis requires a non-empty metadata list")
        if any(not isinstance(meta, dict) or not isinstance(meta.get('id'), str)
               for meta in all_meta):
            raise ValueError("every candidate metadata record requires a string id")
        ids = [meta['id'] for meta in all_meta]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate metadata IDs must be unique")
        all_meta = sorted(all_meta, key=lambda meta: meta['id'])

        print(f"  分析变体数: {len(all_meta)}")
        orig_cache = self.load_original_cache(all_meta)
        
        results = []
        descriptors = []
        
        print("\n  [阶段1] 计算保真度 & 物理可信度...")
        for i, meta in enumerate(all_meta):
            if (i + 1) % 50 == 0:
                print(f"    进度: {i+1}/{len(all_meta)}")
            
            inventory_path = meta.get('_candidate_ply_path')
            if inventory_path is None:
                aug_ply = self.aug_dir / f"{meta['id']}.ply"
            else:
                aug_ply = Path(inventory_path)
                expected_hash = meta.get('_candidate_ply_sha256')
                if (not aug_ply.is_file() or not isinstance(expected_hash, str) or
                        sha256_file(aug_ply) != expected_hash):
                    raise ValueError(
                        f"inventory candidate identity changed before analysis: {meta['id']}")
            self._candidate_paths[meta['id']] = aug_ply
            aug_pts = load_pts(aug_ply)

            src = Path(meta['source']).name
            original = orig_cache[src]
            orig_pts = original['points']

            if aug_pts is None or len(aug_pts) == 0:
                raise ValueError(f"unreadable/empty augmented point cloud: {aug_ply}")

            fid_score, fid_detail = self.fid_eval.compute(orig_pts, aug_pts)
            
            # 物理可信度
            phys_score, phys_detail = self.phys_eval.compute(aug_pts)
            
            # 形状描述符
            try:
                desc = self.div_eval.descriptor(aug_pts)
            except:
                desc = np.zeros(7)
            
            r = {
                'variant_id': meta['id'],
                'source': src,
                'source_ply_sha256': original['sha256'],
                'candidate_ply_sha256': sha256_file(aug_ply),
                'data_type': meta.get('data_type', 'unknown'),
                'scheme': meta.get('scheme', 'N/A'),
                # ★ v6: 记录 applied_methods 供 report() 中按方法统计
                'applied_methods': list(meta.get('applied_methods', [])),
                'deform_types': list(meta.get('deform_params', {}).keys()),
                'retention_ratio': meta.get('retention_ratio', 1.0),
                'fidelity_score': fid_score,
                'physics_score': phys_score,
                'fidelity_detail': fid_detail,
                'physics_detail': phys_detail,
                'shape_descriptor': desc.tolist(),
                'status': 'ok'
            }
            results.append(r)
            descriptors.append(desc)
        
        print(f"\n  [阶段2] 计算多样性贡献...")
        for r in results:
            try:
                desc = np.array(r['shape_descriptor'])
                div_contrib = self.div_eval.diversity_contribution(desc, descriptors)
            except:
                div_contrib = 0.3
            r['diversity_contribution'] = div_contrib
            
            # 综合评分
            total = (self.WEIGHTS['fidelity'] * r['fidelity_score'] +
                     self.WEIGHTS['physics'] * r['physics_score'] +
                     self.WEIGHTS['diversity'] * div_contrib)
            r['total_score'] = float(total)
            
            # 质量等级
            if total >= self.THRESHOLDS['excellent']:
                r['quality_level'] = 'excellent'
            elif total >= self.THRESHOLDS['good']:
                r['quality_level'] = 'good'
            elif total >= self.THRESHOLDS['fair']:
                r['quality_level'] = 'fair'
            else:
                r['quality_level'] = 'poor'
            
            # 通过判定
            r['passed'] = total >= self.THRESHOLDS['total']
        
        return results
    
    def report(self, results):
        """生成并打印报告"""
        total = len(results)
        passed = sum(1 for r in results if r.get('passed'))
        
        dist = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        for r in results:
            dist[r.get('quality_level', 'poor')] += 1
        
        # ★ v6修正: 从 applied_methods 列表检测装载模拟变体
        # step1中每个变体的meta里用 applied_methods=['loading_simulation', ...]
        # step2分析时把 meta['applied_methods'] 存入 r['applied_methods']
        # 按 scheme（增强模式）分组统计质量
        scheme_groups = {}
        for r in results:
            sc = r.get('scheme', 'unknown')
            scheme_groups.setdefault(sc, []).append(r)

        # 按数据类型分组（near_field / aerial）
        type_groups = {}
        for r in results:
            dt = r.get('data_type', 'unknown')
            type_groups.setdefault(dt, []).append(r)

        loading_res = [r for r in results if 'loading_simulation' in r.get('applied_methods', [])]
        nonlinear_res = results
        
        def avg_score(lst):
            return np.mean([r['total_score'] for r in lst]) if lst else 0
        
        def scheme_stats(lst):
            if not lst: return {'count': 0, 'pass_rate': 0, 'avg_f': 0, 'avg_p': 0, 'avg_d': 0, 'avg_q': 0}
            p = [x for x in lst if x.get('passed')]
            return {
                'count': len(lst),
                'passed': len(p),
                'pass_rate': len(p) / len(lst),
                'avg_fidelity': float(np.mean([x['fidelity_score'] for x in lst])),
                'avg_physics':  float(np.mean([x['physics_score'] for x in lst])),
                'avg_diversity': float(np.mean([x.get('diversity_contribution', 0) for x in lst])),
                'avg_total':    float(np.mean([x['total_score'] for x in lst])),
            }
        
        if total == 0:
            print("[WARN] 无有效分析结果，跳过报告生成")
            return {'schema_version': 'quality-report-v1',
                    'generated_by': 'step2_quality.py',
                    'summary': {'total': 0, 'passed': 0, 'pass_rate': 0.0,
                                'analysis_time': str(datetime.now())},
                    'quality_distribution': dist, 'by_method': {},
                    'filter_rule': self.FILTER_RULE,
                    'thresholds': self.THRESHOLDS,
                    'weights': self.WEIGHTS, 'results': []}

        print(f"\n{'='*60}")
        print(f"  质量评估报告")
        print(f"{'='*60}")
        pass_rate = passed / total if total > 0 else 0.0
        print(f"  总计: {total} | 通过: {passed} | 通过率: {pass_rate:.1%}")
        print(f"\n  质量分布:")
        for lvl, cnt in dist.items():
            print(f"    {lvl:10s}: {cnt:4d} ({cnt/total:.1%})")
        print(f"\n  各增强模式质量统计:")
        for sc, lst in sorted(scheme_groups.items()):
            stats = scheme_stats(lst)
            print(f"    {sc:20s}: n={stats['count']:4d}  通过={stats['passed']:4d}({stats['pass_rate']:.1%})  "
                  f"F={stats['avg_fidelity']:.3f}  P={stats['avg_physics']:.3f}  "
                  f"D={stats['avg_diversity']:.3f}  Q={stats['avg_total']:.3f}")
        print(f"\n  各数据类型统计:")
        for dt, lst in sorted(type_groups.items()):
            stats = scheme_stats(lst)
            print(f"    {dt:20s}: n={stats['count']:4d}  通过={stats['passed']:4d}({stats['pass_rate']:.1%})  Q={stats['avg_total']:.3f}")
        
        report_data = {
            'schema_version': 'quality-report-v1',
            'generated_by': 'step2_quality.py',
            'summary': {'total': total, 'passed': passed, 'pass_rate': pass_rate,
                        'analysis_time': str(datetime.now())},
            'quality_distribution': dist,
            'by_scheme': {sc: scheme_stats(lst) for sc, lst in scheme_groups.items()},
            'by_data_type': {dt: scheme_stats(lst) for dt, lst in type_groups.items()},
            'by_method': {
                'loading_sim': {'count': len(loading_res), 'avg_score': float(avg_score(loading_res)) if loading_res else 0.0},
                'all': {'count': len(nonlinear_res), 'avg_score': float(avg_score(nonlinear_res))},
            },
            'thresholds': self.THRESHOLDS,
            'weights': self.WEIGHTS,
            'filter_rule': self.FILTER_RULE,
            'results': results
        }
        return report_data
    
    def filter_and_save(self, results):
        """事务化发布完整 high_quality 快照，不保留上次运行的陈旧文件。"""
        passed = [r for r in results if r.get('passed')]
        passed_ids = [r.get('variant_id') for r in passed]
        if any(not isinstance(item, str) or not item for item in passed_ids):
            raise ValueError("every passed result requires a non-empty variant_id")
        if len(passed_ids) != len(set(passed_ids)):
            raise ValueError("passed variant IDs must be unique")
        sources = [self._candidate_paths.get(item, self.aug_dir / f"{item}.ply")
                   for item in passed_ids]
        missing = [str(path) for path in sources if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing passed candidate PLY files: {missing}")

        out_dir = self.aug_dir.parent / 'high_quality'
        staging = out_dir.with_name(f".{out_dir.name}.staging-{os.getpid()}")
        backup = out_dir.with_name(f".{out_dir.name}.backup-{os.getpid()}")
        if staging.exists() or backup.exists():
            raise FileExistsError("stale high_quality staging/backup directory exists")
        old_moved = False
        try:
            staging.mkdir(parents=True)
            for src in sources:
                shutil.copy2(src, staging / src.name)
            info = {
                'total': len(results), 'passed': len(passed),
                'pass_rate': len(passed) / len(results) if results else 0,
                'passed_ids': passed_ids,
            }
            with (staging / 'filter_info.json').open('w', encoding='utf-8') as handle:
                json.dump(info, handle, ensure_ascii=False, indent=2)
            if out_dir.exists():
                os.replace(out_dir, backup)
                old_moved = True
            os.replace(staging, out_dir)
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            if old_moved and backup.exists() and not out_dir.exists():
                os.replace(backup, out_dir)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        print(f"\n  筛选完成:")
        print(f"    高质量变体: {len(passed)} 个 → {out_dir}")
        print(f"    淘汰变体:   {len(results) - len(passed)} 个")
        return passed


# ══════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════

def load_frozen_quality_config(path):
    """Load and validate the revision-v2 Q-only contract."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load quality_v1.yaml") from exc
    path = Path(path).resolve()
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or cfg.get('version') != 'v1' or cfg.get('rule') != 'Q-only':
        raise ValueError(f"invalid Q-only quality config: {path}")
    weights = {
        'fidelity': cfg.get('components', {}).get('F', {}).get('weight'),
        'physics': cfg.get('components', {}).get('P', {}).get('weight'),
        'diversity': cfg.get('components', {}).get('D', {}).get('weight'),
    }
    threshold = cfg.get('threshold', {}).get('Q')
    if weights != {'fidelity': 0.40, 'physics': 0.35, 'diversity': 0.25} or threshold != 0.55:
        raise ValueError(f"quality config differs from frozen Q-only v1: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, weights, float(threshold), digest


if __name__ == "__main__":
    _script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description='散料堆点云质量评估（冻结 Q-only v1）')
    parser.add_argument('--original-dir', default='./data')
    parser.add_argument('--aug-dir', default='./outputs/augmented')
    parser.add_argument('--quality-config', default='./revision_v2/configs/quality_v1.yaml')
    parser.add_argument('--inventory', default=None,
                        help='Formal candidate_inventory.csv; binds nested assets and report identity')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Compatibility check only; must equal the frozen config threshold')
    args = parser.parse_args()

    # 相对路径统一解析为基于脚本目录的绝对路径
    aug_dir = Path(args.aug_dir) if Path(args.aug_dir).is_absolute() else _script_dir / args.aug_dir
    quality_path = (Path(args.quality_config) if Path(args.quality_config).is_absolute()
                    else _script_dir / args.quality_config)
    quality_path, frozen_weights, frozen_threshold, quality_config_sha256 = \
        load_frozen_quality_config(quality_path)
    if args.threshold is not None and args.threshold != frozen_threshold:
        parser.error(f'--threshold must equal frozen Q threshold {frozen_threshold}')

    inventory_binding = None
    if args.inventory is not None:
        inventory_path = (Path(args.inventory) if Path(args.inventory).is_absolute()
                          else _script_dir / args.inventory).resolve()
        all_meta, inventory_binding = load_candidate_inventory(inventory_path)
        aug_dir = inventory_path.parent
    analyzer = QualityAnalyzer(args.original_dir, str(aug_dir),
                               weights=frozen_weights, threshold=frozen_threshold)
    if inventory_binding is None:
        all_meta = analyzer.load_all_meta()

    print(f"\n开始质量分析 (共 {len(all_meta)} 个变体)...")
    results = analyzer.analyze(all_meta)
    
    report_data = analyzer.report(results)
    report_data['quality_contract'] = {
        'path': quality_path.name,
        'sha256': quality_config_sha256,
        'version': 'v1',
        'rule': 'Q-only',
    }
    if inventory_binding is not None:
        report_data['candidate_inventory'] = inventory_binding

    # 保存报告
    report_path = aug_dir / 'quality_report.json'
    def cvt(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: cvt(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cvt(i) for i in obj]
        return obj
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(cvt(report_data), f, ensure_ascii=False, indent=2)
    print(f"\n  质量报告已保存: {report_path}")
    

    # ★ v6: 导出 CSV 表格，方便后续论文数据查询
    try:
        import csv
        csv_path = aug_dir / 'quality_results_table.csv'
        fieldnames = ['variant_id', 'source', 'data_type', 'fidelity_score',
                      'physics_score', 'diversity_contribution', 'total_score',
                      'quality_level', 'passed', 'applied_methods']
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in results:
                row = {k: r.get(k, '') for k in fieldnames}
                row['applied_methods'] = '|'.join(r.get('applied_methods', []))
                writer.writerow(row)
        print(f"  质量结果CSV已导出: {csv_path}")
        summary_csv = aug_dir / 'quality_summary_table.csv'
        with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['指标', '值'])
            s = report_data['summary']
            for k, v in s.items():
                writer.writerow([k, v])
            writer.writerow([])
            writer.writerow(['质量等级', '数量', '占比'])
            for lvl, cnt in report_data['quality_distribution'].items():
                writer.writerow([lvl, cnt, f"{cnt/s['total']:.1%}"])
        print(f"  摘要CSV已导出: {summary_csv}")
    except Exception as csv_e:
        print(f"  [WARN] CSV导出失败: {csv_e}")

    # 筛选
    analyzer.filter_and_save(results)
    print(f"\n下一步: python step3a_geo_labels.py")
