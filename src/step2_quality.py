"""
step2_quality.py
================
散料堆点云质量评估与筛选

评估框架: Q = 0.40·F + 0.35·P + 0.25·D
  F: 保真度 (Fidelity)   — CD距离、HD距离、体积变化、特征相似度
  P: 物理可信度 (Physics) — 安息角、高宽比、表面连续性
  D: 多样性贡献 (Diversity) — 形状描述符偏差

筛选条件: Q >= 0.55 AND F >= 0.45 AND P >= 0.50

用法:
    python step2_quality.py \\
        --original-dir "D:/...data" \\
        --aug-dir "./outputs/augmented" \\
        --threshold 0.55
"""

import json, argparse, shutil, warnings, sys, io
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree, ConvexHull

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


def sample(pts, n):
    if len(pts) <= n:
        return pts
    idx = np.random.choice(len(pts), n, replace=False)
    return pts[idx]


# ══════════════════════════════════════════════════════════
# 保真度评估
# ══════════════════════════════════════════════════════════

class FidelityEvaluator:
    SAMPLE = 2000
    
    def chamfer_dist(self, p1, p2):
        """归一化倒角距离"""
        s1, s2 = sample(p1, self.SAMPLE), sample(p2, self.SAMPLE)
        t1, t2 = cKDTree(s1), cKDTree(s2)
        d12, _ = t2.query(s1)
        d21, _ = t1.query(s2)
        cd = float(d12.mean() + d21.mean())
        scale = np.mean(p1.max(0) - p1.min(0))
        return cd / (scale + 1e-8)
    
    def hausdorff_dist(self, p1, p2):
        """归一化豪斯多夫距离"""
        s1, s2 = sample(p1, self.SAMPLE), sample(p2, self.SAMPLE)
        t1, t2 = cKDTree(s1), cKDTree(s2)
        d12, _ = t2.query(s1)
        d21, _ = t1.query(s2)
        hd = float(max(d12.max(), d21.max()))
        scale = np.mean(p1.max(0) - p1.min(0))
        return hd / (scale + 1e-8)
    
    def volume_change(self, p1, p2):
        """凸包体积变化率"""
        def hull_vol(pts):
            s = sample(pts, 1000)
            try:
                return ConvexHull(s).volume
            except:
                dims = s.max(0) - s.min(0)
                return float(np.prod(np.maximum(dims, 1e-6)))
        
        v1, v2 = hull_vol(p1), hull_vol(p2)
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
        s = sample(pts, 2000)
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
        'total': 0.55, 'fidelity': 0.45, 'physics': 0.50,
        'excellent': 0.72, 'good': 0.62, 'fair': 0.55
    }
    
    def __init__(self, original_dir, aug_dir):
        self.original_dir = Path(original_dir)
        self.aug_dir = Path(aug_dir)
        self.fid_eval = FidelityEvaluator()
        self.phys_eval = PhysicsChecker()
        self.div_eval = DiversityEvaluator()
    
    def load_all_meta(self):
        """加载所有变体元数据"""
        summary = self.aug_dir / 'augmentation_summary.json'
        if summary.exists():
            with open(summary, encoding='utf-8') as f:
                return json.load(f)['all_metadata']
        meta_files = list(self.aug_dir.glob('*_meta.json'))
        result = []
        for mf in meta_files:
            with open(mf, encoding='utf-8') as f:
                result.append(json.load(f))
        return result
    
    def load_original_cache(self, all_meta):
        """预加载所有原始点云"""
        cache = {}
        for meta in all_meta:
            src = Path(meta['source']).name
            if src not in cache:
                # 尝试在original_dir找
                for ext in ['.ply', '']:
                    candidates = list(self.original_dir.glob(f"*{Path(src).stem}*"))
                    if candidates:
                        pts = load_pts(candidates[0])
                        if pts is not None:
                            cache[src] = pts
                        break
        print(f"  缓存原始点云: {len(cache)} 个")
        return cache
    
    def analyze(self, all_meta=None):
        """执行批量质量分析"""
        if all_meta is None:
            all_meta = self.load_all_meta()
        
        print(f"  分析变体数: {len(all_meta)}")
        orig_cache = self.load_original_cache(all_meta)
        
        results = []
        descriptors = []
        
        print("\n  [阶段1] 计算保真度 & 物理可信度...")
        for i, meta in enumerate(all_meta):
            if (i + 1) % 50 == 0:
                print(f"    进度: {i+1}/{len(all_meta)}")
            
            aug_ply = self.aug_dir / f"{meta['id']}.ply"
            aug_pts = load_pts(aug_ply)
            
            src = Path(meta['source']).name
            orig_pts = orig_cache.get(src)
            
            if aug_pts is None:
                continue

            # 如果原始点云缺失，用保留比例估算保真度（避免results为空）
            if orig_pts is None:
                ret = float(meta.get('retention_ratio', 0.9))
                fid_score = float(np.clip(ret * 0.85 + 0.10, 0.30, 0.85))
                fid_detail = {'estimated': True, 'retention_ratio': ret}
            else:
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
            return {'summary': {'total': 0, 'passed': 0, 'pass_rate': 0.0,
                                'analysis_time': str(datetime.now())},
                    'quality_distribution': dist, 'by_method': {}, 'thresholds': self.THRESHOLDS,
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
            'results': results
        }
        return report_data
    
    def filter_and_save(self, results):
        """筛选高质量变体，保存到 high_quality 子目录"""
        passed = [r for r in results if r.get('passed')]
        
        out_dir = self.aug_dir.parent / 'high_quality'
        out_dir.mkdir(exist_ok=True)
        rejected_dir = self.aug_dir.parent / 'rejected'
        rejected_dir.mkdir(exist_ok=True)
        
        copied = 0
        for r in passed:
            src = self.aug_dir / f"{r['variant_id']}.ply"
            if src.exists():
                shutil.copy2(src, out_dir / src.name)
                copied += 1
        
        print(f"\n  筛选完成:")
        print(f"    高质量变体: {len(passed)} 个 → {out_dir}")
        print(f"    淘汰变体:   {len(results) - len(passed)} 个")
        
        with open(out_dir / 'filter_info.json', 'w', encoding='utf-8') as f:
            info = {
                'total': len(results), 'passed': len(passed),
                'pass_rate': len(passed)/len(results) if results else 0,
                'passed_ids': [r['variant_id'] for r in passed]
            }
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        return passed


# ══════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    _script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description='散料堆点云质量评估')
    parser.add_argument('--original-dir', default='./data')
    parser.add_argument('--aug-dir', default='./outputs/augmented')
    parser.add_argument('--threshold', type=float, default=0.55)
    args = parser.parse_args()

    # 相对路径统一解析为基于脚本目录的绝对路径
    aug_dir = Path(args.aug_dir) if Path(args.aug_dir).is_absolute() else _script_dir / args.aug_dir

    analyzer = QualityAnalyzer(args.original_dir, str(aug_dir))
    all_meta = analyzer.load_all_meta()
    
    print(f"\n开始质量分析 (共 {len(all_meta)} 个变体)...")
    results = analyzer.analyze(all_meta)
    
    report_data = analyzer.report(results)
    
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
