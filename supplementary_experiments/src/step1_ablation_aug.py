"""
step1_ablation_aug.py — LoadSim组件消融实验数据增强
====================================================
支持单独启用/禁用三个子组件:
  1. directional_removal  方向性材料去除
  2. slope_reshaping      坡面重塑/坡度加权挖掘中心
  3. lateral_collapse     侧向坍塌
"""
import sys, io, json, argparse, random, warnings, time
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Windows UTF-8 安全修复 ─────────────────────────────────────────────────
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        elif hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── 导入 step1_augmentation 工具函数 ─────────────────────────────────────────
# 关键: 临时将 sys.platform 设为非 'win32'，防止 step1_augmentation 的模块级
# io.TextIOWrapper 重定向在 Windows 子进程中覆盖已初始化的 stdout 句柄，
# 导致双重包装后 GC 时关闭底层 buffer，引发"I/O operation on closed file"。
sys.path.insert(0, str(Path(__file__).parent))
_orig_platform = sys.platform
try:
    sys.platform = 'linux_import_guard'   # 临时屏蔽 win32 检测
    from step1_augmentation import (
        _load_ply_numpy, _save_ply_numpy,
        estimate_normals_numpy, preprocess, knn_smooth,
        get_data_config, linear_scale_rot,
        nonlinear_surface_noise, nonlinear_rbf,
    )
finally:
    sys.platform = _orig_platform         # 恢复真实 platform

from config import ABLATION_CONFIGS


# ══════════════════════════════════════════════════════════════════════════
# 可消融装载模拟核心函数
# ══════════════════════════════════════════════════════════════════════════

def loading_simulation_ablation(pts, normals, meta, cfg,
                                 enable_directional_removal=True,
                                 enable_slope_reshaping=True,
                                 enable_lateral_collapse=True):
    r"""
    LoadSim 三组件独立控制版

    数学形式:
    [1] directional_removal:
        s_i=(p_i-c)·d, τ=max(s)-α·range(s), α~U[0.15,0.25]
        F={p_i|s_i>τ}, B(p_i)=|(p_i-p_core)·e_side|<W/2
        Remove: P <- P \ {p_i ∈ F∩B}

    [2] slope_reshaping (挖掘中心选取):
        θ_i=arcsin(|n_z^i|), w_i=(π/2-θ_i)^2  → Categorical采样
        若禁用: p_core ~ Uniform(F)

    [3] lateral_collapse:
        C={p_i|W/2≤d_side<1.2W/2 ∧ s_i>τ}
        f(p_i)=1-(d_side-W/2)/(0.2W/2)
        Δz=-δ·f, Δx=γ·f·normalize(p_core-p_i)_xy
    """
    if normals is None:
        return pts

    deformed_pts     = pts.copy()
    deformed_normals = normals.copy()
    centroid         = deformed_pts.mean(axis=0)
    n_ops            = random.randint(2, 4)
    operations       = []

    for _ in range(n_ops):
        if len(deformed_pts) < 500:
            break

        angle       = random.uniform(0, 2 * np.pi)
        d_vec       = np.array([np.cos(angle), np.sin(angle), 0.0])
        side_vec    = np.array([-np.sin(angle), np.cos(angle), 0.0])
        proj        = (deformed_pts - centroid) @ d_vec
        p_min, p_max = proj.min(), proj.max()
        if p_max <= p_min:
            continue

        front_thresh  = p_max - random.uniform(0.15, 0.25) * (p_max - p_min)
        front_idx     = np.where(proj > front_thresh)[0]
        if len(front_idx) < 100:
            continue

        # [2] 坡面重塑: 挖掘中心选取
        if enable_slope_reshaping:
            fn    = deformed_normals[front_idx]
            theta = np.arcsin(np.clip(np.abs(fn[:, 2]), 0, 1))
            w     = (np.pi / 2 - theta) ** 2 + 1e-8
            w    /= w.sum()
            ci    = np.random.choice(len(front_idx), p=w)
        else:
            ci = np.random.randint(0, len(front_idx))

        exc_center = deformed_pts[front_idx[ci]]
        bw         = random.uniform(0.8, 1.5)
        dist_side  = np.abs((deformed_pts - exc_center) @ side_vec)
        exc_mask   = (dist_side < bw / 2) & (proj > front_thresh)
        take_idx   = np.where(exc_mask)[0]
        if len(take_idx) == 0:
            continue

        # [3] 侧向坍塌
        if enable_lateral_collapse:
            col_mask = (dist_side >= bw / 2) & (dist_side < bw * 1.2) & (proj > front_thresh)
            col_idx  = np.where(col_mask)[0]
            if len(col_idx) > 0:
                cd = dist_side[col_idx] - bw / 2
                cf = np.clip(1.0 - cd / (bw * 0.2 + 1e-8), 0, 1)
                deformed_pts[col_idx, 2] -= random.uniform(0.1, 0.3) * cf
                sv = exc_center - deformed_pts[col_idx]
                sv[:, 2] = 0
                sv /= (np.linalg.norm(sv, axis=1, keepdims=True) + 1e-8)
                deformed_pts[col_idx] += sv * (random.uniform(0.05, 0.15) * cf)[:, None]

        # [1] 方向性材料去除
        if enable_directional_removal:
            keep = np.ones(len(deformed_pts), dtype=bool)
            keep[take_idx] = False
            deformed_pts     = deformed_pts[keep]
            deformed_normals = deformed_normals[keep]

        operations.append({
            'angle_deg':    float(np.degrees(angle)),
            'bucket_width': float(bw),
            'removed':      int(len(take_idx)) if enable_directional_removal else 0,
            'components':   {'directional_removal': enable_directional_removal,
                             'slope_reshaping':     enable_slope_reshaping,
                             'lateral_collapse':    enable_lateral_collapse},
        })

    if operations and len(deformed_pts) > 100 and HAS_SCIPY:
        deformed_pts = knn_smooth(deformed_pts, k=15, factor=0.15)

    meta['deform_params']['loading_simulation_ablation'] = {
        'n_operations':  len(operations),
        'removal_ratio': float(1.0 - len(deformed_pts) / max(len(pts), 1)),
        'components':    {'directional_removal': enable_directional_removal,
                          'slope_reshaping':     enable_slope_reshaping,
                          'lateral_collapse':    enable_lateral_collapse},
    }
    return deformed_pts


def augment_one_ablation(pts, normals, filename, var_id, cfg,
                          enable_directional_removal=True,
                          enable_slope_reshaping=True,
                          enable_lateral_collapse=True):
    """
    消融实验变体生成 - 仅装载模拟，不叠加传统方法
    
    消融实验设计原则:
    - 基线: 完整装载模拟 (三个组件全部启用)
    - 消融: 每次移除一个组件，观察性能变化
    - 不叠加传统方法 (Scale/Rot/Noise/RBF)，确保对比的是纯装载模拟效果
    """
    meta = {
        'id':               f"{Path(filename).stem}_abl_var{var_id:03d}",
        'source':           str(filename),
        'data_type':        cfg['type'],
        'variant_id':       var_id,
        'deform_params':    {},
        'applied_methods':  [],
        'scheme':           'ablation',
        'ablation':         {'directional_removal': enable_directional_removal,
                             'slope_reshaping':     enable_slope_reshaping,
                             'lateral_collapse':    enable_lateral_collapse},
        'original_count':   int(len(pts)),
        'timestamp':        str(datetime.now()),
    }
    deformed     = pts.copy()
    deformed_nrm = normals.copy() if normals is not None else None

    try:
        deformed = loading_simulation_ablation(
            deformed, deformed_nrm, meta, cfg,
            enable_directional_removal=enable_directional_removal,
            enable_slope_reshaping=enable_slope_reshaping,
            enable_lateral_collapse=enable_lateral_collapse,
        )
        meta['applied_methods'].append('loading_simulation_ablation')
    except Exception:
        pass

    # 消融实验: 不叠加任何传统增强方法 (Scale/Rot, Noise, RBF)
    # 仅对比装载模拟各组件的贡献

    if len(deformed) < 50:
        deformed = pts.copy()
        meta['fallback'] = True

    meta['final_count']     = int(len(deformed))
    meta['retention_ratio'] = float(len(deformed) / max(len(pts), 1))
    return deformed, meta


def run_ablation_augmentation(data_dir, output_dir, ablation_config_name, n_variants=40):
    if ablation_config_name not in ABLATION_CONFIGS:
        raise ValueError(f"未知消融配置: {ablation_config_name}")

    en_rem, en_sl, en_col, label_cn, label_en = ABLATION_CONFIGS[ablation_config_name]
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(data_dir.glob('*.ply'))
    if not ply_files:
        print(f"[ERROR] {data_dir} 中无PLY文件", flush=True)
        sys.exit(1)   # 非零退出让 run_step 检测到失败

    print(f"\n{'='*65}", flush=True)
    print(f"  消融增强: {ablation_config_name} — {label_en}", flush=True)
    print(f"  组件: removal={en_rem}, slope={en_sl}, collapse={en_col}", flush=True)
    print(f"  文件数: {len(ply_files)}, 每文件变体数: {n_variants}", flush=True)
    print(f"{'='*65}", flush=True)

    all_meta = []
    t0 = time.time()

    for fi, ply_file in enumerate(ply_files):
        try:
            pts, normals = _load_ply_numpy(ply_file)
            if pts is None or len(pts) == 0:
                continue
            pts, normals, _ = preprocess(pts, normals)
            cfg = get_data_config(ply_file.name)
            print(f"  [{fi+1}/{len(ply_files)}] {ply_file.name}: {len(pts):,}点", flush=True)
        except Exception as e:
            print(f"  [ERROR] {ply_file.name}: {e}", flush=True)
            continue

        for var_id in range(n_variants):
            try:
                d_pts, meta = augment_one_ablation(
                    pts, normals, ply_file.name, var_id, cfg,
                    enable_directional_removal=en_rem,
                    enable_slope_reshaping=en_sl,
                    enable_lateral_collapse=en_col,
                )
                d_nrm = estimate_normals_numpy(d_pts, k=20)
                _save_ply_numpy(output_dir / f"{meta['id']}.ply", d_pts, d_nrm)
                with open(output_dir / f"{meta['id']}_meta.json", 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                all_meta.append(meta)
            except Exception as e:
                print(f"  [WARN] {ply_file.name} var{var_id}: {e}", flush=True)

    elapsed = time.time() - t0
    with open(output_dir / 'ablation_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'ablation_config': ablation_config_name, 'label_en': label_en,
                   'total_variants': len(all_meta), 'elapsed_seconds': round(elapsed, 1),
                   'generated_at': str(datetime.now())}, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ 消融增强完成: {len(all_meta)} 变体, 耗时 {elapsed:.1f}s", flush=True)
    return all_meta


def main():
    parser = argparse.ArgumentParser(description='LoadSim消融实验数据增强')
    parser.add_argument('--data-dir',        required=True)
    parser.add_argument('--output-dir',      required=True)
    parser.add_argument('--ablation-config', required=True,
                        choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument('--variants', type=int, default=40)
    args = parser.parse_args()
    run_ablation_augmentation(args.data_dir, args.output_dir,
                               args.ablation_config, args.variants)


if __name__ == '__main__':
    main()
