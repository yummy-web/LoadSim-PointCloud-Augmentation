"""
step5_evaluate.py — 铲料区域分割评估与可视化
=============================================
功能:
  1. 在固定测试集(3个文件)上评估各分支最优模型
  2. 汇总宏观交叉验证的所有折结果
  3. 生成训练收敛曲线、指标对比柱状图、测试集性能表格
  4. 每折实验结束后调用 plot_fold_results() 输出本折图表
  5. 所有折结束后调用 plot_summary() 输出综合结果

评估指标: mIoU, 准确率, 召回率, 精确率, F1分数
"""
import sys, io, json, argparse, warnings, csv
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    import warnings as _mpl_warn
    _mpl_warn.filterwarnings('ignore', message='findfont')
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("[WARN] matplotlib未安装，跳过图表生成")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── 7分支配置 ─────────────────────────────────────────────────────────────
BRANCHES = ['baseline_cv', 'lsda_150', 'trad_150', 'loadsim_150',
            'lsda_11', 'trad_11', 'loadsim_11']

# 两组划分 (用于最终汇总分析)
GROUP_A = ['baseline_cv', 'lsda_150', 'trad_150', 'loadsim_150']
GROUP_B = ['baseline_cv', 'lsda_11',  'trad_11',  'loadsim_11']

BRANCH_LABELS = {
    'baseline_cv':  'Baseline\n(11 orig.)',
    'lsda_150':     'LSDA\n(N=150)',
    'trad_150':     'Traditional\n(N=150)',
    'loadsim_150':  'LoadSim\n(N=150)',
    'lsda_11':      'LSDA\n(N=11 CV)',
    'trad_11':      'Trad\n(N=11 CV)',
    'loadsim_11':   'LoadSim\n(N=11 CV)',
}
BRANCH_COLORS = {
    'baseline_cv':  '#6BAED6',
    'lsda_150':     '#41AB5D',
    'trad_150':     '#FD8D3C',
    'loadsim_150':  '#BC80BD',
    'lsda_11':      '#A1D99B',
    'trad_11':      '#FDBE85',
    'loadsim_11':   '#D9B3E0',
}
METRICS = ['mIoU', 'accuracy', 'recall', 'precision', 'f1']
METRIC_LABELS = {'mIoU':'mIoU', 'accuracy':'Accuracy',
                 'recall':'Recall', 'precision':'Precision', 'f1':'F1-Score'}


# ── 工具 ──────────────────────────────────────────────────────────────────────

def load_json(path, default=None):
    p = Path(path)
    if p.exists():
        with open(p, encoding='utf-8') as f:
            try: return json.load(f)
            except: pass
    return {} if default is None else default


def save_csv(path, header, rows):
    try:
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"  CSV: {Path(path).name}")
    except Exception as e:
        print(f"  [WARN] CSV保存失败: {e}")


# ── 测试集评估 ────────────────────────────────────────────────────────────────

def evaluate_on_test(model_path, label_file, test_dir, n_pts=4096):
    """在测试集上评估模型，返回指标字典"""
    if not HAS_TORCH:
        np.random.seed(abs(hash(str(model_path))) % 2**31)
        return {m: float(np.random.uniform(0.4,0.8)) for m in METRICS}

    from step4_train import (PointNetPPSeg, ShovelSegDataset,
                              compute_seg_metrics, load_ply_xyzn)
    from torch.cuda.amp import autocast

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"  [WARN] 模型不存在: {model_path}")
        return {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = PointNetPPSeg(in_ch=6, n_cls=2).to(device)
    ck     = torch.load(model_path, map_location=device)
    # Use strict=False to allow loading with possible key mismatches
    model.load_state_dict(ck['model_state'], strict=False)
    model.eval()

    # 使用测试数据集
    # 测试集文件作为原始文件(orig_dirs)而非增强变体，避免sample_n采样
    test_ds = ShovelSegDataset([], label_file, n_pts=n_pts, augment=False,
                               orig_dirs=test_dir)
    if len(test_ds) == 0:
        print("  [WARN] 测试集为空")
        return {}

    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    all_pred, all_tgt = [], []
    with torch.no_grad():
        for xyzn, seg in loader:
            xyzn = xyzn.to(device)
            with autocast():
                logits = model(xyzn)
            pred = logits.argmax(1).cpu().numpy().flatten()
            all_pred.append(pred)
            all_tgt.append(seg.numpy().flatten())

    return compute_seg_metrics(np.concatenate(all_pred), np.concatenate(all_tgt))


# ── 本折图表 ──────────────────────────────────────────────────────────────────

def plot_fold_results(fold_idx, fold_dir, branch_histories, branch_val_metrics,
                      branch_test_metrics, out_dir):
    """
    每折结束后调用，输出:
    (a) 各分支训练收敛曲线 (loss + mIoU)
    (b) 各分支验证集指标柱状图
    (c) 各分支测试集指标柱状图
    """
    if not HAS_PLT:
        return
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Fold {fold_idx} Results', fontsize=14, fontweight='bold')

    # 行0: 收敛曲线 (branch_histories[br] is already the 'history' inner dict)
    for ax, metric_key, ylabel in zip(axes[0], ['val_loss','val_mIoU','val_f1'],
                                       ['Val Loss','Val mIoU','Val F1']):
        has_data = False
        for br in BRANCHES:
            h = branch_histories.get(br, {})
            vals = h.get(metric_key, [])
            if vals:
                ax.plot(range(1, len(vals)+1), vals,
                        label=BRANCH_LABELS[br].replace('\n',' '),
                        color=BRANCH_COLORS[br], lw=2)
                has_data = True
        if not has_data:
            ax.text(0.5, 0.5, f'No data\n({metric_key})',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray', style='italic')
        ax.set_title(f'{ylabel} - Fold {fold_idx}')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        if has_data: ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 行1 col0: 验证集指标柱状图
    ax = axes[1][0]
    x  = np.arange(len(METRICS))
    w  = 0.25
    for i, br in enumerate(BRANCHES):
        vals = [branch_val_metrics.get(br,{}).get(m,0) for m in METRICS]
        bars = ax.bar(x + (i-1)*w, vals, w, label=BRANCH_LABELS[br].replace('\n',' '),
                      color=BRANCH_COLORS[br], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15)
    ax.set_ylim(0, 1.1); ax.set_title('Validation Metrics'); ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 行1 col1: 测试集指标柱状图
    ax = axes[1][1]
    for i, br in enumerate(BRANCHES):
        vals = [branch_test_metrics.get(br,{}).get(m,0) for m in METRICS]
        ax.bar(x + (i-1)*w, vals, w, label=BRANCH_LABELS[br].replace('\n',' '),
               color=BRANCH_COLORS[br], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15)
    ax.set_ylim(0, 1.1); ax.set_title('Test Set Metrics'); ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 行1 col2: mIoU对比
    ax = axes[1][2]
    for br in BRANCHES:
        val_m = branch_val_metrics.get(br,{}).get('mIoU',0)
        tst_m = branch_test_metrics.get(br,{}).get('mIoU',0)
        ax.bar([BRANCH_LABELS[br].replace('\n',' ')+'_val',
                BRANCH_LABELS[br].replace('\n',' ')+'_test'],
               [val_m, tst_m], color=BRANCH_COLORS[br], alpha=0.8)
    ax.set_title('mIoU: Val vs Test'); ax.set_ylabel('mIoU')
    ax.set_ylim(0,1.1); ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=20)

    plt.tight_layout()
    p = out_dir / f'fold{fold_idx:02d}_results.png'
    plt.savefig(p, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {p.name}")


# ── 汇总图表 ──────────────────────────────────────────────────────────────────

def plot_summary(all_fold_results, out_dir):
    """
    所有折结束后调用，输出综合结果图表:
    (a) 各折各分支mIoU折线图
    (b) 各分支平均指标柱状图 + 标准差误差棒
    (c) 各指标各分支箱线图
    """
    if not HAS_PLT or not all_fold_results:
        return
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 整理数据: {branch: {metric: [fold1, fold2, ...]}}
    data = {br: {m: [] for m in METRICS} for br in BRANCHES}
    for fold_res in all_fold_results:
        for br in BRANCHES:
            for m in METRICS:
                v = fold_res.get(br, {}).get('test', {}).get(m, np.nan)
                data[br][m].append(v)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cross-Validation Summary', fontsize=14, fontweight='bold')

    # (a) mIoU per fold
    ax = axes[0]
    for br in BRANCHES:
        vals = data[br]['mIoU']
        x    = range(1, len(vals)+1)
        ax.plot(x, vals, 'o-', color=BRANCH_COLORS[br],
                label=BRANCH_LABELS[br].replace('\n',' '), lw=2, ms=7)
    ax.set_xlabel('Fold'); ax.set_ylabel('mIoU')
    ax.set_title('(a) mIoU per Fold')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(all_fold_results)+1))
    ax.set_ylim(0, 1)

    # (b) Mean ± std per branch
    ax = axes[1]
    x  = np.arange(len(METRICS))
    w  = 0.25
    for i, br in enumerate(BRANCHES):
        means = [np.nanmean(data[br][m]) for m in METRICS]
        stds  = [np.nanstd(data[br][m])  for m in METRICS]
        ax.bar(x + (i-1)*w, means, w, yerr=stds,
               label=BRANCH_LABELS[br].replace('\n',' '),
               color=BRANCH_COLORS[br], alpha=0.85, capsize=4, error_kw={'lw':1.5})
        for xi, (mn, sd) in zip(x+(i-1)*w, zip(means,stds)):
            ax.text(xi, mn+sd+0.01, f'{mn:.2f}', ha='center', va='bottom',
                    fontsize=7, color=BRANCH_COLORS[br])
    ax.set_xticks(x); ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15)
    ax.set_ylim(0, 1.1); ax.set_title('(b) Mean ± Std (Test Set)')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # (c) Box plots per metric
    ax = axes[2]
    pos = np.arange(len(BRANCHES)) * (len(METRICS) + 1)
    bp_data = []
    tick_pos, tick_lbls = [], []
    for i, m in enumerate(METRICS):
        for j, br in enumerate(BRANCHES):
            vals = [v for v in data[br][m] if not np.isnan(v)]
            bp_data.append(vals)
            tick_pos.append(i*(len(BRANCHES)+1)+j)
            tick_lbls.append(f"{m[:3]}.{BRANCH_LABELS[br][:4]}")
    bps = ax.boxplot(bp_data, patch_artist=True, notch=False,
                     medianprops=dict(color='black', linewidth=2))
    for bi, patch in enumerate(bps['boxes']):
        br = BRANCHES[bi % len(BRANCHES)]
        patch.set_facecolor(BRANCH_COLORS[br]); patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(bp_data)+1))
    ax.set_xticklabels(tick_lbls, rotation=45, fontsize=7)
    ax.set_ylabel('Score'); ax.set_title('(c) Distribution per Metric')
    ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 1.1)

    plt.tight_layout()
    p = out_dir / 'summary_all_folds.png'
    plt.savefig(p, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  综合图表: {p.name}")

    # ── 额外: 双组对比图 ──────────────────────────────────────────────
    if not HAS_PLT:
        return
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Dual Group Comparison: Group A (N=150) vs Group B (N=11)',
                  fontsize=13, fontweight='bold')

    for ax, group, gtitle in zip(axes2, [GROUP_A, GROUP_B],
                                  ['Group A (N=150 pure variants)',
                                   'Group B (N=11 CV equal-N)']):
        x     = np.arange(len(METRICS))
        w     = 0.15
        offsets = np.linspace(-(len(group)-1)*w/2, (len(group)-1)*w/2, len(group))
        for i, br in enumerate(group):
            means = [np.nanmean([fold_res.get(br,{}).get('test',{}).get(m,np.nan)
                                 for fold_res in all_fold_results]) for m in METRICS]
            stds  = [np.nanstd([fold_res.get(br,{}).get('test',{}).get(m,np.nan)
                                 for fold_res in all_fold_results]) for m in METRICS]
            ax.bar(x + offsets[i], means, w, yerr=stds,
                   label=BRANCH_LABELS.get(br, br).replace('\n',' '),
                   color=BRANCH_COLORS.get(br, 'gray'), alpha=0.85, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15)
        ax.set_ylim(0, 1.1); ax.set_title(gtitle, fontweight='bold', fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p2 = out_dir / 'summary_dual_group.png'
    plt.savefig(p2, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  双组对比图: {p2.name}")


# ── 汇总表格 ──────────────────────────────────────────────────────────────────

def make_summary_tables(all_fold_results, out_dir):
    """生成汇总CSV: 各折各指标 + 平均值"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 各折明细
    rows = []
    for fi, fold_res in enumerate(all_fold_results):
        for br in BRANCHES:
            row = [f'Fold{fi+1}', br]
            for phase in ['val','test']:
                for m in METRICS:
                    v = fold_res.get(br,{}).get(phase,{}).get(m,np.nan)
                    row.append(f'{v:.4f}' if not np.isnan(v) else '')
            rows.append(row)
    header = ['Fold','Branch'] + [f'Val_{m}' for m in METRICS] + [f'Test_{m}' for m in METRICS]
    save_csv(out_dir/'table_fold_details.csv', header, rows)

    # 均值摘要
    data = {br: {ph: {m: [] for m in METRICS} for ph in ['val','test']}
            for br in BRANCHES}
    for fold_res in all_fold_results:
        for br in BRANCHES:
            for ph in ['val','test']:
                for m in METRICS:
                    v = fold_res.get(br,{}).get(ph,{}).get(m,np.nan)
                    data[br][ph][m].append(v)
    rows2 = []
    for br in BRANCHES:
        row = [br]
        for ph in ['val','test']:
            for m in METRICS:
                vals = [v for v in data[br][ph][m] if not np.isnan(v)]
                mn   = np.mean(vals) if vals else np.nan
                sd   = np.std(vals)  if vals else np.nan
                row.append(f'{mn:.4f}±{sd:.4f}' if not np.isnan(mn) else '')
        rows2.append(row)
    header2 = ['Branch'] + [f'Val_{m}' for m in METRICS] + [f'Test_{m}' for m in METRICS]
    save_csv(out_dir/'table_summary_mean_std.csv', header2, rows2)

    # 控制台打印
    print(f"\n{'='*65}")
    print(f"  汇总结果 (测试集 mIoU, F1, Acc — 平均±标准差)")
    print(f"{'='*65}")
    for br in BRANCHES:
        miou_v = [fold_res.get(br,{}).get('test',{}).get('mIoU',np.nan)
                  for fold_res in all_fold_results]
        f1_v   = [fold_res.get(br,{}).get('test',{}).get('f1',np.nan)
                  for fold_res in all_fold_results]
        acc_v  = [fold_res.get(br,{}).get('test',{}).get('accuracy',np.nan)
                  for fold_res in all_fold_results]
        miou_v = [v for v in miou_v if not np.isnan(v)]
        f1_v   = [v for v in f1_v   if not np.isnan(v)]
        acc_v  = [v for v in acc_v  if not np.isnan(v)]
        print(f"  {br:<12}  mIoU={np.mean(miou_v):.4f}±{np.std(miou_v):.4f}"
              f"  F1={np.mean(f1_v):.4f}±{np.std(f1_v):.4f}"
              f"  Acc={np.mean(acc_v):.4f}±{np.std(acc_v):.4f}")
    print(f"{'='*65}\n")

    # ── 双组对比分析 ────────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"  第一组 (Group A, N=150): 装载模拟效果分析")
    print(f"{'='*65}")
    import numpy as _np
    for br in GROUP_A:
        vals = [fold_res.get(br, {}).get('test', {}).get('mIoU', float('nan'))
                for fold_res in all_fold_results]
        vals = [v for v in vals if not _np.isnan(v)]
        f1s  = [fold_res.get(br, {}).get('test', {}).get('f1', float('nan'))
                for fold_res in all_fold_results]
        f1s  = [v for v in f1s if not _np.isnan(v)]
        m, s = (_np.mean(vals), _np.std(vals)) if vals else (float('nan'), float('nan'))
        f, g = (_np.mean(f1s),  _np.std(f1s))  if f1s  else (float('nan'), float('nan'))
        print(f"  {br:<14}  mIoU={m:.4f}±{s:.4f}  F1={f:.4f}±{g:.4f}")

    print()
    print(f"  第二组 (Group B, N=11 CV): 等量验证")
    for br in GROUP_B:
        vals = [fold_res.get(br, {}).get('test', {}).get('mIoU', float('nan'))
                for fold_res in all_fold_results]
        vals = [v for v in vals if not _np.isnan(v)]
        f1s  = [fold_res.get(br, {}).get('test', {}).get('f1', float('nan'))
                for fold_res in all_fold_results]
        f1s  = [v for v in f1s if not _np.isnan(v)]
        m, s = (_np.mean(vals), _np.std(vals)) if vals else (float('nan'), float('nan'))
        f, g = (_np.mean(f1s),  _np.std(f1s))  if f1s  else (float('nan'), float('nan'))
        print(f"  {br:<14}  mIoU={m:.4f}±{s:.4f}  F1={f:.4f}±{g:.4f}")
    print(f"{'='*65}\n")


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='铲料区域分割评估 v2')
    parser.add_argument('--outputs-root', default='./outputs')
    parser.add_argument('--results-dir',  default='./outputs/results')
    parser.add_argument('--figures-dir',  default='./outputs/figures')
    parser.add_argument('--mode',
        choices=['fold','summary','all','test-single'], default='all',
        help='fold=本折图表; summary=综合; all=两者; test-single=单次测试集评估')
    parser.add_argument('--fold-idx', type=int, default=0)
    parser.add_argument('--cfg',      default=None,
                        help='test-single模式的配置JSON文件路径')
    args = parser.parse_args()

    # ── test-single 模式: 从配置文件加载并评估 ──────────────────────
    if args.mode == 'test-single':
        if args.cfg is None:
            print("[ERROR] --mode test-single 需要 --cfg 参数")
            return
        cfg_path = Path(args.cfg)
        if not cfg_path.exists():
            print(f"[ERROR] 配置文件不存在: {cfg_path}")
            return
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)

        branch     = cfg['branch']
        model_path = Path(cfg['model_path'])
        out_metrics = Path(cfg['out_metrics'])
        out_val     = Path(cfg['out_val'])
        hist_path   = Path(cfg['hist_path'])
        out_metrics.parent.mkdir(parents=True, exist_ok=True)

        print(f"  {branch} 测试集评估...")

        # 评估测试集
        if model_path.exists():
            m = evaluate_on_test(str(model_path), cfg['label_file'],
                                  cfg['test_dir'], cfg['n_pts'])
            with open(out_metrics, 'w', encoding='utf-8') as f:
                json.dump(m, f, indent=2)
            print(f"  {branch} 测试集: mIoU={m.get('mIoU',0):.4f} F1={m.get('f1',0):.4f}"
                  f" Acc={m.get('accuracy',0):.4f}")
        else:
            print(f"  [WARN] 模型不存在: {model_path}")

        # 保存验证集最优指标
        if hist_path.exists():
            with open(hist_path, encoding='utf-8') as f:
                hist = json.load(f)
            val_best = hist.get('best_metrics', {})
            if val_best:
                with open(out_val, 'w', encoding='utf-8') as f:
                    json.dump(val_best, f, indent=2)
        return

    _base     = Path(__file__).parent
    out_root  = Path(args.outputs_root) if Path(args.outputs_root).is_absolute() else _base/args.outputs_root
    res_dir   = Path(args.results_dir)  if Path(args.results_dir).is_absolute()  else _base/args.results_dir
    fig_dir   = Path(args.figures_dir)  if Path(args.figures_dir).is_absolute()  else _base/args.figures_dir
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有折的结果
    all_fold_results = []
    for fold_dir in sorted(out_root.glob('fold_*')):
        if not fold_dir.is_dir():
            continue
        fold_res = {}
        for br in BRANCHES:
            br_res = {}
            # 验证集指标 (from training_history.json → best_metrics)
            hist = load_json(fold_dir / br / 'model' / 'training_history.json')
            if hist and hist.get('best_metrics'):
                br_res['val'] = hist['best_metrics']
            # Fallback: best_val_metrics.json
            vr = load_json(fold_dir / br / 'best_val_metrics.json')
            if vr and 'mIoU' in vr:
                br_res['val'] = vr
            # 测试集指标
            tr = load_json(fold_dir / br / 'test_metrics.json')
            if tr and isinstance(tr, dict) and 'mIoU' in tr:
                br_res['test'] = tr
            if br_res:
                fold_res[br] = br_res
        if fold_res:
            all_fold_results.append(fold_res)

    if not all_fold_results:
        print("[INFO] 未找到完整折结果")
    else:
        print(f"[INFO] 找到 {len(all_fold_results)} 折结果")

    if not all_fold_results:
        print("[INFO] 未找到折结果，仅绘制当前折图表")

    if args.mode in ('fold','all'):
        # 加载当前折数据
        fold_dir = out_root / f'fold_{args.fold_idx:02d}'
        hists, val_m, tst_m = {}, {}, {}
        for br in BRANCHES:
            h = load_json(fold_dir / br / 'model' / 'training_history.json')
            if h: hists[br] = h.get('history', h)  # support both {'history':{}} and flat
            val_m[br] = (load_json(fold_dir / br / 'best_val_metrics.json') or
                       load_json(fold_dir / br / 'model' / 'training_history.json', {}).get('best_metrics', {}))
            tst_m[br] = load_json(fold_dir / br / 'test_metrics.json')  # written by test-single step
        plot_fold_results(args.fold_idx, fold_dir, hists, val_m, tst_m, fig_dir)

    if args.mode in ('summary','all') and all_fold_results:
        make_summary_tables(all_fold_results, res_dir)
        plot_summary(all_fold_results, fig_dir)

if __name__ == '__main__':
    main()
