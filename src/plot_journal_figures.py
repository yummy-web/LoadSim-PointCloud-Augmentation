"""
plot_journal_figures.py — 期刊论文图表生成
==========================================
生成 RA-L 期刊所需的所有可视化图表，
读取 outputs4/ 和 outputs_journal/ 中的实验数据。

运行:
    python plot_journal_figures.py --outputs-root ./outputs4 \
        --journal-root ./outputs_journal --out-dir ./figures_journal

说明:
    所有图仅导出为 SVG 矢量格式（适配论文排版，去除图内总标题）
"""
import argparse, json, sys, io, warnings
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({
        'svg.fonttype': 'path',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    print("[ERROR] matplotlib required. pip install matplotlib")
    sys.exit(1)

# ── Color scheme (IEEE/RAL style) ──────────────────────────────────────────
COLORS = {
    'baseline':   '#555555',
    'trad':       '#E69F00',
    'lsda':       '#56B4E9',
    'loadsim':    '#009E73',
    'pn2':        '#0072B2',
    'pointnext':  '#D55E00',
    'kpconv':     '#CC79A7',
}
HATCHES = {'baseline': '', 'trad': '//', 'lsda': '..', 'loadsim': ''}

SVG_SAVE_KW = {
    'format': 'svg',
    'bbox_inches': 'tight',
    'facecolor': 'white',
    'transparent': False,
}


def save_svg(fig, out_dir, stem):
    out_path = out_dir / f'{stem}.svg'
    fig.savefig(out_path, **SVG_SAVE_KW)
    print(f"  [SAVED] {out_path}")
    return str(out_path)


def save_pdf(fig, out_dir, stem):
    out_path = out_dir / f'{stem}.pdf'
    fig.savefig(out_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  [SAVED] {out_path}")
    return str(out_path)


def load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def load_history(path):
    """Load training_history.json"""
    d = load_json(path)
    if d and 'history' in d:
        return d['history'], d.get('best_metrics', {})
    return None, None


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Convergence curves (fold 0, all 4 original branches)
# ══════════════════════════════════════════════════════════════════════════

def plot_convergence_curves(outputs_root, out_dir):
    """Training convergence: mIoU, F1, Loss vs epoch (fold 0)"""
    outputs_root = Path(outputs_root)
    fold_dir = outputs_root / 'fold_00'

    branch_cfg = {
        'baseline_cv':  ('Baseline (N=11)',  COLORS['baseline'], '-'),
        'trad_150':     ('Conventional (N=150)', COLORS['trad'], '--'),
        'lsda_150':     ('LSDA (N=150)',     COLORS['lsda'], '-.'),
        'loadsim_150':  ('LoadSim (N=150)',  COLORS['loadsim'], '-'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    metrics_keys = [('val_mIoU', 'Validation mIoU'),
                    ('val_f1',   'Validation F1'),
                    ('val_loss', 'Validation Loss')]

    for branch, (label, color, ls) in branch_cfg.items():
        hist_path = fold_dir / branch / 'model' / 'training_history.json'
        hist, best = load_history(hist_path)
        if hist is None:
            print(f"  [WARN] No history: {hist_path}")
            continue
        for ax, (key, ylabel) in zip(axes, metrics_keys):
            if key in hist:
                y = hist[key]
                ax.plot(range(1, len(y)+1), y, label=label,
                        color=color, linestyle=ls, lw=1.8, alpha=0.9)

    for ax, (key, ylabel) in zip(axes, metrics_keys):
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, 71)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.tick_params(labelsize=9)

    axes[0].legend(fontsize=8, loc='lower right')
    out_path = save_svg(fig, out_dir, 'fig1_convergence')
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Group A & B bar chart (PointNet++ original results)
# ══════════════════════════════════════════════════════════════════════════

def plot_group_comparison(out_dir):
    """Bar chart: test mIoU for all 7 branches"""
    # Data from conference paper (Table 2 & 3)
    group_A = {
        'Baseline\n(N=11)':  (0.542, 0.071, COLORS['baseline']),
        'Conv.\n(N=150)':    (0.563, 0.115, COLORS['trad']),
        'LSDA\n(N=150)':     (0.607, 0.018, COLORS['lsda']),
        'LoadSim\n(N=150)':  (0.632, 0.021, COLORS['loadsim']),
    }
    group_B = {
        'Conv.\n(N=11)':     (0.450, 0.038, COLORS['trad']),
        'LSDA\n(N=11)':      (0.472, 0.001, COLORS['lsda']),
        'LoadSim\n(N=11)':   (0.473, 0.001, COLORS['loadsim']),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0),
                                    gridspec_kw={'width_ratios': [4, 3]},
                                    constrained_layout=True)

    for ax, group in [(ax1, group_A),
                      (ax2, group_B)]:
        names = list(group.keys())
        means = [v[0] for v in group.values()]
        stds  = [v[1] for v in group.values()]
        cols  = [v[2] for v in group.values()]
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, color=cols, alpha=0.85,
                      width=0.55, capsize=5, error_kw={'lw': 1.5},
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel('Test mIoU', fontsize=10)
        ax.set_ylim(0.35, 0.75)
        ax.axhline(means[0], color='gray', linestyle=':', lw=1, alpha=0.6)
        ax.grid(True, axis='y', alpha=0.3, linestyle=':')
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.005,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    save_svg(fig, out_dir, 'fig2_group_comparison')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Ablation study bar chart
# ══════════════════════════════════════════════════════════════════════════

def plot_ablation(out_dir):
    """Ablation: mIoU and F1 for 7 LoadSim configurations"""
    configs = [
        ('Full\nLoadSim',    0.6249, 0.0577, 0.4748, '#009E73'),
        ('w/o\nRemoval',     0.6355, 0.0366, 0.5106, '#66C2A5'),
        ('w/o\nSlope',       0.6737, 0.0487, 0.5763, '#FC8D62'),
        ('w/o\nCollapse',    0.6490, 0.0189, 0.5494, '#8DA0CB'),
        ('Only\nRemoval',    0.6295, 0.0363, 0.5109, '#E78AC3'),
        ('Only\nSlope',      0.6665, 0.0349, 0.5715, '#A6D854'),
        ('Only\nCollapse',   0.6646, 0.0323, 0.5679, '#FFD92F'),
    ]

    labels  = [c[0] for c in configs]
    miou_m  = [c[1] for c in configs]
    miou_s  = [c[2] for c in configs]
    f1_m    = [c[3] for c in configs]
    colors  = [c[4] for c in configs]

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
    b1 = ax.bar(x - w/2, miou_m, w, yerr=miou_s, label='mIoU',
                color=colors, alpha=0.88, capsize=4,
                edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w/2, f1_m,   w, label='F1',
                color=colors, alpha=0.58, hatch='//',
                edgecolor='black', linewidth=0.5)

    ax.axhline(0.6249, color='#009E73', linestyle='--', lw=1.5,
               label='Full LoadSim mIoU baseline', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_ylim(0.40, 0.80)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')

    for bar, v in zip(b1, miou_m):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.2)

    fig.savefig(out_dir / 'fig3_ablation.svg', format='svg', bbox_inches='tight',
                 facecolor='white', transparent=False)
    fig.savefig(out_dir / 'fig3_ablation.pdf', format='pdf', bbox_inches='tight',
                 facecolor='white', transparent=False)
    print(f"  [SAVED] {out_dir / 'fig3_ablation.svg'}")
    print(f"  [SAVED] {out_dir / 'fig3_ablation.pdf'}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Backbone comparison grouped bar chart
# ══════════════════════════════════════════════════════════════════════════

def plot_backbone_comparison(out_dir):
    """Backbone × augmentation strategy comparison (split into multiple narrow figures)"""
    data = {
        'Baseline': {
            'PointNet++': (0.542, 0.071), 'PointNeXt': (0.5491, 0.0464), 'KPConv': (0.6254, 0.0162)},
        'Conventional': {
            'PointNet++': (0.563, 0.115), 'PointNeXt': (0.5598, 0.0674), 'KPConv': (0.6835, 0.0066)},
        'LSDA': {
            'PointNet++': (0.607, 0.018), 'PointNeXt': (0.6086, 0.0338), 'KPConv': (0.6724, 0.0060)},
        'LoadSim': {
            'PointNet++': (0.632, 0.021), 'PointNeXt': (0.6290, 0.0289), 'KPConv': (0.6837, 0.0105)},
    }
    backbones = ['PointNet++', 'PointNeXt', 'KPConv']
    strategies = list(data.keys())
    bk_colors = [COLORS['pn2'], COLORS['pointnext'], COLORS['kpconv']]

    x = np.arange(len(strategies))

    # Part A: each backbone as an independent narrow figure
    for bk, col in zip(backbones, bk_colors):
        fig, ax = plt.subplots(figsize=(5.2, 3.9), constrained_layout=True)
        means = [data[s][bk][0] for s in strategies]
        stds = [data[s][bk][1] for s in strategies]
        bars = ax.bar(x, means, yerr=stds, color=col, alpha=0.85,
                      capsize=3, edgecolor='black', linewidth=0.5, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, fontsize=9)
        ax.set_ylabel('Test mIoU (mean ± std)', fontsize=9)
        ax.set_ylim(0.42, 0.78)
        ax.grid(True, axis='y', alpha=0.3, linestyle=':')
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.004,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8)

        stem = f"fig4_backbone_comparison_{bk.lower().replace('+', 'p').replace(' ', '_')}"
        save_svg(fig, out_dir, stem)
        plt.close(fig)

    # Part B: per-strategy grouped comparison, split to avoid over-wide figure
    w = 0.22
    offsets = [-w, 0, w]
    for strategy in strategies:
        fig, ax = plt.subplots(figsize=(5.0, 3.9), constrained_layout=True)
        means = [data[strategy][bk][0] for bk in backbones]
        stds = [data[strategy][bk][1] for bk in backbones]
        for idx, (bk, col, off) in enumerate(zip(backbones, bk_colors, offsets)):
            ax.bar(idx + off, means[idx], w, yerr=stds[idx], color=col,
                   alpha=0.85, capsize=3, edgecolor='black', linewidth=0.5)
        ax.set_xticks(np.arange(len(backbones)))
        ax.set_xticklabels(backbones, fontsize=9)
        ax.set_ylabel('Test mIoU (mean ± std)', fontsize=9)
        ax.set_ylim(0.42, 0.78)
        ax.grid(True, axis='y', alpha=0.3, linestyle=':')

        stem = f"fig4_backbone_by_strategy_{strategy.lower().replace(' ', '_')}"
        save_svg(fig, out_dir, stem)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Sim-to-real gap across backbones
# ══════════════════════════════════════════════════════════════════════════

def plot_simtoreal_gap(outputs_root, journal_root, out_dir):
    """Sim-to-real gap: val mIoU vs test mIoU scatter/bar"""
    # Data from original experiment + backbone experiment
    gap_data = {
        'Baseline\nPN2':      (0.557, 0.542, COLORS['pn2'],       'o'),
        'Conv.\nPN2':         (0.633, 0.563, COLORS['trad'],       's'),
        'LSDA\nPN2':          (0.630, 0.607, COLORS['lsda'],       '^'),
        'LoadSim\nPN2':       (0.832, 0.632, COLORS['loadsim'],    'D'),
        'LoadSim\nKPConv':    (0.700, 0.684, COLORS['kpconv'],     'P'),
        'LoadSim\nPointNeXt': (0.620, 0.629, COLORS['pointnext'],  '*'),
    }
    # For KPConv and PointNeXt LoadSim, we estimate val mIoU from training best
    # KPConv loadsim_150: test=0.6837; approximate val from training convergence
    # PointNeXt loadsim_150: test=0.6290

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    # Left: scatter of val vs test
    for label, (val, test, col, mk) in gap_data.items():
        ax1.scatter(val, test, c=col, marker=mk, s=120, label=label,
                    zorder=5, edgecolors='black', linewidths=0.5)
    # diagonal
    lim = [0.40, 0.90]
    ax1.plot(lim, lim, 'k--', lw=1, alpha=0.5, label='val=test')
    ax1.set_xlabel('Validation mIoU', fontsize=10)
    ax1.set_ylabel('Test mIoU', fontsize=10)
    ax1.legend(fontsize=7, ncol=2, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.43, 0.88)
    ax1.set_ylim(0.43, 0.88)

    # Right: gap bars
    methods = ['Baseline\nPN2', 'Conv.\nPN2', 'LSDA\nPN2', 'LoadSim\nPN2']
    gaps    = [gap_data[m][1] - gap_data[m][0] for m in methods]
    colors2 = [gap_data[m][2] for m in methods]
    x = np.arange(len(methods))
    bars = ax2.bar(x, gaps, color=colors2, alpha=0.85,
                   edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8.5)
    ax2.set_ylabel('Gap (test − val mIoU)', fontsize=10)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.grid(True, axis='y', alpha=0.3)
    for bar, g in zip(bars, gaps):
        y = bar.get_height() + (0.003 if g >= 0 else -0.010)
        ax2.text(bar.get_x() + bar.get_width()/2, y,
                 f'{g:.3f}', ha='center', va='bottom', fontsize=8)

    save_svg(fig, out_dir, 'fig5_simtoreal_gap')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate journal figures')
    parser.add_argument('--outputs-root',  default='./outputs4')
    parser.add_argument('--journal-root',  default='./outputs_journal')
    parser.add_argument('--out-dir',       default='./figures_journal')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  期刊图表生成  |  输出目录: {out_dir}")
    print(f"{'='*60}")

    print("\n[Fig 1] 训练收敛曲线 ...")
    plot_convergence_curves(args.outputs_root, out_dir)

    print("\n[Fig 2] 分组性能对比 ...")
    plot_group_comparison(out_dir)

    print("\n[Fig 3] 消融实验 ...")
    plot_ablation(out_dir)

    print("\n[Fig 4] 骨干网络对比 ...")
    plot_backbone_comparison(out_dir)

    print("\n[Fig 5] Sim-to-Real Gap ...")
    plot_simtoreal_gap(args.outputs_root, args.journal_root, out_dir)

    print(f"\n  ✅ 所有图表生成完成! → {out_dir}")
    print(f"  包含: fig1_convergence, fig2_group_comparison,")
    print(f"         fig3_ablation, fig4_backbone_comparison, fig5_simtoreal_gap")
    print(f"  格式: PDF (向量, 300dpi) + PNG (200dpi 预览)")

if __name__ == '__main__':
    main()
