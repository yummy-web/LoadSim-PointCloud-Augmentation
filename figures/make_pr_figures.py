# -*- coding: utf-8 -*-
"""
make_pr_figures.py — Pattern Recognition submission figure generator
=====================================================================
Regenerates all manuscript figures in publication formats:
  * Vector PDF  (preferred for LaTeX/Elsevier final submission)
  * High-resolution JPG (300 dpi, for Word manuscript embedding)

Design rules enforced (per PR submission requirements):
  * NO overall/figure-level title inside the image (caption lives in the paper)
  * Multi-panel figures keep per-subplot (a)/(b)/(c) labels
  * Colour-blind-safe Okabe-Ito palette, consistent across figures

Data sources:
  * Chart values: experiment result tables (4-fold CV, ablation, multi-backbone)
    and training_history.json under outputs4/
  * Pseudo-label point-cloud panels: pre-rendered high-res PNGs under
    outputs4/visualization/  (rendering needs torch+open3d+GPU; rasters reused)

Usage:
    python make_pr_figures.py \
        --outputs-root ../../outputs4 \
        --viz-root     ../../outputs4/visualization \
        --out-dir      ../output
"""
import argparse, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams.update({
    'pdf.fonttype': 42,        # editable TrueType in PDF (Elsevier requirement)
    'ps.fonttype': 42,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.8,
})

# Okabe-Ito colour-blind-safe palette
COLORS = {
    'baseline':  '#555555',
    'trad':      '#E69F00',
    'lsda':      '#56B4E9',
    'loadsim':   '#009E73',
    'pn2':       '#0072B2',
    'pointnext': '#D55E00',
    'kpconv':    '#CC79A7',
}

DPI_JPG = 300


def _save(fig, out_dir, stem):
    """Save a figure as vector PDF + 300 dpi JPG."""
    out_dir = Path(out_dir)
    pdf = out_dir / f'{stem}.pdf'
    jpg = out_dir / f'{stem}.jpg'
    fig.savefig(pdf, format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(jpg, format='jpg', dpi=DPI_JPG, bbox_inches='tight',
                facecolor='white', pil_kwargs={'quality': 95})
    plt.close(fig)
    print(f"  [SAVED] {pdf.name} + {jpg.name}")


def load_history(path):
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except Exception:
        return None
    return d.get('history', d)


# ════════════════════════════════════════════════════════════════════════
# Figure 1 : Convergence curves (a,b,c) + pseudo-label panels (d,e,f,g)
#            Composite 2-row figure used as manuscript Figure 1
# ════════════════════════════════════════════════════════════════════════
def fig1_convergence_and_labels(outputs_root, viz_root, out_dir):
    outputs_root, viz_root = Path(outputs_root), Path(viz_root)
    fold = outputs_root / 'fold_00'
    branch_cfg = {
        'baseline_cv': ('Baseline (N=11)',     COLORS['baseline'], '-'),
        'trad_150':    ('Conventional (N=150)', COLORS['trad'],     '--'),
        'lsda_150':    ('LSDA (N=150)',         COLORS['lsda'],     '-.'),
        'loadsim_150': ('LoadSim (N=150)',      COLORS['loadsim'],  '-'),
    }
    metrics = [('val_mIoU', 'Validation mIoU'),
               ('val_f1',   'Validation F1'),
               ('val_loss', 'Validation Loss')]

    # ---- standalone version: 1x3 convergence only ----
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    for branch, (label, color, ls) in branch_cfg.items():
        hist = load_history(fold / branch / 'model' / 'training_history.json')
        if not hist:
            print(f"  [WARN] missing history: {branch}")
            continue
        for ax, (key, ylabel) in zip(axes, metrics):
            if key in hist:
                y = hist[key]
                ax.plot(range(1, len(y) + 1), y, label=label,
                        color=color, linestyle=ls, lw=1.8, alpha=0.9)
    panel = ['(a)', '(b)', '(c)']
    for ax, (key, ylabel), pl in zip(axes, metrics, panel):
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, 71)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.tick_params(labelsize=9)
        ax.text(0.02, 0.97, pl, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', ha='left')
    axes[0].legend(fontsize=8, loc='lower right')
    _save(fig, out_dir, 'fig1_convergence')

    # ---- pseudo-label qualitative panels on DJI_3 (b,c,d,e) ----
    viz_files = [
        ('baseline_cv', 'Baseline'),
        ('trad_150',    'Conventional'),
        ('lsda_150',    'LSDA'),
        ('loadsim_150', 'LoadSim'),
    ]
    imgs = []
    for branch, name in viz_files:
        p = viz_root / 'DJI_3' / f'DJI_3_{branch}.png'
        if p.exists():
            imgs.append((mpimg.imread(str(p)), name))
    if imgs:
        fig, axes = plt.subplots(1, len(imgs), figsize=(3.0 * len(imgs), 3.2),
                                 constrained_layout=True)
        if len(imgs) == 1:
            axes = [axes]
        labels = ['(a)', '(b)', '(c)', '(d)']
        for ax, (im, name), pl in zip(axes, imgs, labels):
            ax.imshow(im)
            ax.set_xlabel(f'{pl} {name}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        _save(fig, out_dir, 'fig1b_pseudolabel_DJI3')


# ════════════════════════════════════════════════════════════════════════
# Figure 2 : Group A / Group B strategy comparison (PointNet++)
# ════════════════════════════════════════════════════════════════════════
def fig2_group_comparison(out_dir):
    group_A = {
        'Baseline\n(N=11)': (0.542, 0.071, COLORS['baseline']),
        'Conv.\n(N=150)':   (0.563, 0.115, COLORS['trad']),
        'LSDA\n(N=150)':    (0.607, 0.018, COLORS['lsda']),
        'LoadSim\n(N=150)': (0.632, 0.021, COLORS['loadsim']),
    }
    group_B = {
        'Conv.\n(N=11)':    (0.450, 0.038, COLORS['trad']),
        'LSDA\n(N=11)':     (0.472, 0.001, COLORS['lsda']),
        'LoadSim\n(N=11)':  (0.473, 0.001, COLORS['loadsim']),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0),
                                   gridspec_kw={'width_ratios': [4, 3]},
                                   constrained_layout=True)
    for ax, group, pl in [(ax1, group_A, '(a)'), (ax2, group_B, '(b)')]:
        names = list(group.keys())
        means = [v[0] for v in group.values()]
        stds  = [v[1] for v in group.values()]
        cols  = [v[2] for v in group.values()]
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, color=cols, alpha=0.85, width=0.55,
                      capsize=5, error_kw={'lw': 1.5}, edgecolor='black',
                      linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel('Test mIoU (mean ± std)', fontsize=10)
        ax.set_ylim(0.30, 0.72)
        ax.grid(True, axis='y', alpha=0.3, linestyle=':')
        ax.text(0.02, 0.97, pl, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', ha='left')
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, m + 0.008,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8)
    _save(fig, out_dir, 'fig2_group_comparison')


# ════════════════════════════════════════════════════════════════════════
# Figure 3 : LoadSim component ablation (7 configurations)
# ════════════════════════════════════════════════════════════════════════
def fig3_ablation(out_dir):
    configs = [
        ('Full\nLoadSim',  0.6249, 0.0577, 0.4748, '#009E73'),
        ('w/o\nRemoval',   0.6355, 0.0366, 0.5106, '#66C2A5'),
        ('w/o\nSlope',     0.6737, 0.0487, 0.5763, '#FC8D62'),
        ('w/o\nCollapse',  0.6490, 0.0189, 0.5494, '#8DA0CB'),
        ('Only\nRemoval',  0.6295, 0.0363, 0.5109, '#E78AC3'),
        ('Only\nSlope',    0.6665, 0.0349, 0.5715, '#A6D854'),
        ('Only\nCollapse', 0.6646, 0.0323, 0.5679, '#FFD92F'),
    ]
    labels = [c[0] for c in configs]
    miou_m = [c[1] for c in configs]
    miou_s = [c[2] for c in configs]
    f1_m   = [c[3] for c in configs]
    colors = [c[4] for c in configs]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
    b1 = ax.bar(x - w/2, miou_m, w, yerr=miou_s, label='mIoU', color=colors,
                alpha=0.88, capsize=4, edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, f1_m, w, label='F1', color=colors, alpha=0.58, hatch='//',
           edgecolor='black', linewidth=0.5)
    ax.axhline(0.6249, color='#009E73', linestyle='--', lw=1.5,
               label='Full LoadSim mIoU baseline', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('Score', fontsize=10); ax.set_ylim(0.40, 0.80)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    for bar, v in zip(b1, miou_m):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.2)
    _save(fig, out_dir, 'fig3_ablation')


# ════════════════════════════════════════════════════════════════════════
# Figure 4 : Multi-backbone model-agnostic validation (5 backbones)
# ════════════════════════════════════════════════════════════════════════
def fig4_backbone(out_dir):
    # Group A (N=150) test mIoU: baseline(N=11) vs LoadSim(N=150)
    data = {
        'PointNet++': (0.542, 0.632, 0.071, 0.021),
        'PointNeXt':  (0.549, 0.629, 0.046, 0.029),
        'KPConv':     (0.625, 0.684, 0.016, 0.011),
        'RandLA-Net': (0.555, 0.730, 0.079, 0.014),
        'PTv3':       (0.666, 0.761, 0.009, 0.026),
    }
    backbones = list(data.keys())
    baseline = [data[b][0] for b in backbones]
    loadsim  = [data[b][1] for b in backbones]
    b_std    = [data[b][2] for b in backbones]
    l_std    = [data[b][3] for b in backbones]
    x = np.arange(len(backbones)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    ax.bar(x - w/2, baseline, w, yerr=b_std, label='Baseline (N=11)',
           color=COLORS['baseline'], alpha=0.8, capsize=4,
           edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + w/2, loadsim, w, yerr=l_std, label='LoadSim (N=150)',
                   color=COLORS['loadsim'], alpha=0.88, capsize=4,
                   edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(backbones, fontsize=9.5)
    ax.set_ylabel('Test mIoU (mean ± std)', fontsize=10)
    ax.set_ylim(0.40, 0.83)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    for b, base, ls in zip(x, baseline, loadsim):
        ax.text(b + w/2, ls + 0.012, f'+{ls-base:.3f}', ha='center',
                va='bottom', fontsize=8, color='#006644', fontweight='bold')
    _save(fig, out_dir, 'fig4_backbone_comparison')


# ════════════════════════════════════════════════════════════════════════
# Figure 5 : Sim-to-real gap (val vs test scatter + gap bars)
# ════════════════════════════════════════════════════════════════════════
def fig5_simtoreal(out_dir):
    gap = {
        'Baseline\nPN++':     (0.557, 0.542, COLORS['pn2'],       'o'),
        'Conv.\nPN++':        (0.633, 0.563, COLORS['trad'],      's'),
        'LSDA\nPN++':         (0.630, 0.607, COLORS['lsda'],      '^'),
        'LoadSim\nPN++':      (0.832, 0.632, COLORS['loadsim'],   'D'),
        'LoadSim\nKPConv':    (0.700, 0.684, COLORS['kpconv'],    'P'),
        'LoadSim\nPointNeXt': (0.620, 0.629, COLORS['pointnext'], '*'),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0),
                                   constrained_layout=True)
    for label, (val, test, col, mk) in gap.items():
        ax1.scatter(val, test, c=col, marker=mk, s=120, label=label, zorder=5,
                    edgecolors='black', linewidths=0.5)
    lim = [0.40, 0.90]
    ax1.plot(lim, lim, 'k--', lw=1, alpha=0.5, label='val = test')
    ax1.set_xlabel('Validation mIoU', fontsize=10)
    ax1.set_ylabel('Test mIoU', fontsize=10)
    ax1.legend(fontsize=7, ncol=2, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.43, 0.88); ax1.set_ylim(0.43, 0.88)
    ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes, fontsize=11,
             fontweight='bold', va='top')
    methods = ['Baseline\nPN++', 'Conv.\nPN++', 'LSDA\nPN++', 'LoadSim\nPN++']
    gaps = [gap[m][1] - gap[m][0] for m in methods]
    cols2 = [gap[m][2] for m in methods]
    x = np.arange(len(methods))
    bars = ax2.bar(x, gaps, color=cols2, alpha=0.85, edgecolor='black',
                   linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=8.5)
    ax2.set_ylabel('Gap (test − val mIoU)', fontsize=10)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes, fontsize=11,
             fontweight='bold', va='top')
    for bar, g in zip(bars, gaps):
        y = bar.get_height() + (0.003 if g >= 0 else -0.012)
        ax2.text(bar.get_x() + bar.get_width()/2, y, f'{g:.3f}',
                 ha='center', va='bottom', fontsize=8)
    _save(fig, out_dir, 'fig5_simtoreal_gap')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs-root', default='../../outputs4')
    ap.add_argument('--viz-root',     default='../../outputs4/visualization')
    ap.add_argument('--out-dir',      default='../output')
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("=== PR figure generation (PDF + 300dpi JPG, no titles) ===")
    fig1_convergence_and_labels(a.outputs_root, a.viz_root, out)
    fig2_group_comparison(out)
    fig3_ablation(out)
    fig4_backbone(out)
    fig5_simtoreal(out)
    print(f"Done -> {out}")


if __name__ == '__main__':
    main()
