"""
kitti_visualize.py  ——  SemanticKITTI 实验结果可视化 [v3]
==========================================================
生成:
  - Val mIoU 收敛曲线（4 策略）
  - 测试集对比条形图（mIoU / F1_ground / IoU_ground）
  - LaTeX 格式结果表格（可直接复制到论文）

用法:
    python kitti_visualize.py --result-dir ./kitti_outputs
"""
import sys, json, argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 不可用，跳过图表生成")

# v3: traditional（而非 conventional）
STRATEGIES = ["baseline", "traditional", "lsda", "loadsim"]

COLORS = {
    "baseline":    "#555555",
    "traditional": "#E69F00",
    "lsda":        "#56B4E9",
    "loadsim":     "#009E73",
}
LABELS = {
    "baseline":    "Baseline",
    "traditional": "Conventional",
    "lsda":        "LSDA",
    "loadsim":     "LoadSim (ours)",
}


def plot_convergence(log_dir: Path, out_dir: Path):
    """Val mIoU 收敛曲线（4 策略）。"""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for strat, color in COLORS.items():
        hf = log_dir / f"{strat}_history.json"
        if not hf.exists():
            continue
        with open(hf) as f:
            d = json.load(f)
        hist     = d.get("history", {})
        best_ep  = d.get("best_metrics", {}).get("epoch", None)

        if "val_mIoU" in hist:
            y = hist["val_mIoU"]
            x = list(range(1, len(y) + 1))
            axes[0].plot(x, y, label=LABELS[strat], color=color,
                         lw=2.0, alpha=0.9)
            if best_ep:
                axes[0].axvline(x=best_ep, color=color, ls="--",
                                lw=0.8, alpha=0.5)

        if "val_F1_ground" in hist:
            y = hist["val_F1_ground"]
            x = list(range(1, len(y) + 1))
            axes[1].plot(x, y, label=LABELS[strat], color=color,
                         lw=2.0, alpha=0.9)

    for ax, ylabel, title in zip(axes,
        ["Validation mIoU", "Validation F1 (Ground)"],
        ["mIoU Convergence", "F1_ground Convergence"]):
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, ls=":")

    fig.suptitle(
        "Training Convergence — SemanticKITTI Seq.00 (Ground Segmentation)",
        fontsize=12, y=1.02)
    plt.tight_layout()

    out = out_dir / "kitti_convergence.pdf"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [图] {out}")


def plot_test_comparison(result_file: Path, out_dir: Path):
    """测试集对比条形图（3 指标并排）。"""
    if not HAS_MPL:
        return
    if not result_file.exists():
        print(f"  [SKIP] 结果文件不存在: {result_file}")
        return

    with open(result_file) as f:
        data = json.load(f)
    results = data.get("results", {})

    metrics_cfg = [
        ("mIoU",       "Test mIoU"),
        ("F1_ground",  "Test F1 (Ground)"),
        ("IoU_ground", "Test IoU (Ground)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (mk, ylabel) in zip(axes, metrics_cfg):
        vals  = [results.get(s, {}).get("test", {}).get(mk, 0)
                 for s in STRATEGIES]
        cols  = [COLORS[s] for s in STRATEGIES]
        xlabs = [LABELS[s] for s in STRATEGIES]

        bars = ax.bar(range(len(STRATEGIES)), vals, color=cols,
                      alpha=0.85, edgecolor="black", linewidth=0.6,
                      zorder=3)
        ax.set_xticks(range(len(STRATEGIES)))
        ax.set_xticklabels(xlabs, rotation=18, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max(max(vals) * 1.15, 0.1))
        ax.grid(True, axis="y", alpha=0.3, ls=":", zorder=0)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        # 基准线
        if vals[0] > 0:
            ax.axhline(vals[0], color=COLORS["baseline"], ls="--",
                       lw=1.0, alpha=0.6)

    n_tr = data.get("n_train", "?")
    n_vl = data.get("n_val", "?")
    n_ts = data.get("n_test", "?")
    fig.suptitle(
        f"Augmentation Strategy Comparison — SemanticKITTI Seq.00\n"
        f"Train: {n_tr}  Val: {n_vl}  Test: {n_ts} (original frames)",
        fontsize=11, y=1.04)
    plt.tight_layout()

    out = out_dir / "kitti_test_comparison.pdf"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [图] {out}")


def print_latex_table(result_file: Path):
    """输出 LaTeX 格式结果表格（可直接复制到论文）。"""
    if not result_file.exists():
        return

    with open(result_file) as f:
        data = json.load(f)
    results  = data.get("results", {})
    n_tr     = data.get("n_train", "?")
    n_ts     = data.get("n_test",  "?")

    print("\n" + "=" * 72)
    print("  LaTeX 表格（可直接复制到论文）")
    print("=" * 72)
    print(r"\begin{table}[ht]")
    print(r"\caption{Generalization on SemanticKITTI Seq.~00: "
          r"Ground/Non-ground Segmentation}")
    print(r"\label{tab:kitti}")
    print(r"\centering\small")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Strategy & Val mIoU & Test mIoU "
          r"& F1$_{\rm ground}$ & IoU$_{\rm ground}$ \\")
    print(r"\midrule")

    for s in STRATEGIES:
        vm = results.get(s, {}).get("val",  {})
        tm = results.get(s, {}).get("test", {})
        v_miou     = vm.get("mIoU",       0)
        t_miou     = tm.get("mIoU",       0)
        t_f1g      = tm.get("F1_ground",  0)
        t_ioug     = tm.get("IoU_ground", 0)
        best_mark  = r" $\dagger$" if s == "loadsim" else ""
        print(f"{LABELS[s]:22s} & "
              f"{v_miou:.4f} & "
              f"{t_miou:.4f}{best_mark} & "
              f"{t_f1g:.4f} & "
              f"{t_ioug:.4f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(rf"\vspace{{1pt}}")
    print(rf"\begin{{flushleft}}\footnotesize "
          rf"Training/Val/Test: {n_tr}/{data.get('n_val','?')}/{n_ts} frames. "
          rf"All branches use equal sample counts. "
          rf"Test set: original frames (no augmentation).\end{{flushleft}}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="SemanticKITTI 实验结果可视化 v3")
    parser.add_argument("--result-dir", default="./kitti_outputs")
    args = parser.parse_args()

    result_dir  = Path(args.result_dir)
    log_dir     = result_dir / "logs"
    figure_dir  = result_dir / "figures"
    result_file = result_dir / "results" / "kitti_results.json"

    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  生成论文图表 ({result_dir})...", flush=True)
    plot_convergence(log_dir, figure_dir)
    plot_test_comparison(result_file, figure_dir)
    print_latex_table(result_file)
    print(f"\n  图表输出目录: {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
