"""
kitti_train.py  ——  SemanticKITTI 泛化性实验训练主控 [v3]
==========================================================

实验设计（v3 等量公平对比）:
  各分支数量完全相同（3200 训练 / 600 验证 / 741 测试）:
    baseline:    原始帧直接训练
    traditional: 传统几何增强变体（每帧 1 个，仅变体，不含原始帧）
    lsda:        LoadSim + 传统组合变体
    loadsim:     纯 LoadSim 物理增强变体

  测试集: 741 帧原始帧，所有分支共享，固定不变

断点续传支持:
  缓存生成: 每 200 帧保存一次临时进度，中断后自动从最后断点恢复
  模型训练: 若 best_model.pth + training_history.json 均已存在则跳过
  测试评估: 若 test_metrics.json 已存在则跳过

用法:
    # 完整实验（4 个分支，约 8-12 小时）
    python kitti_train.py --data-dir E:/BaiduNetdiskDownload/sequences

    # 只运行某一分支（支持中断后继续）
    python kitti_train.py --data-dir ./sequences --strategy loadsim

    # 跳过已完成的缓存生成（直接用已有 .npz）
    python kitti_train.py --data-dir ./sequences --skip-aug

    # 调试模式（100 帧 + 5 epoch，几分钟完成）
    python kitti_train.py --data-dir ./sequences --debug

    # 生成结果图表
    python kitti_visualize.py --result-dir ./kitti_outputs
"""

import sys, json, argparse, warnings, time
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch 未安装，训练步骤将跳过", flush=True)

from config_kitti import (
    DATA_DIR, SPLIT, AUG, MODEL, TRAIN, DEBUG,
    OUT_ROOT, AUG_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR,
    N_POINTS, GROUND_CLASSES
)
from kitti_dataset import (
    KITTIGroundDataset, get_frame_paths,
    build_aug_cache, load_bin, load_label, preprocess_frame
)


# ══════════════════════════════════════════════════════════════════════════
# PointNet++ 分割网络（与原始散料堆实验完全相同架构）
# ══════════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def _index_pts(pts, idx):
        B, dev = pts.shape[0], pts.device
        bidx = (torch.arange(B, device=dev)
                .view(B, *([1] * (idx.dim() - 1)))
                .expand_as(idx))
        return pts[bidx, idx]

    def _fps(xyz, n_pt):
        """最远点采样: (B,N,3) → (B,n_pt) 索引。"""
        B, N, _ = xyz.shape
        dev  = xyz.device
        sel  = torch.zeros(B, n_pt, dtype=torch.long, device=dev)
        dist = torch.full((B, N), float("inf"), device=dev)
        far  = torch.randint(0, N, (B,), device=dev)
        for i in range(n_pt):
            sel[:, i] = far
            ctr  = xyz[torch.arange(B, device=dev), far].unsqueeze(1)
            d    = ((xyz - ctr) ** 2).sum(-1)
            dist = torch.minimum(dist, d)
            far  = dist.max(-1)[1]
        return sel

    def _ball(r, k, xyz, nxyz):
        """球查询: 返回 (B,S,k) 邻居索引。"""
        B, N, _ = xyz.shape
        _, S, _ = nxyz.shape
        dist = torch.cdist(nxyz, xyz)
        dm   = dist.clone()
        dm[dist > r] = float("inf")
        if N <= k:
            _, top = dm.topk(N, dim=-1, largest=False)
            idx = torch.cat([top, top[:, :, :1].expand(B, S, k - N)], dim=-1)
        else:
            _, idx = dm.topk(k, dim=-1, largest=False)
        inv = dist.gather(2, idx) > r
        idx = idx.clone()
        idx[inv] = idx[:, :, :1].expand_as(idx)[inv]
        return idx

    class _SA(nn.Module):
        def __init__(self, npt, r, k, in_ch, out_ch):
            super().__init__()
            self.npt, self.r, self.k = npt, r, k
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + 3, out_ch, 1),
                nn.BatchNorm2d(out_ch), nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch), nn.ReLU(True))

        def forward(self, xyz, f):
            B, N, _ = xyz.shape
            S     = min(self.npt, N)
            nidx  = _fps(xyz, S)
            nxyz  = _index_pts(xyz, nidx)
            bidx  = _ball(self.r, min(self.k, N), xyz, nxyz)
            gxyz  = _index_pts(xyz, bidx) - nxyz.unsqueeze(2)
            if f is not None:
                gf = _index_pts(f.permute(0, 2, 1), bidx)
                g  = torch.cat([gxyz, gf], -1).permute(0, 3, 2, 1)
            else:
                g  = gxyz.permute(0, 3, 2, 1)
            return nxyz, self.conv(g).max(2)[0]

    class _FP(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch), nn.ReLU(True),
                nn.Conv1d(out_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch), nn.ReLU(True))

        def forward(self, xyz1, xyz2, f1, f2):
            B, N, _ = xyz1.shape
            _, S, _ = xyz2.shape
            k  = min(3, S)
            d  = torch.cdist(xyz1, xyz2)
            dk, ki = d.topk(k, dim=-1, largest=False)
            dk = dk.clamp(1e-10)
            w  = (1.0 / dk) / (1.0 / dk).sum(-1, keepdim=True)
            C2 = f2.shape[1]
            interp = (f2.unsqueeze(2).expand(B, C2, N, S)
                      .gather(3, ki.unsqueeze(1).expand(B, C2, -1, -1))
                      * w.unsqueeze(1)).sum(-1)
            out = torch.cat([f1, interp], 1) if f1 is not None else interp
            return self.mlp(out)

    class PointNetPPKITTI(nn.Module):
        """
        PointNet++ 分割网络 — KITTI 米制版本。

        架构与原始散料堆实验完全相同，SA 层球查询半径 ×10 (cm→m):
          1.0 / 2.5 / 5.0 / 10.0 m（适配城市驾驶场景尺度）
        """

        def __init__(self, in_ch: int = 6, n_cls: int = 2):
            super().__init__()
            ex = in_ch - 3
            self.sa1 = _SA(1024,  1.0,  32, ex,    64)
            self.sa2 = _SA(256,   2.5,  64, 64,   128)
            self.sa3 = _SA(64,    5.0, 128, 128,  256)
            self.sa4 = _SA(16,   10.0, 256, 256,  512)
            self.fp4 = _FP(256 + 512, 256)
            self.fp3 = _FP(128 + 256, 256)
            self.fp2 = _FP(64  + 256, 128)
            self.fp1 = _FP(ex  + 128, 128)
            self.head = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
                nn.Dropout(0.5), nn.Conv1d(128, n_cls, 1))
            self.n_params = sum(p.numel() for p in self.parameters())

        def forward(self, xyzn):
            B, C, N = xyzn.shape
            xyz = xyzn[:, :3, :].permute(0, 2, 1).contiguous()
            f0  = xyzn[:, 3:, :] if C > 3 else None
            x1, f1 = self.sa1(xyz, f0)
            x2, f2 = self.sa2(x1,  f1)
            x3, f3 = self.sa3(x2,  f2)
            x4, f4 = self.sa4(x3,  f3)
            f3b = self.fp4(x3, x4, f3, f4)
            f2b = self.fp3(x2, x3, f2, f3b)
            f1b = self.fp2(x1, x2, f1, f2b)
            f0b = self.fp1(xyz, x1, f0, f1b)
            return self.head(f0b)


# ══════════════════════════════════════════════════════════════════════════
# 评估指标
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """计算 mIoU、F1、accuracy（与原始散料堆实验完全一致）。"""
    ious, f1s, precs, recs = [], [], [], []
    for c in range(2):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        iou  = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        ious.append(float(iou))
        f1s.append(float(f1))
        precs.append(float(prec))
        recs.append(float(rec))
    return {
        "mIoU":               float(np.mean(ious)),
        "IoU_bg":             ious[0],
        "IoU_ground":         ious[1],
        "accuracy":           float((preds == labels).mean()),
        "F1_ground":          f1s[1],
        "F1_mean":            float(np.mean(f1s)),
        "precision":          precs[1],
        "recall":             recs[1],
        "pred_ground_ratio":  float((preds == 1).mean()),
        "label_ground_ratio": float((labels == 1).mean()),
    }


# ══════════════════════════════════════════════════════════════════════════
# 单 epoch 训练 / 验证
# ══════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    def run_epoch(model, loader, optimizer, scaler, dev,
                  criterion, train: bool = True) -> dict:
        """执行一个 epoch，返回评估指标。"""
        model.train() if train else model.eval()
        tot_loss = 0.0
        all_pred = []
        all_lbl  = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                if train:
                    optimizer.zero_grad()
                with autocast():
                    logits = model(x)
                    loss   = criterion(logits, y)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                tot_loss += loss.item()
                all_pred.append(logits.argmax(1).detach().cpu().numpy().reshape(-1))
                all_lbl.append(y.detach().cpu().numpy().reshape(-1))

        all_pred = np.concatenate(all_pred)
        all_lbl  = np.concatenate(all_lbl)
        metrics  = compute_metrics(all_pred, all_lbl)
        metrics["loss"] = tot_loss / max(len(loader), 1)
        return metrics


# ══════════════════════════════════════════════════════════════════════════
# 单策略训练（支持断点续传）
# ══════════════════════════════════════════════════════════════════════════

def train_one_strategy(strategy: str,
                       train_cache, val_cache,
                       model_dir: Path, log_dir: Path,
                       P_TR: dict, P_MDL: dict,
                       debug: bool = False) -> dict:
    """
    训练一个策略的 PointNet++ 模型并返回最优验证指标。

    断点续传:
      - 若 {strategy}_best.pth 和 {strategy}_history.json 均已存在，
        则跳过该策略的训练。
      - 基于 val_mIoU 单一指标早停（稳定、清晰）。

    Args:
        strategy:    'baseline' | 'traditional' | 'lsda' | 'loadsim'
        train_cache: list of (pts6, labels) — 训练集增强缓存
        val_cache:   list of (pts6, labels) — 验证集增强缓存
        model_dir:   模型保存目录
        log_dir:     日志目录
        P_TR, P_MDL: 训练/模型超参
        debug:       调试模式

    Returns:
        best_metrics: 最优验证集指标 dict
    """
    if not HAS_TORCH:
        return {}

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_pth  = model_dir / f"{strategy}_best.pth"
    hist_json = log_dir   / f"{strategy}_history.json"

    # ── 断点续传：跳过已完成的训练 ────────────────────────────────────
    if best_pth.exists() and hist_json.exists():
        print(f"\n  [SKIP] {strategy}: 模型已存在，跳过训练", flush=True)
        with open(hist_json, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("best_metrics", {})

    dev = torch.device(
        P_TR["device"] if torch.cuda.is_available() else "cpu")

    # ── 构建 DataLoader ────────────────────────────────────────────────
    train_ds = KITTIGroundDataset(
        None, aug_cache=train_cache,
        augment=True, n_pts=P_MDL["n_points"])
    val_ds = KITTIGroundDataset(
        None, aug_cache=val_cache,
        augment=False, n_pts=P_MDL["n_points"])

    bs = P_TR["batch_size"]
    trn_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=P_TR["n_workers"], drop_last=True,
        pin_memory=(dev.type == "cuda"))
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=P_TR["n_workers"],
        pin_memory=(dev.type == "cuda"))

    # ── 统计地面比例 ──────────────────────────────────────────────────
    gr_tr = float(np.mean([lbl.mean() for _, lbl in train_cache]))
    gr_vl = float(np.mean([lbl.mean() for _, lbl in val_cache]))

    # ── 模型与优化器 ─────────────────────────────────────────────────
    model     = PointNetPPKITTI(
        in_ch=P_MDL["in_ch"], n_cls=P_MDL["n_cls"]).to(dev)
    optimizer = optim.Adam(
        model.parameters(), lr=P_TR["lr"],
        weight_decay=P_TR["weight_decay"])
    epochs    = P_TR["epochs"] if not debug else DEBUG["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    scaler    = GradScaler()
    cw        = torch.tensor(P_TR["class_weight"],
                             dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=cw)

    early_stop = P_TR["early_stop"] if not debug else DEBUG["early_stop"]

    print(f"\n  {'='*62}", flush=True)
    print(f"  策略: {strategy.upper()}  |  设备: {dev}", flush=True)
    print(f"  训练: {len(train_ds)} 样本 (地面比={gr_tr:.4f})"
          f"  验证: {len(val_ds)} 样本 (地面比={gr_vl:.4f})", flush=True)
    print(f"  参数量: {model.n_params:,}  Epochs: {epochs}"
          f"  EarlyStop: {early_stop}", flush=True)
    print(f"  {'Epoch':>6} {'TrnLoss':>9} {'TrnmIoU':>9} "
          f"{'ValLoss':>9} {'ValmIoU':>9} {'ValF1g':>8} {'PredGR':>7}",
          flush=True)
    print(f"  {'-'*62}", flush=True)

    history = {k: [] for k in [
        "trn_loss", "trn_mIoU",
        "val_loss", "val_mIoU",
        "val_F1_ground", "val_IoU_ground",
        "val_pred_ground_ratio"]}
    best_miou, best_metrics, patience = -1.0, {}, 0

    for ep in range(1, epochs + 1):
        t_m = run_epoch(model, trn_loader, optimizer, scaler,
                        dev, criterion, train=True)
        v_m = run_epoch(model, val_loader, optimizer, scaler,
                        dev, criterion, train=False)
        scheduler.step()

        for hk, src, mk in [
            ("trn_loss",  t_m, "loss"),   ("trn_mIoU",  t_m, "mIoU"),
            ("val_loss",  v_m, "loss"),   ("val_mIoU",  v_m, "mIoU"),
            ("val_F1_ground",  v_m, "F1_ground"),
            ("val_IoU_ground", v_m, "IoU_ground"),
            ("val_pred_ground_ratio", v_m, "pred_ground_ratio"),
        ]:
            history[hk].append(round(float(src.get(mk, 0)), 4))

        if ep % 10 == 0 or ep in (1, epochs):
            pgr = v_m.get("pred_ground_ratio", 0)
            print(f"  {ep:>6} {t_m['loss']:>9.4f} {t_m['mIoU']:>9.4f} "
                  f"{v_m['loss']:>9.4f} {v_m['mIoU']:>9.4f} "
                  f"{v_m['F1_ground']:>8.4f} {pgr:>7.4f}", flush=True)

        cur = v_m.get("mIoU", -1.0)
        if cur > best_miou:
            best_miou    = cur
            best_metrics = {k: round(float(v), 4) for k, v in v_m.items()}
            best_metrics["epoch"] = ep
            patience = 0
            torch.save(
                {"model_state": model.state_dict(),
                 "strategy": strategy, "epoch": ep,
                 "metrics": best_metrics},
                best_pth)
        else:
            patience += 1
            if patience >= early_stop:
                print(f"  早停 @ epoch {ep}  (best mIoU={best_miou:.4f})",
                      flush=True)
                break

    # 保存训练历史
    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump({"strategy": strategy, "history": history,
                   "best_metrics": best_metrics,
                   "n_train": len(train_ds), "n_val": len(val_ds),
                   "saved_at": str(datetime.now())},
                  f, indent=2)

    print(f"\n  ✅ 最优 Val mIoU={best_miou:.4f} "
          f"(F1_ground={best_metrics.get('F1_ground', 0):.4f}) "
          f"@ epoch {best_metrics.get('epoch')}", flush=True)
    return best_metrics


# ══════════════════════════════════════════════════════════════════════════
# 测试集评估（支持断点续传）
# ══════════════════════════════════════════════════════════════════════════

def evaluate_on_test(strategy: str, test_pairs,
                     model_dir: Path, result_dir: Path,
                     P_MDL: dict, P_TR: dict,
                     debug: bool = False) -> dict:
    """
    在时序测试集（原始帧）上评估已保存的最优模型。

    断点续传: 若 {strategy}_test_metrics.json 已存在则直接读取并返回。

    测试集始终使用原始帧（未经任何增强），所有分支共享。
    """
    metrics_json = result_dir / f"{strategy}_test_metrics.json"

    # ── 断点续传：跳过已完成的评估 ────────────────────────────────────
    if metrics_json.exists():
        print(f"  [SKIP] {strategy}: 测试结果已存在，直接读取", flush=True)
        with open(metrics_json, encoding="utf-8") as f:
            return json.load(f)

    if not HAS_TORCH:
        return {}

    best_pth = model_dir / f"{strategy}_best.pth"
    if not best_pth.exists():
        print(f"  [SKIP] {strategy}: 模型文件不存在 ({best_pth})",
              flush=True)
        return {}

    dev  = torch.device(
        P_TR["device"] if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(best_pth), map_location=dev)

    model = PointNetPPKITTI(
        in_ch=P_MDL["in_ch"], n_cls=P_MDL["n_cls"]).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pairs = test_pairs[:DEBUG["n_val_frames"]] if debug else test_pairs
    bs    = P_TR["batch_size"]

    all_pred, all_lbl = [], []
    print(f"  评估 {strategy}: {len(pairs)} 帧（原始测试集）...", flush=True)

    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            batch_pts6, batch_lbl = [], []
            for bin_path, lbl_path in pairs[i:i + bs]:
                pts4   = load_bin(bin_path)
                lbls   = load_label(lbl_path)
                pts6, labels = preprocess_frame(
                    pts4, lbls, P_MDL["n_points"], seed=i)
                if pts6 is None:
                    continue
                batch_pts6.append(pts6)
                batch_lbl.append(labels)

            if not batch_pts6:
                continue

            x = torch.tensor(
                np.stack(batch_pts6), dtype=torch.float32
            ).permute(0, 2, 1).to(dev)
            logits = model(x)
            pred   = logits.argmax(1).cpu().numpy()
            all_pred.append(pred.reshape(-1))
            all_lbl.append(np.array(batch_lbl).reshape(-1))

    if not all_pred:
        return {}

    all_pred = np.concatenate(all_pred)
    all_lbl  = np.concatenate(all_lbl)
    metrics  = compute_metrics(all_pred, all_lbl)

    print(f"    mIoU={metrics['mIoU']:.4f}  "
          f"F1_ground={metrics['F1_ground']:.4f}  "
          f"IoU_ground={metrics['IoU_ground']:.4f}  "
          f"Acc={metrics['accuracy']:.4f}  "
          f"PredGR={metrics['pred_ground_ratio']:.4f}", flush=True)

    # 保存结果（供断点续传使用）
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SemanticKITTI 泛化性实验 v3 —— 等量公平对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument(
        "--data-dir", default=None,
        help="sequences 文件夹路径（覆盖 config 中的 KITTI_ROOT）")
    parser.add_argument(
        "--debug", action="store_true",
        help="调试模式: 100 帧 + 5 epoch，几分钟内完成")
    parser.add_argument(
        "--strategy", default="all",
        choices=["all", "baseline", "traditional", "lsda", "loadsim"],
        help="运行指定策略（默认: all）")
    parser.add_argument(
        "--skip-aug", action="store_true",
        help="跳过增强生成步骤（使用已有 .npz 缓存）")
    args = parser.parse_args()

    # 路径覆盖
    global DATA_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir) / "00"

    debug      = args.debug
    strategies = (["baseline", "traditional", "lsda", "loadsim"]
                  if args.strategy == "all"
                  else [args.strategy])

    P_TR  = {**TRAIN}
    P_MDL = {**MODEL}
    if debug:
        P_TR.update({"epochs":     DEBUG["epochs"],
                     "batch_size": DEBUG["batch_size"],
                     "early_stop": DEBUG["early_stop"]})
        print("\n[调试模式] 已激活", flush=True)

    for d in [OUT_ROOT, AUG_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}", flush=True)
    print(f"  SemanticKITTI 泛化性实验 v3  |  Sequence 00", flush=True)
    print(f"  任务: Ground/Non-ground 二分类"
          f"  Ground={sorted(GROUND_CLASSES)}", flush=True)
    print(f"  数据路径: {DATA_DIR}", flush=True)
    print(f"  实验策略: {strategies}", flush=True)
    print(f"{'='*68}", flush=True)

    # ── 获取帧路径 ───────────────────────────────────────────────────────
    seq_dir     = DATA_DIR
    train_pairs = get_frame_paths(seq_dir, "train", debug=debug)
    val_pairs   = get_frame_paths(seq_dir, "val",   debug=debug)
    test_pairs  = get_frame_paths(seq_dir, "test",  debug=debug)

    print(f"\n  数据划分（严格按时间顺序）:", flush=True)
    print(f"    训练集: {len(train_pairs)} 帧"
          f"  [{SPLIT['train'][0]}-{SPLIT['train'][1]-1}]", flush=True)
    print(f"    验证集: {len(val_pairs)} 帧"
          f"  [{SPLIT['val'][0]}-{SPLIT['val'][1]-1}]", flush=True)
    print(f"    测试集: {len(test_pairs)} 帧"
          f"  [{SPLIT['test'][0]}-{SPLIT['test'][1]-1}]"
          f"  （原始帧，固定不变）", flush=True)

    if not train_pairs:
        print(f"\n[ERROR] 未找到训练帧，请检查路径: {seq_dir}/velodyne/",
              flush=True)
        sys.exit(1)

    # ── Step 1: 生成增强缓存（含断点续传）───────────────────────────────
    print(f"\n{'#'*68}", flush=True)
    print(f"  Step 1: 预生成增强缓存", flush=True)
    print(f"  每帧 1 个变体；各分支训练/验证样本数: "
          f"{len(train_pairs)}/{len(val_pairs)}", flush=True)
    print(f"{'#'*68}", flush=True)

    train_caches = {}
    val_caches   = {}

    for strat in strategies:
        print(f"\n  [{strat}] 训练集缓存:", flush=True)
        train_caches[strat] = build_aug_cache(
            train_pairs, strat,
            AUG, P_MDL["n_points"],
            debug=debug,
            cache_dir=AUG_DIR / "train")

        print(f"\n  [{strat}] 验证集缓存:", flush=True)
        val_caches[strat] = build_aug_cache(
            val_pairs, strat,
            AUG, P_MDL["n_points"],
            debug=debug,
            cache_dir=AUG_DIR / "val")

    # ── Step 2: 模型训练（含断点续传）───────────────────────────────────
    print(f"\n{'#'*68}", flush=True)
    print(f"  Step 2: 模型训练", flush=True)
    print(f"  说明: 各分支训练/验证集均为增强（或原始）变体，", flush=True)
    print(f"        数量完全相同，保证实验公平性", flush=True)
    print(f"{'#'*68}", flush=True)

    val_metrics_all = {}
    for strat in strategies:
        if strat not in train_caches or strat not in val_caches:
            continue
        val_metrics_all[strat] = train_one_strategy(
            strat,
            train_caches[strat],
            val_caches[strat],
            MODEL_DIR, LOG_DIR,
            P_TR, P_MDL, debug=debug)

    # ── Step 3: 测试集评估（含断点续传）─────────────────────────────────
    print(f"\n{'#'*68}", flush=True)
    print(f"  Step 3: 测试集评估（原始帧，所有分支共享）", flush=True)
    print(f"{'#'*68}", flush=True)

    test_metrics_all = {}
    for strat in strategies:
        test_metrics_all[strat] = evaluate_on_test(
            strat, test_pairs,
            MODEL_DIR, RESULT_DIR,
            P_MDL, P_TR, debug=debug)

    # ── Step 4: 汇总打印 & 保存 ──────────────────────────────────────────
    print(f"\n{'='*72}", flush=True)
    print(f"  实验结果汇总 (SemanticKITTI Seq.00, Ground/Non-ground 二分类)",
          flush=True)
    print(f"  各分支训练/验证/测试样本数: "
          f"{len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'策略':14s} | {'ValMIoU':8s} | {'TestMIoU':9s} | "
          f"{'F1_gnd':7s} | {'IoU_gnd':8s} | {'PredGR':7s}", flush=True)
    print(f"  {'-'*72}", flush=True)

    summary = {}
    for strat in strategies:
        vm = val_metrics_all.get(strat,  {})
        tm = test_metrics_all.get(strat, {})
        summary[strat] = {"val": vm, "test": tm}
        print(f"  {strat:14s} | "
              f"{vm.get('mIoU',0):.4f}   | "
              f"{tm.get('mIoU',0):.4f}    | "
              f"{tm.get('F1_ground',0):.4f}  | "
              f"{tm.get('IoU_ground',0):.4f}   | "
              f"{tm.get('pred_ground_ratio',0):.4f}",
              flush=True)

    out_json = RESULT_DIR / "kitti_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment":     "SemanticKITTI_ground_segmentation_v3",
            "sequence":       "00",
            "design":         "equal_sample_fair_comparison",
            "split":          {k: list(v) for k, v in SPLIT.items()},
            "ground_classes": sorted(GROUND_CLASSES),
            "n_train":        len(train_pairs),
            "n_val":          len(val_pairs),
            "n_test":         len(test_pairs),
            "n_points":       P_MDL["n_points"],
            "results":        summary,
            "generated_at":   str(datetime.now()),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_json}", flush=True)
    print(f"{'='*72}\n", flush=True)


if __name__ == "__main__":
    main()
