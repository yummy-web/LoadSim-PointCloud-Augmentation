"""
step4_train.py — 铲料区域分割模型训练
=======================================
PointNet++ 语义分割 (yanx27/Pointnet_Pointnet2_pytorch 风格实现)

任务: 逐点二分类 — 铲料区域(1) vs 非铲料区域(0)
输入: xyz + normals (6通道)
输出: 每点的类别概率 [P(bg), P(shovel)]

三个分支:
  baseline:    11个原始训练文件，LeaveOneOut交叉验证
  traditional: 200个高质量传统增强变体 + 11个原始 → 8:2划分
  lsda:        200个高质量LSDA增强变体   + 11个原始 → 8:2划分

评估指标: mIoU, 准确率(Acc), 召回率(Recall), F1分数
混合精度: torch.cuda.amp (AMP)
早停:     验证集mIoU连续patience轮不提升则停止

输出:
  best_model.pth         最优模型权重
  training_history.json  训练记录
  fold_result.json       本折最优指标
"""
import sys, io, json, argparse, warnings, random, time
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    try:
        # 避免重复包装已经包装过的 stdout/stderr
        if not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if not isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, Subset
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch未安装，将运行模拟模式")

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


# ══════════════════════════════════════════════════════════
# PLY 加载
# ══════════════════════════════════════════════════════════

def load_ply_xyzn(path):
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
    pts, nrms, has_nx = [], [], False
    n_vert, is_bin = 0, False
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
                    is_bin = True
                elif ln == 'end_header':
                    break
            if is_bin:
                df = [('x','f4'),('y','f4'),('z','f4')]
                if has_nx: df += [('nx','f4'),('ny','f4'),('nz','f4')]
                dt = np.dtype(df)
                d  = np.frombuffer(f.read(dt.itemsize*n_vert), dtype=dt)
                p  = np.column_stack([d['x'],d['y'],d['z']]).astype(np.float32)
                n  = np.column_stack([d['nx'],d['ny'],d['nz']]).astype(np.float32) if has_nx else None
                return p, n
            else:
                for _ in range(n_vert):
                    vs = f.readline().decode('ascii', errors='ignore').split()
                    if len(vs)>=3:
                        try:
                            pts.append([float(v) for v in vs[:3]])
                            if has_nx and len(vs)>=6:
                                nrms.append([float(v) for v in vs[3:6]])
                        except ValueError: pass
    except Exception: pass
    p = np.array(pts, dtype=np.float32) if pts else None
    n = np.array(nrms,dtype=np.float32) if (has_nx and nrms and len(nrms)==len(pts)) else None
    return p, n


# ══════════════════════════════════════════════════════════
# PointNet++ 分割网络 (yanx27风格，无第三方依赖)
# ══════════════════════════════════════════════════════════

if HAS_TORCH:

    def index_points(pts, idx):
        """
        pts: (B, N, C)
        idx: (B, S) or (B, S, K)
        returns: (B, S, C) or (B, S, K, C)
        """
        B = pts.shape[0]
        device = pts.device
        batch_idx = torch.arange(B, device=device).view(B, *([1] * (idx.dim() - 1)))
        batch_idx = batch_idx.expand_as(idx)
        out = pts[batch_idx, idx]
        return out

    def farthest_point_sample(xyz, n_pt):
        B, N, _ = xyz.shape
        device   = xyz.device
        sel  = torch.zeros(B, n_pt, dtype=torch.long, device=device)
        dist = torch.full((B, N), float('inf'), device=device)
        far  = torch.randint(0, N, (B,), device=device)
        for i in range(n_pt):
            sel[:,i] = far
            ctr = xyz[torch.arange(B,device=device), far].unsqueeze(1)
            d   = ((xyz - ctr)**2).sum(-1)
            dist = torch.minimum(dist, d)
            far  = dist.max(-1)[1]
        return sel

    def ball_query(r, k, xyz, new_xyz):
        """Ball query using torch.cdist for memory efficiency. Returns (B,S,k)."""
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape
        dev = xyz.device
        dist = torch.cdist(new_xyz, xyz)   # B,S,N
        dist_masked = dist.clone()
        dist_masked[dist > r] = float('inf')
        if N <= k:
            _, top_idx = dist_masked.topk(N, dim=-1, largest=False)
            pad = top_idx[:,:,:1].expand(B, S, k - N)
            idx = torch.cat([top_idx, pad], dim=-1)
        else:
            _, idx = dist_masked.topk(k, dim=-1, largest=False)
        invalid = dist.gather(2, idx) > r
        first   = idx[:,:,:1].expand_as(idx)
        idx = idx.clone()
        idx[invalid] = first[invalid]
        return idx

    class PointNetSALayer(nn.Module):
        """Set Abstraction Layer"""
        def __init__(self, n_pt, radius, k_sample, in_ch, mlp_dims):
            super().__init__()
            self.n_pt, self.r, self.k = n_pt, radius, k_sample
            layers, c = [], in_ch + 3
            for d in mlp_dims:
                layers += [nn.Conv2d(c, d, 1), nn.BatchNorm2d(d), nn.ReLU(inplace=True)]
                c = d
            self.mlp = nn.Sequential(*layers)
            self.out_ch = mlp_dims[-1]

        def forward(self, xyz, feats):
            # xyz: B,N,3  feats: B,C,N
            B, N, _ = xyz.shape
            S = min(self.n_pt, N)
            # FPS
            new_xyz_idx = farthest_point_sample(xyz, S)          # B,S
            new_xyz = index_points(xyz, new_xyz_idx)  # (B,S,3)
            # ball query
            nb_idx  = ball_query(self.r, min(self.k,N), xyz, new_xyz) # B,S,k
            # gather features
            grouped_xyz  = index_points(xyz, nb_idx)              # B,S,k,3
            grouped_xyz -= new_xyz.unsqueeze(2)                   # relative
            if feats is not None:
                grouped_f = index_points(feats.permute(0,2,1), nb_idx) # B,S,k,C
                grouped   = torch.cat([grouped_xyz, grouped_f], -1).permute(0,3,2,1) # B,C+3,k,S
            else:
                grouped   = grouped_xyz.permute(0,3,2,1)          # B,3,k,S
            out = self.mlp(grouped).max(2)[0]                      # B,out_ch,S
            return new_xyz, out

    class PointNetFPLayer(nn.Module):
        """Feature Propagation Layer"""
        def __init__(self, in_ch, mlp_dims):
            super().__init__()
            layers, c = [], in_ch
            for d in mlp_dims:
                layers += [nn.Conv1d(c,d,1), nn.BatchNorm1d(d), nn.ReLU(inplace=True)]
                c = d
            self.mlp = nn.Sequential(*layers)

        def forward(self, xyz1, xyz2, f1, f2):
            # Upsample f2 (coarse) to xyz1 (fine) by inverse distance weighting
            B, N, _ = xyz1.shape
            _, S, _ = xyz2.shape
            k = min(3, S)
            dist = torch.cdist(xyz1, xyz2)       # B,N,S
            dk, ki = dist.topk(k, dim=-1, largest=False)
            dk = dk.clamp(min=1e-10)
            w  = 1.0 / dk                        # B,N,k
            w /= w.sum(-1, keepdim=True)
            C2 = f2.shape[1]
            ki_exp  = ki.unsqueeze(1).expand(B, C2, -1, -1)    # B,C2,N,k
            f2_exp  = f2.unsqueeze(2).expand(B, C2, N, S)      # B,C2,N,S
            gathered = f2_exp.gather(3, ki_exp)                  # B,C2,N,k
            interp   = (gathered * w.unsqueeze(1)).sum(-1)      # B,C2,N
            if f1 is not None:
                out = torch.cat([f1, interp], 1)
            else:
                out = interp
            return self.mlp(out)

    class PointNetPPSeg(nn.Module):
        """
        PointNet++ 语义分割网络
        in_ch: 输入通道数 (xyz=3 or xyz+normal=6)
        n_cls: 类别数
        """
        def __init__(self, in_ch: int = 6, n_cls: int = 2):
            super().__init__()
            extra = in_ch - 3   # normal channels (0 or 3)
            # SA layers
            self.sa1 = PointNetSALayer(1024, 0.1, 32,  extra,  [32,32,64])
            self.sa2 = PointNetSALayer(256,  0.2, 64,  64,     [64,64,128])
            self.sa3 = PointNetSALayer(64,   0.4, 128, 128,    [128,128,256])
            self.sa4 = PointNetSALayer(16,   0.8, 256, 256,    [256,256,512])
            # FP layers
            self.fp4 = PointNetFPLayer(256+512,  [256,256])
            self.fp3 = PointNetFPLayer(128+256,  [256,128])
            self.fp2 = PointNetFPLayer(64+128,   [128,128])
            self.fp1 = PointNetFPLayer(extra+128,[128,128])
            # Head
            self.head = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(128, n_cls, 1)
            )

        def forward(self, xyzn):
            # xyzn: B, C, N  (C=6: xyz+normal)
            B, C, N = xyzn.shape
            xyz  = xyzn[:,:3,:].permute(0,2,1).contiguous()  # B,N,3
            if C > 3:
                feats0 = xyzn[:,3:,:]                          # B,3,N  (normal)
            else:
                feats0 = None

            xyz1, f1 = self.sa1(xyz, feats0)
            xyz2, f2 = self.sa2(xyz1, f1)
            xyz3, f3 = self.sa3(xyz2, f2)
            xyz4, f4 = self.sa4(xyz3, f3)

            f3b = self.fp4(xyz3, xyz4, f3, f4)
            f2b = self.fp3(xyz2, xyz3, f2, f3b)
            f1b = self.fp2(xyz1, xyz2, f1, f2b)
            f0b = self.fp1(xyz, xyz1, feats0, f1b)

            return self.head(f0b)   # B, n_cls, N


# ══════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════

if HAS_TORCH:
    class ShovelSegDataset(Dataset):
        """
        点云分割数据集
        label_file: step3_labels.py 生成的 JSON
        每个样本: (xyzn: 6×N tensor, labels: N tensor int64)
        """
        def __init__(self, ply_dirs, label_file: str,
                     n_pts: int = 4096, augment: bool = False,
                     sample_n: int = 0, seed: int = 42,
                     orig_dirs=None,
                     split: str = 'all',      # 'train' / 'val' / 'all'
                     split_ratio: float = 0.8, # train:val split
                     split_seed: int = 0):
            """
            数据集说明:
            ply_dirs:    增强变体目录列表 (sample_n从这里取样)
            orig_dirs:   原始文件目录列表 (全部保留，不受sample_n限制)
            label_file:  伪标签JSON文件
            sample_n:    从ply_dirs中最多取多少个增强样本(0=全取)
            split:       'train'=取前split_ratio部分, 'val'=取后(1-split_ratio)部分
                         'all'=不分割，全部使用
            split_ratio: 训练/验证分割比例 (默认0.8)
            split_seed:  分割随机种子(固定保证train/val不重叠)
            """
            self.n_pts   = n_pts
            self.augment = augment
            self.data    = []

            # 加载标签
            lbl_path = Path(label_file)
            if not lbl_path.exists():
                print(f"  [WARN] 标签文件不存在: {label_file}")
                return
            with open(lbl_path, encoding='utf-8') as f:
                raw = json.load(f)
            all_labels = raw.get('labels', {})

            # 收集增强变体PLY文件
            aug_files = []
            if isinstance(ply_dirs, (str, Path)):
                ply_dirs = [ply_dirs]
            seen = set()
            for d in ply_dirs:
                d = Path(d)
                if not d.exists():
                    print(f"  [WARN] 目录不存在: {d}")
                    continue
                for p in sorted(d.glob('*.ply')):
                    if p.stem not in seen and p.stem in all_labels:
                        seen.add(p.stem)
                        aug_files.append(p)

            # 对增强变体随机子采样 (sample_n)
            if sample_n > 0 and len(aug_files) > sample_n:
                rng = random.Random(seed)
                aug_files = rng.sample(aug_files, sample_n)

            # 收集原始文件 (全部保留)
            orig_files = []
            if orig_dirs is not None:
                if isinstance(orig_dirs, (str, Path)):
                    orig_dirs = [orig_dirs]
                for d in orig_dirs:
                    d = Path(d)
                    if not d.exists():
                        continue
                    for p in sorted(d.glob('*.ply')):
                        if p.stem not in seen and p.stem in all_labels:
                            seen.add(p.stem)
                            orig_files.append(p)

            all_files = aug_files + orig_files
            n_aug_found  = len(aug_files)
            n_orig_found = len(orig_files)

            # 80/20 分割: 对整体文件列表按固定种子shuffle后取前/后部分
            if split != 'all' and len(all_files) > 1:
                rng_split = random.Random(split_seed)
                shuffled = all_files[:]
                rng_split.shuffle(shuffled)
                cut = max(1, int(len(shuffled) * split_ratio))
                if split == 'train':
                    all_files = shuffled[:cut]
                elif split == 'val':
                    all_files = shuffled[cut:]

            for ply in all_files:
                pts, nrm = load_ply_xyzn(ply)
                if pts is None or len(pts) < 100:
                    continue
                lbl_entry = all_labels.get(ply.stem)
                if lbl_entry is None:
                    continue
                seg = np.array(lbl_entry['labels'], dtype=np.int64)
                if len(seg) != len(pts):
                    if len(seg) > 0:
                        idx = np.random.choice(len(seg), len(pts), replace=(len(seg)<len(pts)))
                        seg = seg[idx]
                    else:
                        continue
                self.data.append({'pts': pts, 'nrm': nrm, 'seg': seg,
                                   'name': ply.stem})

            split_tag = f" [{split} {split_ratio:.0%}/{1-split_ratio:.0%}]" if split != 'all' else ""
            print(f"    {len(self.data)} 样本 (增强:{n_aug_found} + 原始:{n_orig_found}){split_tag}")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            d   = self.data[idx]
            pts = d['pts'].copy()
            nrm = d['nrm'].copy() if d['nrm'] is not None else np.zeros_like(pts)
            seg = d['seg'].copy()

            # 随机采样或补齐
            N = len(pts)
            if N >= self.n_pts:
                choice = np.random.choice(N, self.n_pts, replace=False)
            else:
                choice = np.concatenate([np.arange(N),
                          np.random.choice(N, self.n_pts - N, replace=True)])
            pts = pts[choice]; nrm = nrm[choice]; seg = seg[choice]

            # 归一化
            ctr   = pts.mean(0)
            scale = np.abs(pts - ctr).max() + 1e-8
            pts   = (pts - ctr) / scale
            nrm   = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)

            # 数据增强
            if self.augment:
                # 随机旋转Z轴 (全角度)
                a = np.random.uniform(0, 2*np.pi)
                c, s = np.cos(a), np.sin(a)
                R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
                pts = pts @ R.T; nrm = nrm @ R.T
                # 随机X轴小角度旋转 (±15°, 模拟不同俯仰角)
                tilt = np.random.uniform(-0.26, 0.26)
                ct, st = np.cos(tilt), np.sin(tilt)
                Rx = np.array([[1,0,0],[0,ct,-st],[0,st,ct]], dtype=np.float32)
                pts = pts @ Rx.T; nrm = nrm @ Rx.T
                # 随机翻转 (X轴)
                if np.random.rand() > 0.5:
                    pts[:,0] *= -1; nrm[:,0] *= -1
                # 随机缩放 (±10%)
                scale_aug = np.random.uniform(0.9, 1.1)
                pts *= scale_aug
                # 随机抖动
                pts += np.clip(np.random.randn(*pts.shape)*0.01, -0.02, 0.02).astype(np.float32)

            xyzn = np.concatenate([pts, nrm], axis=1).T.astype(np.float32)  # 6×N
            return torch.from_numpy(xyzn), torch.from_numpy(seg)


# ══════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════

def compute_seg_metrics(pred: np.ndarray, target: np.ndarray, n_cls: int = 2):
    """
    计算分割指标
    pred, target: shape (N,) int

    Returns dict with: mIoU, acc, recall_shovel, precision_shovel, f1_shovel
    """
    # per-class IoU
    ious = []
    for c in range(n_cls):
        tp = ((pred == c) & (target == c)).sum()
        fp = ((pred == c) & (target != c)).sum()
        fn = ((pred != c) & (target == c)).sum()
        denom = tp + fp + fn
        ious.append(float(tp / denom) if denom > 0 else 0.0)
    miou = float(np.mean(ious))

    # overall accuracy
    acc = float((pred == target).sum() / len(pred))

    # shovel-region metrics (class 1)
    tp1 = int(((pred == 1) & (target == 1)).sum())
    fp1 = int(((pred == 1) & (target == 0)).sum())
    fn1 = int(((pred == 0) & (target == 1)).sum())
    prec   = tp1 / (tp1 + fp1 + 1e-8)
    recall = tp1 / (tp1 + fn1 + 1e-8)
    f1     = 2 * prec * recall / (prec + recall + 1e-8)

    return {
        'mIoU':      miou,
        'IoU_bg':    ious[0],
        'IoU_shovel': ious[1],
        'accuracy':  acc,
        'precision': float(prec),
        'recall':    float(recall),
        'f1':        float(f1),
    }


# ══════════════════════════════════════════════════════════
# 训练与验证
# ══════════════════════════════════════════════════════════

if HAS_TORCH:

    def run_epoch(model, loader, optimizer, scaler, device,
                  class_weights, is_train: bool):
        model.train() if is_train else model.eval()
        total_loss = 0.0
        all_pred, all_tgt = [], []
        # 类别权重交叉熵: 处理铲料区域(~6%)与背景(~94%)的严重不平衡
        # class_weights = [bg_weight, shovel_weight], shovel大权重迫使模型学习少数类
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)  # no label smoothing with imbalanced weights

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for xyzn, seg in loader:
                xyzn = xyzn.to(device); seg = seg.to(device)
                if is_train:
                    optimizer.zero_grad()
                    with autocast():
                        logits = model(xyzn)           # B,2,N
                        loss   = criterion(logits, seg)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    with autocast():
                        logits = model(xyzn)
                        loss   = criterion(logits, seg)

                total_loss += loss.item() * xyzn.size(0)
                pred = logits.argmax(1).cpu().numpy().flatten()
                tgt  = seg.cpu().numpy().flatten()
                all_pred.append(pred); all_tgt.append(tgt)

        all_pred = np.concatenate(all_pred)
        all_tgt  = np.concatenate(all_tgt)
        metrics  = compute_seg_metrics(all_pred, all_tgt)
        metrics['loss'] = total_loss / max(len(loader.dataset), 1)
        return metrics


def train_branch(train_dirs, label_file, model_dir,
                 n_pts=4096, epochs=50, batch_size=24, lr=1e-3,
                 weight_decay=1e-4, early_stop=15,
                 class_weights=(0.2, 0.8), device='cuda',
                 sample_n=0, fold_tag='', debug=False,
                 orig_train_dirs=None):
    """
    训练单个分支（baseline / traditional / lsda）的一次完整实验

    train_dirs:      增强变体目录（subject to sample_n subsampling）
    orig_train_dirs: 原始训练文件目录（全部保留，不受sample_n限制）
    # 验证集由内部80/20 split自动产生 (基于split_seed固定分割)

    Returns: best_metrics dict
    """
    if not HAS_TORCH:
        print("  [模拟模式] PyTorch未安装，返回随机指标")
        return _simulate_metrics(fold_tag)

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    model_dir  = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  设备: {device_obj}")

    # 数据集: 增强变体(sample_n采样) + 原始文件(全部) → 各自内部80/20分割
    # 使用相同的 split_seed 确保 train/val 不重叠
    SPLIT_SEED = abs(hash(fold_tag)) % 10000

    # 训练集 = 80% 增强变体 + 80% 原始文件
    train_ds = ShovelSegDataset(
        train_dirs, label_file, n_pts, augment=True,
        sample_n=sample_n, orig_dirs=orig_train_dirs,
        split='train', split_ratio=0.8, split_seed=SPLIT_SEED)

    # 验证集 = 20% 增强变体 + 20% 原始文件
    val_ds   = ShovelSegDataset(
        train_dirs, label_file, n_pts, augment=False,
        sample_n=sample_n, orig_dirs=orig_train_dirs,
        split='val', split_ratio=0.8, split_seed=SPLIT_SEED)

    if len(train_ds) == 0:
        print("  [WARN] 训练集为空，跳过")
        return _simulate_metrics(fold_tag)

    n_w  = 0  # Windows: num_workers=0
    trn_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)),
                            shuffle=True,  num_workers=n_w,
                            drop_last=(len(train_ds) > batch_size),
                            pin_memory=(device_obj.type=='cuda'))
    val_loader = DataLoader(val_ds,   batch_size=min(batch_size, max(len(val_ds),1)),
                            shuffle=False, num_workers=n_w,
                            pin_memory=(device_obj.type=='cuda'))

    model = PointNetPPSeg(in_ch=6, n_cls=2).to(device_obj)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler()
    cw_list   = list(class_weights)

    history     = {'train_loss':[], 'train_mIoU':[], 'train_f1':[],
                   'val_loss':[],   'val_mIoU':[],   'val_f1':[],
                   'val_acc':[], 'val_recall':[], 'val_precision':[]}
    best_miou   = -1.0
    best_metrics = {}
    patience_cnt = 0
    best_path   = model_dir / 'best_model.pth'

    print(f"  训练集: {len(train_ds)} | 验证集: {len(val_ds)} | Epochs: {epochs}")
    print(f"  {'Epoch':>6} {'TrnLoss':>9} {'TrnmIoU':>9} {'ValLoss':>9} {'ValmIoU':>9} {'ValF1':>8}")
    print(f"  {'-'*55}")

    for epoch in range(1, epochs+1):
        t_m = run_epoch(model, trn_loader, optimizer, scaler,
                        device_obj, cw_list, is_train=True)
        v_m = run_epoch(model, val_loader,  optimizer, scaler,
                        device_obj, cw_list, is_train=False) if len(val_ds) > 0 else t_m
        scheduler.step()

        history['train_loss'].append(round(t_m['loss'],4))
        history['train_mIoU'].append(round(t_m['mIoU'],4))
        history['train_f1'].append(round(t_m['f1'],4))
        history['val_loss'].append(round(v_m['loss'],4))
        history['val_mIoU'].append(round(v_m['mIoU'],4))
        history['val_f1'].append(round(v_m['f1'],4))
        history['val_acc'].append(round(v_m['accuracy'],4))
        history['val_recall'].append(round(v_m['recall'],4))
        history['val_precision'].append(round(v_m['precision'],4))

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  {epoch:>6} {t_m['loss']:>9.4f} {t_m['mIoU']:>9.4f}"
                  f" {v_m['loss']:>9.4f} {v_m['mIoU']:>9.4f} {v_m['f1']:>8.4f}")

        if v_m['mIoU'] > best_miou:
            best_miou    = v_m['mIoU']
            best_metrics = {k: round(float(v),4) for k,v in v_m.items()}
            best_metrics['epoch'] = epoch
            patience_cnt = 0
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'metrics': best_metrics,
                        'fold_tag': fold_tag}, best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= early_stop:
                print(f"  早停触发 (epoch {epoch}, patience={early_stop})")
                break

    # 保存训练历史
    hist_path = model_dir / 'training_history.json'
    with open(hist_path, 'w', encoding='utf-8') as f:
        json.dump({'fold_tag': fold_tag, 'history': history,
                   'best_metrics': best_metrics,
                   'total_epochs': epoch,
                   'saved_at': str(datetime.now())}, f, indent=2)

    # 保存本折结果
    res_path = model_dir / 'fold_result.json'
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump({'fold_tag': fold_tag, 'best_metrics': best_metrics,
                   'saved_at': str(datetime.now())}, f, indent=2)

    print(f"\n  最优 mIoU={best_miou:.4f} @ epoch {best_metrics.get('epoch')}")
    print(f"  模型: {best_path}")
    return best_metrics


def _simulate_metrics(fold_tag=''):
    """PyTorch不可用时的随机模拟指标"""
    np.random.seed(abs(hash(fold_tag)) % 2**31)
    return {
        'mIoU':      float(np.random.uniform(0.45, 0.75)),
        'accuracy':  float(np.random.uniform(0.70, 0.95)),
        'recall':    float(np.random.uniform(0.50, 0.85)),
        'precision': float(np.random.uniform(0.50, 0.85)),
        'f1':        float(np.random.uniform(0.50, 0.80)),
        'loss':      float(np.random.uniform(0.1, 0.5)),
        'simulated': True,
    }


# ══════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════

def main():
    _base = Path(__file__).parent
    parser = argparse.ArgumentParser(description='铲料区域分割训练 v3')
    # ── 数据目录 ──────────────────────────────────────────────
    parser.add_argument('--train-dirs',      nargs='*', default=None,
                        help='增强变体目录(可多个，sample-n从中采样; 不传则无增强变体)')
    parser.add_argument('--orig-train-dirs', nargs='*', default=None,
                        help='原始文件目录(全部保留，不受sample-n限制)')
    # ── 标签与模型 ────────────────────────────────────────────
    parser.add_argument('--label-file',   required=True)
    parser.add_argument('--model-dir',    required=True)
    # ── 超参数 ────────────────────────────────────────────────
    parser.add_argument('--sample-n',     type=int,   default=0,
                        help='从train-dirs中最多取多少增强样本(0=全取)')
    parser.add_argument('--n-pts',        type=int,   default=4096)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch-size',   type=int,   default=24)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--early-stop',   type=int,   default=15)
    parser.add_argument('--class-weight', nargs=2, type=float, default=[0.1, 0.9],
                        help='[背景权重, 铲料区域权重] (铲料区域稀疏，应大权重)')
    parser.add_argument('--device',       default='cuda')
    parser.add_argument('--fold-tag',     default='')
    args = parser.parse_args()

    def res(p):
        """相对路径 → 脚本所在目录的绝对路径"""
        p = Path(p)
        return str(p if p.is_absolute() else _base / p)

    # ── 解析路径 ─────────────────────────────────────────────
    train_dirs      = [res(d) for d in args.train_dirs]      if args.train_dirs      else []
    orig_train_dirs = [res(d) for d in args.orig_train_dirs] if args.orig_train_dirs else None

    best = train_branch(
        train_dirs      = train_dirs,
        orig_train_dirs = orig_train_dirs,
        label_file      = res(args.label_file),
        model_dir       = res(args.model_dir),
        n_pts           = args.n_pts,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        early_stop      = args.early_stop,
        class_weights   = args.class_weight,
        device          = args.device,
        sample_n        = args.sample_n,
        fold_tag        = args.fold_tag,
    )
    print(f"\n  最终指标: {best}")

if __name__ == '__main__':
    main()
