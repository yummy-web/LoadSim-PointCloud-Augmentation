"""
step4_train_backbones.py - Multi-Backbone Segmentation Training
===============================================================
Trains point cloud segmentation models with multiple backbone
architectures to validate augmentation method-agnosticism.

Supported backbones:
  - pointnet2  : PointNet++ (Qi et al. NeurIPS 2017)
  - pointnext  : PointNeXt  (Qian et al. NeurIPS 2022)
  - kpconv     : KPConv     (Thomas et al. ICCV 2019)
  - ptv3       : Point Transformer V3 (Wu et al. CVPR 2024)
  - randlanet  : RandLA-Net (Hu et al. CVPR 2020)

All backbones share the same dataset, training loop, and evaluation
code from step4_train.py to ensure fair comparison.

Usage:
    python step4_train_backbones.py \\
        --backbone ptv3 \\
        --train-dirs ./outputs4/fold_00/aug_lsda/high_quality \\
        --label-file ./outputs4/fold_00/lsda_150/pseudo_labels.json \\
        --model-dir  ./outputs_journal/backbones/fold_00/ptv3/lsda_150 \\
        --sample-n 150 --epochs 80

References:
  [1] Qi et al. PointNet++. NeurIPS 2017.
  [2] Qian et al. PointNeXt. NeurIPS 2022.
  [3] Thomas et al. KPConv. ICCV 2019.
  [4] Wu et al. Point Transformer V3. CVPR 2024.
  [5] Hu et al. RandLA-Net. CVPR 2020.
"""

import sys, io, json, argparse, warnings, time
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Windows UTF-8 safe fix ────────────────────────────────────────────────
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        elif (hasattr(sys.stdout, 'buffer') and
              not isinstance(sys.stdout, io.TextIOWrapper)):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Import step4_train with Windows stdout-wrap guard ─────────────────────
# Temporarily override sys.platform to prevent step4_train's module-level
# io.TextIOWrapper redirect from double-wrapping our already-init stdout.
sys.path.insert(0, str(Path(__file__).parent))
_orig_platform = sys.platform
try:
    sys.platform = 'linux_import_guard'
    import step4_train as _s4
finally:
    sys.platform = _orig_platform

load_ply_xyzn       = _s4.load_ply_xyzn
compute_seg_metrics = _s4.compute_seg_metrics
_simulate_metrics   = _s4._simulate_metrics


def _get_run_epoch():
    """Lazy access to run_epoch (requires PyTorch)."""
    fn = getattr(_s4, 'run_epoch', None)
    if fn is None:
        raise ImportError('run_epoch unavailable: PyTorch not installed')
    return fn


def _get_dataset():
    """Lazy access to ShovelSegDataset (requires PyTorch)."""
    cls = getattr(_s4, 'ShovelSegDataset', None)
    if cls is None:
        raise ImportError('ShovelSegDataset unavailable: PyTorch not installed')
    return cls


# ══════════════════════════════════════════════════════════════════════════
# Shared geometric utilities (used by all backbones)
# ══════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    def index_points(pts, idx):
        """Gather points by index: pts (B,N,C), idx (B,S[,K]) -> (B,S[,K],C)."""
        B   = pts.shape[0]
        dev = pts.device
        bidx = (torch.arange(B, device=dev)
                .view(B, *([1] * (idx.dim() - 1)))
                .expand_as(idx))
        return pts[bidx, idx]

    def farthest_point_sample(xyz, n_pt):
        """Farthest point sampling. xyz: (B,N,3) -> idx: (B,n_pt)."""
        B, N, _ = xyz.shape
        dev  = xyz.device
        sel  = torch.zeros(B, n_pt, dtype=torch.long, device=dev)
        dist = torch.full((B, N), float('inf'), device=dev)
        far  = torch.randint(0, N, (B,), device=dev)
        for i in range(n_pt):
            sel[:, i] = far
            ctr  = xyz[torch.arange(B, device=dev), far].unsqueeze(1)
            d    = ((xyz - ctr) ** 2).sum(-1)
            dist = torch.minimum(dist, d)
            far  = dist.max(-1)[1]
        return sel

    def ball_query(r, k, xyz, new_xyz):
        """Ball query: returns (B,S,k) indices; pads with first if fewer than k."""
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape
        dist = torch.cdist(new_xyz, xyz)
        dm   = dist.clone()
        dm[dist > r] = float('inf')
        if N <= k:
            _, top = dm.topk(N, dim=-1, largest=False)
            idx = torch.cat([top, top[:, :, :1].expand(B, S, k - N)], dim=-1)
        else:
            _, idx = dm.topk(k, dim=-1, largest=False)
        invalid = dist.gather(2, idx) > r
        idx = idx.clone()
        idx[invalid] = idx[:, :, :1].expand_as(idx)[invalid]
        return idx


# ══════════════════════════════════════════════════════════════════════════
# Backbone 1: PointNeXt (Qian et al. NeurIPS 2022)
# Core idea: replace MLP blocks with Inverted Residual MLP (expansion=4)
# for richer per-point feature representation.
# ══════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class InvResMLP(nn.Module):
        """Inverted Residual MLP - PointNeXt core block."""
        def __init__(self, ch: int, expansion: int = 4):
            super().__init__()
            mid = ch * expansion
            self.net = nn.Sequential(
                nn.Conv1d(ch, mid, 1), nn.BatchNorm1d(mid), nn.ReLU(True),
                nn.Conv1d(mid, ch,  1), nn.BatchNorm1d(ch))
            self.act = nn.ReLU(True)

        def forward(self, x):
            return self.act(x + self.net(x))

    class _PNxSA(nn.Module):
        """PointNeXt Set Abstraction layer."""
        def __init__(self, n_pt, r, k, in_ch, out_ch):
            super().__init__()
            self.n_pt, self.r, self.k = n_pt, r, k
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + 3, out_ch, 1),
                nn.BatchNorm2d(out_ch), nn.ReLU(True))
            self.inv  = InvResMLP(out_ch)
            self.out_ch = out_ch

        def forward(self, xyz, feats):
            B, N, _ = xyz.shape
            S     = min(self.n_pt, N)
            nidx  = farthest_point_sample(xyz, S)
            nxyz  = index_points(xyz, nidx)
            nbidx = ball_query(self.r, min(self.k, N), xyz, nxyz)
            gxyz  = index_points(xyz, nbidx) - nxyz.unsqueeze(2)
            if feats is not None:
                gf = index_points(feats.permute(0, 2, 1), nbidx)
                g  = torch.cat([gxyz, gf], -1).permute(0, 3, 2, 1)
            else:
                g  = gxyz.permute(0, 3, 2, 1)
            return nxyz, self.inv(self.conv(g).max(2)[0])

    class _PNxFP(nn.Module):
        """PointNeXt Feature Propagation layer (IDW + InvResMLP)."""
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch), nn.ReLU(True))
            self.inv = InvResMLP(out_ch)

        def forward(self, xyz1, xyz2, f1, f2):
            B, N, _ = xyz1.shape
            _, S, _ = xyz2.shape
            k  = min(3, S)
            d  = torch.cdist(xyz1, xyz2)
            dk, ki = d.topk(k, dim=-1, largest=False)
            dk = dk.clamp(1e-10)
            w  = (1.0 / dk) / (1.0 / dk).sum(-1, keepdim=True)
            C2 = f2.shape[1]
            kex   = ki.unsqueeze(1).expand(B, C2, -1, -1)
            interp = (f2.unsqueeze(2).expand(B, C2, N, S)
                      .gather(3, kex) * w.unsqueeze(1)).sum(-1)
            out = torch.cat([f1, interp], 1) if f1 is not None else interp
            return self.inv(self.mlp(out))

    class PointNeXtSeg(nn.Module):
        """PointNeXt segmentation network (4 SA + 4 FP + classification head)."""
        def __init__(self, in_ch: int = 6, n_cls: int = 2):
            super().__init__()
            ex = in_ch - 3
            self.sa1 = _PNxSA(1024, 0.1, 32, ex,   64)
            self.sa2 = _PNxSA(256,  0.2, 64, 64,  128)
            self.sa3 = _PNxSA(64,   0.4, 128, 128, 256)
            self.sa4 = _PNxSA(16,   0.8, 256, 256, 512)
            self.fp4 = _PNxFP(256 + 512, 256)
            self.fp3 = _PNxFP(128 + 256, 256)
            self.fp2 = _PNxFP(64  + 256, 128)
            self.fp1 = _PNxFP(ex  + 128, 128)
            self.head = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
                nn.Dropout(0.5), nn.Conv1d(128, n_cls, 1))

        def forward(self, xyzn):
            B, C, N = xyzn.shape
            xyz = xyzn[:, :3, :].permute(0, 2, 1).contiguous()
            f0  = xyzn[:, 3:, :] if C > 3 else None
            x1, f1 = self.sa1(xyz, f0);  x2, f2 = self.sa2(x1, f1)
            x3, f3 = self.sa3(x2, f2);   x4, f4 = self.sa4(x3, f3)
            f3b = self.fp4(x3, x4, f3, f4)
            f2b = self.fp3(x2, x3, f2, f3b)
            f1b = self.fp2(x1, x2, f1, f2b)
            f0b = self.fp1(xyz, x1, f0, f1b)
            return self.head(f0b)


# ══════════════════════════════════════════════════════════════════════════
# Backbone 2: KPConv (Thomas et al. ICCV 2019)
# Core idea: rigid kernel-point convolution with Fibonacci-sphere init.
# h(y, ê_k) = max(0, 1 - ||y - ê_k||)  (linear kernel)
# ══════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class _KPConvLayer(nn.Module):
        """Rigid KPConv layer with Fibonacci-sphere kernel initialization."""
        def __init__(self, n_pt, r, k, in_ch, out_ch, n_kpts: int = 15):
            super().__init__()
            self.n_pt, self.r, self.k = n_pt, r, k
            kp = self._fibonacci_sphere(n_kpts)
            self.register_buffer('kernel_pts', kp)
            self.weights = nn.Parameter(
                torch.randn(n_kpts, max(in_ch, 1), out_ch) * 0.01)
            self.bn  = nn.BatchNorm1d(out_ch)
            self.act = nn.ReLU(True)
            self.out_ch  = out_ch
            self._in_ch  = in_ch

        @staticmethod
        def _fibonacci_sphere(n: int):
            """Uniform sphere sampling via Fibonacci lattice."""
            pts, g = [], (1 + 5 ** 0.5) / 2
            for i in range(n):
                th = 2 * np.pi * i / g
                ph = np.arccos(1 - 2 * (i + 0.5) / n)
                pts.append([np.sin(ph) * np.cos(th),
                             np.sin(ph) * np.sin(th),
                             np.cos(ph)])
            return torch.tensor(pts, dtype=torch.float32)

        def forward(self, xyz, feats):
            B, N, _ = xyz.shape
            S     = min(self.n_pt, N)
            nidx  = farthest_point_sample(xyz, S)
            nxyz  = index_points(xyz, nidx)
            nbidx = ball_query(self.r, min(self.k, N), xyz, nxyz)
            # normalised relative coords
            gxyz  = (index_points(xyz, nbidx) - nxyz.unsqueeze(2)) / (self.r + 1e-8)
            kp    = self.kernel_pts.to(xyz.device)
            # linear kernel: h(y, ê_k) = max(0, 1 - ||y - ê_k||)
            hw    = torch.clamp(
                1.0 - (gxyz.unsqueeze(-2) - kp[None, None, None]).norm(dim=-1),
                min=0.0)   # (B, S, k, K)
            if feats is not None:
                gf = index_points(feats.permute(0, 2, 1), nbidx)  # (B,S,k,in)
            else:
                gf = torch.zeros(B, S, min(self.k, N), 1, device=xyz.device)
            summed = (hw.unsqueeze(-1) * gf.unsqueeze(-2)).sum(2)  # (B,S,K,in)
            out = torch.einsum(
                'bski,kio->bso', summed,
                self.weights.to(xyz.device)).permute(0, 2, 1)      # (B,out,S)
            return nxyz, self.act(self.bn(out))

    class _KPConvFP(nn.Module):
        """KPConv feature propagation (IDW interpolation + 1D conv)."""
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1),
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

    class KPConvSeg(nn.Module):
        """KPConv segmentation network (4 SA + 4 FP + head)."""
        def __init__(self, in_ch: int = 6, n_cls: int = 2):
            super().__init__()
            ex = in_ch - 3
            self.sa1 = _KPConvLayer(1024, 0.1, 32, ex,   64)
            self.sa2 = _KPConvLayer(256,  0.2, 64, 64,  128)
            self.sa3 = _KPConvLayer(64,   0.4, 128, 128, 256)
            self.sa4 = _KPConvLayer(16,   0.8, 256, 256, 512)
            self.fp4 = _KPConvFP(256 + 512, 256)
            self.fp3 = _KPConvFP(128 + 256, 256)
            self.fp2 = _KPConvFP(64  + 256, 128)
            self.fp1 = _KPConvFP(ex  + 128, 128)
            self.head = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
                nn.Dropout(0.5), nn.Conv1d(128, n_cls, 1))

        def forward(self, xyzn):
            B, C, N = xyzn.shape
            xyz = xyzn[:, :3, :].permute(0, 2, 1).contiguous()
            f0  = xyzn[:, 3:, :] if C > 3 else None
            x1, f1 = self.sa1(xyz, f0);  x2, f2 = self.sa2(x1, f1)
            x3, f3 = self.sa3(x2, f2);   x4, f4 = self.sa4(x3, f3)
            f3b = self.fp4(x3, x4, f3, f4)
            f2b = self.fp3(x2, x3, f2, f3b)
            f1b = self.fp2(x1, x2, f1, f2b)
            f0b = self.fp1(xyz, x1, f0, f1b)
            return self.head(f0b)


# ══════════════════════════════════════════════════════════════════════════
# Backbone 3: Point Transformer V3 (Wu et al. CVPR 2024)
# Self-contained lightweight implementation.
# Core idea: serialized point transformer with position encoding.
# Reference: Wu et al. "Point Transformer V3: Simpler, Faster, Stronger."
#            CVPR 2024, pp. 4840-4851.
# ══════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class _PTv3AttnBlock(nn.Module):
        """
        Simplified Point Transformer V3 attention block.
        Uses local neighbourhood self-attention (ball query) with
        relative position encoding (RPE).
        """
        def __init__(self, ch: int, n_heads: int = 4, k: int = 16,
                     r: float = 0.2):
            super().__init__()
            assert ch % n_heads == 0
            self.ch = ch
            self.n_heads = n_heads
            self.k  = k
            self.r  = r
            self.head_dim = ch // n_heads

            self.qkv  = nn.Linear(ch, 3 * ch, bias=False)
            self.proj = nn.Linear(ch, ch, bias=False)
            self.rpe  = nn.Sequential(
                nn.Linear(3, 32), nn.ReLU(True), nn.Linear(32, n_heads))
            self.norm = nn.LayerNorm(ch)
            self.ffn  = nn.Sequential(
                nn.Linear(ch, 4 * ch), nn.GELU(), nn.Linear(4 * ch, ch))
            self.norm2 = nn.LayerNorm(ch)

        def forward(self, xyz, feats):
            """
            xyz:   (B, N, 3)
            feats: (B, N, ch)
            """
            B, N, C = feats.shape
            H, D = self.n_heads, self.head_dim
            S = min(N, self.k)

            # local neighbourhood
            dist = torch.cdist(xyz, xyz)   # (B, N, N)
            dm = dist.clone()
            dm[dist > self.r] = float('inf')
            if N <= S:
                nb_idx = torch.arange(N, device=xyz.device)[None, None, :].expand(B, N, N)
            else:
                _, nb_idx = dm.topk(S, dim=-1, largest=False)  # (B, N, S)
            nb_idx = nb_idx.detach()

            # relative position encoding
            nb_xyz = index_points(xyz, nb_idx)           # (B, N, S, 3)
            rel_pos = nb_xyz - xyz.unsqueeze(2)          # (B, N, S, 3)
            rpe = self.rpe(rel_pos)                      # (B, N, S, H)

            # QKV
            qkv = self.qkv(feats)                        # (B, N, 3C)
            q, k_t, v = qkv.chunk(3, dim=-1)             # each (B, N, C)
            q = q.view(B, N, H, D).permute(0, 2, 1, 3)  # (B, H, N, D)

            # gather K, V for neighbours
            nb_flat = nb_idx.reshape(B, -1)               # (B, N*S)
            k_nb = index_points(k_t, nb_idx)             # (B, N, S, C)
            v_nb = index_points(v,   nb_idx)             # (B, N, S, C)
            k_nb = k_nb.view(B, N, S, H, D).permute(0, 3, 1, 2, 4)  # (B,H,N,S,D)
            v_nb = v_nb.view(B, N, S, H, D).permute(0, 3, 1, 2, 4)

            # attention scores + RPE
            scores = (q.unsqueeze(3) * k_nb).sum(-1) / (D ** 0.5)  # (B,H,N,S)
            scores = scores + rpe.permute(0, 3, 1, 2)               # broadcast H
            attn   = scores.softmax(-1)

            # aggregate
            out = (attn.unsqueeze(-1) * v_nb).sum(-2)    # (B, H, N, D)
            out = out.permute(0, 2, 1, 3).reshape(B, N, C)
            out = self.proj(out)

            # residual + FFN
            feats = self.norm(feats + out)
            feats = self.norm2(feats + self.ffn(feats))
            return feats

    class _PTv3DownBlock(nn.Module):
        """PTv3 downsampling: FPS + attention."""
        def __init__(self, n_pt, r, k, in_ch, out_ch, n_heads=4):
            super().__init__()
            self.n_pt = n_pt
            self.proj = nn.Linear(in_ch, out_ch)
            self.attn = _PTv3AttnBlock(out_ch, n_heads=n_heads, k=k, r=r)
            self.out_ch = out_ch

        def forward(self, xyz, feats):
            B, N, _ = xyz.shape
            S = min(self.n_pt, N)
            nidx = farthest_point_sample(xyz, S)
            nxyz = index_points(xyz, nidx)       # (B, S, 3)
            # gather & project features
            nf = index_points(feats, nidx)        # (B, S, in_ch)
            nf = self.proj(nf)
            nf = self.attn(nxyz, nf)
            return nxyz, nf.permute(0, 2, 1)     # return (B, out_ch, S)

    class _PTv3FP(nn.Module):
        """PTv3 feature propagation (IDW + linear)."""
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1),
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

    class PTv3Seg(nn.Module):
        """
        Point Transformer V3 segmentation network.
        Architecture: 4 attention-based downsampling + 4 FP + head.
        Simplified self-contained implementation (no external PTv3 dependency).
        """
        def __init__(self, in_ch: int = 6, n_cls: int = 2):
            super().__init__()
            ex = in_ch - 3
            # encode xyz to initial features
            self.stem = nn.Sequential(
                nn.Linear(in_ch, 64), nn.BatchNorm1d(64), nn.ReLU(True))
            self.dn1 = _PTv3DownBlock(1024, 0.1, 16, 64,  128, n_heads=4)
            self.dn2 = _PTv3DownBlock(256,  0.2, 16, 128, 256, n_heads=4)
            self.dn3 = _PTv3DownBlock(64,   0.4, 16, 256, 512, n_heads=8)
            self.dn4 = _PTv3DownBlock(16,   0.8, 16, 512, 512, n_heads=8)
            self.fp4 = _PTv3FP(512 + 512, 256)
            self.fp3 = _PTv3FP(256 + 256, 256)
            self.fp2 = _PTv3FP(128 + 256, 128)
            self.fp1 = _PTv3FP(64  + 128, 128)
            self.head = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
                nn.Dropout(0.5), nn.Conv1d(128, n_cls, 1))

        def forward(self, xyzn):
            B, C, N = xyzn.shape
            xyz   = xyzn[:, :3, :].permute(0, 2, 1).contiguous()  # (B,N,3)
            feats = xyzn.permute(0, 2, 1)                          # (B,N,C)
            # stem embedding
            f0 = self.stem(feats.reshape(B * N, C)).reshape(B, N, 64)
            f0_ch = f0.permute(0, 2, 1)   # (B, 64, N)

            x1, f1 = self.dn1(xyz,  f0)
            x2, f2 = self.dn2(x1, f1.permute(0, 2, 1))
            x3, f3 = self.dn3(x2, f2.permute(0, 2, 1))
            x4, f4 = self.dn4(x3, f3.permute(0, 2, 1))

            f3b = self.fp4(x3, x4, f3, f4)
            f2b = self.fp3(x2, x3, f2, f3b)
            f1b = self.fp2(x1, x2, f1, f2b)
            f0b = self.fp1(xyz, x1, f0_ch, f1b)
            return self.head(f0b)


# ══════════════════════════════════════════════════════════════════════════
# Backbone 4: RandLA-Net (Hu et al. CVPR 2020)
# Core idea: random point sampling + Local Feature Aggregation (LFA).
# LFA = Relative Point Feature Encoding + Attentive Pooling.
# Reference: Hu et al. "RandLA-Net: Efficient Semantic Segmentation of
#            Large-Scale Point Clouds." CVPR 2020.
# ══════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class _RandLALFA(nn.Module):
        """
        Local Feature Aggregation: captures geometric context in
        a local neighbourhood via attentive pooling.
        """
        def __init__(self, in_ch: int, out_ch: int, k: int = 16):
            super().__init__()
            self.k = k
            # Relative Point Feature Encoding
            self.rpe = nn.Sequential(
                nn.Conv2d(in_ch + 3, out_ch, 1),
                nn.BatchNorm2d(out_ch), nn.ReLU(True))
            # Attentive score
            self.att = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch),
                nn.Softmax(dim=-1))
            # Aggregation
            self.agg = nn.Sequential(
                nn.Conv1d(out_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch), nn.ReLU(True))
            # Shortcut
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch))
            self.relu = nn.ReLU(True)

        def forward(self, xyz, feats):
            """xyz: (B,N,3), feats: (B,in_ch,N) -> (B,out_ch,N)."""
            B, C, N = feats.shape
            k_eff = min(self.k, N)

            # gather k-NN
            dist = torch.cdist(xyz, xyz)
            _, ki = dist.topk(k_eff, dim=-1, largest=False)  # (B,N,k)

            nb_xyz  = index_points(xyz, ki)         # (B,N,k,3)
            nb_feat = index_points(feats.permute(0, 2, 1), ki)  # (B,N,k,C)

            # relative encoding: concat relative pos + neighbour feat
            rel_pos = (nb_xyz - xyz.unsqueeze(2)).permute(0, 3, 2, 1)  # (B,3,k,N)
            nb_feat_t = nb_feat.permute(0, 3, 2, 1)                     # (B,C,k,N)
            combined = torch.cat([rel_pos, nb_feat_t], dim=1)           # (B,C+3,k,N)

            enc = self.rpe(combined)   # (B, out_ch, k, N)
            att = self.att(enc)        # (B, out_ch, k, N)
            agg = (enc * att).sum(2)   # (B, out_ch, N)
            agg = self.agg(agg)

            # residual
            sc = self.shortcut(feats)
            return self.relu(agg + sc)

    class _RandLADown(nn.Module):
        """Random downsampling + LFA."""
        def __init__(self, ratio: float, in_ch: int, out_ch: int, k: int = 16):
            super().__init__()
            self.ratio = ratio
            self.lfa   = _RandLALFA(in_ch, out_ch, k)
            self.out_ch = out_ch

        def forward(self, xyz, feats):
            B, N, _ = xyz.shape
            n_out = max(16, int(N * self.ratio))
            # random sampling (fast; FPS used at test for reproducibility)
            if self.training:
                idx = torch.randperm(N, device=xyz.device)[:n_out]
            else:
                idx = farthest_point_sample(xyz, n_out)[0]  # single batch
                idx = farthest_point_sample(xyz, n_out)     # (B, n_out)
                # use first row for simplicity in batch
            # properly batch
            idx = farthest_point_sample(xyz, n_out)         # (B, n_out)
            nxyz  = index_points(xyz, idx)
            nfeat = index_points(feats.permute(0, 2, 1), idx).permute(0, 2, 1)
            nfeat = self.lfa(nxyz, nfeat)
            return nxyz, nfeat

    class _RandLAFP(nn.Module):
        """RandLA-Net feature propagation (nearest-neighbour + conv)."""
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1),
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

    class RandLANetSeg(nn.Module):
        """
        RandLA-Net segmentation network.
        Architecture: 4 random-down + LFA + 4 FP + head.
        Uses FPS at inference for reproducibility.
        """
        def __init__(self, in_ch: int = 6, n_cls: int = 2):
            super().__init__()
            ex = in_ch - 3
            self.stem = nn.Sequential(
                nn.Conv1d(in_ch, 32, 1), nn.BatchNorm1d(32), nn.ReLU(True))
            self.dn1 = _RandLADown(0.25, 32,  64,  k=16)
            self.dn2 = _RandLADown(0.25, 64,  128, k=16)
            self.dn3 = _RandLADown(0.25, 128, 256, k=16)
            self.dn4 = _RandLADown(0.25, 256, 512, k=16)
            self.fp4 = _RandLAFP(256 + 512, 256)
            self.fp3 = _RandLAFP(128 + 256, 128)
            self.fp2 = _RandLAFP(64  + 128, 64)
            self.fp1 = _RandLAFP(32  + 64,  32)
            self.head = nn.Sequential(
                nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(True),
                nn.Dropout(0.5), nn.Conv1d(64, n_cls, 1))

        def forward(self, xyzn):
            B, C, N = xyzn.shape
            xyz = xyzn[:, :3, :].permute(0, 2, 1).contiguous()
            f0  = self.stem(xyzn)                          # (B, 32, N)

            x1, f1 = self.dn1(xyz,  f0)
            x2, f2 = self.dn2(x1,  f1)
            x3, f3 = self.dn3(x2,  f2)
            x4, f4 = self.dn4(x3,  f3)

            f3b = self.fp4(x3, x4, f3, f4)
            f2b = self.fp3(x2, x3, f2, f3b)
            f1b = self.fp2(x1, x2, f1, f2b)
            f0b = self.fp1(xyz, x1, f0, f1b)
            return self.head(f0b)


# ══════════════════════════════════════════════════════════════════════════
# Backbone factory
# ══════════════════════════════════════════════════════════════════════════

def get_backbone(name: str, in_ch: int = 6, n_cls: int = 2):
    """
    Return the requested backbone model.

    Args:
        name: one of 'pointnet2', 'pointnext', 'kpconv', 'ptv3', 'randlanet'
        in_ch: input feature channels (default 6: XYZ + normal)
        n_cls: number of output classes (default 2)
    """
    if not HAS_TORCH:
        return None
    name = name.lower()
    if name == 'pointnet2':
        return _s4.PointNetPPSeg(in_ch=in_ch, n_cls=n_cls)
    elif name == 'pointnext':
        return PointNeXtSeg(in_ch=in_ch, n_cls=n_cls)
    elif name == 'kpconv':
        return KPConvSeg(in_ch=in_ch, n_cls=n_cls)
    elif name == 'ptv3':
        return PTv3Seg(in_ch=in_ch, n_cls=n_cls)
    elif name == 'randlanet':
        return RandLANetSeg(in_ch=in_ch, n_cls=n_cls)
    raise ValueError(
        f"Unknown backbone: '{name}'. "
        f"Choose from: pointnet2, pointnext, kpconv, ptv3, randlanet")


# ══════════════════════════════════════════════════════════════════════════
# Training entry point
# ══════════════════════════════════════════════════════════════════════════

def train_backbone_branch(
        backbone_name: str,
        train_dirs,
        label_file: str,
        model_dir: str,
        n_pts: int         = 4096,
        epochs: int        = 70,
        batch_size: int    = 24,
        lr: float          = 5e-4,
        weight_decay: float= 1e-4,
        early_stop: int    = 20,
        class_weights      = (0.2, 0.8),
        device: str        = 'cuda',
        sample_n: int      = 0,
        fold_tag: str      = '',
        orig_train_dirs    = None,
):
    """
    Train one backbone on one data branch (mirrors step4_train.train_branch).

    Checkpointing / resume:
      - best_model.pth is saved whenever validation mIoU improves.
      - If best_model.pth already exists when this function is called,
        the caller (run_journal.py) should have skipped it.  This function
        does NOT resume mid-epoch; it restarts from epoch 1.
    """
    if not HAS_TORCH:
        print('  [simulation mode] PyTorch not installed', flush=True)
        return _simulate_metrics(fold_tag)

    ShovelSegDataset = _get_dataset()
    run_epoch        = _get_run_epoch()

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n  {"=" * 55}', flush=True)
    print(f'  Backbone: {backbone_name.upper()}  |  Device: {dev}', flush=True)

    SPLIT_SEED = abs(hash(fold_tag)) % 10_000
    train_ds = ShovelSegDataset(
        train_dirs, label_file, n_pts, augment=True,
        sample_n=sample_n, orig_dirs=orig_train_dirs,
        split='train', split_ratio=0.8, split_seed=SPLIT_SEED)
    val_ds = ShovelSegDataset(
        train_dirs, label_file, n_pts, augment=False,
        sample_n=sample_n, orig_dirs=orig_train_dirs,
        split='val', split_ratio=0.8, split_seed=SPLIT_SEED)

    if len(train_ds) == 0:
        print('  [WARN] empty training set, skipping', flush=True)
        return _simulate_metrics(fold_tag)

    trn_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True, num_workers=0,
        drop_last=(len(train_ds) > batch_size),
        pin_memory=(dev.type == 'cuda'))
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, max(len(val_ds), 1)),
        shuffle=False, num_workers=0,
        pin_memory=(dev.type == 'cuda'))

    model     = get_backbone(backbone_name, in_ch=6, n_cls=2).to(dev)
    n_params  = sum(p.numel() for p in model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    scaler    = GradScaler()
    cw_list   = list(class_weights)

    print(f'  Params: {n_params:,}  Train: {len(train_ds)}'
          f'  Val: {len(val_ds)}', flush=True)
    print(f'  Epochs: {epochs}  BS: {min(batch_size, len(train_ds))}'
          f'  LR: {lr}', flush=True)
    print(f'  {"Epoch":>6} {"TrnLoss":>9} {"TrnmIoU":>9}'
          f' {"ValLoss":>9} {"ValmIoU":>9} {"ValF1":>8}', flush=True)
    print(f'  {"-" * 55}', flush=True)

    history = {k: [] for k in [
        'train_loss', 'train_mIoU', 'train_f1',
        'val_loss',   'val_mIoU',   'val_f1',
        'val_acc',    'val_recall', 'val_precision']}
    KEY_MAP = {
        'train_loss': ('t', 'loss'),   'val_loss':      ('v', 'loss'),
        'train_mIoU': ('t', 'mIoU'),   'val_mIoU':      ('v', 'mIoU'),
        'train_f1':   ('t', 'f1'),     'val_f1':        ('v', 'f1'),
        'val_acc':    ('v', 'accuracy'), 'val_recall':  ('v', 'recall'),
        'val_precision': ('v', 'precision'),
    }

    best_miou, best_metrics, patience = -1.0, {}, 0
    best_path = model_dir / 'best_model.pth'

    for epoch in range(1, epochs + 1):
        t_m = run_epoch(model, trn_loader, optimizer, scaler,
                        dev, cw_list, True)
        v_m = (run_epoch(model, val_loader, optimizer, scaler,
                         dev, cw_list, False)
               if len(val_ds) > 0 else t_m)
        scheduler.step()

        for hk, (src_tag, mk) in KEY_MAP.items():
            src = t_m if src_tag == 't' else v_m
            history[hk].append(round(float(src.get(mk, 0.0)), 4))

        if epoch % 10 == 0 or epoch in (1, epochs):
            print(f'  {epoch:>6} {t_m["loss"]:>9.4f} {t_m["mIoU"]:>9.4f}'
                  f' {v_m["loss"]:>9.4f} {v_m["mIoU"]:>9.4f}'
                  f' {v_m["f1"]:>8.4f}', flush=True)

        if v_m['mIoU'] > best_miou:
            best_miou    = v_m['mIoU']
            best_metrics = {k: round(float(v), 4) for k, v in v_m.items()}
            best_metrics['epoch'] = epoch
            patience = 0
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict(),
                 'metrics': best_metrics, 'backbone': backbone_name,
                 'fold_tag': fold_tag}, best_path)
        else:
            patience += 1
            if patience >= early_stop:
                print(f'  Early stop @ epoch {epoch}', flush=True)
                break

    # Save training history
    with open(model_dir / 'training_history.json', 'w',
              encoding='utf-8') as f:
        json.dump(
            {'backbone': backbone_name, 'fold_tag': fold_tag,
             'history': history, 'best_metrics': best_metrics,
             'n_params': n_params, 'saved_at': str(datetime.now())},
            f, indent=2)
    with open(model_dir / 'fold_result.json', 'w',
              encoding='utf-8') as f:
        json.dump(
            {'backbone': backbone_name, 'fold_tag': fold_tag,
             'best_metrics': best_metrics,
             'saved_at': str(datetime.now())},
            f, indent=2)

    print(f'\n  Best mIoU={best_miou:.4f} @ epoch'
          f' {best_metrics.get("epoch")}', flush=True)
    return best_metrics


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    _base = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description='Multi-backbone segmentation training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument(
        '--backbone', required=True,
        choices=['pointnet2', 'pointnext', 'kpconv', 'ptv3', 'randlanet'],
        help='Backbone architecture to train')
    parser.add_argument('--train-dirs',       nargs='*', default=None)
    parser.add_argument('--orig-train-dirs',  nargs='*', default=None)
    parser.add_argument('--label-file',       required=True)
    parser.add_argument('--model-dir',        required=True)
    parser.add_argument('--sample-n',         type=int,   default=0)
    parser.add_argument('--n-pts',            type=int,   default=4096)
    parser.add_argument('--epochs',           type=int,   default=70)
    parser.add_argument('--batch-size',       type=int,   default=24)
    parser.add_argument('--lr',               type=float, default=5e-4)
    parser.add_argument('--weight-decay',     type=float, default=1e-4)
    parser.add_argument('--early-stop',       type=int,   default=20)
    parser.add_argument('--class-weight',     nargs=2, type=float,
                        default=[0.2, 0.8])
    parser.add_argument('--device',           default='cuda')
    parser.add_argument('--fold-tag',         default='')
    a = parser.parse_args()

    res = lambda p: str(Path(p) if Path(p).is_absolute() else _base / p)

    best = train_backbone_branch(
        backbone_name   = a.backbone,
        train_dirs      = ([res(d) for d in a.train_dirs]
                           if a.train_dirs else []),
        orig_train_dirs = ([res(d) for d in a.orig_train_dirs]
                           if a.orig_train_dirs else None),
        label_file      = res(a.label_file),
        model_dir       = res(a.model_dir),
        n_pts           = a.n_pts,
        epochs          = a.epochs,
        batch_size      = a.batch_size,
        lr              = a.lr,
        weight_decay    = a.weight_decay,
        early_stop      = a.early_stop,
        class_weights   = a.class_weight,
        device          = a.device,
        sample_n        = a.sample_n,
        fold_tag        = a.fold_tag,
    )
    print(f'\n  Final metrics: {best}', flush=True)


if __name__ == '__main__':
    main()
