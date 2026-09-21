"""Isolated multi-backbone module for the A3MB matched-budget supplement.

Reviewer request: on FIVE backbones (PointNet++, PointNeXt, KPConv, RandLA-Net,
PTv3), under the SAME M / B / U / training策略 / seeds, compare RAW_REPEAT vs
LOADSIM. A3 currently proves the augmentation-saturation result only on
PointNet++; this module lets the frozen A3 matched-budget engine run the other
four backbones so the multi-backbone claim is itself matched-budget-controlled.

The four non-PointNet++ classes are copied VERBATIM from the repo's existing
`step4_train_backbones.py` (the same self-contained pure-PyTorch reimplementations
that produced the original paper's multi-backbone numbers) so this supplement is
a controlled re-run of the paper's own code, not a new architecture. They are
simplified reimplementations, NOT official reference implementations — state this
honestly in the response letter. Copied here (rather than imported) to avoid
pulling in the legacy step4_train training stack at import time.

Interface contract (identical to a3_model.PointNetPPSeg):
    forward(values: FloatTensor[B, C, N]) -> FloatTensor[B, n_cls, N]
    C == 6 (XYZ + normal) when use_normals else 3 (XYZ). n_cls == 2.

build_backbone(name, in_ch, n_cls, profile) dispatches by name; pointnet2 routes
to the frozen A3 PointNetPPSeg (with its formal/smoke profile), the other four to
the copied classes (which have a single fixed width profile).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from a3_model import PointNetPPSeg  # frozen A3 PointNet++ (profile-aware)

BACKBONES = ("pointnet2", "pointnext", "kpconv", "ptv3", "randlanet")

# --- shared geometry utilities (verbatim from step4_train_backbones.py) -------
def index_points(pts, idx):
    """Gather points by index: pts (B,N,C), idx (B,S[,K]) -> (B,S[,K],C)."""
    B = pts.shape[0]
    dev = pts.device
    bidx = (torch.arange(B, device=dev)
            .view(B, *([1] * (idx.dim() - 1)))
            .expand_as(idx))
    return pts[bidx, idx]


def farthest_point_sample(xyz, n_pt):
    """Farthest point sampling. xyz: (B,N,3) -> idx: (B,n_pt)."""
    B, N, _ = xyz.shape
    dev = xyz.device
    sel = torch.zeros(B, n_pt, dtype=torch.long, device=dev)
    dist = torch.full((B, N), float('inf'), device=dev)
    far = torch.randint(0, N, (B,), device=dev)
    for i in range(n_pt):
        sel[:, i] = far
        ctr = xyz[torch.arange(B, device=dev), far].unsqueeze(1)
        d = ((xyz - ctr) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.max(-1)[1]
    return sel


def ball_query(r, k, xyz, new_xyz):
    """Ball query: returns (B,S,k) indices; pads with first if fewer than k."""
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    dist = torch.cdist(new_xyz, xyz)
    dm = dist.clone()
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
# --- Backbone: PointNeXt (Qian et al. NeurIPS 2022) --------------------------
class InvResMLP(nn.Module):
    """Inverted Residual MLP - PointNeXt core block."""
    def __init__(self, ch: int, expansion: int = 4):
        super().__init__()
        mid = ch * expansion
        self.net = nn.Sequential(
            nn.Conv1d(ch, mid, 1), nn.BatchNorm1d(mid), nn.ReLU(True),
            nn.Conv1d(mid, ch, 1), nn.BatchNorm1d(ch))
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
        self.inv = InvResMLP(out_ch)
        self.out_ch = out_ch

    def forward(self, xyz, feats):
        B, N, _ = xyz.shape
        S = min(self.n_pt, N)
        nidx = farthest_point_sample(xyz, S)
        nxyz = index_points(xyz, nidx)
        nbidx = ball_query(self.r, min(self.k, N), xyz, nxyz)
        gxyz = index_points(xyz, nbidx) - nxyz.unsqueeze(2)
        if feats is not None:
            gf = index_points(feats.permute(0, 2, 1), nbidx)
            g = torch.cat([gxyz, gf], -1).permute(0, 3, 2, 1)
        else:
            g = gxyz.permute(0, 3, 2, 1)
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
        k = min(3, S)
        d = torch.cdist(xyz1, xyz2)
        dk, ki = d.topk(k, dim=-1, largest=False)
        dk = dk.clamp(1e-10)
        w = (1.0 / dk) / (1.0 / dk).sum(-1, keepdim=True)
        C2 = f2.shape[1]
        kex = ki.unsqueeze(1).expand(B, C2, -1, -1)
        interp = (f2.unsqueeze(2).expand(B, C2, N, S)
                  .gather(3, kex) * w.unsqueeze(1)).sum(-1)
        out = torch.cat([f1, interp], 1) if f1 is not None else interp
        return self.inv(self.mlp(out))


class PointNeXtSeg(nn.Module):
    """PointNeXt segmentation network (4 SA + 4 FP + classification head)."""
    def __init__(self, in_ch: int = 6, n_cls: int = 2):
        super().__init__()
        ex = in_ch - 3
        self.sa1 = _PNxSA(1024, 0.1, 32, ex, 64)
        self.sa2 = _PNxSA(256, 0.2, 64, 64, 128)
        self.sa3 = _PNxSA(64, 0.4, 128, 128, 256)
        self.sa4 = _PNxSA(16, 0.8, 256, 256, 512)
        self.fp4 = _PNxFP(256 + 512, 256)
        self.fp3 = _PNxFP(128 + 256, 256)
        self.fp2 = _PNxFP(64 + 256, 128)
        self.fp1 = _PNxFP(ex + 128, 128)
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(0.5), nn.Conv1d(128, n_cls, 1))

    def forward(self, xyzn):
        B, C, N = xyzn.shape
        xyz = xyzn[:, :3, :].permute(0, 2, 1).contiguous()
        f0 = xyzn[:, 3:, :] if C > 3 else None
        x1, f1 = self.sa1(xyz, f0); x2, f2 = self.sa2(x1, f1)
        x3, f3 = self.sa3(x2, f2); x4, f4 = self.sa4(x3, f3)
        f3b = self.fp4(x3, x4, f3, f4)
        f2b = self.fp3(x2, x3, f2, f3b)
        f1b = self.fp2(x1, x2, f1, f2b)
        f0b = self.fp1(xyz, x1, f0, f1b)
        return self.head(f0b)
# --- Backbone: KPConv (Thomas et al. ICCV 2019) ------------------------------
class _KPConvLayer(nn.Module):
    """Rigid KPConv layer with Fibonacci-sphere kernel initialization."""
    def __init__(self, n_pt, r, k, in_ch, out_ch, n_kpts: int = 15):
        super().__init__()
        self.n_pt, self.r, self.k = n_pt, r, k
        kp = self._fibonacci_sphere(n_kpts)
        self.register_buffer('kernel_pts', kp)
        self.weights = nn.Parameter(
            torch.randn(n_kpts, max(in_ch, 1), out_ch) * 0.01)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(True)
        self.out_ch = out_ch
        self._in_ch = in_ch

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
        S = min(self.n_pt, N)
        nidx = farthest_point_sample(xyz, S)
        nxyz = index_points(xyz, nidx)
        nbidx = ball_query(self.r, min(self.k, N), xyz, nxyz)
        gxyz = (index_points(xyz, nbidx) - nxyz.unsqueeze(2)) / (self.r + 1e-8)
        kp = self.kernel_pts.to(xyz.device)
        hw = torch.clamp(
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
        k = min(3, S)
        d = torch.cdist(xyz1, xyz2)
        dk, ki = d.topk(k, dim=-1, largest=False)
        dk = dk.clamp(1e-10)
        w = (1.0 / dk) / (1.0 / dk).sum(-1, keepdim=True)
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
        self.sa1 = _KPConvLayer(1024, 0.1, 32, ex, 64)
        self.sa2 = _KPConvLayer(256, 0.2, 64, 64, 128)
        self.sa3 = _KPConvLayer(64, 0.4, 128, 128, 256)
        self.sa4 = _KPConvLayer(16, 0.8, 256, 256, 512)
        self.fp4 = _KPConvFP(256 + 512, 256)
        self.fp3 = _KPConvFP(128 + 256, 256)
        self.fp2 = _KPConvFP(64 + 256, 128)
        self.fp1 = _KPConvFP(ex + 128, 128)
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(0.5), nn.Conv1d(128, n_cls, 1))

    def forward(self, xyzn):
        B, C, N = xyzn.shape
        xyz = xyzn[:, :3, :].permute(0, 2, 1).contiguous()
        f0 = xyzn[:, 3:, :] if C > 3 else None
        x1, f1 = self.sa1(xyz, f0); x2, f2 = self.sa2(x1, f1)
        x3, f3 = self.sa3(x2, f2); x4, f4 = self.sa4(x3, f3)
        f3b = self.fp4(x3, x4, f3, f4)
        f2b = self.fp3(x2, x3, f2, f3b)
        f1b = self.fp2(x1, x2, f1, f2b)
        f0b = self.fp1(xyz, x1, f0, f1b)
        return self.head(f0b)
# --- Backbone: Point Transformer V3 (simplified) -----------------------------
class _PTv3AttnBlock(nn.Module):
    """Simplified PTv3 attention block: local ball-query self-attention + RPE."""
    def __init__(self, ch: int, n_heads: int = 4, k: int = 16, r: float = 0.2):
        super().__init__()
        assert ch % n_heads == 0
        self.ch = ch
        self.n_heads = n_heads
        self.k = k
        self.r = r
        self.head_dim = ch // n_heads
        self.qkv = nn.Linear(ch, 3 * ch, bias=False)
        self.proj = nn.Linear(ch, ch, bias=False)
        self.rpe = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(True), nn.Linear(32, n_heads))
        self.norm = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(
            nn.Linear(ch, 4 * ch), nn.GELU(), nn.Linear(4 * ch, ch))
        self.norm2 = nn.LayerNorm(ch)

    def forward(self, xyz, feats):
        B, N, C = feats.shape
        H, D = self.n_heads, self.head_dim
        S = min(N, self.k)
        dist = torch.cdist(xyz, xyz)
        dm = dist.clone()
        dm[dist > self.r] = float('inf')
        if N <= S:
            nb_idx = torch.arange(N, device=xyz.device)[None, None, :].expand(B, N, N)
        else:
            _, nb_idx = dm.topk(S, dim=-1, largest=False)
        nb_idx = nb_idx.detach()
        nb_xyz = index_points(xyz, nb_idx)
        rel_pos = nb_xyz - xyz.unsqueeze(2)
        rpe = self.rpe(rel_pos)
        qkv = self.qkv(feats)
        q, k_t, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, D).permute(0, 2, 1, 3)
        k_nb = index_points(k_t, nb_idx)
        v_nb = index_points(v, nb_idx)
        k_nb = k_nb.view(B, N, S, H, D).permute(0, 3, 1, 2, 4)
        v_nb = v_nb.view(B, N, S, H, D).permute(0, 3, 1, 2, 4)
        scores = (q.unsqueeze(3) * k_nb).sum(-1) / (D ** 0.5)
        scores = scores + rpe.permute(0, 3, 1, 2)
        attn = scores.softmax(-1)
        out = (attn.unsqueeze(-1) * v_nb).sum(-2)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
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
        nxyz = index_points(xyz, nidx)
        nf = index_points(feats, nidx)
        nf = self.proj(nf)
        nf = self.attn(nxyz, nf)
        return nxyz, nf.permute(0, 2, 1)


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
        k = min(3, S)
        d = torch.cdist(xyz1, xyz2)
        dk, ki = d.topk(k, dim=-1, largest=False)
        dk = dk.clamp(1e-10)
        w = (1.0 / dk) / (1.0 / dk).sum(-1, keepdim=True)
        C2 = f2.shape[1]
        interp = (f2.unsqueeze(2).expand(B, C2, N, S)
                  .gather(3, ki.unsqueeze(1).expand(B, C2, -1, -1))
                  * w.unsqueeze(1)).sum(-1)
        out = torch.cat([f1, interp], 1) if f1 is not None else interp
        return self.mlp(out)


class PTv3Seg(nn.Module):
    """Point Transformer V3 segmentation (simplified self-contained)."""
    def __init__(self, in_ch: int = 6, n_cls: int = 2):
        super().__init__()
        ex = in_ch - 3
        self.stem = nn.Sequential(
            nn.Linear(in_ch, 64), nn.BatchNorm1d(64), nn.ReLU(True))
        self.dn1 = _PTv3DownBlock(1024, 0.1, 16, 64, 128, n_heads=4)
        self.dn2 = _PTv3DownBlock(256, 0.2, 16, 128, 256, n_heads=4)
        self.dn3 = _PTv3DownBlock(64, 0.4, 16, 256, 512, n_heads=8)
        self.dn4 = _PTv3DownBlock(16, 0.8, 16, 512, 512, n_heads=8)
        self.fp4 = _PTv3FP(512 + 512, 256)
        self.fp3 = _PTv3FP(256 + 256, 256)
        self.fp2 = _PTv3FP(128 + 256, 128)
        self.fp1 = _PTv3FP(64 + 128, 128)
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(0.5), nn.Conv1d(128, n_cls, 1))

    def forward(self, xyzn):
        B, C, N = xyzn.shape
        xyz = xyzn[:, :3, :].permute(0, 2, 1).contiguous()
        feats = xyzn.permute(0, 2, 1)
        f0 = self.stem(feats.reshape(B * N, C)).reshape(B, N, 64)
        f0_ch = f0.permute(0, 2, 1)
        x1, f1 = self.dn1(xyz, f0)
        x2, f2 = self.dn2(x1, f1.permute(0, 2, 1))
        x3, f3 = self.dn3(x2, f2.permute(0, 2, 1))
        x4, f4 = self.dn4(x3, f3.permute(0, 2, 1))
        f3b = self.fp4(x3, x4, f3, f4)
        f2b = self.fp3(x2, x3, f2, f3b)
        f1b = self.fp2(x1, x2, f1, f2b)
        f0b = self.fp1(xyz, x1, f0_ch, f1b)
        return self.head(f0b)
# --- Backbone: RandLA-Net (Hu et al. CVPR 2020) ------------------------------
class _RandLALFA(nn.Module):
    """Local Feature Aggregation: relative encoding + attentive pooling."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 16):
        super().__init__()
        self.k = k
        self.rpe = nn.Sequential(
            nn.Conv2d(in_ch + 3, out_ch, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.att = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch),
            nn.Softmax(dim=-1))
        self.agg = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch), nn.ReLU(True))
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch))
        self.relu = nn.ReLU(True)

    def forward(self, xyz, feats):
        """xyz: (B,N,3), feats: (B,in_ch,N) -> (B,out_ch,N)."""
        B, C, N = feats.shape
        k_eff = min(self.k, N)
        dist = torch.cdist(xyz, xyz)
        _, ki = dist.topk(k_eff, dim=-1, largest=False)  # (B,N,k)
        nb_xyz = index_points(xyz, ki)
        nb_feat = index_points(feats.permute(0, 2, 1), ki)
        rel_pos = (nb_xyz - xyz.unsqueeze(2)).permute(0, 3, 2, 1)
        nb_feat_t = nb_feat.permute(0, 3, 2, 1)
        combined = torch.cat([rel_pos, nb_feat_t], dim=1)
        enc = self.rpe(combined)
        att = self.att(enc)
        agg = (enc * att).sum(2)
        agg = self.agg(agg)
        sc = self.shortcut(feats)
        return self.relu(agg + sc)


class _RandLADown(nn.Module):
    """Random downsampling + LFA. NOTE: the redundant farthest_point_sample
    calls below are preserved VERBATIM from step4_train_backbones.py so the RNG
    stream (and thus reproducibility vs. the paper's own code) is unchanged."""
    def __init__(self, ratio: float, in_ch: int, out_ch: int, k: int = 16):
        super().__init__()
        self.ratio = ratio
        self.lfa = _RandLALFA(in_ch, out_ch, k)
        self.out_ch = out_ch

    def forward(self, xyz, feats):
        B, N, _ = xyz.shape
        n_out = max(16, int(N * self.ratio))
        if self.training:
            idx = torch.randperm(N, device=xyz.device)[:n_out]
        else:
            idx = farthest_point_sample(xyz, n_out)[0]
            idx = farthest_point_sample(xyz, n_out)
        idx = farthest_point_sample(xyz, n_out)
        nxyz = index_points(xyz, idx)
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
        k = min(3, S)
        d = torch.cdist(xyz1, xyz2)
        dk, ki = d.topk(k, dim=-1, largest=False)
        dk = dk.clamp(1e-10)
        w = (1.0 / dk) / (1.0 / dk).sum(-1, keepdim=True)
        C2 = f2.shape[1]
        interp = (f2.unsqueeze(2).expand(B, C2, N, S)
                  .gather(3, ki.unsqueeze(1).expand(B, C2, -1, -1))
                  * w.unsqueeze(1)).sum(-1)
        out = torch.cat([f1, interp], 1) if f1 is not None else interp
        return self.mlp(out)


class RandLANetSeg(nn.Module):
    """RandLA-Net segmentation (random-down + LFA + FP + head)."""
    def __init__(self, in_ch: int = 6, n_cls: int = 2):
        super().__init__()
        ex = in_ch - 3
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 32, 1), nn.BatchNorm1d(32), nn.ReLU(True))
        self.dn1 = _RandLADown(0.25, 32, 64, k=16)
        self.dn2 = _RandLADown(0.25, 64, 128, k=16)
        self.dn3 = _RandLADown(0.25, 128, 256, k=16)
        self.dn4 = _RandLADown(0.25, 256, 512, k=16)
        self.fp4 = _RandLAFP(256 + 512, 256)
        self.fp3 = _RandLAFP(128 + 256, 128)
        self.fp2 = _RandLAFP(64 + 128, 64)
        self.fp1 = _RandLAFP(32 + 64, 32)
        self.head = nn.Sequential(
            nn.Conv1d(32, 64, 1), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Dropout(0.5), nn.Conv1d(64, n_cls, 1))

    def forward(self, xyzn):
        B, C, N = xyzn.shape
        xyz = xyzn[:, :3, :].permute(0, 2, 1).contiguous()
        f0 = self.stem(xyzn)
        x1, f1 = self.dn1(xyz, f0)
        x2, f2 = self.dn2(x1, f1)
        x3, f3 = self.dn3(x2, f2)
        x4, f4 = self.dn4(x3, f3)
        f3b = self.fp4(x3, x4, f3, f4)
        f2b = self.fp3(x2, x3, f2, f3b)
        f1b = self.fp2(x1, x2, f1, f2b)
        f0b = self.fp1(xyz, x1, f0, f1b)
        return self.head(f0b)


# --- factory -----------------------------------------------------------------
def build_backbone(name: str, in_ch: int = 6, n_cls: int = 2,
                   profile: str = "formal") -> nn.Module:
    """Return the requested backbone. `profile` only applies to pointnet2
    (formal/smoke, per a3_model); the other four have a single fixed width."""
    key = name.lower()
    if key == "pointnet2":
        return PointNetPPSeg(in_ch, n_cls, profile)
    if key == "pointnext":
        return PointNeXtSeg(in_ch=in_ch, n_cls=n_cls)
    if key == "kpconv":
        return KPConvSeg(in_ch=in_ch, n_cls=n_cls)
    if key == "ptv3":
        return PTv3Seg(in_ch=in_ch, n_cls=n_cls)
    if key == "randlanet":
        return RandLANetSeg(in_ch=in_ch, n_cls=n_cls)
    raise ValueError(f"Unknown backbone {name!r}; choose from {BACKBONES}")
