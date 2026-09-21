"""Self-contained PointNet++ SSG segmentation model used by formal A3 runs."""
from __future__ import annotations

import torch
from torch import nn


def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.view(points.shape[0], *([1] * (indices.ndim - 1))).expand_as(indices)
    return points[batch, indices]


def farthest_point_sample(xyz: torch.Tensor, count: int) -> torch.Tensor:
    batch_size, point_count, _ = xyz.shape
    count = min(count, point_count)
    selected = torch.zeros(batch_size, count, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch_size, point_count), float("inf"), device=xyz.device)
    farthest = torch.zeros(batch_size, dtype=torch.long, device=xyz.device)
    rows = torch.arange(batch_size, device=xyz.device)
    for index in range(count):
        selected[:, index] = farthest
        centroid = xyz[rows, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.max(-1).indices
    return selected


def ball_query(radius: float, sample_count: int, xyz: torch.Tensor,
               centers: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(centers, xyz)
    k = min(sample_count, xyz.shape[1])
    masked = distances.masked_fill(distances > radius, float("inf"))
    indices = masked.topk(k, dim=-1, largest=False).indices
    nearest = distances.argmin(dim=-1, keepdim=True)
    invalid = ~torch.isfinite(masked.gather(2, indices))
    indices = torch.where(invalid, nearest.expand_as(indices), indices)
    if k < sample_count:
        indices = torch.cat([indices, nearest.expand(*nearest.shape[:-1], sample_count - k)], -1)
    return indices


class SetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int,
                 feature_channels: int, mlp: list[int]):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers: list[nn.Module] = []
        channels = feature_channels + 3
        for output in mlp:
            layers.extend((nn.Conv2d(channels, output, 1), nn.BatchNorm2d(output), nn.ReLU(True)))
            channels = output
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None):
        center_indices = farthest_point_sample(xyz, self.npoint)
        centers = index_points(xyz, center_indices)
        neighbor_indices = ball_query(self.radius, self.nsample, xyz, centers)
        relative_xyz = index_points(xyz, neighbor_indices) - centers.unsqueeze(2)
        if features is None:
            grouped = relative_xyz
        else:
            grouped_features = index_points(features.transpose(1, 2), neighbor_indices)
            grouped = torch.cat((relative_xyz, grouped_features), dim=-1)
        output = self.mlp(grouped.permute(0, 3, 2, 1)).max(dim=2).values
        return centers, output


class FeaturePropagation(nn.Module):
    def __init__(self, input_channels: int, mlp: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        channels = input_channels
        for output in mlp:
            layers.extend((nn.Conv1d(channels, output, 1), nn.BatchNorm1d(output), nn.ReLU(True)))
            channels = output
        self.mlp = nn.Sequential(*layers)

    def forward(self, fine_xyz: torch.Tensor, coarse_xyz: torch.Tensor,
                fine_features: torch.Tensor | None, coarse_features: torch.Tensor):
        distances = torch.cdist(fine_xyz, coarse_xyz)
        k = min(3, coarse_xyz.shape[1])
        nearest_distance, nearest_indices = distances.topk(k, dim=-1, largest=False)
        weights = nearest_distance.clamp_min(1e-10).reciprocal()
        weights = weights / weights.sum(-1, keepdim=True)
        coarse_by_point = coarse_features.transpose(1, 2)
        gathered = index_points(coarse_by_point, nearest_indices)
        interpolated = (gathered * weights.unsqueeze(-1)).sum(2).transpose(1, 2)
        combined = interpolated if fine_features is None else torch.cat((fine_features, interpolated), 1)
        return self.mlp(combined)


class PointNetPPSeg(nn.Module):
    """PointNet++ SSG per-point segmentation network; input is B x C x N."""

    def __init__(self, input_channels: int = 6, num_classes: int = 2,
                 profile: str = "formal"):
        super().__init__()
        if input_channels not in {3, 6}:
            raise ValueError("PointNetPPSeg supports XYZ or XYZ+normal input")
        extra = input_channels - 3
        if profile == "formal":
            npoints = (1024, 256, 64, 16)
            widths = ([32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512])
            fp_widths = ([256, 256], [256, 128], [128, 128], [128, 128])
        elif profile == "smoke":
            npoints = (32, 16, 8, 4)
            widths = ([8, 8, 16], [16, 16, 32], [32, 32, 64], [64, 64, 128])
            fp_widths = ([64, 64], [64, 32], [32, 32], [32, 32])
        else:
            raise ValueError(f"Unknown model profile: {profile}")
        c1, c2, c3, c4 = (item[-1] for item in widths)
        p4, p3, p2, p1 = fp_widths
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.profile = profile
        self.sa1 = SetAbstraction(npoints[0], 0.1, 32, extra, list(widths[0]))
        self.sa2 = SetAbstraction(npoints[1], 0.2, 32, c1, list(widths[1]))
        self.sa3 = SetAbstraction(npoints[2], 0.4, 32, c2, list(widths[2]))
        self.sa4 = SetAbstraction(npoints[3], 0.8, 32, c3, list(widths[3]))
        self.fp4 = FeaturePropagation(c3 + c4, list(p4))
        self.fp3 = FeaturePropagation(c2 + p4[-1], list(p3))
        self.fp2 = FeaturePropagation(c1 + p3[-1], list(p2))
        self.fp1 = FeaturePropagation(extra + p2[-1], list(p1))
        self.head = nn.Sequential(
            nn.Conv1d(p1[-1], p1[-1], 1), nn.BatchNorm1d(p1[-1]), nn.ReLU(True),
            nn.Dropout(0.5), nn.Conv1d(p1[-1], num_classes, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError(f"Expected Bx{self.input_channels}xN input, got {tuple(values.shape)}")
        xyz = values[:, :3].transpose(1, 2).contiguous()
        features0 = values[:, 3:] if self.input_channels > 3 else None
        xyz1, features1 = self.sa1(xyz, features0)
        xyz2, features2 = self.sa2(xyz1, features1)
        xyz3, features3 = self.sa3(xyz2, features2)
        xyz4, features4 = self.sa4(xyz3, features3)
        decoded3 = self.fp4(xyz3, xyz4, features3, features4)
        decoded2 = self.fp3(xyz2, xyz3, features2, decoded3)
        decoded1 = self.fp2(xyz1, xyz2, features1, decoded2)
        decoded0 = self.fp1(xyz, xyz1, features0, decoded1)
        return self.head(decoded0)
