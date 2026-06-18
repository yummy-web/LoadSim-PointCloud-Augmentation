"""
benchmark_inference.py
======================

独立的点云模型实时性评测脚本，适用于 RA-L 论文中的推理速度基准测试。

功能：
1) 构建 GPU 上的 dummy 输入，避免重新加载大规模数据集
2) 计算参数量 Params (Million)
3) 进行 GPU warm-up
4) 使用 CUDA 同步的方式测量 100 次前向推理延迟
5) 计算平均 Latency(ms) 与 FPS
6) 可选统计 FLOPs / MACs（优先尝试 thop，其次 fvcore）

默认 dummy 输入格式：
    [B, C, N] = [1, 6, 100000]
其中 C=6 表示 xyz + normal。

说明：
- 不修改任何现有工程代码，可直接作为独立脚本使用。
- 由于不同点云模型的 forward 入参形式可能不同，本脚本提供了
  一个较为通用的兼容封装，会自动尝试若干常见调用方式。
- 如果你的模型 forward 需要额外参数，请在 `MODEL_BUILDERS` 中
  按自己的工程实际情况补充。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from step4_train import load_ply_xyzn
from step4_train_backbones import get_backbone


# -----------------------------------------------------------------------------
# 可选 FLOPs 依赖
# -----------------------------------------------------------------------------
try:
    from thop import profile as thop_profile  # type: ignore
except Exception:
    thop_profile = None

try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
except Exception:
    FlopCountAnalysis = None


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------
def get_num_params(model: nn.Module) -> int:
    """统计模型总参数量。"""
    return sum(p.numel() for p in model.parameters())


def format_params(num_params: int) -> str:
    return f"{num_params / 1e6:.3f} M"


def format_flops(flops: Optional[float]) -> str:
    if flops is None:
        return "N/A"
    return f"{flops / 1e9:.3f} G"


def _unwrap_output(output: Any) -> Any:
    """尽量把模型输出规整一下，兼容各种返回类型。"""
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return output[0]
    return output


@torch.no_grad()
def _try_forward(model: nn.Module, dummy_input: torch.Tensor) -> Any:
    """尝试若干常见的点云模型 forward 调用方式。"""
    # 方式1：直接喂 [B, C, N]
    try:
        return model(dummy_input)
    except Exception:
        pass

    # 方式2：喂 [B, N, C]
    try:
        return model(dummy_input.permute(0, 2, 1).contiguous())
    except Exception:
        pass

    # 方式3：拆成 xyz 和 feature
    xyz = dummy_input[:, :3, :].contiguous()       # [B, 3, N]
    feat = dummy_input[:, 3:, :].contiguous()      # [B, 3, N]
    xyz_bn3 = xyz.permute(0, 2, 1).contiguous()    # [B, N, 3]
    feat_bnc = feat.permute(0, 2, 1).contiguous()  # [B, N, 3]

    # 3.1 model(xyz, feat)
    try:
        return model(xyz_bn3, feat_bnc)
    except Exception:
        pass

    # 3.2 model(xyz, feat, None)
    try:
        return model(xyz_bn3, feat_bnc, None)
    except Exception:
        pass

    # 3.3 model(xyz)
    try:
        return model(xyz_bn3)
    except Exception:
        pass

    # 3.4 model(points)
    try:
        points = dummy_input.permute(0, 2, 1).contiguous()
        return model(points)
    except Exception as e:
        raise RuntimeError(
            "模型 forward 调用失败。请根据你的模型接口修改 `_try_forward()` 中的调用方式。"
        ) from e


@torch.no_grad()
def benchmark_model(model: nn.Module, dummy_input: torch.Tensor, warmup: int = 30, iters: int = 100) -> Dict[str, float]:
    """
    评测单个模型的 Params / Latency / FPS / FLOPs。

    流程严格按照论文基准测试习惯：
    1) 参数量统计
    2) warm-up 30 次
    3) torch.cuda.synchronize() 后开始计时
    4) 正式推理 100 次
    5) torch.cuda.synchronize() 后结束计时
    6) 计算平均延迟与 FPS

    返回：
        dict, 包含 params_m, latency_ms, fps, flops_g, macs_g 等字段
    """
    assert dummy_input.is_cuda, "dummy_input 必须放在 GPU 上"

    model = model.cuda().eval()
    num_params = get_num_params(model)

    print("\n" + "=" * 80)
    print(f"开始评测模型：{model.__class__.__name__}")
    print("=" * 80)
    print(f"[1/4] 参数量 Params: {format_params(num_params)} ({num_params:,} 参数)")

    # ------------------------------------------------------------------
    # 可选 FLOPs / MACs
    # ------------------------------------------------------------------
    flops_value: Optional[float] = None
    macs_value: Optional[float] = None

    try:
        if thop_profile is not None:
            model_for_profile = model.eval()
            # thop 对一些自定义算子可能不完整，但可作为快速近似
            macs, params = thop_profile(model_for_profile, inputs=(dummy_input,), verbose=False)
            macs_value = float(macs)
            flops_value = float(macs) * 2.0
            print(f"[2/4] FLOPs/MACs (thop): MACs={format_flops(macs_value)}, FLOPs≈{format_flops(flops_value)}")
        elif FlopCountAnalysis is not None:
            model_for_profile = model.eval()
            flops = FlopCountAnalysis(model_for_profile, dummy_input)
            flops_value = float(flops.total())
            print(f"[2/4] FLOPs (fvcore): {format_flops(flops_value)}")
        else:
            print("[2/4] FLOPs/MACs: 未安装 thop 或 fvcore，已跳过")
    except Exception as e:
        print(f"[2/4] FLOPs/MACs 统计失败，已跳过。原因：{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------
    print(f"[3/4] GPU warm-up: {warmup} 次前向推理，不计时")
    for _ in range(warmup):
        _ = _unwrap_output(_try_forward(model, dummy_input))

    # ------------------------------------------------------------------
    # 正式计时
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(iters):
        _ = _unwrap_output(_try_forward(model, dummy_input))

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    elapsed_s = end_time - start_time
    latency_ms = (elapsed_s / iters) * 1000.0
    fps = iters / elapsed_s

    print(f"[4/4] 正式测试: {iters} 次完成")
    print(f"      总耗时: {elapsed_s:.4f} s")
    print(f"      平均延迟 Latency: {latency_ms:.4f} ms / iter")
    print(f"      吞吐量 FPS: {fps:.2f} frames/s")
    print("=" * 80)

    return {
        "params_m": num_params / 1e6,
        "latency_ms": latency_ms,
        "fps": fps,
        "macs_g": (macs_value / 1e9) if macs_value is not None else float("nan"),
        "flops_g": (flops_value / 1e9) if flops_value is not None else float("nan"),
    }


# -----------------------------------------------------------------------------
# 示例：在这里接入你工程中已经定义好的五个模型实例
# -----------------------------------------------------------------------------
#
# 你只需要把下面的占位函数替换成你自己的模型构建逻辑即可。
# 如果你的模型已经在别的脚本里实例化完成，也可以直接把实例传入
# benchmark_model(model, dummy_input)。
#
# 例如：
#   from your_model_file import pointnet2_model, pointnext_model, kpconv_model
#   ...
#   MODEL_BUILDERS = {
#       "PointNet++": lambda: pointnet2_model,
#       ...
#   }
# -----------------------------------------------------------------------------


def _raise_not_configured(name: str) -> Callable[[], nn.Module]:
    def _inner() -> nn.Module:
        raise RuntimeError(
            f"请先在 benchmark_inference.py 中配置 `{name}` 的模型构建逻辑。"
        )
    return _inner


def _build_backbone(name: str) -> nn.Module:
    model = get_backbone(name, in_ch=6, n_cls=2)
    if model is None:
        raise RuntimeError("当前环境未安装 PyTorch，无法构建模型。")
    return model


def _find_real_ply_file() -> Path:
    """优先从源实验输出目录中找一个真实 PLY 文件。"""
    base = Path(__file__).parent
    priority_dirs = [
        base / "outputs4",
        base / "outputs_journal",
        base / "outputs",
    ]
    keywords = ["fold", "train", "val", "valid", "validation", "test", "high_quality"]

    candidates = []
    for root in priority_dirs:
        if not root.exists():
            continue
        for p in root.rglob("*.ply"):
            s = str(p).lower()
            if any(k in s for k in keywords):
                candidates.append(p)

    if not candidates:
        candidates = list(base.rglob("*.ply"))
    if not candidates:
        raise FileNotFoundError("工作区里没有找到任何 .ply 文件，无法进行真实点云测速。")

    candidates.sort(key=lambda p: (len(str(p)), str(p).lower()))
    return candidates[0]


MODEL_BUILDERS: Dict[str, Callable[[], nn.Module]] = {
    "PointNet++": lambda: _build_backbone("pointnet2"),
    "PointNeXt": lambda: _build_backbone("pointnext"),
    "KPConv": lambda: _build_backbone("kpconv"),
    "PTv3": lambda: _build_backbone("ptv3"),
    "RandLA-Net": lambda: _build_backbone("randlanet"),
}


SOURCE_N_PTS = 4096


def _prepare_real_input_from_ply(ply_path: Path, n_points: int = SOURCE_N_PTS) -> torch.Tensor:
    pts, nrm = load_ply_xyzn(ply_path)
    if pts is None or len(pts) == 0:
        raise RuntimeError(f"无法读取点云: {ply_path}")

    if nrm is None:
        nrm = np.zeros_like(pts, dtype=np.float32)

    pts = np.asarray(pts, dtype=np.float32)
    nrm = np.asarray(nrm, dtype=np.float32)
    n = len(pts)
    if n >= n_points:
        idx = np.random.choice(n, n_points, replace=False)
    else:
        idx = np.concatenate([np.arange(n), np.random.choice(n, n_points - n, replace=True)])

    pts = pts[idx]
    nrm = nrm[idx]

    ctr = pts.mean(axis=0)
    scale = np.abs(pts - ctr).max() + 1e-8
    pts = (pts - ctr) / scale
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)

    xyzn = np.concatenate([pts, nrm], axis=1).T.astype(np.float32)
    return torch.from_numpy(xyzn).unsqueeze(0)


def run_benchmark_suite() -> None:
    """批量评测五个模型，输入来自源实验一致的真实 `.ply` 文件。"""
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境未检测到 CUDA，请在 GPU 环境中运行该脚本。")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    ply_path = _find_real_ply_file()
    dummy_input = _prepare_real_input_from_ply(ply_path, n_points=4096).to(device)
    print(f"Using real PLY: {ply_path}")
    print(f"Input Shape: {tuple(dummy_input.shape)}, device={dummy_input.device}")

    results = {}
    for model_name, builder in MODEL_BUILDERS.items():
        print(f"\n>>> 即将测试：{model_name}")
        model = builder().to(device).eval()
        results[model_name] = benchmark_model(model, dummy_input)

    print("\n" + "#" * 80)
    print("基准测试汇总结果")
    print("#" * 80)
    print(f"{'Model':<15} {'Params(M)':>12} {'Latency(ms)':>14} {'FPS':>12} {'MACs(G)':>12} {'FLOPs(G)':>12}")
    for name, r in results.items():
        print(
            f"{name:<15} {r['params_m']:>12.3f} {r['latency_ms']:>14.4f} {r['fps']:>12.2f} "
            f"{r['macs_g']:>12.3f} {r['flops_g']:>12.3f}"
        )
    print("#" * 80)


if __name__ == "__main__":
    run_benchmark_suite()
