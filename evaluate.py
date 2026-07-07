"""Evaluation and visualization helpers for shallow Vp inversion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_datasets, denormalize_vp
from model_gprinvnet_like import build_model

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - optional dependency
    structural_similarity = None


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _prediction_vp(prediction):
    if isinstance(prediction, dict):
        return prediction["vp"]
    return prediction


def _prepare_metric_mask(mask: torch.Tensor | None,
                         reference: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    import torch.nn.functional as F

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(
            f"metric mask must be [B,1,H,W] or [B,H,W], got {tuple(mask.shape)}")
    mask = mask.to(device=reference.device, dtype=reference.dtype)
    if tuple(mask.shape[-2:]) != tuple(reference.shape[-2:]):
        mask = F.interpolate(
            mask, size=tuple(reference.shape[-2:]), mode="nearest")
    if mask.shape[1] == 1 and reference.shape[1] != 1:
        mask = mask.expand(-1, reference.shape[1], -1, -1)
    elif mask.shape[1] != reference.shape[1]:
        raise ValueError("metric mask channel count must be 1 or match data")
    return mask > 0.5


def compute_metrics(
        pred_mps: torch.Tensor,
        target_mps: torch.Tensor,
        mask: torch.Tensor | None = None,
        compute_ssim: bool = True) -> dict[str, float | None]:
    diff = pred_mps - target_mps
    metric_mask = _prepare_metric_mask(mask, diff)
    if metric_mask is not None and torch.any(metric_mask):
        diff_values = diff[metric_mask]
        target_values = target_mps[metric_mask]
        mask_fraction = float(metric_mask.float().mean().detach().cpu())
    else:
        diff_values = diff.reshape(-1)
        target_values = target_mps.reshape(-1)
        mask_fraction = None
    mae = torch.mean(torch.abs(diff_values)).item()
    rmse = torch.sqrt(torch.mean(diff_values * diff_values)).item()
    rel = torch.mean(
        torch.abs(diff_values)
        / torch.clamp(torch.abs(target_values), min=1.0)).item()

    ssim_value = None
    if compute_ssim and structural_similarity is not None:
        pred_np = _to_numpy(pred_mps[:, 0])
        target_np = _to_numpy(target_mps[:, 0])
        values = []
        for pred_i, target_i in zip(pred_np, target_np):
            data_range = float(max(target_i.max() - target_i.min(), 1.0))
            win_size = min(11, pred_i.shape[0], pred_i.shape[1])
            if win_size % 2 == 0:
                win_size -= 1
            if win_size >= 3:
                values.append(
                    structural_similarity(
                        target_i,
                        pred_i,
                        data_range=data_range,
                        win_size=win_size,
                    )
                )
        if values:
            ssim_value = float(np.mean(values))

    return {
        "mae_mps": float(mae),
        "rmse_mps": float(rmse),
        "relative_error": float(rel),
        "metric_mask_fraction": mask_fraction,
        "ssim": ssim_value,
    }


class MetricAverager:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, metrics: dict[str, float | None], n: int) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
            self.counts[key] = self.counts.get(key, 0) + n

    def average(self) -> dict[str, float]:
        return {key: self.sums[key] / max(self.counts[key], 1) for key in sorted(self.sums)}


def save_prediction_figure(
    input_bscan: torch.Tensor,
    target_vp: torch.Tensor,
    pred_vp: torch.Tensor,
    output_path: str | Path,
    sample_name: str = "",
    curve_positions: list[float] | tuple[float, ...] = (0.25, 0.5, 0.75),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bscan = _to_numpy(input_bscan[0])
    target = _to_numpy(target_vp[0])
    pred = _to_numpy(pred_vp[0])
    error = pred - target

    vmin = float(min(target.min(), pred.min()))
    vmax = float(max(target.max(), pred.max()))
    err_abs = float(max(np.max(np.abs(error)), 1.0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(sample_name)

    im0 = axes[0, 0].imshow(bscan, aspect="auto", cmap="gray")
    axes[0, 0].set_title("Input B-scan")
    axes[0, 0].set_xlabel("Trace")
    axes[0, 0].set_ylabel("Time sample")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(target, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("True Vp (m/s)")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Depth sample")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(pred, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Predicted Vp (m/s)")
    axes[0, 2].set_xlabel("X")
    axes[0, 2].set_ylabel("Depth sample")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(error, aspect="auto", cmap="seismic", vmin=-err_abs, vmax=err_abs)
    axes[1, 0].set_title("Prediction Error (m/s)")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Depth sample")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].axis("off")
    curve_ax = axes[1, 2]
    depth = np.arange(target.shape[0])
    for frac in curve_positions:
        col = int(round(float(frac) * (target.shape[1] - 1)))
        col = max(0, min(target.shape[1] - 1, col))
        curve_ax.plot(target[:, col], depth, linestyle="-", label=f"true x={col}")
        curve_ax.plot(pred[:, col], depth, linestyle="--", label=f"pred x={col}")
    curve_ax.invert_yaxis()
    curve_ax.set_title("Vp-depth Curves")
    curve_ax.set_xlabel("Vp (m/s)")
    curve_ax.set_ylabel("Depth sample")
    curve_ax.legend(fontsize=8)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    vp_min: float,
    vp_max: float,
    compute_ssim_flag: bool = True,
    visualization_dir: str | Path | None = None,
    max_visualization_batches: int = 0,
    max_visualization_samples: int = 2,
    curve_positions: list[float] | tuple[float, ...] = (0.25, 0.5, 0.75),
) -> dict[str, float]:
    model.eval()
    averager = MetricAverager()

    for batch_idx, batch in enumerate(loader):
        inputs = batch["input"].to(device, non_blocking=True)
        target_norm = batch["target"].to(device, non_blocking=True)
        vp_loss_mask = batch.get("vp_loss_mask")
        if vp_loss_mask is not None:
            vp_loss_mask = vp_loss_mask.to(device, non_blocking=True)
        pred_norm = _prediction_vp(model(inputs))
        pred_mps = denormalize_vp(pred_norm, vp_min, vp_max)
        target_mps = denormalize_vp(target_norm, vp_min, vp_max)
        metrics = compute_metrics(
            pred_mps, target_mps, mask=vp_loss_mask,
            compute_ssim=compute_ssim_flag)
        averager.update(metrics, n=inputs.shape[0])

        if visualization_dir is not None and batch_idx < max_visualization_batches:
            for sample_idx in range(min(inputs.shape[0], max_visualization_samples)):
                name = batch["name"][sample_idx]
                output_path = Path(visualization_dir) / f"{batch_idx:03d}_{sample_idx:02d}_{name}.png"
                save_prediction_figure(
                    inputs[sample_idx].cpu(),
                    target_mps[sample_idx].cpu(),
                    pred_mps[sample_idx].cpu(),
                    output_path,
                    sample_name=name,
                    curve_positions=curve_positions,
                )

    return averager.average()


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GPRInvNet-like shallow Vp inversion model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = build_datasets(config)
    dataset = datasets[args.split]
    loader = DataLoader(dataset, batch_size=config["train"].get("batch_size", 1), shuffle=False, num_workers=0)

    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state)

    output_dir = args.output_dir or str(Path(args.checkpoint).with_suffix("")) + "_eval"
    metrics = evaluate_model(
        model,
        loader,
        device,
        vp_min=float(config["data"]["vp_min"]),
        vp_max=float(config["data"]["vp_max"]),
        compute_ssim_flag=bool(config.get("eval", {}).get("ssim", True)),
        visualization_dir=output_dir,
        max_visualization_batches=999999,
        max_visualization_samples=999999,
        curve_positions=config.get("eval", {}).get("curve_positions", [0.25, 0.5, 0.75]),
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(output_dir) / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
