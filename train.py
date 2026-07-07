"""Training entry point for the GPRInvNet-like shallow Vp inversion baseline."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets, denormalize_vp
from evaluate import MetricAverager, compute_metrics, evaluate_model, save_prediction_figure
from losses import build_loss
from model_gprinvnet_like import build_model

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.sum += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    import yaml

    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.save_dir is not None:
        config["save_dir"] = args.save_dir
    if args.resume is not None:
        config["resume"] = args.resume
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    if args.data_root is not None:
        config["data"]["root_dir"] = args.data_root
    return config


def make_loader(dataset, config: dict[str, Any], shuffle: bool) -> DataLoader:
    train_cfg = config["train"]
    num_workers = int(train_cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=shuffle and len(dataset) > 1,
        num_workers=num_workers,
        pin_memory=bool(train_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _prediction_vp(prediction):
    if isinstance(prediction, dict):
        return prediction["vp"]
    return prediction


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: torch.device | str = "cpu",
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(_strip_module_prefix(state))
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return start_epoch, best_val_loss


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    meters = {key: AverageMeter() for key in ("total", "primary", "mse", "gradient")}
    use_amp = bool(config["train"].get("amp", False)) and device.type == "cuda"
    grad_clip_norm = config["train"].get("grad_clip_norm", None)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} train", leave=False)
    for batch in pbar:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        class_label = batch.get("class_label")
        if class_label is not None:
            class_label = class_label.to(device, non_blocking=True)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        vp_loss_mask = batch.get("vp_loss_mask")
        if vp_loss_mask is not None:
            vp_loss_mask = vp_loss_mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            preds = model(inputs)
            loss, parts = criterion(
                preds, targets, class_label=class_label, mask=mask,
                vp_loss_mask=vp_loss_mask)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()

        batch_size = inputs.shape[0]
        for key in parts:
            if key not in meters:
                meters[key] = AverageMeter()
        for key in parts:
            meters[key].update(parts[key], batch_size)
        pbar.set_postfix(loss=f"{meters['total'].avg:.4f}")

    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
    visualization_dir: Path | None,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    meters = {key: AverageMeter() for key in ("total", "primary", "mse", "gradient")}
    metric_averager = MetricAverager()
    vp_min = float(config["data"]["vp_min"])
    vp_max = float(config["data"]["vp_max"])
    curve_positions = config.get("eval", {}).get("curve_positions", [0.25, 0.5, 0.75])
    max_vis_batches = int(config["train"].get("visualize_batches", 1))
    max_vis_samples = int(config["train"].get("visualize_samples", 2))

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} val", leave=False)
    for batch_idx, batch in enumerate(pbar):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        class_label = batch.get("class_label")
        if class_label is not None:
            class_label = class_label.to(device, non_blocking=True)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        vp_loss_mask = batch.get("vp_loss_mask")
        if vp_loss_mask is not None:
            vp_loss_mask = vp_loss_mask.to(device, non_blocking=True)
        preds = model(inputs)
        loss, parts = criterion(
            preds, targets, class_label=class_label, mask=mask,
            vp_loss_mask=vp_loss_mask)

        batch_size = inputs.shape[0]
        for key in parts:
            if key not in meters:
                meters[key] = AverageMeter()
        for key in parts:
            meters[key].update(parts[key], batch_size)

        pred_mps = denormalize_vp(_prediction_vp(preds), vp_min, vp_max)
        target_mps = denormalize_vp(targets, vp_min, vp_max)
        metrics = compute_metrics(
            pred_mps,
            target_mps,
            mask=vp_loss_mask,
            compute_ssim=bool(config.get("eval", {}).get("ssim", True)),
        )
        metric_averager.update(metrics, batch_size)

        if visualization_dir is not None and batch_idx < max_vis_batches:
            for sample_idx in range(min(batch_size, max_vis_samples)):
                name = batch["name"][sample_idx]
                output_path = visualization_dir / f"epoch_{epoch:03d}_{batch_idx:03d}_{sample_idx:02d}_{name}.png"
                save_prediction_figure(
                    inputs[sample_idx].cpu(),
                    target_mps[sample_idx].cpu(),
                    pred_mps[sample_idx].cpu(),
                    output_path,
                    sample_name=name,
                    curve_positions=curve_positions,
                )

        pbar.set_postfix(loss=f"{meters['total'].avg:.4f}")

    return {key: meter.avg for key, meter in meters.items()}, metric_averager.average()


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPRInvNet-like shallow Vp inversion baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--data_root", default=None)
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config.get("seed", 42)))

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir / "resolved_config.yaml")
    if Path(args.config).exists():
        shutil.copy2(args.config, save_dir / "source_config.yaml")

    datasets = build_datasets(config)
    for split, dataset in datasets.items():
        groups = sorted({record.group_id for record in dataset.records})
        print(f"[Data] {split}: {len(dataset)} samples, groups={groups}")

    input_channels = len(config["data"].get("input_channels", ["processed_envelope"]))
    model_in_channels = int(config.get("model", {}).get("in_channels", input_channels))
    if model_in_channels != input_channels:
        warnings.warn(
            f"model.in_channels={model_in_channels} but data has {input_channels} channels; overriding model.in_channels.",
            RuntimeWarning,
        )
        config.setdefault("model", {})["in_channels"] = input_channels

    device_name = config.get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    print(f"[Model] parameters={sum(p.numel() for p in model.parameters()) / 1e6:.3f}M device={device}")
    if bool(config.get("semantic", {}).get("enabled", False)):
        train_dataset = datasets["train"]
        print(
            "[Semantic] enabled "
            f"classes={getattr(train_dataset, 'num_class_labels', 'unknown')} "
            f"ignore_index={getattr(train_dataset, 'class_ignore_index', -1)} "
            f"raw_to_train={getattr(train_dataset, 'raw_to_train_class', {})}")

    train_loader = make_loader(datasets["train"], config, shuffle=True)
    val_loader = make_loader(datasets["val"], config, shuffle=False) if len(datasets["val"]) else None
    test_loader = make_loader(datasets["test"], config, shuffle=False) if len(datasets["test"]) else None

    criterion = build_loss(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-3)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["train"].get("epochs", 1))),
        eta_min=float(config["train"].get("min_lr", 1e-6)),
    )
    scaler = GradScaler(enabled=bool(config["train"].get("amp", False)) and device.type == "cuda")
    writer = SummaryWriter(str(save_dir / "tensorboard")) if SummaryWriter is not None else None

    start_epoch = 1
    best_val_loss = float("inf")
    resume = config.get("resume")
    if resume:
        start_epoch, best_val_loss = load_checkpoint(resume, model, optimizer, scheduler, device)
        print(f"[Resume] {resume} from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    metrics_path = save_dir / "metrics.jsonl"
    epochs = int(config["train"].get("epochs", 1))
    if len(datasets["train"]) == 0:
        raise RuntimeError("Training split is empty.")

    for epoch in range(start_epoch, epochs + 1):
        tic = time.time()
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config)
        scheduler.step()

        val_losses: dict[str, float] = {}
        val_metrics: dict[str, float] = {}
        should_visualize = epoch % int(config["train"].get("visualize_interval", 1)) == 0
        vis_dir = save_dir / "visualizations" if should_visualize else None
        if val_loader is not None:
            val_losses, val_metrics = validate_one_epoch(model, val_loader, criterion, device, epoch, config, vis_dir)
            current_val_loss = val_losses["total"]
        else:
            warnings.warn("Validation split is empty; best_model.pth will follow training loss.", RuntimeWarning)
            current_val_loss = train_losses["total"]

        is_best = current_val_loss < best_val_loss
        if is_best:
            best_val_loss = current_val_loss
            save_checkpoint(save_dir / "best_model.pth", epoch, model, optimizer, scheduler, best_val_loss, config)
        save_checkpoint(save_dir / "last_model.pth", epoch, model, optimizer, scheduler, best_val_loss, config)

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "seconds": time.time() - tic,
            "train": train_losses,
            "val": val_losses,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
        }
        write_jsonl(metrics_path, row)

        if writer is not None:
            writer.add_scalar("loss/train_total", train_losses["total"], epoch)
            writer.add_scalar("lr", lr, epoch)
            for key, value in val_losses.items():
                writer.add_scalar(f"loss/val_{key}", value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"metrics/val_{key}", value, epoch)

        metric_text = " ".join(f"{k}={v:.3f}" for k, v in val_metrics.items())
        print(
            f"[Epoch {epoch:03d}] train_loss={train_losses['total']:.5f} "
            f"val_loss={current_val_loss:.5f} best={best_val_loss:.5f} lr={lr:.2e} {metric_text}"
        )

    if writer is not None:
        writer.close()

    if test_loader is not None and (save_dir / "best_model.pth").exists():
        load_checkpoint(save_dir / "best_model.pth", model, device=device)
        test_metrics = evaluate_model(
            model,
            test_loader,
            device,
            vp_min=float(config["data"]["vp_min"]),
            vp_max=float(config["data"]["vp_max"]),
            compute_ssim_flag=bool(config.get("eval", {}).get("ssim", True)),
            visualization_dir=save_dir / "test_visualizations",
            max_visualization_batches=999999,
            max_visualization_samples=int(config["train"].get("visualize_samples", 2)),
            curve_positions=config.get("eval", {}).get("curve_positions", [0.25, 0.5, 0.75]),
        )
        with (save_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"[Test] {test_metrics}")

    print(f"[Done] checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()
