"""Loss functions for normalized Vp inversion and optional facies labels."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """L1 loss on vertical and lateral finite differences."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dz = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dz = target[:, :, 1:, :] - target[:, :, :-1, :]
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        loss_z = F.l1_loss(pred_dz, target_dz, reduction=self.reduction)
        loss_x = F.l1_loss(pred_dx, target_dx, reduction=self.reduction)
        return loss_z + loss_x


class MultiScaleSSIMLoss(nn.Module):
    """Differentiable 1 - MS-SSIM for normalized Vp images.

    The implementation uses local average pooling instead of a fixed Gaussian
    kernel to avoid an extra dependency. Inputs are expected to be normalized to
    roughly [0, 1], matching this project's Vp targets.
    """

    def __init__(
        self,
        scales: int = 3,
        window_size: int = 11,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.scales = max(1, int(scales))
        self.window_size = max(3, int(window_size) | 1)
        self.data_range = float(data_range)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = []
        x = pred
        y = target
        current_mask = mask
        for _ in range(self.scales):
            scores.append(self._ssim(x, y, current_mask))
            if min(x.shape[-2:]) < 4:
                break
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
            if current_mask is not None:
                current_mask = F.avg_pool2d(
                    current_mask.float(), kernel_size=2, stride=2)
                current_mask = (current_mask > 0.5).to(x.dtype)
        return 1.0 - torch.stack(scores).mean().clamp(0.0, 1.0)

    def _ssim(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        window = min(self.window_size, x.shape[-2], x.shape[-1])
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return 1.0 - F.l1_loss(x, y)
        padding = window // 2
        mu_x = F.avg_pool2d(
            x, kernel_size=window, stride=1, padding=padding,
            count_include_pad=False)
        mu_y = F.avg_pool2d(
            y, kernel_size=window, stride=1, padding=padding,
            count_include_pad=False)
        sigma_x = F.avg_pool2d(
            x * x, kernel_size=window, stride=1, padding=padding,
            count_include_pad=False) - mu_x * mu_x
        sigma_y = F.avg_pool2d(
            y * y, kernel_size=window, stride=1, padding=padding,
            count_include_pad=False) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(
            x * y, kernel_size=window, stride=1, padding=padding,
            count_include_pad=False) - mu_x * mu_y

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim_map = numerator / torch.clamp(denominator, min=1e-12)
        if mask is None:
            return ssim_map.mean()
        if tuple(mask.shape) != tuple(ssim_map.shape):
            mask = mask.expand_as(ssim_map)
        denom = mask.sum()
        if denom <= 0:
            return ssim_map.mean()
        return torch.sum(ssim_map * mask) / denom


class VelocityLoss(nn.Module):
    """Baseline Vp loss plus optional semantic-facies auxiliary loss."""

    def __init__(
        self,
        primary: str = "smooth_l1",
        l1_weight: float = 1.0,
        mse_weight: float = 0.5,
        gradient_weight: float = 0.1,
        semantic_weight: float = 0.0,
        semantic_ignore_index: int = -1,
        semantic_require_labels: bool = False,
        semantic_use_visibility_weight: bool = True,
        visibility_mask_channel: int = 8,
        uncertainty_mask_channel: int = 9,
        uncertainty_weight: float = 0.5,
        use_vp_mask: bool = True,
        msssim_weight: float = 0.0,
        msssim_scales: int = 3,
        msssim_window_size: int = 11,
    ) -> None:
        super().__init__()
        if primary not in ("smooth_l1", "l1"):
            raise ValueError(f"Unsupported primary loss: {primary}")
        self.primary = primary
        self.l1_weight = float(l1_weight)
        self.mse_weight = float(mse_weight)
        self.gradient_weight = float(gradient_weight)
        self.semantic_weight = float(semantic_weight)
        self.semantic_ignore_index = int(semantic_ignore_index)
        self.semantic_require_labels = bool(semantic_require_labels)
        self.semantic_use_visibility_weight = bool(
            semantic_use_visibility_weight)
        self.visibility_mask_channel = int(visibility_mask_channel)
        self.uncertainty_mask_channel = int(uncertainty_mask_channel)
        self.uncertainty_weight = float(uncertainty_weight)
        self.use_vp_mask = bool(use_vp_mask)
        self.msssim_weight = float(msssim_weight)
        self.msssim = (
            MultiScaleSSIMLoss(
                scales=int(msssim_scales),
                window_size=int(msssim_window_size),
                data_range=1.0,
            )
            if self.msssim_weight > 0.0 else None
        )

    @staticmethod
    def velocity_prediction(pred) -> torch.Tensor:
        if isinstance(pred, dict):
            if "vp" not in pred:
                raise KeyError("Model output dict must contain key 'vp'")
            return pred["vp"]
        return pred

    def forward(
        self,
        pred,
        target: torch.Tensor,
        class_label: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        vp_loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred_vp = self.velocity_prediction(pred)
        vp_mask = (
            self._prepare_vp_mask(vp_loss_mask, target)
            if self.use_vp_mask and vp_loss_mask is not None else None)
        primary = self._primary_loss(pred_vp, target, vp_mask)
        mse = self._mse_loss(pred_vp, target, vp_mask)
        grad = self._gradient_loss(pred_vp, target, vp_mask)
        total = self.l1_weight * primary + self.mse_weight * mse + self.gradient_weight * grad
        parts = {
            "primary": float(primary.detach().cpu()),
            "mse": float(mse.detach().cpu()),
            "gradient": float(grad.detach().cpu()),
        }
        if vp_mask is not None:
            parts["vp_mask_fraction"] = float(
                vp_mask.detach().float().mean().cpu())
        if self.msssim is not None:
            msssim = self.msssim(pred_vp, target, vp_mask)
            total = total + self.msssim_weight * msssim
            parts["msssim"] = float(msssim.detach().cpu())
            parts["msssim_weighted"] = float(
                (self.msssim_weight * msssim).detach().cpu())
        semantic = self._semantic_loss(pred, class_label, mask)
        if semantic is not None:
            total = total + self.semantic_weight * semantic
            parts["semantic"] = float(semantic.detach().cpu())
            parts["semantic_weighted"] = float(
                (self.semantic_weight * semantic).detach().cpu())
        parts["total"] = float(total.detach().cpu())
        return total, parts

    def _elementwise_primary(
            self, pred_vp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.primary == "smooth_l1":
            return F.smooth_l1_loss(pred_vp, target, reduction="none")
        return torch.abs(pred_vp - target)

    @staticmethod
    def _masked_mean(values: torch.Tensor,
                     mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return values.mean()
        mask = mask.to(device=values.device, dtype=values.dtype)
        if tuple(mask.shape) != tuple(values.shape):
            mask = mask.expand_as(values)
        denom = mask.sum()
        if denom <= 0:
            return values.sum() * 0.0
        return torch.sum(values * mask) / denom

    def _primary_loss(
            self, pred_vp: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor | None) -> torch.Tensor:
        return self._masked_mean(
            self._elementwise_primary(pred_vp, target), mask)

    def _mse_loss(
            self, pred_vp: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor | None) -> torch.Tensor:
        return self._masked_mean((pred_vp - target) ** 2, mask)

    def _gradient_loss(
            self, pred_vp: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor | None) -> torch.Tensor:
        pred_dz = pred_vp[:, :, 1:, :] - pred_vp[:, :, :-1, :]
        target_dz = target[:, :, 1:, :] - target[:, :, :-1, :]
        pred_dx = pred_vp[:, :, :, 1:] - pred_vp[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        if mask is None:
            return (
                F.l1_loss(pred_dz, target_dz)
                + F.l1_loss(pred_dx, target_dx))
        mask = mask.to(device=pred_vp.device, dtype=pred_vp.dtype)
        if tuple(mask.shape) != tuple(pred_vp.shape):
            mask = mask.expand_as(pred_vp)
        mask_dz = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        return (
            self._masked_mean(torch.abs(pred_dz - target_dz), mask_dz)
            + self._masked_mean(torch.abs(pred_dx - target_dx), mask_dx))

    @staticmethod
    def _prepare_vp_mask(
            mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(
                f"vp_loss_mask must be [B,1,H,W] or [B,H,W], got {tuple(mask.shape)}")
        mask = mask.to(device=target.device, dtype=target.dtype)
        if tuple(mask.shape[-2:]) != tuple(target.shape[-2:]):
            mask = F.interpolate(
                mask, size=tuple(target.shape[-2:]), mode="nearest")
        if mask.shape[1] == 1 and target.shape[1] != 1:
            mask = mask.expand(-1, target.shape[1], -1, -1)
        elif mask.shape[1] != target.shape[1]:
            raise ValueError(
                "vp_loss_mask channel count must be 1 or match target")
        return (mask > 0.5).to(target.dtype)

    def _semantic_loss(
        self,
        pred,
        class_label: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.semantic_weight <= 0.0:
            return None
        class_logits = pred.get("class_logits") if isinstance(pred, dict) else None
        if class_logits is None:
            if self.semantic_require_labels:
                raise RuntimeError(
                    "semantic.enabled is true but model produced no class_logits")
            return None
        if class_label is None:
            if self.semantic_require_labels:
                raise RuntimeError(
                    "semantic.enabled is true but batch has no class_label")
            return None
        if class_label.ndim == 4 and class_label.shape[1] == 1:
            labels = class_label[:, 0].long()
        elif class_label.ndim == 3:
            labels = class_label.long()
        else:
            raise ValueError(
                f"class_label must be [B,1,H,W] or [B,H,W], got {tuple(class_label.shape)}")
        if tuple(class_logits.shape[-2:]) != tuple(labels.shape[-2:]):
            class_logits = F.interpolate(
                class_logits,
                size=tuple(labels.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
        valid = labels != self.semantic_ignore_index
        if not torch.any(valid):
            return class_logits.sum() * 0.0
        per_pixel = F.cross_entropy(
            class_logits, labels,
            ignore_index=self.semantic_ignore_index,
            reduction="none",
        )
        weights = valid.to(per_pixel.dtype)
        if self.semantic_use_visibility_weight and mask is not None:
            if mask.ndim != 4:
                raise ValueError(f"mask must be [B,M,H,W], got {tuple(mask.shape)}")
            if tuple(mask.shape[-2:]) != tuple(labels.shape[-2:]):
                mask = F.interpolate(
                    mask.float(),
                    size=tuple(labels.shape[-2:]),
                    mode="nearest",
                )
            if mask.shape[1] > self.visibility_mask_channel:
                visibility = mask[:, self.visibility_mask_channel].float()
                if float(visibility.detach().max().cpu()) > 1.0:
                    visibility = visibility / 255.0
                weights = weights * visibility.clamp(0.0, 1.0)
            if mask.shape[1] > self.uncertainty_mask_channel:
                uncertain = mask[:, self.uncertainty_mask_channel] > 0.5
                weights = weights * torch.where(
                    uncertain,
                    torch.full_like(weights, self.uncertainty_weight),
                    torch.ones_like(weights),
                )
        weight_sum = torch.sum(weights)
        if weight_sum <= 0:
            return per_pixel[valid].mean()
        return torch.sum(per_pixel * weights) / weight_sum


def build_loss(config: dict) -> VelocityLoss:
    loss_cfg = config.get("loss", {})
    semantic_cfg = config.get("semantic", {})
    data_class_cfg = config.get("data", {}).get("class_label", {})
    semantic_enabled = bool(semantic_cfg.get("enabled", False))
    return VelocityLoss(
        primary=loss_cfg.get("primary", "smooth_l1"),
        l1_weight=float(loss_cfg.get("l1_weight", 1.0)),
        mse_weight=float(loss_cfg.get("mse_weight", 0.5)),
        gradient_weight=float(loss_cfg.get("gradient_weight", 0.1)),
        semantic_weight=(
            float(semantic_cfg.get("weight", 0.2))
            if semantic_enabled else 0.0),
        semantic_ignore_index=int(semantic_cfg.get(
            "ignore_index", data_class_cfg.get("ignore_index", -1))),
        semantic_require_labels=bool(
            semantic_cfg.get("require_labels", semantic_enabled)),
        semantic_use_visibility_weight=bool(
            semantic_cfg.get("use_visibility_weight", True)),
        visibility_mask_channel=int(
            semantic_cfg.get("visibility_mask_channel", 8)),
        uncertainty_mask_channel=int(
            semantic_cfg.get("uncertainty_mask_channel", 9)),
        uncertainty_weight=float(
            semantic_cfg.get("uncertainty_weight", 0.5)),
        use_vp_mask=bool(loss_cfg.get("use_vp_mask", True)),
        msssim_weight=float(loss_cfg.get("msssim_weight", 0.0)),
        msssim_scales=int(loss_cfg.get("msssim_scales", 3)),
        msssim_window_size=int(loss_cfg.get("msssim_window_size", 11)),
    )
