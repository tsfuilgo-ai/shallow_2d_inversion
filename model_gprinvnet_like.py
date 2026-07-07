"""GPRInvNet-style baseline for shallow acoustic Vp inversion.

The original GPRInvNet maps GPR B-scan data to a subsurface property image
with a trace-to-trace encoder. This implementation keeps the same important
idea for shallow velocity inversion:

  input [B, C, T, R]
    -> preserve trace order and fuse neighboring traces with 5x5 convolutions
    -> compress each trace's time samples with a shared MLP
    -> decode the aligned latent traces to a normalized Vp image [B, 1, H, W]

An optional semantic decoder can share the same latent representation and
produce facies logits for the existing auxiliary loss.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm2d(kind: str, channels: int) -> nn.Module:
    kind = str(kind).lower()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"Unsupported 2D norm: {kind}")


class ConvNormReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, tuple):
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            _norm2d(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TraceNeighborEncoder(nn.Module):
    """Neighbor-trace feature fusion from the GPRInvNet encoder.

    The model receives tensors as [B, C, T, R]. Internally it uses [B, C, R, T]
    so that the convolution height is trace index and width is time sample.
    Five 5x5 stride-1 layers preserve the B-scan geometry while expanding
    feature channels, matching the open-source GPRInvNet design.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (4, 8, 16, 32, 64),
        norm: str = "batch",
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("encoder channels must not be empty")
        layers = []
        prev = int(in_channels)
        for out_ch in channels:
            layers.append(
                ConvNormReLU(prev, int(out_ch), kernel_size=5, norm=norm)
            )
            prev = int(out_ch)
        self.layers = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PerTraceMLPCompressor(nn.Module):
    """Shared per-trace MLP that compresses time to latent depth.

    This mirrors the open-source Generator module:
    Linear(time -> hidden...) is applied to every trace independently while
    the trace axis is kept intact. BatchNorm1d over the configured trace count
    follows the original implementation closely and makes the model fixed-size,
    which is acceptable because the dataset already resizes windows.
    """

    def __init__(
        self,
        input_time_samples: int,
        input_traces: int,
        latent_depth: int,
        hidden_dims: Sequence[int] = (1024, 512, 256, 256),
        norm: str = "trace_batch",
    ) -> None:
        super().__init__()
        self.input_time_samples = int(input_time_samples)
        self.input_traces = int(input_traces)
        self.latent_depth = int(latent_depth)
        self.norm = str(norm).lower()
        if self.input_time_samples <= 0:
            raise ValueError("input_time_samples must be positive")
        if self.input_traces <= 0:
            raise ValueError("input_traces must be positive")
        if self.latent_depth <= 0:
            raise ValueError("latent_depth must be positive")

        dims = [self.input_time_samples, *[int(v) for v in hidden_dims], self.latent_depth]
        self.linears = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.norms = nn.ModuleList(
            self._make_norm() for _ in range(len(self.linears))
        )
        self.relu = nn.ReLU(inplace=True)

    def _make_norm(self) -> nn.Module:
        if self.norm in ("trace_batch", "batch"):
            return nn.BatchNorm1d(self.input_traces)
        if self.norm == "layer":
            return nn.Identity()
        if self.norm in ("none", "identity", ""):
            return nn.Identity()
        raise ValueError(f"Unsupported compressor norm: {self.norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,F,R,T], got {tuple(x.shape)}")
        batch, features, traces, time_samples = x.shape
        if time_samples != self.input_time_samples or traces != self.input_traces:
            raise ValueError(
                "PerTraceMLPCompressor received shape "
                f"[R={traces}, T={time_samples}], expected "
                f"[R={self.input_traces}, T={self.input_time_samples}]. "
                "Check data.target_time_samples/data.target_traces or enable "
                "model.auto_resize_input."
            )

        y = x.reshape(batch * features, traces, time_samples)
        for linear, norm in zip(self.linears, self.norms):
            y = linear(y)
            if self.norm == "layer":
                y = F.layer_norm(y, y.shape[-1:])
            else:
                y = norm(y)
            y = self.relu(y)
        y = y.transpose(1, 2).contiguous()
        return y.view(batch, features, self.latent_depth, traces)


class SpatialDecoder(nn.Module):
    """Convolutional decoder used for Vp and optional semantic logits."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        target_depth: int,
        target_width: int,
        stem_channels: int = 128,
        decoder_channels: Iterable[int] = (64, 64, 32, 32),
        upsample_mode: str = "depth_only",
        norm: str = "batch",
        dropout: float = 0.2,
        output_activation: str | None = "sigmoid",
    ) -> None:
        super().__init__()
        self.target_depth = int(target_depth)
        self.target_width = int(target_width)
        self.upsample_mode = str(upsample_mode).lower()
        self.output_activation = output_activation
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()

        if self.upsample_mode == "both":
            first: nn.Module = nn.ConvTranspose2d(
                in_channels,
                stem_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        elif self.upsample_mode == "depth_only":
            first = nn.ConvTranspose2d(
                in_channels,
                stem_channels,
                kernel_size=(4, 3),
                stride=(2, 1),
                padding=(1, 1),
            )
        elif self.upsample_mode in ("none", "identity"):
            first = nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            raise ValueError(f"Unsupported decoder upsample_mode: {upsample_mode}")

        self.stem = nn.Sequential(
            first,
            _norm2d(norm, stem_channels),
            nn.ReLU(inplace=True),
            ConvNormReLU(stem_channels, stem_channels, kernel_size=3, norm=norm),
        )

        layers = []
        prev = int(stem_channels)
        for out_ch in decoder_channels:
            layers.append(ConvNormReLU(prev, int(out_ch), kernel_size=3, norm=norm))
            prev = int(out_ch)
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(prev, int(out_channels), kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if tuple(x.shape[-2:]) != (self.target_depth, self.target_width):
            x = F.interpolate(
                x,
                size=(self.target_depth, self.target_width),
                mode="bilinear",
                align_corners=False,
            )
        x = self.dropout(x)
        x = self.body(x)
        x = self.head(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "relu":
            return torch.relu(x)
        if self.output_activation == "softplus":
            return F.softplus(x)
        if self.output_activation in ("none", None):
            return x
        raise ValueError(f"Unsupported output_activation: {self.output_activation}")


class GPRInvNetLike(nn.Module):
    """Trace-to-trace encoder, shared trace MLP, and spatial decoder."""

    def __init__(
        self,
        in_channels: int = 1,
        input_time_samples: int = 800,
        input_traces: int = 256,
        target_depth: int = 256,
        target_width: int = 256,
        latent_depth: int | None = None,
        encoder_channels: Sequence[int] = (4, 8, 16, 32, 64),
        mlp_hidden_dims: Sequence[int] = (1024, 512, 256, 256),
        decoder_stem_channels: int = 128,
        decoder_channels: Sequence[int] = (64, 64, 32, 32),
        decoder_upsample_mode: str = "depth_only",
        encoder_norm: str = "batch",
        decoder_norm: str = "batch",
        compressor_norm: str = "trace_batch",
        dropout: float = 0.2,
        output_activation: str = "sigmoid",
        auto_resize_input: bool = False,
        semantic_num_classes: int | None = None,
        semantic_decoder_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.input_time_samples = int(input_time_samples)
        self.input_traces = int(input_traces)
        self.target_depth = int(target_depth)
        self.target_width = int(target_width)
        self.auto_resize_input = bool(auto_resize_input)
        if latent_depth is None:
            latent_depth = math.ceil(self.target_depth / 2)
        self.latent_depth = int(latent_depth)

        self.encoder = TraceNeighborEncoder(
            in_channels=self.in_channels,
            channels=encoder_channels,
            norm=encoder_norm,
        )
        self.trace_compressor = PerTraceMLPCompressor(
            input_time_samples=self.input_time_samples,
            input_traces=self.input_traces,
            latent_depth=self.latent_depth,
            hidden_dims=mlp_hidden_dims,
            norm=compressor_norm,
        )
        self.decoder = SpatialDecoder(
            in_channels=self.encoder.out_channels,
            out_channels=1,
            target_depth=self.target_depth,
            target_width=self.target_width,
            stem_channels=int(decoder_stem_channels),
            decoder_channels=decoder_channels,
            upsample_mode=decoder_upsample_mode,
            norm=decoder_norm,
            dropout=dropout,
            output_activation=output_activation,
        )

        self.semantic_decoder = None
        if semantic_num_classes is not None and int(semantic_num_classes) > 1:
            self.semantic_decoder = SpatialDecoder(
                in_channels=self.encoder.out_channels,
                out_channels=int(semantic_num_classes),
                target_depth=self.target_depth,
                target_width=self.target_width,
                stem_channels=int(decoder_stem_channels),
                decoder_channels=(
                    tuple(semantic_decoder_channels)
                    if semantic_decoder_channels is not None
                    else tuple(decoder_channels)
                ),
                upsample_mode=decoder_upsample_mode,
                norm=decoder_norm,
                dropout=dropout,
                output_activation="none",
            )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,C,T,R], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        expected_size = (self.input_time_samples, self.input_traces)
        if tuple(x.shape[-2:]) != expected_size:
            if not self.auto_resize_input:
                raise ValueError(
                    f"Expected input size [T,R]={expected_size}, got "
                    f"{tuple(x.shape[-2:])}. Set model.auto_resize_input=true "
                    "or align data.target_time_samples/data.target_traces."
                )
            x = F.interpolate(
                x,
                size=expected_size,
                mode="bilinear",
                align_corners=False,
            )
        return x.transpose(2, 3).contiguous()

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        x = self.encoder(x)
        return self.trace_compressor(x)

    def forward(self, x: torch.Tensor):
        features = self.encode_features(x)
        vp = self.decoder(features)
        if self.semantic_decoder is None:
            return vp
        return {
            "vp": vp,
            "class_logits": self.semantic_decoder(features),
        }


def _resolve_semantic_classes(config: dict) -> int:
    data_cfg = config.get("data", {})
    semantic_cfg = config.get("semantic", {})
    class_cfg = data_cfg.get("class_label", {})
    raw_ids = class_cfg.get("raw_ids", "marine_facies_v1")
    if raw_ids in (None, "", "marine_facies_v1", "all"):
        inferred = 27
    elif isinstance(raw_ids, str):
        inferred = len([part for part in raw_ids.split(",") if part.strip()])
    else:
        inferred = len(raw_ids)
    return int(semantic_cfg.get("num_classes", inferred))


def build_model(config: dict) -> GPRInvNetLike:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    semantic_cfg = config.get("semantic", {})
    semantic_enabled = bool(semantic_cfg.get("enabled", False))

    target_depth = int(data_cfg.get("target_depth", 256))
    upsample_mode = model_cfg.get("decoder_upsample_mode", "depth_only")
    latent_depth = model_cfg.get("latent_depth")
    if latent_depth is None:
        latent_depth = (
            math.ceil(target_depth / 2)
            if str(upsample_mode).lower() in ("both", "depth_only")
            else target_depth
        )

    return GPRInvNetLike(
        in_channels=int(
            model_cfg.get(
                "in_channels",
                len(data_cfg.get("input_channels", ["processed_envelope"])),
            )
        ),
        input_time_samples=int(
            model_cfg.get(
                "input_time_samples",
                data_cfg.get("target_time_samples", 800),
            )
        ),
        input_traces=int(
            model_cfg.get("input_traces", data_cfg.get("target_traces", 256))
        ),
        target_depth=target_depth,
        target_width=int(
            data_cfg.get("target_width", data_cfg.get("target_traces", 256))
        ),
        latent_depth=int(latent_depth),
        encoder_channels=tuple(model_cfg.get("encoder_channels", (4, 8, 16, 32, 64))),
        mlp_hidden_dims=tuple(model_cfg.get("mlp_hidden_dims", (1024, 512, 256, 256))),
        decoder_stem_channels=int(model_cfg.get("decoder_stem_channels", 128)),
        decoder_channels=tuple(model_cfg.get("decoder_channels", (64, 64, 32, 32))),
        decoder_upsample_mode=upsample_mode,
        encoder_norm=model_cfg.get("encoder_norm", "batch"),
        decoder_norm=model_cfg.get("decoder_norm", "batch"),
        compressor_norm=model_cfg.get("compressor_norm", "trace_batch"),
        dropout=float(model_cfg.get("dropout", 0.2)),
        output_activation=model_cfg.get("output_activation", "sigmoid"),
        auto_resize_input=bool(model_cfg.get("auto_resize_input", False)),
        semantic_num_classes=(
            _resolve_semantic_classes(config)
            if semantic_enabled else None
        ),
        semantic_decoder_channels=semantic_cfg.get("decoder_channels"),
    )


if __name__ == "__main__":
    model = GPRInvNetLike(
        in_channels=1,
        input_time_samples=800,
        input_traces=256,
        target_depth=256,
        target_width=256,
        semantic_num_classes=27,
    )
    x = torch.randn(2, 1, 800, 256)
    y = model(x)
    vp = y["vp"] if isinstance(y, dict) else y
    params = sum(p.numel() for p in model.parameters())
    print(f"input={tuple(x.shape)} output={tuple(vp.shape)} params={params}")
