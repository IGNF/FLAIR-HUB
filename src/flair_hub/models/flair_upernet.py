"""UPerNet pyramid pooling, feature fusion and auxiliary decoder heads."""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm(norm_type: str, channels: int) -> nn.Module:
    """Build one of the normalization layers supported by the project heads."""
    norm_name = norm_type.lower().replace("-", "_")
    if norm_name in {"sync_batchnorm", "syncbn"}:
        return nn.SyncBatchNorm(channels)
    if norm_name in {"batchnorm", "batchnorm2d", "bn"}:
        return nn.BatchNorm2d(channels)
    if norm_name in {"groupnorm", "group_norm", "gn"}:
        if channels < 2:
            raise ValueError("GroupNorm heads require at least two channels")
        num_groups = min(32, channels // 2)
        while channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, channels)
    raise ValueError(f"Unsupported decoder normalization: {norm_type}")


class ConvNormReLU(nn.Sequential):
    """Convolution, normalization and activation block for UPerNet heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: str,
        *,
        padding: int = 0,
        inplace: bool = True,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            build_norm(norm_type, out_channels),
            nn.ReLU(inplace=inplace),
        )


class PyramidPoolingModule(nn.ModuleList):
    """Pyramid pooling where every branch maps to the decoder channel width."""

    def __init__(
        self,
        pool_scales: Sequence[int],
        in_channels: int,
        channels: int,
        norm_type: str,
        align_corners: bool,
    ) -> None:
        super().__init__(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvNormReLU(
                        in_channels,
                        channels,
                        kernel_size=1,
                        norm_type=norm_type,
                    ),
                )
                for pool_scale in pool_scales
            ]
        )
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [
            F.interpolate(
                branch(x),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for branch in self
        ]


class FLAIRUPerHead(nn.Module):
    """FLAIR-HUB UPerNet head with a configurable wide pyramid topology."""

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int,
        num_classes: int,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        dropout: float = 0.1,
        norm_type: str = "sync_batchnorm",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("FLAIRUPerHead requires at least two encoder stages")

        self.in_channels = tuple(int(value) for value in in_channels)
        self.channels = int(channels)
        self.align_corners = align_corners

        self.psp_modules = PyramidPoolingModule(
            pool_scales=pool_scales,
            in_channels=self.in_channels[-1],
            channels=self.channels,
            norm_type=norm_type,
            align_corners=align_corners,
        )
        self.psp_bottleneck = ConvNormReLU(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm_type=norm_type,
        )

        self.lateral_convs = nn.ModuleList(
            [
                ConvNormReLU(
                    stage_channels,
                    self.channels,
                    kernel_size=1,
                    norm_type=norm_type,
                    inplace=False,
                )
                for stage_channels in self.in_channels[:-1]
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                ConvNormReLU(
                    self.channels,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_type=norm_type,
                    inplace=False,
                )
                for _ in self.in_channels[:-1]
            ]
        )
        self.fpn_bottleneck = ConvNormReLU(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm_type=norm_type,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(self.channels, num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Mirror MMCV defaults and its Normal(std=.01) classifier override."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def _select_features(
        self,
        features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Select the last encoder stages and validate their channel contract."""
        selected = list(features)[-len(self.in_channels) :]
        actual_channels = [feature.shape[1] for feature in selected]
        expected_channels = list(self.in_channels)
        if actual_channels != expected_channels:
            raise ValueError(
                "FLAIRUPerHead expected final feature channels "
                f"{expected_channels}, received {actual_channels}"
            )
        return selected

    def _psp_forward(self, deepest_feature: torch.Tensor) -> torch.Tensor:
        psp_outputs = [deepest_feature]
        psp_outputs.extend(self.psp_modules(deepest_feature))
        return self.psp_bottleneck(torch.cat(psp_outputs, dim=1))

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if len(features) == 1 and isinstance(features[0], (list, tuple)):
            features = tuple(features[0])
        inputs = self._select_features(features)

        laterals = [
            lateral_conv(inputs[index])
            for index, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self._psp_forward(inputs[-1]))

        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        fpn_outputs = [
            fpn_conv(laterals[index])
            for index, fpn_conv in enumerate(self.fpn_convs)
        ]
        fpn_outputs.append(laterals[-1])
        target_size = fpn_outputs[0].shape[-2:]

        fpn_outputs = [
            output
            if output.shape[-2:] == target_size
            else F.interpolate(
                output,
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for output in fpn_outputs
        ]
        fused = self.fpn_bottleneck(torch.cat(fpn_outputs, dim=1))
        return self.classifier(self.dropout(fused))


class FLAIRUPerDecoder(nn.Module):
    """Adapter preserving the ``.seg_model(*features)`` project interface."""

    architecture_name = "upernet_flair_hub"

    def __init__(self, **head_kwargs) -> None:
        super().__init__()
        self.seg_model = FLAIRUPerHead(**head_kwargs)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        return self.seg_model(*features)
