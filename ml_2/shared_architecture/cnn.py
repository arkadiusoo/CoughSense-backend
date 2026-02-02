from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class CNNConfig:
    input_channels: int
    num_classes: int
    conv_channels: Sequence[int] = (16, 32, 64)
    kernel_size: int = 3
    dropout: float = 0.3
    dense_units: int = 128
    pool_kernel: int = 2


class CNN2DClassifier(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.input_channels
        for out_channels in config.conv_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=config.kernel_size,
                        padding=config.kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(config.pool_kernel),
                ]
            )
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(in_channels, config.dense_units),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_units, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return self.classifier(x)
