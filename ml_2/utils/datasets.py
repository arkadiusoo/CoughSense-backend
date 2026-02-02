from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import SampleRecord


@dataclass(frozen=True)
class ImageConfig:
    image_size: tuple[int, int]


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        label_to_index: dict[str, int],
        image_config: ImageConfig,
    ) -> None:
        self.records = list(records)
        self.label_to_index = label_to_index
        self.image_config = image_config
        self.sample_ids = [record.sample_id for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        if record.image_path is None:
            raise ValueError(f"Missing image for sample {record.sample_id}")
        image = _load_image(record.image_path, self.image_config.image_size)
        label = self.label_to_index[record.label]
        return image, label


def _load_image(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image_size:
        image = image.resize(image_size, Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor
