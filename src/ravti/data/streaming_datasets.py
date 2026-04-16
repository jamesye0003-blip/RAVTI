from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from datasets import IterableDataset, load_dataset
from PIL import Image


@dataclass
class HfStreamConfig:
    dataset_id: str
    split: str
    image_column: str
    label_column: str
    taxonomy_column: Optional[str]
    streaming: bool = True
    config_name: Optional[str] = None


class HfStreamingImageIterable:
    """Thin wrapper around `datasets.load_dataset(..., streaming=True)`."""

    def __init__(self, cfg: HfStreamConfig) -> None:
        self.cfg = cfg
        self._ds: IterableDataset = load_dataset(
            cfg.dataset_id,
            name=cfg.config_name,
            split=cfg.split,
            streaming=cfg.streaming,
        )

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        for row in self._ds:
            yield row

    def map_image_label(self, row: dict[str, Any]) -> tuple[Image.Image, str, Optional[str]]:
        img = row[self.cfg.image_column]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        label = row[self.cfg.label_column]
        taxon: Optional[str] = None
        if self.cfg.taxonomy_column and self.cfg.taxonomy_column in row:
            taxon = str(row[self.cfg.taxonomy_column])
        species = taxon or str(label)
        return img, str(label), taxon


def synthetic_demo_stream(
    n: int,
    size: tuple[int, int] = (256, 256),
) -> Iterator[dict[str, Any]]:
    """Offline-friendly iterator mimicking HF rows (no download)."""
    for i in range(n):
        yield {
            "image": Image.new("RGB", size, color=(i * 7 % 255, 128, 64)),
            "labels": f"synthetic_species_{i % 17}",
            "taxonomy": f"Animalia;Chordata;species_{i % 17}",
        }
