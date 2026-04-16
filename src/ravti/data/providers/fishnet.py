from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


@dataclass
class FishNetDatasetConfig:
    root: Path
    layout: str = "imagefolder"  # imagefolder | manifest_csv
    manifest_csv: Optional[Path] = None
    image_column: str = "image_path"
    species_column: str = "species"
    taxonomy_column: Optional[str] = None
    image_transform: Optional[Callable[[Image.Image], Any]] = None


def _species_from_class_folder(name: str) -> str:
    return name.replace("_", " ")


class FishNetImageFolderDataset(Dataset):
    """FishNet as torchvision ImageFolder: each class directory = one species label."""

    def __init__(self, root: Path, image_transform: Optional[Callable[[Image.Image], Any]] = None) -> None:
        self._ds = ImageFolder(str(root))
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, target = self._ds.samples[index]
        img = Image.open(path).convert("RGB")
        class_name = self._ds.classes[target]
        species_text = _species_from_class_folder(class_name)
        if self.image_transform is not None:
            img = self.image_transform(img)
        return {
            "image": img,
            "species_text": species_text,
            "taxonomy_line": species_text,
            "dataset": "fishnet",
            "image_path": str(path),
            "index": index,
        }


class FishNetManifestCSVDataset(Dataset):
    """FishNet from CSV manifest: paths + species (+ optional taxonomy)."""

    def __init__(self, cfg: FishNetDatasetConfig) -> None:
        if cfg.manifest_csv is None:
            raise ValueError("manifest_csv is required for layout=manifest_csv")
        self.cfg = cfg
        self.rows: list[dict[str, str]] = []
        with cfg.manifest_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rel = row[self.cfg.image_column]
        path = Path(rel)
        if not path.is_absolute():
            path = self.cfg.root / path
        img = Image.open(path).convert("RGB")
        species = row[self.cfg.species_column].strip()
        tax_line = row[self.cfg.taxonomy_column].strip() if self.cfg.taxonomy_column and self.cfg.taxonomy_column in row else species
        if self.cfg.image_transform is not None:
            img = self.cfg.image_transform(img)
        return {
            "image": img,
            "species_text": species,
            "taxonomy_line": tax_line,
            "dataset": "fishnet",
            "image_path": str(path),
            "index": index,
        }


def build_fishnet_dataset(cfg: FishNetDatasetConfig) -> Dataset:
    layout = (cfg.layout or "imagefolder").lower()
    if layout == "imagefolder":
        return FishNetImageFolderDataset(cfg.root, image_transform=cfg.image_transform)
    if layout == "manifest_csv":
        return FishNetManifestCSVDataset(cfg)
    raise ValueError(f"Unknown fishnet layout: {layout}")
