from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import INaturalist


def parse_inaturalist_2021_folder_name(folder_name: str) -> tuple[str, str]:
    """
    Torchvision iNat2021 folder names look like:
        04237_Animalia_Arthropoda_Insecta_Hymenoptera_Apidae_Apis_mellifera
    Returns (species_binomial, taxonomy_line).
    """
    parts = folder_name.split("_")
    if len(parts) >= 8 and len(parts[0]) == 5 and parts[0].isdigit():
        *_, genus, species_ep = parts[1:8]
        species_binomial = f"{genus} {species_ep}"
        taxonomy_line = " > ".join(parts[1:8])
        return species_binomial, taxonomy_line
    return folder_name, folder_name


@dataclass
class INaturalistDatasetConfig:
    root: Path
    version: str = "2021_train_mini"
    download: bool = False
    image_transform: Optional[Callable[[Image.Image], Any]] = None


class INaturalistRAVTIDataset(Dataset):
    """iNaturalist (torchvision) exposed as dict samples for RAVTI."""

    def __init__(self, cfg: INaturalistDatasetConfig) -> None:
        self.cfg = cfg
        self._ds = INaturalist(
            root=str(cfg.root),
            version=cfg.version,
            target_type="full",
            transform=None,
            download=cfg.download,
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img, cat_id = self._ds[index]
        folder = self._ds.category_name("full", int(cat_id))
        species_text, taxonomy_line = parse_inaturalist_2021_folder_name(folder)
        if self.cfg.image_transform is not None:
            img = self.cfg.image_transform(img)
        return {
            "image": img,
            "species_text": species_text,
            "taxonomy_line": taxonomy_line,
            "dataset": "inaturalist",
            "sample_id": f"inaturalist:{index}",
            "raw_category_dir": folder,
            "index": index,
        }
