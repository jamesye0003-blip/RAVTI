from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchvision.transforms as T

from ravti.config import resolve_paths
from ravti.data.providers import (
    FishNetDatasetConfig,
    INaturalistDatasetConfig,
    INaturalistRAVTIDataset,
    ManifestDatasetConfig,
    ManifestRAVTIDataset,
    TreeOfLifeDatasetConfig,
    TreeOfLifeStreamingDataset,
    build_fishnet_dataset,
)
from ravti.paths import project_root as repo_root
from ravti.taxonomy_text import canonical_taxonomy_line


def _image_transform(image_size: int) -> T.Compose:
    """Create the image transform"""
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ]
    )


class _SyntheticRAVTIDataset(Dataset):
    """Tiny offline dataset for CI / import checks (set env ``RAVTI_SYNTHETIC_DATA=1``)."""

    def __init__(self, n: int, image_size: int) -> None:
        """Initialize the synthetic dataset"""
        self.n = max(int(n), 1)
        self.image_size = int(image_size)
        self.tf = _image_transform(self.image_size)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, Any]:
        i = int(index) % self.n
        pil = Image.new("RGB", (self.image_size, self.image_size), (i * 7 % 255, 90, 120))
        return {
            "image": self.tf(pil),
            "species_text": f"synthetic_species_{i % 17}",
            "taxonomy_line": canonical_taxonomy_line(f"Animalia;Chordata;synthetic_{i % 17}", f"synthetic_species_{i % 17}"),
            "dataset": "synthetic",
            "index": i,
        }


def build_ravti_dataset(cfg: dict) -> Union[Dataset, IterableDataset]:
    """
    Construct the training dataset selected by `cfg["dataset"]["provider"]`.

    Supported providers:
      - ``inaturalist_mini`` (default): torchvision ``INaturalist`` ``2021_train_mini``
      - ``fishnet``: local ImageFolder or CSV manifest
      - ``treeoflife_10m``: HuggingFace streaming (large; use with care)
    """
    # Get the dataset configuration
    ds_cfg = cfg.get("dataset") or {}
    if os.environ.get("RAVTI_SYNTHETIC_DATA") == "1":
        train_cfg = cfg.get("training") or {}
        image_size = int(train_cfg.get("image_size", 512))
        n = int((ds_cfg.get("synthetic") or {}).get("num_samples", 128))
        return _SyntheticRAVTIDataset(n=n, image_size=image_size)
    provider = str(ds_cfg.get("provider", "inaturalist_mini")).lower()  # Get the provider
    paths = resolve_paths(cfg)  # Resolve the paths
    train_cfg = cfg.get("training") or {}
    image_size = int(train_cfg.get("image_size", 512))
    transform = _image_transform(image_size)

    # Get the split configuration
    split_cfg = ds_cfg.get("split") or {}
    split_manifest_raw = split_cfg.get("train_manifest_jsonl")

    # ------------------------------------------------------------ Build the datasets ------------------------------------------------------------
    if split_manifest_raw:  # Manifest split wins: ignore dataset.provider for the training file list.
        split_manifest = Path(str(split_manifest_raw))
        if not split_manifest.is_absolute():
            split_manifest = (repo_root() / split_manifest).resolve()
        return ManifestRAVTIDataset(
            ManifestDatasetConfig(
                manifest_jsonl=split_manifest,
                image_transform=transform,
            )
        )
    else:  # No train manifest: build from dataset.provider
        if provider in ("inaturalist", "inaturalist_mini", "inat_mini", "inat2021_mini"):  # If the provider is inaturalist, build the inaturalist dataset
            icfg = ds_cfg.get("inaturalist") or {}
            raw_root = icfg.get("root")
            if raw_root is None:
                root = (paths.data_root / "datasets" / "inaturalist").resolve()
            else:
                root = Path(raw_root)
                if not root.is_absolute():
                    root = (repo_root() / root).resolve()
            return INaturalistRAVTIDataset(
                INaturalistDatasetConfig(
                    root=root,
                    version=str(icfg.get("version", "2021_train_mini")),
                    download=bool(icfg.get("download", True)),
                    image_transform=transform,
                )
            )
        elif provider in ("fishnet",): # If the provider is fishnet, build the fishnet dataset
            fcfg = ds_cfg.get("fishnet") or {}
            raw_fr = fcfg.get("root")
            if raw_fr is None:
                root = (paths.data_root / "datasets" / "fishnet").resolve()
            else:
                root = Path(str(raw_fr))
                if not root.is_absolute():
                    root = (repo_root() / root).resolve()
            manifest = fcfg.get("manifest_csv")
            manifest_path = Path(manifest) if manifest else None
            if manifest_path and not manifest_path.is_absolute():
                manifest_path = (repo_root() / manifest_path).resolve()
            return build_fishnet_dataset(
                FishNetDatasetConfig(
                    root=root,
                    layout=str(fcfg.get("layout", "imagefolder")),
                    manifest_csv=manifest_path,
                    image_column=str(fcfg.get("image_column", "image_path")),
                    species_column=str(fcfg.get("species_column", "species")),
                    taxonomy_column=fcfg.get("taxonomy_column"),
                    image_transform=transform,
                )
            )
        elif provider in ("treeoflife_10m", "tol_10m", "treeoflife"):  # If the provider is treeoflife, build the Tree-of-Life dataset
            tcfg = ds_cfg.get("treeoflife") or {}
            return TreeOfLifeStreamingDataset(
                TreeOfLifeDatasetConfig(
                    hf_repo=str(tcfg.get("hf_repo", "imageomics/TreeOfLife-10M")),
                    split=str(tcfg.get("split", "train")),
                    streaming=bool(tcfg.get("streaming", True)),
                    trust_remote_code=bool(tcfg.get("trust_remote_code", False)),
                    image_keys=tuple(tcfg.get("image_keys", ("image", "jpg", "bytes"))),
                    text_keys=tuple(
                        tcfg.get("text_keys", ("taxonomy", "scientific_name", "species", "canonicalName", "text"))
                    ),
                    image_transform=transform,
                )
            )
        raise ValueError(
            f"Unknown dataset.provider={provider!r}. "
            f"Use one of: inaturalist_mini, fishnet, treeoflife_10m, or set dataset.split.train_manifest_jsonl."
        )


def ravti_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack the images and collate the batch into a dictionary"""
    images = torch.stack([b["image"] for b in batch], dim=0)
    # Return the dictionary
    return {
        "pixels": images,
        "species_texts": [b["species_text"] for b in batch],
        "taxonomy_lines": [canonical_taxonomy_line(b.get("taxonomy_line"), b["species_text"]) for b in batch],
        "sample_ids": [str(b.get("sample_id", b.get("index", i))) for i, b in enumerate(batch)],
        "image_paths": [b.get("image_path") for b in batch],
        "dataset": batch[0].get("dataset", "unknown"),
    }


def build_ravti_train_dataloader(cfg: dict) -> DataLoader:
    """Build the training dataloader from the configuration"""
    # Get the dataset and dataloader configuration
    ds_cfg = cfg.get("dataset") or {}
    dl_cfg = ds_cfg.get("dataloader") or {}
    train_cfg = cfg.get("training") or {}
    batch_size = int(dl_cfg.get("batch_size", train_cfg.get("train_batch_size", 1)))
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    # Build the dataset
    dataset = build_ravti_dataset(cfg)
    # Set the shuffle flag
    shuffle = bool(dl_cfg.get("shuffle", True))
    if isinstance(dataset, IterableDataset):
        shuffle = False
        if num_workers > 0:
            num_workers = 0

    # Build the dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ravti_collate_fn,
    )
