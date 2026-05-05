from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from datasets import load_dataset
from PIL import Image
from torch.utils.data import IterableDataset


@dataclass
class TreeOfLifeDatasetConfig:
    hf_repo: str = "imageomics/TreeOfLife-10M"
    split: str = "train"
    streaming: bool = True
    trust_remote_code: bool = False
    image_keys: tuple[str, ...] = ("image", "jpg", "bytes")
    text_keys: tuple[str, ...] = ("taxonomy", "scientific_name", "species", "canonicalName", "text")
    image_transform: Optional[Callable[[Image.Image], Any]] = None


class TreeOfLifeStreamingDataset(IterableDataset):
    """
    Streaming access to a Tree-of-Life style HF dataset.

    Schemas differ across revisions; this reader picks the first usable image field
    and the first usable taxonomy / species string field from `text_keys`.
    """

    def __init__(self, cfg: TreeOfLifeDatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._ds = load_dataset(
            cfg.hf_repo,
            split=cfg.split,
            streaming=cfg.streaming,
            trust_remote_code=cfg.trust_remote_code,
        )

    def _pick_image(self, row: dict[str, Any]) -> Image.Image:
        for k in self.cfg.image_keys:
            if k not in row or row[k] is None:
                continue
            v = row[k]
            if isinstance(v, Image.Image):
                im = v.convert("RGB")
            elif isinstance(v, dict) and "bytes" in v:
                from io import BytesIO

                im = Image.open(BytesIO(v["bytes"])).convert("RGB")
            else:
                continue
            if self.cfg.image_transform is not None:
                im = self.cfg.image_transform(im)
            return im
        raise KeyError(f"No image field found in row keys {list(row.keys())}")

    def _pick_taxon(self, row: dict[str, Any]) -> tuple[str, str]:
        for k in self.cfg.text_keys:
            if k in row and row[k] is not None and str(row[k]).strip():
                s = str(row[k]).strip()
                return s, s
        return "unknown_taxon", "unknown_taxon"

    def __iter__(self) -> Iterator[dict[str, Any]]:
        # Streaming + multi-worker sharding is non-trivial; keep single-threaded iteration by default.
        for i, row in enumerate(self._ds):
            yield self._row_to_sample(row, i)

    def _row_to_sample(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        img = self._pick_image(row)
        species_text, taxonomy_line = self._pick_taxon(row)
        return {
            "image": img,
            "species_text": species_text,
            "taxonomy_line": taxonomy_line,
            "dataset": "treeoflife_10m",
            "sample_id": f"treeoflife:{index}",
            "index": index,
        }
