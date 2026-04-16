from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image
from torchvision.datasets import ImageFolder


def iter_imagefolder(root: Path | str) -> Iterator[tuple[Image.Image, str]]:
    """Yield (PIL image, class folder name) from a torchvision ImageFolder tree."""
    ds = ImageFolder(str(root))
    for path, target in ds.samples:
        im = Image.open(path).convert("RGB")
        label = ds.classes[target]
        yield im, label
