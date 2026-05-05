"""
Image-quality metrics (planned).

Fréchet Inception Distance (FID) and Learned Perceptual Image Patch Similarity (LPIPS)
require batched real/fake image pipelines and optional extra dependencies; stubs are kept here
so benchmarks can import a single module when those paths are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FIDMetric:
    """Placeholder for FID between real and generated distributions."""

    _real_features: list[Any] = field(default_factory=list)
    _fake_features: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self._real_features.clear()
        self._fake_features.clear()

    def compute(self) -> float:
        raise NotImplementedError(
            "FID is not wired yet; integrate pytorch-fid / clean-fid style feature banks, then implement compute()."
        )


@dataclass
class LPIPSMetric:
    """Placeholder for LPIPS against reference images (e.g. one real image per species)."""

    def score(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError(
            "LPIPS is not wired yet; use lpips package or torchvision-backed perceptual loss on paired tensors."
        )
