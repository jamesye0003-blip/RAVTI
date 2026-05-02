from __future__ import annotations

from dataclasses import dataclass

import open_clip
import torch
import torch.nn as nn
from PIL import Image


@dataclass
class BioCLIP2VisualConfig:
    hub_id: str = "hf-hub:imageomics/bioclip-2"


class BioCLIP2VisualEncoder(nn.Module):
    """Frozen BioCLIP-2 image tower for reference morphology."""

    def __init__(self, hub_id: str) -> None:
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(hub_id)
        for p in self.model.parameters():
            p.requires_grad = False
        self.eval()

    @property
    def embed_dim(self) -> int:
        vis = self.model.visual
        if hasattr(vis, "output_dim"):
            return int(vis.output_dim)
        if hasattr(vis, "proj") and vis.proj is not None:
            return int(vis.proj.shape[0])
        return 768

    @torch.inference_mode()
    def forward(self, images: list[Image.Image], device: torch.device) -> torch.Tensor:
        """
        Encode a list of images into a tensor of features, and normalize the features to unit length.
        The output tensor has shape [N, D] where N is the number of images and D is the embedding dimension.
        """
        tensors = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(
            device=device, dtype=next(self.model.parameters()).dtype
        )
        feats = self.model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats
