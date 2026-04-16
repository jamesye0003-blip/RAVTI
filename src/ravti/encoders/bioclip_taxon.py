from __future__ import annotations

from dataclasses import dataclass

import open_clip
import torch
import torch.nn as nn


@dataclass
class BioCLIPTaxonConfig:
    hub_id: str = "hf-hub:imageomics/bioclip-vit-b-16-inat-only"


class BioCLIPTaxonEncoder(nn.Module):
    """Frozen BioCLIP text tower for Linnaean / taxonomic strings."""

    def __init__(self, hub_id: str) -> None:
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(hub_id)
        self.tokenizer = open_clip.get_tokenizer(hub_id)
        for p in self.model.parameters():
            p.requires_grad = False
        self.eval()

    @property
    def embed_dim(self) -> int:
        tp = self.model.text_projection
        if hasattr(tp, "out_features"):
            return int(tp.out_features)
        if isinstance(tp, torch.Tensor):
            return int(tp.shape[-1])
        return 512

    @torch.inference_mode()
    def forward(self, taxon_strings: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(taxon_strings)
        tokens = tokens.to(next(self.model.parameters()).device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats
