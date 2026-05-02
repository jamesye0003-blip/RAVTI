"""
Text-to-image alignment metrics (TaxaAdapter-style).

- **CLIP**: cosine similarity between the generated image and the species **common name** string.
- **BioCLIP**: cosine similarity in the BioCLIP (iNat) joint space between the generated image
  and the species **taxonomic** string (Linnaean line / scientific naming).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class OpenCLIPCommonNameMetric:
    """OpenCLIP image–text score with **common name** (not the full generation prompt)."""

    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    _model: object | None = None
    _preprocess: object | None = None
    _tokenizer: object | None = None

    def _ensure_ready(self, device: torch.device) -> None:
        if self._model is not None:
            return
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        tokenizer = open_clip.get_tokenizer(self.model_name)
        model.eval().to(device)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer

    @torch.inference_mode()
    def score(self, image: Image.Image, common_name: str, device: torch.device) -> float:
        text = (common_name or "").strip()
        if not text:
            return 0.0
        self._ensure_ready(device)
        assert self._model is not None and self._preprocess is not None and self._tokenizer is not None
        img = self._preprocess(image.convert("RGB")).unsqueeze(0).to(device=device)
        tok = self._tokenizer([text]).to(device=device)
        image_feat = self._model.encode_image(img)  # type: ignore[attr-defined]
        text_feat = self._model.encode_text(tok)  # type: ignore[attr-defined]
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        return float((image_feat * text_feat).sum(dim=-1).item())


@dataclass
class BioCLIPTaxonomicTextMetric:
    """
    BioCLIP (Imageomics iNat) joint image–text score with a **taxonomic** string.

    Uses the same hub for both towers so image/text embeddings are comparable.
    """

    hub_id: str = "hf-hub:imageomics/bioclip-vit-b-16-inat-only"
    _model: Optional[torch.nn.Module] = None
    _preprocess: object | None = None
    _tokenizer: object | None = None

    def _ensure_ready(self, device: torch.device) -> None:
        if self._model is not None:
            return
        model, _, preprocess = open_clip.create_model_and_transforms(self.hub_id)
        tokenizer = open_clip.get_tokenizer(self.hub_id)
        for p in model.parameters():
            p.requires_grad = False
        model.eval().to(device)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer

    @torch.inference_mode()
    def score(self, image: Image.Image, taxonomic_text: str, device: torch.device) -> float:
        text = (taxonomic_text or "").strip()
        if not text:
            return 0.0
        self._ensure_ready(device)
        assert self._model is not None and self._preprocess is not None and self._tokenizer is not None
        img = self._preprocess(image.convert("RGB")).unsqueeze(0).to(device=device)
        tok = self._tokenizer([text]).to(device=device)
        image_feat = self._model.encode_image(img)  # type: ignore[attr-defined]
        text_feat = self._model.encode_text(tok)  # type: ignore[attr-defined]
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        return float((image_feat * text_feat).sum(dim=-1).item())


def build_t2i_metrics_from_config(cfg: dict) -> tuple[OpenCLIPCommonNameMetric, BioCLIPTaxonomicTextMetric]:
    ev = cfg.get("evaluation") or {}
    models_cfg = cfg.get("models") or {}
    clip = OpenCLIPCommonNameMetric(
        model_name=str(ev.get("clip_model_name", "ViT-B-32")),
        pretrained=str(ev.get("clip_pretrained", "openai")),
    )
    bioclip = BioCLIPTaxonomicTextMetric(
        hub_id=str(models_cfg.get("bioclip_text_hub", "hf-hub:imageomics/bioclip-vit-b-16-inat-only"))
    )
    return clip, bioclip
