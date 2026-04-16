from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import StableDiffusionXLPipeline


def iter_sdxl_text_encoders(pipe: StableDiffusionXLPipeline):
    """
    Diffusers versions differ: some expose ``text_encoders`` (list), others
    ``text_encoder`` + ``text_encoder_2``. Yield non-None encoders in SDXL order.
    """
    encs = getattr(pipe, "text_encoders", None)
    if encs is not None:
        for e in encs:
            if e is not None:
                yield e
        return
    for name in ("text_encoder", "text_encoder_2"):
        e = getattr(pipe, name, None)
        if e is not None:
            yield e


@dataclass
class SemanticEmbeddingBundle:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor


class SDXLSemanticTextEncoder:
    """Frozen SDXL dual text encoders (CLIP-L + OpenCLIP) for style / scene prompts."""

    def __init__(self, pipe: StableDiffusionXLPipeline) -> None:
        self.pipe = pipe

    @torch.inference_mode()
    def encode(
        self,
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
        do_classifier_free_guidance: bool = False,
    ) -> SemanticEmbeddingBundle:
        (
            prompt_embeds,
            _,
            pooled_prompt_embeds,
            _,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
        )
        return SemanticEmbeddingBundle(
            prompt_embeds=prompt_embeds.to(device=device, dtype=dtype),
            pooled_prompt_embeds=pooled_prompt_embeds.to(device=device, dtype=dtype),
        )
