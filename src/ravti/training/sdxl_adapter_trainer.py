from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline

from ravti.encoders.bioclip2_visual import BioCLIP2VisualEncoder
from ravti.encoders.bioclip_taxon import BioCLIPTaxonEncoder
from ravti.encoders.semantic_sdxl import SDXLSemanticTextEncoder, iter_sdxl_text_encoders
from ravti.models.decoupled_attention import attach_decoupled_cross_attention
from ravti.models.projections import TaxonRefProjectionBundle
from ravti.models.reliability import reliability_lambdas
from ravti.models.triple_stream_conditioning import TripleStreamConditioning


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-4
    mixed_precision: str = "fp32"
    lambda_tax_base: float = 0.35
    lambda_ref_base: float = 0.25
    use_reference_condition: bool = True


class SDXLAdapterTrainer:
    """
    Adapter-only training: frozen SDXL stack, train tax/ref projections + optional gating MLP later.

    Uses standard diffusion MSE on noise prediction with extended encoder hidden states.
    """

    def __init__(
        self,
        pipe: StableDiffusionXLPipeline,
        tax_encoder: BioCLIPTaxonEncoder,
        ref_encoder: BioCLIP2VisualEncoder,
        semantic_encoder: SDXLSemanticTextEncoder,
        conditioning: TripleStreamConditioning,
        cfg: TrainerConfig,
    ) -> None:
        self.pipe = pipe
        self.tax_encoder = tax_encoder
        self.ref_encoder = ref_encoder
        self.semantic_encoder = semantic_encoder
        self.conditioning = conditioning
        self.cfg = cfg

        # Freeze encoders and UNet
        self.tax_encoder.requires_grad_(False)
        self.ref_encoder.requires_grad_(False)
        self.tax_encoder.eval()
        self.ref_encoder.eval()
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        for enc in iter_sdxl_text_encoders(self.pipe):
            enc.requires_grad_(False)
        
        # Trainable conditioning
        self.conditioning.train()
        params = list(self.conditioning.parameters())
        self.decoupled_attn_processors: Optional[torch.nn.ModuleList] = attach_decoupled_cross_attention(self.pipe.unet)
        ref_param = next(self.pipe.unet.parameters())
        self.decoupled_attn_processors.to(device=ref_param.device, dtype=ref_param.dtype)
        for p in self.decoupled_attn_processors.parameters():
            p.requires_grad = True
        params.extend(list(self.decoupled_attn_processors.parameters()))
        self.optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler("cuda") if cfg.mixed_precision == "fp16" and torch.cuda.is_available() else None
        )

    def training_step(
        self,
        pixels: torch.Tensor,
        prompt: str,
        taxon_string: str,
        ref_images: Optional[list],
        cmc: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        pixels: [1,3,H,W] in [0,1] float on device
        ref_images: list of PIL images for the visual reference stream
        """
        lambda_tax, lambda_ref = reliability_lambdas(
            cmc, self.cfg.lambda_tax_base, self.cfg.lambda_ref_base
        )
        if not self.cfg.use_reference_condition:
            lambda_ref = 0.0
        with torch.no_grad():
            latents = self.pipe.vae.encode(pixels).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                self.pipe.scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noisy = self.pipe.scheduler.add_noise(latents, noise, timesteps)
            sem = self.semantic_encoder.encode(prompt, device=device, dtype=dtype)

        # Frozen encoders should not build grads, but outputs must be normal tensors
        # so downstream trainable projection linears can save activations for backward.
        with torch.no_grad():
            tax = self.tax_encoder([taxon_string]).to(device=device).clone()
            ref = None
            if self.cfg.use_reference_condition and ref_images:
                ref = self.ref_encoder(ref_images, device=device).to(device=device).clone()
                if ref.shape[0] > 1:
                    ref = ref.mean(dim=0, keepdim=True)
        prompt_embeds = self.conditioning(
            sem.prompt_embeds,
            tax,
            ref,
            lambda_tax=lambda_tax,
            lambda_ref=lambda_ref,
            use_reference_condition=self.cfg.use_reference_condition,
        )

        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16) if self.scaler is not None else contextlib.nullcontext()
        )
        with amp_ctx:
            time_ids = self._time_ids(pixels.shape[2], pixels.shape[3], bsz, device, sem.pooled_prompt_embeds.dtype)
            model_pred = self.pipe.unet(
                noisy,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": sem.pooled_prompt_embeds,
                    "time_ids": time_ids,
                },
                cross_attention_kwargs={
                    "lambda_tax": float(lambda_tax),
                    "lambda_ref": float(lambda_ref),
                    "use_reference_condition": bool(self.cfg.use_reference_condition),
                },
            ).sample
            target = noise
            loss = F.mse_loss(model_pred.float(), target.float())

        if not torch.isfinite(loss):
            # Skip optimizer update on non-finite steps to avoid poisoning weights.
            return torch.tensor(float("nan"), device=device)

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return loss.detach()

    def _time_ids(
        self, height: int, width: int, bsz: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # SDXL `time_ids`: (orig_h, orig_w, crop_y, crop_x, target_h, target_w)
        row = [height, width, 0, 0, height, width]
        t = torch.tensor([row], device=device, dtype=dtype)
        return t.repeat(bsz, 1)
