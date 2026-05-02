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
from ravti.models.reliability import CMCGate, reliability_lambdas
from ravti.models.triple_stream_conditioning import TripleStreamConditioning


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-4
    mixed_precision: str = "fp32"
    lambda_tax_base: float = 0.35  # The base can be learned in future work!!!
    lambda_ref_base: float = 0.25  # The base can be learned in future work!!!
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
        cmc_gate: CMCGate | None,
        cfg: TrainerConfig,
    ) -> None:
        self.pipe = pipe
        self.tax_encoder = tax_encoder
        self.ref_encoder = ref_encoder
        self.semantic_encoder = semantic_encoder
        self.conditioning = conditioning
        self.cmc_gate = cmc_gate
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
        if self.cmc_gate is not None:
            self.cmc_gate.train()
            params.extend(list(self.cmc_gate.parameters()))
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
        prompts: list[str],
        taxon_strings: list[str],
        ref_images: Optional[list],
        ref_vectors: Optional[list[torch.Tensor | None]],
        cmc_features: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Parameters:
            pixels: [B,3,H,W] in [0,1] float on device
            prompts: list[str], one prompt per sample
            taxon_strings: list[str], one taxonomy string per sample
            ref_images: list[Optional[PIL.Image]], optional fallback images
            ref_vectors: list[Optional[Tensor]], optional retrieved vectors
            cmc_features: Optional[torch.Tensor], the CMC features for the image
            device: torch.device, the device to use
            dtype: torch.dtype, the dtype to use
        Returns:
            torch.Tensor, the loss
        """
        # Check errors
        if self.cmc_gate is None:
            raise RuntimeError("CMCGate is required but missing in SDXLAdapterTrainer.")
        if cmc_features is None:
            raise RuntimeError("cmc_features is required when CMCGate is enabled.")
        bsz = int(pixels.shape[0])
        if len(prompts) != bsz or len(taxon_strings) != bsz:
            raise ValueError(f"Batch size mismatch: pixels={bsz}, prompts={len(prompts)}, taxon_strings={len(taxon_strings)}")
        if ref_images is not None and len(ref_images) != bsz:
            raise ValueError(f"ref_images size mismatch: {len(ref_images)} vs batch {bsz}")
        if ref_vectors is not None and len(ref_vectors) != bsz:
            raise ValueError(f"ref_vectors size mismatch: {len(ref_vectors)} vs batch {bsz}")
        
        # Calculate the CMC score using the CMCGate
        cmc_input = cmc_features.to(device=device, dtype=dtype)
        cmc_t = self.cmc_gate(cmc_input).reshape(-1)

        # Calculate the lambda_tax and lambda_ref using the reliability_lambdas function    
        lambda_tax, lambda_ref = reliability_lambdas(cmc_t, self.cfg.lambda_tax_base, self.cfg.lambda_ref_base)
        if not self.cfg.use_reference_condition:
            # If the reference condition is not used, set the lambda_ref to 0.0
            if torch.is_tensor(lambda_ref):
                lambda_ref = torch.zeros_like(lambda_ref)
            else:
                lambda_ref = 0.0
        
        # Encode the pixels to latents
        with torch.no_grad():
            latents = self.pipe.vae.encode(pixels).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)  # Sample noise from the standard normal distribution
            bsz = latents.shape[0]
            # Sample timesteps from the scheduler
            timesteps = torch.randint(
                0,
                self.pipe.scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noisy = self.pipe.scheduler.add_noise(latents, noise, timesteps)  # Add noise to the latents
            sem = self.semantic_encoder.encode(prompts, device=device, dtype=dtype)  # Encode the prompt embeddings

        # Frozen encoders should not build grads, but outputs must be normal tensors
        # so downstream trainable projection linears can save activations for backward.
        with torch.no_grad():
            tax = self.tax_encoder(taxon_strings).to(device=device).clone()
            ref = None
            if self.cfg.use_reference_condition:
                ref_dtype = next(self.ref_encoder.parameters()).dtype
                ref_buf = torch.zeros((bsz, self.ref_encoder.embed_dim), device=device, dtype=ref_dtype)
                ref_has_any = False

                if ref_vectors is not None:
                    for i, rv in enumerate(ref_vectors):
                        if rv is None:
                            continue
                        rr = rv.to(device=device).clone()
                        if rr.ndim > 1:
                            rr = rr.reshape(-1, rr.shape[-1])[0]
                        ref_buf[i] = rr.to(dtype=ref_dtype)
                        ref_has_any = True

                if ref_images is not None:
                    enc_idx: list[int] = []
                    enc_imgs = []
                    for i, img in enumerate(ref_images):
                        if img is None:
                            continue
                        if ref_vectors is not None and ref_vectors[i] is not None:
                            continue
                        enc_idx.append(i)
                        enc_imgs.append(img)
                    if enc_imgs:
                        enc = self.ref_encoder(enc_imgs, device=device).to(device=device).clone()
                        for j, i in enumerate(enc_idx):
                            ref_buf[i] = enc[j].to(dtype=ref_dtype)
                        ref_has_any = True

                if ref_has_any:
                    ref = ref_buf
        
        # Apply the conditioning to the prompt embeddings
        prompt_embeds = self.conditioning(
            sem.prompt_embeds,
            tax,
            ref,
            lambda_tax=lambda_tax,
            lambda_ref=lambda_ref,
            use_reference_condition=self.cfg.use_reference_condition,
        )

        # Apply the autocast context to the MSE loss calculation between the model prediction and the noise
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16) if self.scaler is not None else contextlib.nullcontext()
        )
        with amp_ctx:
            time_ids = self._time_ids(pixels.shape[2], pixels.shape[3], bsz, device, sem.pooled_prompt_embeds.dtype)
            # Decoupled attention processors currently consume scalar lambdas per UNet call.
            # Use batch mean here while keeping per-sample token scaling in conditioning().
            lambda_tax_attn = float(lambda_tax.mean().detach().item()) if torch.is_tensor(lambda_tax) else float(lambda_tax)
            lambda_ref_attn = float(lambda_ref.mean().detach().item()) if torch.is_tensor(lambda_ref) else float(lambda_ref)
            # Generate the model prediction
            model_pred = self.pipe.unet(
                noisy,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": sem.pooled_prompt_embeds,
                    "time_ids": time_ids,
                },
                cross_attention_kwargs={
                    "lambda_tax": lambda_tax_attn,
                    "lambda_ref": lambda_ref_attn,
                    "use_reference_condition": bool(self.cfg.use_reference_condition),
                },
            ).sample
            target = noise # The target is the noise
            loss = F.mse_loss(model_pred.float(), target.float())

        if not torch.isfinite(loss):
            # Skip optimizer update on non-finite steps to avoid poisoning weights.
            return torch.tensor(float("nan"), device=device)

        # Update the optimizer
        self.optimizer.zero_grad(set_to_none=True)
        # Scale the loss and update the optimizer
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
        """SDXL `time_ids`: (orig_h, orig_w, crop_y, crop_x, target_h, target_w)"""
        row = [height, width, 0, 0, height, width]
        t = torch.tensor([row], device=device, dtype=dtype)
        return t.repeat(bsz, 1)  # Repeat the tensor for each batch element
