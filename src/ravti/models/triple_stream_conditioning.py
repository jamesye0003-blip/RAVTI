from __future__ import annotations

import torch
import torch.nn as nn

from ravti.models.projections import TaxonRefProjectionBundle


class TripleStreamConditioning(nn.Module):
    """
    Extend Stable Diffusion XL `prompt_embeds` with taxonomy token: [prompt_embeds, tax_tok] or [prompt_embeds, tax_tok, ref_tok].

    The SDXL UNet cross-attention treats `encoder_hidden_states` as K/V; the extra tokens will participate in the attention at all spatial locations.

    If the projection layer or lambda scaling makes the tax/ref token norm too large relative to the text token,
    it is equivalent to adding a global bias error to the entire image, commonly seen as symmetrical structures, color bands, and high contrast noise.
    """

    def __init__(self, projections: TaxonRefProjectionBundle) -> None:
        super().__init__()
        self.projections = projections

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        taxon_vec: torch.Tensor,
        ref_vec: torch.Tensor | None,
        lambda_tax: float | torch.Tensor,
        lambda_ref: float | torch.Tensor,
        use_reference_condition: bool = True,
    ) -> torch.Tensor:
        # Use TaxonRefProjectionBundle to project the taxon_vec and ref_vec to the same hidden dimension as prompt_embeds
        tax_tok, ref_tok = self.projections(
            taxon_vec,
            ref_vec if use_reference_condition else None,
        )

        # Apply the lambda values to the tax tokens, and convert the tax tokens to the output dimension
        out_dtype = prompt_embeds.dtype
        bsz = int(prompt_embeds.shape[0])
        lt = lambda_tax
        if not torch.is_tensor(lt):
            lt = torch.tensor([float(lt)] * bsz, device=prompt_embeds.device, dtype=out_dtype)
        else:
            lt = lt.to(device=prompt_embeds.device, dtype=out_dtype).reshape(-1)
            if lt.numel() == 1:
                lt = lt.repeat(bsz)
            elif lt.numel() != bsz:
                raise ValueError(f"lambda_tax size mismatch: got {lt.numel()}, expected 1 or {bsz}")
        tax_tok = (tax_tok.to(out_dtype) * lt[:, None])[:, None, :]
        tokens = [prompt_embeds, tax_tok]

        # If the reference condition is specified and the reference token is not None, apply the lambda values to the reference token
        if use_reference_condition and ref_tok is not None:
            lr = lambda_ref
            if not torch.is_tensor(lr):
                lr = torch.tensor([float(lr)] * bsz, device=prompt_embeds.device, dtype=out_dtype)
            else:
                lr = lr.to(device=prompt_embeds.device, dtype=out_dtype).reshape(-1)
                if lr.numel() == 1:
                    lr = lr.repeat(bsz)
                elif lr.numel() != bsz:
                    raise ValueError(f"lambda_ref size mismatch: got {lr.numel()}, expected 1 or {bsz}")
            ref_tok = (ref_tok.to(out_dtype) * lr[:, None])[:, None, :]
            tokens.append(ref_tok)
        
        # Concatenate the tax and reference tokens with the prompt embeddings, which will be used as the condition for the Stable Diffusion XL model
        return torch.cat(tokens, dim=1)
