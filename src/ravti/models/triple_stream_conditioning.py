from __future__ import annotations

import torch
import torch.nn as nn

from ravti.models.projections import TaxonRefProjectionBundle


class TripleStreamConditioning(nn.Module):
    """
    Unified conditioning: extend SDXL `prompt_embeds` with taxonomy token,
    and optionally a reference-image token.
    """

    def __init__(self, projections: TaxonRefProjectionBundle) -> None:
        super().__init__()
        self.projections = projections

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        taxon_vec: torch.Tensor,
        ref_vec: torch.Tensor | None,
        lambda_tax: float,
        lambda_ref: float,
        use_reference_condition: bool = True,
    ) -> torch.Tensor:
        tax_tok, ref_tok = self.projections(
            taxon_vec,
            ref_vec if use_reference_condition else None,
        )
        out_dtype = prompt_embeds.dtype
        tax_tok = (tax_tok.to(out_dtype) * float(lambda_tax))[:, None, :]
        tokens = [prompt_embeds, tax_tok]
        if use_reference_condition and ref_tok is not None:
            ref_tok = (ref_tok.to(out_dtype) * float(lambda_ref))[:, None, :]
            tokens.append(ref_tok)
        return torch.cat(tokens, dim=1)
