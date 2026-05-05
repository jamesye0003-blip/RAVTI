from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class TaxonRefProjectionBundle(nn.Module):
    """Learnable linear maps from BioCLIP / BioCLIP-2 spaces into Stable Diffusion XL cross-attn width."""

    def __init__(self, taxon_dim: int, ref_dim: int, sdxl_hidden: int = 2048) -> None:
        super().__init__()
        self.taxon_proj = nn.Linear(taxon_dim, sdxl_hidden)
        self.ref_proj = nn.Linear(ref_dim, sdxl_hidden)
        # Keep initial perturbation tiny: start close to baseline Stable Diffusion XL conditioning.
        # nn.init.xavier_uniform_(self.taxon_proj.weight)
        nn.init.normal_(self.taxon_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.taxon_proj.bias)
        # nn.init.xavier_uniform_(self.ref_proj.weight)
        nn.init.normal_(self.ref_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.ref_proj.bias)

    def forward(
        self, taxon_vec: torch.Tensor, ref_vec: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Inputs are [B, D]; returns tax token and optional reference token."""
        # Frozen encoders may emit fp16 on CUDA while Linear defaults to fp32 weights.
        w_dtype = self.taxon_proj.weight.dtype
        tax_tok = self.taxon_proj(taxon_vec.to(w_dtype))
        ref_tok = self.ref_proj(ref_vec.to(w_dtype)) if ref_vec is not None else None
        return tax_tok, ref_tok
