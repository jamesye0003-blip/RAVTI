from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

def reliability_lambdas(
    cmc: float | torch.Tensor,
    lambda_tax_base: float,
    lambda_ref_base: float,
) -> tuple[float | torch.Tensor, float | torch.Tensor]:
    """
    Map cross-modal consistency (CMC) in [0, 1] to (lambda_tax, lambda_ref).

    Interpretation (tunable): when CMC is high, increase reliance on retrieved
    visuals; when low, lean on taxonomic text. This is a minimal differentiable
    prior for sweeps; replace with learned gating as the project matures.
    """
    if torch.is_tensor(cmc):
        cmc_t = torch.clamp(cmc, 0.0, 1.0)
        lambda_tax = lambda_tax_base * (1.0 - cmc_t)
        lambda_ref = lambda_ref_base * cmc_t
        return lambda_tax, lambda_ref
    cmc_f = float(min(1.0, max(0.0, cmc)))
    lambda_tax = lambda_tax_base * (1.0 - cmc_f)
    lambda_ref = lambda_ref_base * cmc_f
    return lambda_tax, lambda_ref


class CMCGate(nn.Module):
    """Minimal learnable gate: CMC = sigmoid(Wx + b)."""

    def __init__(self, in_dim: int = 4) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.linear = nn.Linear(self.in_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.in_dim:
            raise ValueError(f"CMCGate expects feature dim={self.in_dim}, got {features.shape[-1]}")
        return torch.sigmoid(self.linear(features)).squeeze(-1)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    # FAISS IP scores are usually in [-1,1] after cosine normalization.
    return np.clip((scores + 1.0) * 0.5, 0.0, 1.0)


def build_cmc_features(
    species_query: str,
    taxonomy_line: str,
    hits: list[Any],
    has_retrieved_ref: bool,
) -> np.ndarray:
    """Return [sim_conf, concentration, tax_consistency, has_ref] in [0,1]."""
    if not hits:
        return np.array([0.0, 0.0, 0.0, 1.0 if has_retrieved_ref else 0.0], dtype=np.float32)
    # Calculate the average cosine similarity confidence between the query and the hits
    scores = np.array([float(h.score) for h in hits], dtype=np.float32)
    s01 = _normalize_scores(scores)
    sim_conf = float(s01.mean())

    # Calculate the probability of each hit (softmax)
    shifted = scores - float(scores.max())
    probs = np.exp(shifted)
    z = float(probs.sum())
    if z <= 0.0:
        probs = np.ones_like(probs) / max(len(probs), 1)
    else:
        probs = probs / z
    k = max(len(probs), 1)
    if k == 1:
        concentration = 1.0
    else:
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
        concentration = 1.0 - entropy / np.log(float(k))

    # Calculate the taxonomic consistency between the query and the hits
    species_q = species_query.strip().lower()
    genus_q = species_q.split(" ")[0] if species_q else ""
    tax_q = taxonomy_line.strip().lower()
    tax_matches: list[float] = []
    for h in hits:  # For each hit, check if the species and taxonomy match the query
        meta = h.metadata or {}
        hs = str(meta.get("species", "")).strip().lower()
        ht = str(meta.get("taxonomy_line", "")).strip().lower()
        if hs and hs == species_q:  # If the species matches the query, add 1.0 to the tax_matches list
            tax_matches.append(1.0)
            continue
        if hs and genus_q and hs.split(" ")[0] == genus_q:  # If the genus matches the query, add 0.7 to the tax_matches list
            tax_matches.append(0.7)
            continue
        if tax_q and ht and (tax_q in ht or ht in tax_q):  # If the taxonomy matches the query, add 0.7 to the tax_matches list
            tax_matches.append(0.7)
            continue
        tax_matches.append(0.0)  # If the species and taxonomy do not match the query, add 0.0 to the tax_matches list
    tax_consistency = float(np.mean(tax_matches)) if tax_matches else 0.0  # Calculate the average of the tax_matches list, which is the taxonomic consistency

    # Return the features as a numpy array
    return np.array(
        [sim_conf, concentration, tax_consistency, 1.0 if has_retrieved_ref else 0.0],
        dtype=np.float32,
    )
