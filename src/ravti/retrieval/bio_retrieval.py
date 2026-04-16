from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ravti.encoders.bioclip_taxon import BioCLIPTaxonEncoder
from ravti.retrieval.faiss_index import FaissRetrievalIndex, RetrievalHit


class BioRetriever:
    """Species-level retrieval: encode taxon with BioCLIP text, query FAISS gallery."""

    def __init__(self, taxon_encoder: BioCLIPTaxonEncoder, index: FaissRetrievalIndex) -> None:
        self.taxon_encoder = taxon_encoder
        self.index = index

    @torch.inference_mode()
    def retrieve(
        self,
        species_query: str,
        k: int,
        device: torch.device,
        fallback_queries: Optional[list[str]] = None,
    ) -> list[RetrievalHit]:
        queries = [species_query]
        if fallback_queries:
            queries.extend(fallback_queries)
        for q in queries:
            emb = self.taxon_encoder([q]).detach().float().cpu().numpy()
            hits = self.index.search(emb[0], k=k)
            if hits:
                return hits
        return []
