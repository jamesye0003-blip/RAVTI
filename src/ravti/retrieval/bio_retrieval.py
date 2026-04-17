from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ravti.encoders.bioclip2_visual import BioCLIP2VisualEncoder
from ravti.encoders.bioclip_taxon import BioCLIPTaxonEncoder
from ravti.retrieval.faiss_index import FaissRetrievalIndex, RetrievalHit


class RetrievalBackend(ABC):
    """Public retrieval interface for swapping different retrieval methods."""

    @abstractmethod
    def retrieve(
        self,
        species_query: str,
        k: int,
        device: torch.device,
        fallback_queries: Optional[list[str]] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[RetrievalHit]:
        raise NotImplementedError


class BioRetriever(RetrievalBackend):
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
        exclude_ids: Optional[set[str]] = None,
    ) -> list[RetrievalHit]:
        queries = [species_query]
        if fallback_queries:
            queries.extend(fallback_queries)
        for q in queries:
            emb = self.taxon_encoder([q]).detach().float().cpu().numpy()
            hits = self.index.search(emb[0], k=max(int(k), 1))
            if exclude_ids:
                hits = [
                    h
                    for h in hits
                    if str(h.metadata.get("sample_id", "")) not in exclude_ids
                ]
                hits = hits[:k]
            if hits:
                return hits
        return []


class PrecomputedVisualBioRetriever(RetrievalBackend):
    """
    Retrieve top-k gallery samples, then return precomputed BioCLIP-2 embeddings.

    Expected side files in `index_dir`:
      - `{index_name}.faiss` + `{index_name}.jsonl` (FAISS index and metadata)
      - `{embedding_name}.npy` (precomputed image embeddings aligned with metadata rows)
    """

    def __init__(
        self,
        text_retriever: BioRetriever,
        embedding_matrix: np.ndarray,
    ) -> None:
        if embedding_matrix.ndim != 2:
            raise ValueError("embedding_matrix must be 2D")
        self._text_retriever = text_retriever
        self._embedding_matrix = embedding_matrix.astype("float32")

    @classmethod
    def from_precomputed(
        cls,
        index_dir: Path,
        index_name: str,
        embedding_name: str,
        taxon_encoder: BioCLIPTaxonEncoder,
    ) -> "PrecomputedVisualBioRetriever":
        index = FaissRetrievalIndex.load(index_dir, index_name)
        embedding_path = index_dir / f"{embedding_name}.npy"
        emb = np.load(embedding_path).astype("float32")
        return cls(BioRetriever(taxon_encoder=taxon_encoder, index=index), emb)

    def retrieve(
        self,
        species_query: str,
        k: int,
        device: torch.device,
        fallback_queries: Optional[list[str]] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[RetrievalHit]:
        return self._text_retriever.retrieve(
            species_query=species_query,
            k=k,
            device=device,
            fallback_queries=fallback_queries,
            exclude_ids=exclude_ids,
        )

    def retrieve_embedding(
        self,
        species_query: str,
        k: int,
        device: torch.device,
        fallback_queries: Optional[list[str]] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> tuple[Optional[torch.Tensor], list[RetrievalHit]]:
        hits = self.retrieve(
            species_query=species_query,
            k=k,
            device=device,
            fallback_queries=fallback_queries,
            exclude_ids=exclude_ids,
        )
        if not hits:
            return None, []
        rows = []
        for h in hits:
            row_idx = h.metadata.get("embedding_row")
            if row_idx is None:
                continue
            idx = int(row_idx)
            if idx < 0 or idx >= self._embedding_matrix.shape[0]:
                continue
            rows.append(self._embedding_matrix[idx])
        if not rows:
            return None, hits
        mat = np.stack(rows, axis=0)
        # Trainer currently consumes one reference token, so we aggregate top-k.
        mean_vec = mat.mean(axis=0, keepdims=True)
        vec = torch.from_numpy(mean_vec).to(device=device)
        return vec, hits


def precompute_visual_embeddings(
    image_encoder: BioCLIP2VisualEncoder,
    image_paths: list[Path],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    feats: list[np.ndarray] = []
    from PIL import Image

    batch_size = max(int(batch_size), 1)
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        emb = image_encoder(images, device=device).detach().float().cpu().numpy()
        feats.append(emb)
    if not feats:
        raise ValueError("No image embeddings generated from image_paths.")
    return np.concatenate(feats, axis=0).astype("float32")
