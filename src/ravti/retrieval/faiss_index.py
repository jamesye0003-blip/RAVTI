from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import faiss
import numpy as np


@dataclass
class RetrievalHit:
    rank: int
    metadata: dict[str, Any]
    score: float


class FaissRetrievalIndex:
    """k-NN retrieval over precomputed embeddings with JSON sidecar metadata."""

    def __init__(self, vectors: np.ndarray, records: list[dict[str, Any]]) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        if len(records) != vectors.shape[0]:
            raise ValueError("records length must match number of vectors")
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        self._index = index
        self._records = records

    @classmethod
    def from_embeddings_file(cls, npy_path: Path, jsonl_path: Path) -> "FaissRetrievalIndex":
        vectors = np.load(npy_path).astype("float32")
        records = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        return cls(vectors, records)

    def save(self, index_dir: Path, name: str) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_dir / f"{name}.faiss"))
        with (index_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in self._records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_dir: Path, name: str) -> "FaissRetrievalIndex":
        idx = faiss.read_index(str(index_dir / f"{name}.faiss"))
        records = []
        with (index_dir / f"{name}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        obj = object.__new__(cls)
        obj._index = idx
        obj._records = records
        return obj

    def search(self, query: np.ndarray, k: int) -> list[RetrievalHit]:
        if query.ndim == 1:
            query = query[None, :]
        q = query.astype("float32")
        faiss.normalize_L2(q)
        scores, ids = self._index.search(q, k)
        hits: list[RetrievalHit] = []
        for rank, (s, i) in enumerate(zip(scores[0].tolist(), ids[0].tolist())):
            if i < 0:
                continue
            hits.append(RetrievalHit(rank=rank, metadata=self._records[i], score=float(s)))
        return hits


def build_index_from_iter(
    embedding_rows: Iterable[tuple[np.ndarray, dict[str, Any]]],
) -> FaissRetrievalIndex:
    vecs: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for v, m in embedding_rows:
        vecs.append(v.astype("float32"))
        meta.append(m)
    mat = np.stack(vecs, axis=0)
    return FaissRetrievalIndex(mat, meta)
