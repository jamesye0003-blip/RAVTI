"""
Build a FAISS species gallery from a JSONL manifest + on-the-fly BioCLIP embeddings.

Manifest format (one JSON object per line):
  {"species": "...", "image_path": "E:/Datasets/Image/..."}

Example:
  conda activate ai_full
  python scripts/build_retrieval_index.py --manifest data/metadata/gallery.jsonl --name species_index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ravti.config import load_yaml_config, resolve_paths
from ravti.encoders.bioclip_taxon import BioCLIPTaxonEncoder
from ravti.paths import project_root
from ravti.retrieval.faiss_index import build_index_from_iter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(project_root() / "configs" / "default.yaml"))
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL with species + image_path")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = resolve_paths(cfg)
    models_cfg = cfg.get("models") or {}
    retrieval_cfg = cfg.get("retrieval") or {}
    name = args.name or retrieval_cfg.get("index_name", "species_index")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tax = BioCLIPTaxonEncoder(models_cfg.get("bioclip_text_hub")).to(device)

    rows = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sp = str(rec["species"])
            emb = tax([sp]).detach().float().cpu().numpy()[0]
            rows.append((emb, rec))

    index = build_index_from_iter(rows)
    index.save(paths.index_dir, name)
    print("Wrote FAISS index:", paths.index_dir / f"{name}.faiss")


if __name__ == "__main__":
    main()
