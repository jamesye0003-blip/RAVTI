"""
Build retrieval assets from a JSONL manifest:
1) species text FAISS index (for lookup),
2) sample_id -> image_path dictionary,
3) precomputed BioCLIP-2 image embedding matrix.

Manifest format (one JSON object per line):
  {"sample_id": "...", "species": "...", "image_path": "...", "taxonomy_line": "..."}

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
from tqdm import tqdm

from ravti.config import load_yaml_config, resolve_paths
from ravti.encoders.bioclip2_visual import BioCLIP2VisualEncoder
from ravti.encoders.bioclip_taxon import BioCLIPTaxonEncoder
from ravti.paths import project_root
from ravti.retrieval.bio_retrieval import precompute_visual_embeddings
from ravti.retrieval.faiss_index import build_index_from_iter


def _normalize_record(rec: dict, default_id: int) -> dict:
    if "species" not in rec or "image_path" not in rec:
        raise ValueError(f"manifest row missing required fields: {rec}")
    out = dict(rec)
    out["species"] = str(out["species"])
    out["image_path"] = str(out["image_path"])
    out["sample_id"] = str(out.get("sample_id") or out.get("id") or f"sample_{default_id:08d}")
    out["taxonomy_line"] = str(out.get("taxonomy_line") or out["species"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(project_root() / "configs" / "inaturalist.yaml"))
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL with species + image_path")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for image embedding precompute")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = resolve_paths(cfg)
    models_cfg = cfg.get("models") or {}
    retrieval_cfg = cfg.get("retrieval") or {}
    name = args.name or retrieval_cfg.get("index_name", "species_index")
    embedding_name = str(retrieval_cfg.get("embedding_name", f"{name}_image_embeddings"))
    id_map_name = str(retrieval_cfg.get("id_map_name", f"{name}_id_to_image.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tax = BioCLIPTaxonEncoder(models_cfg.get("bioclip_text_hub")).to(device)
    vis = BioCLIP2VisualEncoder(models_cfg.get("bioclip2_image_hub")).to(device)

    records: list[dict] = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = _normalize_record(json.loads(line), default_id=i)
            records.append(rec)

    rows = []
    image_paths: list[Path] = []
    for i, rec in enumerate(tqdm(records, desc="encode species text", dynamic_ncols=True)):
        sp = rec["species"]
        emb = tax([sp]).detach().float().cpu().numpy()[0]
        rec["embedding_row"] = i
        rows.append((emb, rec))
        image_paths.append(Path(rec["image_path"]))

    index = build_index_from_iter(rows)
    index.save(paths.index_dir, name)

    image_emb = precompute_visual_embeddings(
        image_encoder=vis,
        image_paths=image_paths,
        device=device,
        batch_size=args.batch_size,
    )
    paths.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.index_dir / f"{embedding_name}.npy", image_emb)

    id_map = {str(rec["sample_id"]): str(rec["image_path"]) for rec in records}
    with (paths.index_dir / id_map_name).open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    print("Wrote FAISS index:", paths.index_dir / f"{name}.faiss")
    print("Wrote precomputed image embeddings:", paths.index_dir / f"{embedding_name}.npy")
    print("Wrote sample_id -> image_path map:", paths.index_dir / id_map_name)


if __name__ == "__main__":
    main()
