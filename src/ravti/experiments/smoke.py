from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from ravti.config import load_yaml_config, resolve_paths
from ravti.data.build import build_ravti_train_dataloader
from ravti.data.metadata_store import MetadataStore
from ravti.data.streaming_datasets import synthetic_demo_stream
from ravti.encoders.bioclip2_visual import BioCLIP2VisualEncoder
from ravti.encoders.bioclip_taxon import BioCLIPTaxonEncoder
from ravti.encoders.semantic_sdxl import SDXLSemanticTextEncoder
from ravti.eval.cas_metric import CASMetricStub
from ravti.eval.semantic_metric import SemanticSimilarityMetric
from ravti.models.projections import TaxonRefProjectionBundle
from ravti.models.triple_stream_conditioning import TripleStreamConditioning
from ravti.paths import project_root
from ravti.retrieval.bio_retrieval import BioRetriever
from ravti.retrieval.faiss_index import build_index_from_iter
from ravti.training.sdxl_adapter_trainer import SDXLAdapterTrainer, TrainerConfig
from ravti.training.train_loop import _tensor_to_pil


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_runtime_dtype(cfg: dict, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    mp = str((cfg.get("training") or {}).get("mixed_precision", "fp16")).lower()
    if mp in ("fp32", "float32", "no"):
        return torch.float32
    if mp in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


def build_demo_retrieval(device: torch.device, tax_hub: str) -> BioRetriever:
    tax = BioCLIPTaxonEncoder(tax_hub).to(device)
    rows = []
    for sp in ["Apis mellifera", "Panthera leo", "Ursus maritimus"]:
        emb = tax([sp]).detach().float().cpu().numpy()[0]
        rows.append((emb, {"species": sp, "uri": f"demo://{sp}"}))
    index = build_index_from_iter(rows)
    return BioRetriever(tax, index)


def run_data_stage(cfg: dict) -> None:
    if os.environ.get("RAVTI_SYNTHETIC_DATA") == "1":
        row = next(iter(synthetic_demo_stream(1)))
        print("[data] synthetic row keys:", list(row.keys()))
        return
    ds_cfg = cfg.get("dataset") or {}
    dl = build_ravti_train_dataloader(cfg)
    batch = next(iter(dl))
    print(
        "[data] provider=",
        ds_cfg.get("provider"),
        "pixels=",
        tuple(batch["pixels"].shape),
        "species[0]=",
        batch["species_texts"][0],
    )


def run_metadata_stage(paths) -> None:
    store = MetadataStore(paths.metadata_db)
    store.upsert_sample("demo-1", "Apis mellifera", taxonomy={"family": "Apidae"})
    store.close()
    print("[metadata] sqlite ready at", paths.metadata_db)


def run_retrieval_stage(cfg: dict, device: torch.device) -> None:
    models_cfg = cfg.get("models") or {}
    retriever = build_demo_retrieval(device, models_cfg.get("bioclip_text_hub"))
    hits = retriever.retrieve("Apis mellifera", k=2, device=device)
    print("[retrieval] hits:", [(h.metadata.get("species"), round(h.score, 4)) for h in hits])


def run_train_smoke(cfg: dict, device: torch.device, dtype: torch.dtype) -> None:
    models_cfg = cfg.get("models") or {}
    train_cfg = cfg.get("training") or {}
    use_reference_condition = bool(train_cfg.get("use_reference_condition", True))
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        models_cfg.get("sdxl_model_id"),
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.to(device)
    tax = BioCLIPTaxonEncoder(models_cfg.get("bioclip_text_hub")).to(device)
    ref = BioCLIP2VisualEncoder(models_cfg.get("bioclip2_image_hub")).to(device)
    sem = SDXLSemanticTextEncoder(pipe)
    proj = TaxonRefProjectionBundle(tax.embed_dim, ref.embed_dim, int(train_cfg.get("sdxl_hidden_size", 2048))).to(
        device
    )
    conditioning = TripleStreamConditioning(proj).to(device)
    trainer = SDXLAdapterTrainer(
        pipe=pipe,
        tax_encoder=tax,
        ref_encoder=ref,
        semantic_encoder=sem,
        conditioning=conditioning,
        cfg=TrainerConfig(
            learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
            mixed_precision=str(train_cfg.get("mixed_precision", "fp16")),
            lambda_tax_base=float(train_cfg.get("lambda_tax_base", 0.35)),
            lambda_ref_base=float(train_cfg.get("lambda_ref_base", 0.25)),
            use_reference_condition=use_reference_condition,
        ),
    )

    dl = build_ravti_train_dataloader(cfg)
    batch = next(iter(dl))
    pixels = batch["pixels"].to(device=device, dtype=dtype)
    species = batch["species_texts"][0]
    taxon_line = batch["taxonomy_lines"][0]
    prompt_tmpl = str(
        train_cfg.get("prompt_template", "wildlife photograph of {species}, natural lighting, sharp focus")
    )
    prompt = prompt_tmpl.format(species=species, taxonomy=taxon_line)
    pil_ref = _tensor_to_pil(pixels[0]) if use_reference_condition else None
    
    # In smoke test, we don't have reference images, so we pass None for ref_images
    loss = trainer.training_step(
        pixels=pixels[0:1],
        prompt=prompt,
        taxon_string=taxon_line,
        ref_images=([pil_ref] if pil_ref is not None else None),
        ref_vector=None,
        cmc=float(train_cfg.get("cmc_train_default", 0.5)),
        device=device,
        dtype=dtype,
    )
    print("[train] one-step loss:", float(loss), "| species[0]=", species)


def run_eval_smoke() -> None:
    cas = CASMetricStub()
    sem = SemanticSimilarityMetric()
    print("[eval] cas stub:", cas.score(torch.zeros(1, 3, 224, 224), 0))
    print("[eval] semantic:", sem.compare_texts("orange wings", "orange wing"))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAVTI smoke / workflow driver")
    parser.add_argument("--config", type=str, default=str(project_root() / "configs" / "default.yaml"))
    parser.add_argument(
        "--stage",
        choices=["data", "metadata", "retrieval", "train", "eval", "all"],
        default="all",
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip heavy SDXL training smoke")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    paths = resolve_paths(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_runtime_dtype(cfg, device)

    if args.stage in ("data", "all"):
        run_data_stage(cfg)
    if args.stage in ("metadata", "all"):
        run_metadata_stage(paths)
    if args.stage in ("retrieval", "all"):
        run_retrieval_stage(cfg, device)
    if args.stage in ("eval", "all"):
        run_eval_smoke()
    if args.stage in ("train", "all") and not args.skip_train:
        run_train_smoke(cfg, device, dtype)
    elif args.stage in ("train", "all"):
        print("[train] skipped (--skip-train)")


if __name__ == "__main__":
    main()
