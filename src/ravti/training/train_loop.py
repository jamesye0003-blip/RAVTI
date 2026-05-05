from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import IterableDataset
from tqdm import tqdm

from ravti.data.build import build_ravti_train_dataloader
from ravti.encoders.bioclip2_visual import BioCLIP2VisualEncoder
from ravti.encoders.bioclip_taxon import BioCLIPTaxonEncoder
from ravti.encoders.semantic_sdxl import SDXLSemanticTextEncoder
from ravti.models.projections import TaxonRefProjectionBundle
from ravti.models.reliability import CMCGate, build_cmc_features
from ravti.models.triple_stream_conditioning import TripleStreamConditioning
from ravti.retrieval.bio_retrieval import PrecomputedVisualBioRetriever
from ravti.retrieval.faiss_index import RetrievalHit
from ravti.training.sdxl_adapter_trainer import SDXLAdapterTrainer, TrainerConfig


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a tensor to a PIL image"""
    t = x.detach().float().cpu().clamp(0.0, 1.0)
    return T.ToPILImage()(t)


def _checkpoint_prefix(use_reference_condition: bool) -> str:
    """
     - taxonomy-only training → taxaAdapter; 
     - with reference condition training → ravti.
    """
    return "ravti" if use_reference_condition else "taxaAdapter"


def _build_payload(step: int, trainer: SDXLAdapterTrainer, cfg: dict) -> dict:
    """Build the payload for the checkpoint"""
    payload: dict = {
        "step": int(step),
        "conditioning_state_dict": trainer.conditioning.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "trainer_config": {
            "learning_rate": trainer.cfg.learning_rate,
            "mixed_precision": trainer.cfg.mixed_precision,
            "lambda_tax_base": trainer.cfg.lambda_tax_base,
            "lambda_ref_base": trainer.cfg.lambda_ref_base,
            "use_reference_condition": trainer.cfg.use_reference_condition,
        },
        "experiment_name": cfg.get("experiment_name", "ravti_default"),
    }
    if trainer.decoupled_attn_processors is not None:
        payload["decoupled_attn_processors_state_dict"] = trainer.decoupled_attn_processors.state_dict()
    if trainer.cmc_gate is not None:
        payload["cmc_gate_state_dict"] = trainer.cmc_gate.state_dict()
    return payload


def _save_step_checkpoint(output_dir: Path, prefix: str, step: int, trainer: SDXLAdapterTrainer, cfg: dict) -> Path:
    """Save the checkpoint for the current step"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{prefix}_{step:06d}.pt"
    torch.save(_build_payload(step, trainer, cfg), ckpt_path)
    return ckpt_path


def _write_loss_csv(path: Path, history: list[tuple[int, float]], x_label: str = "step") -> None:
    """Write the loss history to a CSV file"""
    # Create the parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the loss history to the CSV file
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([x_label, "loss"])
        for s, lo in history:
            if lo != lo:
                w.writerow([s, ""])
            else:
                w.writerow([s, f"{lo:.8f}"])


def _write_loss_plot_png(path: Path, history: list[tuple[int, float]], x_label: str = "step") -> bool:
    """Write the loss history to a PNG plot"""
    finite = [(s, float(lo)) for s, lo in history if lo == lo]
    if len(finite) < 2:
        return False
    # Plot the loss history
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    steps, losses = zip(*finite)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, linewidth=0.9)
    plt.xlabel(x_label)
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()
    return True


def _ref_source_label(
    use_reference_condition: bool,
    retriever: PrecomputedVisualBioRetriever | None,
    ref_vector: torch.Tensor | None,
    pil_ref: Image.Image | None,
) -> str:
    """
    Returns:
        str, the label for the reference stream was supplied this step (for logs / debugging).
    """
    if not use_reference_condition:
        return "no_ref"
    if ref_vector is not None:
        return "retrieved"
    if pil_ref is not None:
        return "fallback_pil" if retriever is not None else "local_pil"
    return "empty"


def _append_train_log(
    log_path: Path,
    step: int,
    loss: float,
    species: str,
    checkpoint_path: Path,
    is_final: bool,
    ref_source: str,
) -> None:
    """Append the train log of one training step to the log file"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"| {int(step)} | {float(loss):.6f} | {species} | {ref_source} | {checkpoint_path.name} | {int(is_final)} |\n"
        )


def _append_epoch_log(
    log_path: Path,
    epoch: int,
    mean_loss: float,
    n_batches: int,
    checkpoint_path: Path,
    is_final: bool,
) -> None:
    """Append one epoch-level log row."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        if mean_loss == mean_loss:
            loss_txt = f"{float(mean_loss):.6f}"
        else:
            loss_txt = ""
        f.write(f"| {int(epoch)} | {loss_txt} | {int(n_batches)} | {checkpoint_path.name} | {int(is_final)} |\n")


def _init_train_log(log_path: Path, cfg: dict, dtype: torch.dtype, mode: str = "step") -> None:
    """Initialize the train log file with the experiment configuration"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg.get("training") or {}
    with log_path.open("w", encoding="utf-8") as f:
        f.write("# RAVTI Train Log\n\n")
        f.write(f"- experiment_name: {cfg.get('experiment_name', 'ravti_default')}\n")
        f.write(f"- started_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- runtime_dtype: {dtype}\n")
        f.write(f"- mixed_precision: {train_cfg.get('mixed_precision', 'fp32')}\n")
        f.write(f"- learning_rate: {train_cfg.get('learning_rate', 1e-4)}\n")
        f.write(f"- max_train_steps: {train_cfg.get('max_train_steps', 1000)}\n")
        f.write(f"- max_train_epochs: {train_cfg.get('max_train_epochs', None)}\n")
        f.write(f"- save_every_steps: {train_cfg.get('save_every_steps', 10)}\n")
        f.write(f"- save_every_epochs: {train_cfg.get('save_every_epochs', 1)}\n")
        f.write(f"- use_early_stopping: {train_cfg.get('use_early_stopping', True)}\n")
        f.write(f"- tolerance: {train_cfg.get('tolerance', 5)}\n")
        f.write(f"- train_mode: {mode}\n")
        f.write(f"- use_reference_condition: {train_cfg.get('use_reference_condition', True)}\n")
        ret_cfg = cfg.get("retrieval") or {}
        f.write(f"- retrieval.enabled: {bool(ret_cfg.get('enabled', False))}\n")
        f.write("\n")
        if mode == "epoch":
            f.write("| epoch | mean_loss | n_batches | checkpoint | is_final |\n")
            f.write("|-------|-----------|-----------|------------|----------|\n")
        else:
            f.write("| step | loss | species | ref_source | checkpoint | is_final |\n")
            f.write("|------|------|---------|------------|------------|----------|\n")


def train_loop_from_config(cfg: dict, device: torch.device, dtype: torch.dtype) -> None:
    """Minimal multi-step training loop over `build_ravti_train_dataloader(cfg)`."""
    from diffusers import StableDiffusionXLPipeline

    # Get the configurations
    models_cfg = cfg.get("models") or {}
    train_cfg = cfg.get("training") or {}
    use_reference_condition = bool(train_cfg.get("use_reference_condition", True))
    retrieval_cfg = cfg.get("retrieval") or {}

    # Get the checkpoint prefix
    ckpt_prefix = _checkpoint_prefix(use_reference_condition)

    # Load the pipeline for the SDXL model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        models_cfg.get("sdxl_model_id"),
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.to(device)

    # ------------------------------------------------------------ Model Loading ------------------------------------------------------------
    # Load the encoders for the BioCLIP text and image streams
    tax = BioCLIPTaxonEncoder(models_cfg.get("bioclip_text_hub")).to(device)
    ref = BioCLIP2VisualEncoder(models_cfg.get("bioclip2_image_hub")).to(device)
    sem = SDXLSemanticTextEncoder(pipe)
    # Load the projection bundle for the taxon and reference streams
    proj = TaxonRefProjectionBundle(tax.embed_dim, ref.embed_dim, int(train_cfg.get("sdxl_hidden_size", 2048))).to(
        device
    )
    # Load the conditioning module for the taxon and reference streams
    conditioning = TripleStreamConditioning(proj).to(device)
    # Load the CMC gate for the reliability module
    cmc_gate = CMCGate(in_dim=4).to(device=device, dtype=dtype)

    # ------------------------------------------------------------ Trainer loading ------------------------------------------------------------
    # Load the trainer for the SDXL adapter
    trainer = SDXLAdapterTrainer(
        pipe=pipe,
        tax_encoder=tax,
        ref_encoder=ref,
        semantic_encoder=sem,
        conditioning=conditioning,
        cmc_gate=cmc_gate,
        cfg=TrainerConfig(
            learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
            mixed_precision=str(train_cfg.get("mixed_precision", "fp16")),
            lambda_tax_base=float(train_cfg.get("lambda_tax_base", 0.35)),
            lambda_ref_base=float(train_cfg.get("lambda_ref_base", 0.25)),
            use_reference_condition=use_reference_condition,
        ),
    )

    # ------------------------------------------------------------ Retriever Loading ------------------------------------------------------------
    # Initialize the retriever for the reference stream
    retriever = None
    # Load the retriever for the reference stream if enabled
    if use_reference_condition and bool(retrieval_cfg.get("enabled", False)):
        from ravti.config import resolve_paths

        paths = resolve_paths(cfg)
        index_name = str(retrieval_cfg.get("index_name", "species_index"))
        embedding_name = str(retrieval_cfg.get("embedding_name", f"{index_name}_image_embeddings"))
        retriever = PrecomputedVisualBioRetriever.from_precomputed(
            index_dir=paths.index_dir,
            index_name=index_name,
            embedding_name=embedding_name,
            taxon_encoder=tax,
        )

    # ------------------------------------------------- Training Configuration Initialization -------------------------------------------------
    # Get the training configuration
    max_epochs_raw = train_cfg.get("max_train_epochs", None)
    use_epoch_mode = max_epochs_raw is not None
    max_epochs = int(max_epochs_raw) if use_epoch_mode else 0
    save_every_epochs = int(train_cfg.get("save_every_epochs", 1))
    use_early_stopping = bool(train_cfg.get("use_early_stopping", True))
    early_stop_tolerance = int(train_cfg.get("tolerance", 5))
    max_steps = int(train_cfg.get("max_train_steps", 1000))
    save_every = int(train_cfg.get("save_every_steps", 10))
    out_dir_cfg = train_cfg.get("output_dir", "outputs/train_runs")
    prompt_tmpl = str(train_cfg.get("prompt_template", "wildlife photograph of {species}"))

    # Create the output directory if it doesn't exist
    output_dir = Path(out_dir_cfg)
    if not output_dir.is_absolute():
        from ravti.paths import project_root
        output_dir = (project_root() / output_dir).resolve()
    
    # Create the path, according the timestamp
    ts = datetime.now().strftime("%Y%m%d%H%M")
    log_path = output_dir / f"train_log_{ts}.txt"
    loss_csv_path = output_dir / f"{ckpt_prefix}_loss_{ts}.csv"
    loss_png_path = output_dir / f"{ckpt_prefix}_loss_{ts}.png"
    best_ckpt_path = output_dir / f"{ckpt_prefix}_best_{ts}.pt"
    # Initialize the train log
    _init_train_log(log_path, cfg, dtype, mode="epoch" if use_epoch_mode else "step")

    # Initialize the loss history
    loss_history: list[tuple[int, float]] = []
    # Build the data loader
    dl = build_ravti_train_dataloader(cfg)
    # Epoch mode: iterate full dataset each epoch
    if use_epoch_mode:
        if isinstance(dl.dataset, IterableDataset):
            raise ValueError("max_train_epochs mode requires finite dataset; IterableDataset is not supported.")
        if max_epochs <= 0:
            raise ValueError(f"max_train_epochs must be > 0, got {max_epochs}.")
        train_samples = len(dl.dataset)  # type: ignore[arg-type]
        max_steps = int(max_epochs * train_samples)
        pbar = tqdm(total=max_epochs, desc="train(epoch)", dynamic_ncols=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"- derived_train_samples: {train_samples}\n")
            f.write(f"- derived_max_train_steps_from_epochs: {max_steps}\n")
    else:
        # Initialize the progress bar
        pbar = tqdm(total=max_steps, desc="train", dynamic_ncols=True)
    # Initialize the parameters for the training loop
    step = 0
    ref_stats: dict[str, int] = {
        "retrieved": 0,
        "fallback_pil": 0,
        "local_pil": 0,
        "no_ref": 0,
        "empty": 0,
    }
    oom_steps = 0
    last_ref_source = "n/a"
    loss_val = float("nan")
    species = "n/a"
    last_ckpt_path: Path | None = None
    best_loss = float("inf")
    best_step = -1
    best_saved = False

    # -------------------------------------------------------------- Training Loop --------------------------------------------------------------
    if use_epoch_mode:
        epoch_last_ckpt: Path | None = None
        best_epoch_loss = float("inf")
        epochs_worse_than_best = 0
        early_stopped = False
        last_completed_epoch = 0
        for epoch in range(1, max_epochs + 1):
            epoch_losses: list[float] = []
            epoch_batches = 0
            for batch in dl:
                pixels = batch["pixels"].to(device=device, dtype=dtype)
                species_list = [str(x) for x in batch["species_texts"]]
                taxon_list = [str(x) for x in batch["taxonomy_lines"]]
                sample_ids = [str(x) for x in (batch.get("sample_ids") or [f"sample_{i}" for i in range(len(species_list))])]
                prompts = [prompt_tmpl.format(species=s, taxonomy=t) for s, t in zip(species_list, taxon_list)]

                ref_vectors: list[torch.Tensor | None] = []
                ref_images: list[Image.Image | None] = []
                cmc_feats_rows = []
                batch_ref_sources: list[str] = []
                for i, (species_i, taxon_i, sid_i) in enumerate(zip(species_list, taxon_list, sample_ids)):
                    ref_vector_i = None
                    hits_i: list[RetrievalHit] = []
                    if retriever is not None:
                        ref_vector_i, hits_i = retriever.retrieve_embedding(
                            species_query=species_i,
                            k=int(retrieval_cfg.get("k_default", 3)),
                            device=device,
                            fallback_queries=[taxon_i] if taxon_i != species_i else None,
                            exclude_ids={sid_i},
                        )
                    pil_ref_i = _tensor_to_pil(pixels[i]) if (use_reference_condition and ref_vector_i is None) else None
                    ref_source_i = _ref_source_label(use_reference_condition, retriever, ref_vector_i, pil_ref_i)
                    ref_stats[ref_source_i] = ref_stats.get(ref_source_i, 0) + 1
                    batch_ref_sources.append(ref_source_i)
                    cmc_feats_np_i = build_cmc_features(
                        species_query=species_i,
                        taxonomy_line=taxon_i,
                        hits=hits_i,
                        has_retrieved_ref=ref_vector_i is not None,
                    )
                    cmc_feats_rows.append(cmc_feats_np_i)
                    ref_vectors.append(ref_vector_i)
                    ref_images.append(pil_ref_i)

                cmc_feats = torch.tensor(cmc_feats_rows, device=device, dtype=dtype)
                species = species_list[0] if species_list else "n/a"
                last_ref_source = batch_ref_sources[0] if batch_ref_sources else "n/a"
                loss_val = float("nan")
                try:
                    loss = trainer.training_step(
                        pixels=pixels,
                        prompts=prompts,
                        taxon_strings=taxon_list,
                        ref_images=ref_images,
                        ref_vectors=ref_vectors,
                        cmc_features=cmc_feats,
                        device=device,
                        dtype=dtype,
                    )
                    loss_val = float(loss)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        trainer.optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        oom_steps += 1
                    else:
                        raise
                if math.isfinite(loss_val):
                    epoch_losses.append(loss_val)
                epoch_batches += 1

            mean_epoch_loss = (
                float(sum(epoch_losses) / max(len(epoch_losses), 1)) if epoch_losses else float("nan")
            )
            last_completed_epoch = epoch

            if math.isfinite(mean_epoch_loss):
                if mean_epoch_loss < best_epoch_loss:
                    best_epoch_loss = mean_epoch_loss
                    if use_early_stopping and early_stop_tolerance > 0:
                        epochs_worse_than_best = 0
                elif mean_epoch_loss > best_epoch_loss:
                    if use_early_stopping and early_stop_tolerance > 0:
                        epochs_worse_than_best += 1
                        if epochs_worse_than_best >= early_stop_tolerance:
                            early_stopped = True
                elif use_early_stopping and early_stop_tolerance > 0:
                    epochs_worse_than_best = 0
            elif use_early_stopping and early_stop_tolerance > 0:
                epochs_worse_than_best += 1
                if epochs_worse_than_best >= early_stop_tolerance:
                    early_stopped = True

            loss_history.append((epoch, mean_epoch_loss))
            pbar.set_postfix(epoch_loss=mean_epoch_loss, best=best_epoch_loss, worse_streak=epochs_worse_than_best)
            pbar.update(1)
            if save_every_epochs > 0 and (epoch % save_every_epochs == 0):
                ckpt_path = _save_step_checkpoint(
                    output_dir=output_dir,
                    prefix=ckpt_prefix,
                    step=epoch,
                    trainer=trainer,
                    cfg=cfg,
                )
                epoch_last_ckpt = ckpt_path
                _append_epoch_log(
                    log_path=log_path,
                    epoch=epoch,
                    mean_loss=mean_epoch_loss,
                    n_batches=epoch_batches,
                    checkpoint_path=ckpt_path,
                    is_final=False,
                )
            if early_stopped:
                break
        pbar.close()
        if epoch_last_ckpt is None and last_completed_epoch > 0:
            epoch_last_ckpt = _save_step_checkpoint(
                output_dir=output_dir,
                prefix=ckpt_prefix,
                step=last_completed_epoch,
                trainer=trainer,
                cfg=cfg,
            )
        _write_loss_csv(loss_csv_path, loss_history, x_label="epoch")
        plotted = _write_loss_plot_png(loss_png_path, loss_history, x_label="epoch")
        if epoch_last_ckpt is not None:
            _append_epoch_log(
                log_path=log_path,
                epoch=last_completed_epoch,
                mean_loss=loss_history[-1][1] if loss_history else float("nan"),
                n_batches=0,
                checkpoint_path=epoch_last_ckpt,
                is_final=True,
            )
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n## Loss curve\n\n")
            f.write(f"- csv: {loss_csv_path.name}\n")
            f.write(f"- plot: {loss_png_path.name} ({'ok' if plotted else 'skipped (matplotlib or <2 finite points)'})\n")
            if use_early_stopping:
                f.write(
                    f"- early_stopping: enabled, tolerance={early_stop_tolerance}, "
                    f"best_epoch_loss={best_epoch_loss if math.isfinite(best_epoch_loss) else 'n/a'}, "
                    f"early_stopped={early_stopped}, completed_epochs={last_completed_epoch}/{max_epochs}\n"
                )
            f.write("\n## Checkpoints\n\n")
            f.write(f"- per-epoch: `{ckpt_prefix}_{{epoch:06d}}.pt`\n")
            if epoch_last_ckpt is not None:
                f.write(f"- last_saved: {epoch_last_ckpt.name}\n")
            f.write("\n## Reference stream mix (completed epochs)\n\n")
            total_ref = sum(ref_stats.values())
            for label, n in sorted(ref_stats.items(), key=lambda x: -x[1]):
                pct = 100.0 * n / total_ref if total_ref else 0.0
                f.write(f"- {label}: {n} ({pct:.1f}%)\n")
            if oom_steps:
                f.write(f"- oom_skipped_steps: {oom_steps}\n")
        return
    # Iterate over the data loader
    while step < max_steps:
        for batch in dl:  # Iterate over the data loader
            pixels = batch["pixels"].to(device=device, dtype=dtype)
            if step >= max_steps:
                break

            # Get the metadata features from the batch, then initialize the reference vector and the retrieval hits
            species_list = [str(x) for x in batch["species_texts"]]
            taxon_list = [str(x) for x in batch["taxonomy_lines"]]
            sample_ids = [str(x) for x in (batch.get("sample_ids") or [f"sample_{i}" for i in range(len(species_list))])]
            prompts = [prompt_tmpl.format(species=s, taxonomy=t) for s, t in zip(species_list, taxon_list)]

            ref_vectors: list[torch.Tensor | None] = []
            ref_images: list[Image.Image | None] = []
            cmc_feats_rows = []
            batch_ref_sources: list[str] = []

            # Iterate over the batch
            for i, (species_i, taxon_i, sid_i) in enumerate(zip(species_list, taxon_list, sample_ids)):
                ref_vector_i = None
                hits_i: list[RetrievalHit] = []
                if retriever is not None:
                    ref_vector_i, hits_i = retriever.retrieve_embedding(
                        species_query=species_i,
                        k=int(retrieval_cfg.get("k_default", 3)),
                        device=device,
                        fallback_queries=[taxon_i] if taxon_i != species_i else None,
                        exclude_ids={sid_i},
                    )
                # Process the reference images if enabled
                pil_ref_i = _tensor_to_pil(pixels[i]) if (use_reference_condition and ref_vector_i is None) else None
                ref_source_i = _ref_source_label(use_reference_condition, retriever, ref_vector_i, pil_ref_i)
                ref_stats[ref_source_i] = ref_stats.get(ref_source_i, 0) + 1
                batch_ref_sources.append(ref_source_i)

                # Build the CMC features
                cmc_feats_np_i = build_cmc_features(
                    species_query=species_i,
                    taxonomy_line=taxon_i,
                    hits=hits_i,
                    has_retrieved_ref=ref_vector_i is not None,
                )
                cmc_feats_rows.append(cmc_feats_np_i)
                ref_vectors.append(ref_vector_i)
                ref_images.append(pil_ref_i)

            cmc_feats = torch.tensor(cmc_feats_rows, device=device, dtype=dtype)

            # Calculate the loss
            loss_val = float("nan")
            species = species_list[0] if species_list else "n/a"
            last_ref_source = batch_ref_sources[0] if batch_ref_sources else "n/a"
            try:
                loss = trainer.training_step(
                    pixels=pixels,
                    prompts=prompts,
                    taxon_strings=taxon_list,
                    ref_images=ref_images,
                    ref_vectors=ref_vectors,
                    cmc_features=cmc_feats,
                    device=device,
                    dtype=dtype,
                )
                loss_val = float(loss)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    trainer.optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    oom_steps += 1
                    pbar.set_postfix(loss=float("nan"), species=species[:32], ref=last_ref_source, oom=1)
                    pbar.update(1)
                    step += 1
                    loss_history.append((step, float("nan")))
                    continue
                raise

            # Update the progress bar
            pbar.set_postfix(loss=loss_val, species=species[:32], ref=last_ref_source, bsz=len(species_list))
            pbar.update(1)
            step += 1
            loss_history.append((step, loss_val))

            # Save all the checkpoints and the best checkpoint
            if save_every > 0 and step % save_every == 0:
                # Save the current checkpoint
                ckpt_path = _save_step_checkpoint(
                    output_dir=output_dir,
                    prefix=ckpt_prefix,
                    step=step,
                    trainer=trainer,
                    cfg=cfg,
                )
                last_ckpt_path = ckpt_path
                # Update the best checkpoint if the current loss is less than the best loss
                if math.isfinite(loss_val) and loss_val < best_loss:
                    best_loss = float(loss_val)
                    best_step = int(step)
                    torch.save(_build_payload(step, trainer, cfg), best_ckpt_path)
                    best_saved = True
                # Append the train log
                _append_train_log(
                    log_path=log_path,
                    step=step,
                    loss=loss_val,
                    species=species,
                    checkpoint_path=ckpt_path,
                    is_final=False,
                    ref_source=last_ref_source,
                )
            # If the current step reaches the maximum number of steps, break the loop
            if step >= max_steps:
                break
        else:
            if isinstance(dl.dataset, IterableDataset):
                break
    pbar.close()

    # -------------------------------------------------------------- Training Loop End --------------------------------------------------------------
    if max_steps > 0 and step > 0:  # If the model is trained for at least one step
        if save_every > 0 and (step % save_every) != 0:  # If training is not completed
            tail_path = _save_step_checkpoint(
                output_dir=output_dir,
                prefix=ckpt_prefix,
                step=step,
                trainer=trainer,
                cfg=cfg,
            )
            last_ckpt_path = tail_path
            # Update the best checkpoint if the current loss is less than the best loss
            if math.isfinite(loss_val) and loss_val < best_loss:
                best_loss = float(loss_val)
                best_step = int(step)
                torch.save(_build_payload(step, trainer, cfg), best_ckpt_path)
                best_saved = True
        elif save_every <= 0:  # If save_every_steps <= 0, still write one final checkpoint.
            only_path = _save_step_checkpoint(
                output_dir=output_dir,
                prefix=ckpt_prefix,
                step=step,
                trainer=trainer,
                cfg=cfg,
            )
            last_ckpt_path = only_path

    # Write the loss history to a CSV file and a PNG plot
    _write_loss_csv(loss_csv_path, loss_history)
    plotted = _write_loss_plot_png(loss_png_path, loss_history)

    # Append the final training log
    summary_ckpt = best_ckpt_path if best_saved and best_ckpt_path.is_file() else last_ckpt_path
    final_ckpt_for_log = summary_ckpt if summary_ckpt is not None else (output_dir / "(no_checkpoint)")
    _append_train_log(
        log_path=log_path,
        step=step,
        loss=float(loss_val) if math.isfinite(loss_val) else float("nan"),
        species=species,
        checkpoint_path=final_ckpt_for_log,
        is_final=True,
        ref_source=last_ref_source,
    )

    # ------------------------------------------------------ Training Summary ------------------------------------------------------
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n## Loss curve\n\n")
        f.write(f"- csv: {loss_csv_path.name}\n")
        f.write(f"- plot: {loss_png_path.name} ({'ok' if plotted else 'skipped (matplotlib or <2 finite points)'})\n")
        f.write("\n## Checkpoints\n\n")
        f.write(f"- per-step: `{ckpt_prefix}_{{step:06d}}.pt`\n")
        if last_ckpt_path is not None:
            f.write(f"- last_saved: {last_ckpt_path.name}\n")
        if best_saved:
            f.write(f"- best (save the checkpoint with the lowest loss): {best_ckpt_path.name} @ step={best_step}, loss={best_loss:.6f}\n")
        else:
            f.write("- best: no available checkpoint (no finite loss or no checkpoint was generated)\n")
        f.write("\n## Reference stream mix (completed steps)\n\n")
        total_ref = sum(ref_stats.values())
        for label, n in sorted(ref_stats.items(), key=lambda x: -x[1]):
            pct = 100.0 * n / total_ref if total_ref else 0.0
            f.write(f"- {label}: {n} ({pct:.1f}%)\n")
        if oom_steps:
            f.write(f"- oom_skipped_steps: {oom_steps}\n")
