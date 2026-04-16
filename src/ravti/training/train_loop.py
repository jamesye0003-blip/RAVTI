from __future__ import annotations

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
from ravti.models.triple_stream_conditioning import TripleStreamConditioning
from ravti.training.sdxl_adapter_trainer import SDXLAdapterTrainer, TrainerConfig


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    t = x.detach().float().cpu().clamp(0.0, 1.0)
    return T.ToPILImage()(t)


def _save_checkpoint(
    output_dir: Path,
    step: int,
    trainer: SDXLAdapterTrainer,
    cfg: dict,
    is_final: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "adapter_final.pt" if is_final else f"adapter_step_{step:06d}.pt"
    ckpt_path = output_dir / filename
    payload = {
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
    torch.save(payload, ckpt_path)
    return ckpt_path


def _append_train_log(
    log_path: Path,
    step: int,
    loss: float,
    species: str,
    checkpoint_path: Path,
    is_final: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"| {int(step)} | {float(loss):.6f} | {species} | {checkpoint_path.name} | {int(is_final)} |\n"
        )


def _init_train_log(log_path: Path, cfg: dict, dtype: torch.dtype) -> None:
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
        f.write(f"- save_every_steps: {train_cfg.get('save_every_steps', 10)}\n")
        f.write(f"- use_reference_condition: {train_cfg.get('use_reference_condition', True)}\n")
        f.write("\n")
        f.write("| step | loss | species | checkpoint | is_final |\n")
        f.write("|------|------|---------|------------|----------|\n")


def train_loop_from_config(cfg: dict, device: torch.device, dtype: torch.dtype) -> None:
    """Minimal multi-step training loop over `build_ravti_train_dataloader(cfg)`."""
    from diffusers import StableDiffusionXLPipeline

    models_cfg = cfg.get("models") or {}
    train_cfg = cfg.get("training") or {}
    use_reference_condition = bool(train_cfg.get("use_reference_condition", True))

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
    max_steps = int(train_cfg.get("max_train_steps", 1000))
    save_every = int(train_cfg.get("save_every_steps", 10))
    out_dir_cfg = train_cfg.get("output_dir", "outputs/train_runs")
    output_dir = Path(out_dir_cfg)
    if not output_dir.is_absolute():
        from ravti.paths import project_root

        output_dir = (project_root() / output_dir).resolve()
    ts = datetime.now().strftime("%Y%m%d%H%M")
    log_path = output_dir / f"train_log_{ts}.txt"
    _init_train_log(log_path, cfg, dtype)
    prompt_tmpl = str(train_cfg.get("prompt_template", "wildlife photograph of {species}, natural lighting, sharp focus"))
    cmc_default = float(train_cfg.get("cmc_train_default", 0.5))

    pbar = tqdm(total=max_steps, desc="train", dynamic_ncols=True)
    step = 0
    while step < max_steps:
        for batch in dl:
            pixels = batch["pixels"].to(device=device, dtype=dtype)
            for i in range(pixels.shape[0]):
                if step >= max_steps:
                    break
                species = batch["species_texts"][i]
                taxon_line = batch["taxonomy_lines"][i]
                prompt = prompt_tmpl.format(species=species, taxonomy=taxon_line)
                pil_ref = _tensor_to_pil(pixels[i]) if use_reference_condition else None
                try:
                    loss = trainer.training_step(
                        pixels=pixels[i : i + 1],
                        prompt=prompt,
                        taxon_string=taxon_line,
                        ref_images=([pil_ref] if pil_ref is not None else None),
                        cmc=cmc_default,
                        device=device,
                        dtype=dtype,
                    )
                except RuntimeError as e:
                    # Handle OOM errors
                    if "out of memory" in str(e).lower():
                        trainer.optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        pbar.set_postfix(loss=float("nan"), species=species[:32], oom=1)
                        pbar.update(1)
                        step += 1
                        continue
                    raise
                pbar.set_postfix(loss=float(loss), species=species[:32])
                pbar.update(1)
                step += 1
                if save_every > 0 and step % save_every == 0:
                    ckpt_path = _save_checkpoint(output_dir, step, trainer, cfg, is_final=False)
                    _append_train_log(
                        log_path=log_path,
                        step=step,
                        loss=float(loss),
                        species=species,
                        checkpoint_path=ckpt_path,
                        is_final=False,
                    )
            if step >= max_steps:
                break
        else:
            if isinstance(dl.dataset, IterableDataset):
                break
    pbar.close()
    final_ckpt = _save_checkpoint(output_dir, step, trainer, cfg, is_final=True)
    _append_train_log(
        log_path=log_path,
        step=step,
        loss=float(loss) if "loss" in locals() else float("nan"),
        species=species if "species" in locals() else "n/a",
        checkpoint_path=final_ckpt,
        is_final=True,
    )
