from __future__ import annotations

import copy
import hashlib
import json
import warnings
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from tqdm import tqdm

from ravti.config import load_yaml_config, resolve_paths, resolve_runtime_dtype
from ravti.encoders.bioclip2_visual import BioCLIP2VisualEncoder
from ravti.encoders.bioclip_taxon import BioCLIPTaxonEncoder
from ravti.encoders.semantic_sdxl import SDXLSemanticTextEncoder, iter_sdxl_text_encoders
from ravti.eval.classification_accuracy import ClassificationAccuracyMetric
from ravti.eval.species_checklist import (
    build_species_balanced_checklist_from_index,
    build_species_balanced_checklist_from_manifest,
    common_name_from_meta,
)
from ravti.eval.t2i_alignment import build_t2i_metrics_from_config
from ravti.models.decoupled_attention import attach_decoupled_cross_attention
from ravti.models.projections import TaxonRefProjectionBundle
from ravti.models.reliability import CMCGate, build_cmc_features, reliability_lambdas
from ravti.models.triple_stream_conditioning import TripleStreamConditioning
from ravti.paths import project_root
from ravti.retrieval.bio_retrieval import PrecomputedVisualBioRetriever

GenerationMode = Literal["b0_sdxl", "b1_taxonomy_only", "ravti"]
AdapterWeightsMode = Literal["trained", "random_init"]

MODE_ALIASES: dict[str, GenerationMode] = {
    "b0": "b0_sdxl",
    "b0_sdxl": "b0_sdxl",
    "sdxl": "b0_sdxl",
    "b1": "b1_taxonomy_only",
    "b1_taxonomy_only": "b1_taxonomy_only",
    "taxonomy_only": "b1_taxonomy_only",
    "tax_only": "b1_taxonomy_only",
    "ravti": "ravti",
}


def normalize_generation_mode(raw: str) -> GenerationMode:
    key = str(raw or "").strip().lower().replace(" ", "_")
    if key not in MODE_ALIASES:
        allowed = ", ".join(sorted(set(MODE_ALIASES.values())))
        raise ValueError(f"Unknown evaluation.generation.mode={raw!r}; use one of: {allowed}")
    return MODE_ALIASES[key]


def normalize_adapter_weights(raw: str | None) -> AdapterWeightsMode:
    """
    ``evaluation.generation.adapter_weights``：

    - ``trained``（默认）：从 checkpoint 加载 ``conditioning``、解耦 attention processors、CMCGate。
    - ``random_init``：P0 诊断用 — **不**加载 checkpoint，保持 ``_load_adapter_bundle`` 后的随机初始化，
      等价「零训练步 adapter」；无需提供 ``adapter_checkpoint``。用于区分实现/管线问题与训练权重问题。
    """
    key = str(raw or "trained").strip().lower().replace(" ", "_")
    if key in ("trained", "train", "checkpoint", "from_checkpoint"):
        return "trained"
    if key in ("random_init", "random", "untrained", "no_checkpoint", "zero_train", "scratch"):
        return "random_init"
    raise ValueError(
        f"Unknown evaluation.generation.adapter_weights={raw!r}; use 'trained' or 'random_init'."
    )


def _normalize_ckpt_components(raw: Any) -> tuple[bool, bool, bool]:
    """
    Parse ``evaluation.generation.checkpoint_components`` into three toggles:
    ``(load_conditioning, load_processors, load_cmc_gate)``.

    Accepted examples:
    - omitted / "all" / ["all"] -> all True
    - ["conditioning"] -> only conditioning
    - ["processors"] -> only decoupled attention processors
    - ["conditioning", "processors"] -> load two trainable branches, skip gate
    """
    if raw is None:
        return True, True, True
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(x) for x in raw]
    else:
        raise ValueError("evaluation.generation.checkpoint_components must be string or list of strings.")
    keys = {str(x).strip().lower().replace(" ", "_") for x in items if str(x).strip()}
    if not keys or "all" in keys:
        return True, True, True
    aliases = {
        "conditioning": "conditioning",
        "cond": "conditioning",
        "triple_stream": "conditioning",
        "projection": "conditioning",
        "projections": "conditioning",
        "processors": "processors",
        "processor": "processors",
        "decoupled_attn_processors": "processors",
        "decoupled_attention": "processors",
        "attn": "processors",
        "cmc_gate": "cmc_gate",
        "cmc": "cmc_gate",
        "gate": "cmc_gate",
    }
    normalized: set[str] = set()
    for k in keys:
        if k not in aliases:
            raise ValueError(
                f"Unknown checkpoint component {k!r}; use any of ['conditioning', 'processors', 'cmc_gate', 'all']."
            )
        normalized.add(aliases[k])
    return "conditioning" in normalized, "processors" in normalized, "cmc_gate" in normalized


def _git_commit_short(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()[:12] or None
    except Exception:
        return None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_checklist(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Resolution order (first match wins):

    1. ``evaluation.generation.prompts_file`` — external YAML list or ``{checklist: [...]}``.
    2. Non-empty ``evaluation.generation.checklist`` inline list.
    3. ``balanced_checklist.enabled`` — ``num_species`` distinct species, one checklist row each,
       with optional Animalia/Plantae quota balancing (see ``species_checklist``).

    If both ``prompts_file`` and ``balanced_checklist.enabled`` are set, only the file is used;
    a warning is emitted because this is easy to misconfigure.
    """
    ev = cfg.get("evaluation") or {}
    gen = ev.get("generation") or {}
    balanced = gen.get("balanced_checklist") or {}
    prompts_file = gen.get("prompts_file")
    if prompts_file and bool(balanced.get("enabled", False)):
        warnings.warn(
            "evaluation.generation.prompts_file is set, so balanced_checklist is ignored. "
            "Remove prompts_file or set balanced_checklist.enabled: false to use index-based species balancing.",
            UserWarning,
            stacklevel=2,
        )
    if prompts_file:
        p = Path(str(prompts_file))
        if not p.is_absolute():
            p = (project_root() / p).resolve()
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return [dict(x) for x in data]
        if isinstance(data, dict):
            ch = data.get("checklist")
            if isinstance(ch, list):
                return [dict(x) for x in ch]
        raise ValueError(f"prompts_file {p} must be a YAML list or mapping with 'checklist'")
    ch = gen.get("checklist") or []
    if not isinstance(ch, list):
        raise ValueError("evaluation.generation.checklist must be a list")
    if ch:
        return [dict(x) for x in ch]
    if bool(balanced.get("enabled", False)):
        if balanced.get("num_samples") is not None and balanced.get("num_species") is None:
            warnings.warn(
                "balanced_checklist.num_samples is deprecated; use num_species (one row per species).",
                DeprecationWarning,
                stacklevel=2,
            )
        n = int(balanced.get("num_species", balanced.get("num_samples", 500)))
        seed = int(balanced.get("seed", cfg.get("seed", 42)))
        eval_manifest_raw = gen.get("eval_manifest_file")
        if not eval_manifest_raw:
            eval_manifest_raw = ((cfg.get("dataset") or {}).get("split") or {}).get("eval_manifest_jsonl")
        if eval_manifest_raw:
            p = Path(str(eval_manifest_raw))
            if not p.is_absolute():
                p = (project_root() / p).resolve()
            return build_species_balanced_checklist_from_manifest(p, num_species=n, seed=seed)
        return build_species_balanced_checklist_from_index(cfg, num_species=n, seed=seed)
    return []


def _resolve_ckpt_path(cfg: dict[str, Any], path_raw: str | None) -> Path | None:
    if not path_raw:
        return None
    p = Path(str(path_raw))
    if not p.is_absolute():
        p = (project_root() / p).resolve()
    return p


def _mode_expected_use_reference(mode: GenerationMode) -> bool:
    return mode == "ravti"


def _mode_ckpt_prefix(mode: GenerationMode) -> str:
    return "ravti" if mode == "ravti" else "taxaAdapter"


def _resolve_processor_alpha(gen_cfg: dict[str, Any], mode: GenerationMode) -> float:
    """
    Resolve decoupled-attention residual scale for eval:
    1) evaluation.generation.processor_alpha_by_mode.<mode>
    2) evaluation.generation.fixed.processor_alpha
    3) default 1.0
    """
    by_mode = gen_cfg.get("processor_alpha_by_mode")
    if isinstance(by_mode, dict):
        raw = by_mode.get(mode)
        if raw is not None:
            return float(raw)
        for k, v in by_mode.items():
            try:
                if normalize_generation_mode(str(k)) == mode:
                    return float(v)
            except Exception:
                continue
    fixed = gen_cfg.get("fixed") or {}
    return float(fixed.get("processor_alpha", 1.0))


def _latest_matching_checkpoint(output_dir: Path, prefix: str) -> Path | None:
    # Prefer explicit best checkpoints, then fall back to latest step checkpoint.
    best = sorted(output_dir.glob(f"{prefix}_best_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    step = sorted(output_dir.glob(f"{prefix}_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if step:
        return step[0]
    return None


def _resolve_mode_checkpoint(cfg: dict[str, Any], mode: GenerationMode) -> Path | None:
    if mode == "b0_sdxl":
        return None
    ev = cfg.get("evaluation") or {}
    gen = ev.get("generation") or {}
    # Highest priority: explicit mode mapping.
    by_mode = gen.get("adapter_checkpoint_by_mode")
    if isinstance(by_mode, dict):
        raw = by_mode.get(mode)
        if raw:
            p = _resolve_ckpt_path(cfg, str(raw))
            if p is not None and p.is_file():
                return p
    # Next: single path shared by the run.
    single = _resolve_ckpt_path(cfg, gen.get("adapter_checkpoint"))
    if single is not None and single.is_file():
        return single
    # Finally: auto discover in training.output_dir from naming convention.
    train_cfg = cfg.get("training") or {}
    out_dir = Path(str(train_cfg.get("output_dir", "outputs/train_runs")))
    if not out_dir.is_absolute():
        out_dir = (project_root() / out_dir).resolve()
    return _latest_matching_checkpoint(out_dir, _mode_ckpt_prefix(mode))


def _ref_source_label(
    use_reference_condition: bool,
    retriever: PrecomputedVisualBioRetriever | None,
    ref_vector: torch.Tensor | None,
    pil_ref: Image.Image | None,
) -> str:
    if not use_reference_condition:
        return "no_ref"
    if ref_vector is not None:
        return "retrieved"
    if pil_ref is not None:
        return "fallback_pil" if retriever is not None else "local_pil"
    return "empty"


def _build_conditioned_embeds(
    *,
    sem_encoder: SDXLSemanticTextEncoder,
    conditioning: TripleStreamConditioning,
    tax_enc: BioCLIPTaxonEncoder,
    ref_enc: BioCLIP2VisualEncoder,
    prompt: str,
    taxon_line: str,
    ref_vector: torch.Tensor | None,
    pil_ref: Image.Image | None,
    use_reference_condition: bool,
    lambda_tax: float,
    lambda_ref: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct ``prompt_embeds`` (sequence dimension concatenation of semantic / taxonomy / reference image token) and ``pooled_prompt_embeds`` used by SDXL UNet.

    Key points (differences from pure SDXL):
    - ``sem_encoder`` only responsible for the two-stage CLIP text embedding of the "Natural Language prompt" (consistent with diffusers default).
    - ``tax_enc`` encodes ``taxon_line`` (such as taxonomy_line in the checklist) into a vector, and projects it as an **additional 1 token** to concatenate to the cross-attn condition sequence.
    - In RAVTI mode, ``ref_enc`` or precomputed ``ref_vector`` provides the **3rd token**; B1 only when ``use_reference_condition=False``, does not concatenate the reference token.
    - ``pooled_prompt_embeds`` **not** mix tax/ref, still only from the semantic branch - consistent with the original SDXL design, the time step condition is still mainly carried by the pooled vector.
    - If the distribution difference between ``lambda_tax`` / ``lambda_ref`` and during training is large, or the tax/ref vector does not match the scale of the SDXL text stream, the decoupled attention in the UNet will receive the condition of "shape legal but semantic混乱",容易出现条纹 / 对称伪影 / 彩噪。
    """
    with torch.inference_mode():
        # Equivalent to pipe(prompt=...) inside: OpenCLIP + CLIP-G dual encoder text embedding.
        sem = sem_encoder.encode(prompt, device=device, dtype=dtype)
        # BioCLIP text tower: one species / taxonomic string → fixed dimension vector; format needs to be consistent with the checklist during training.
        tax = tax_enc([taxon_line]).to(device=device).clone()
        ref: torch.Tensor | None = None
        if use_reference_condition:
            if ref_vector is not None:
                # From FAISS species-level reference embedding (already a vector in the BioCLIP2 image space).
                ref = ref_vector.to(device=device).clone()
                if ref.ndim == 1:
                    ref = ref.unsqueeze(0)
            elif pil_ref is not None:
                # When the reference image path is explicitly given in the checklist, use ref_enc to encode on-site (same space as the retrieval vector).
                ref = ref_enc([pil_ref], device=device).to(device=device).clone()
                if ref.shape[0] > 1:
                    ref = ref.mean(dim=0, keepdim=True)
        # In TripleStreamConditioning, concatenate along the seq_len dimension; the output is the prompt_embeds of the pipe under the adapter path.
        prompt_embeds = conditioning(
            sem.prompt_embeds,
            tax,
            ref,
            lambda_tax=lambda_tax,
            lambda_ref=lambda_ref,
            use_reference_condition=use_reference_condition,
        )
    return prompt_embeds, sem.pooled_prompt_embeds


def _pil_to_model_input(pil: Image.Image) -> torch.Tensor:
    t = T.ToTensor()(pil.convert("RGB")).unsqueeze(0)
    return t


def _parse_optional_class_index(row: dict[str, Any]) -> int | None:
    raw = row.get("class_index", row.get("target_class_index"))
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _safe_image_stem(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        s = "sample"
    # Windows-safe filename chars: replace reserved characters and collapse spaces.
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = "_".join(s.split())
    return s[:180]


def _resolve_metric_switches(cfg: dict[str, Any]) -> dict[str, bool]:
    ev = cfg.get("evaluation") or {}
    raw = ev.get("metrics") or {}
    if not isinstance(raw, dict):
        raw = {}
    switches = {
        "clip_common_name": bool(raw.get("clip_common_name", True)),
        "bioclip_taxonomic": bool(raw.get("bioclip_taxonomic", True)),
        "cas_at_1": bool(raw.get("cas_at_1", True)),
        "cas_at_5": bool(raw.get("cas_at_5", True)),
    }
    # CAS@5 implies CAS classifier path; keep independent toggles explicit.
    if switches["cas_at_5"] and not switches["cas_at_1"]:
        warnings.warn(
            "evaluation.metrics.cas_at_5=true while cas_at_1=false; keeping this config, "
            "but note both rely on the same classifier inference path.",
            UserWarning,
            stacklevel=2,
        )
    return switches


def _runtime_cmc_for_eval(
    species: str,
    taxon_line: str,
    hits: list[Any],
    has_retrieved_ref: bool,
    use_reference_condition: bool,
    cmc_gate: CMCGate | None,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """B1 returns 0 (no reference branch); RAVTI uses retrieval hits + whether hit or not scalar features through CMCGate to get the confidence, used by reliability_lambdas."""
    if not use_reference_condition:
        return 0.0
    if cmc_gate is None:
        raise RuntimeError("CMCGate is required in eval for reference-conditioned mode.")
    feats = build_cmc_features(
        species_query=species,
        taxonomy_line=taxon_line,
        hits=hits,
        has_retrieved_ref=has_retrieved_ref,
    )
    with torch.inference_mode():
        ft = torch.tensor(feats, device=device, dtype=dtype)
        return float(cmc_gate(ft).reshape(-1)[0].item())


@dataclass
class _AdapterBundle:
    pipe: Any
    tax_encoder: BioCLIPTaxonEncoder
    ref_encoder: BioCLIP2VisualEncoder
    semantic_encoder: SDXLSemanticTextEncoder
    conditioning: TripleStreamConditioning
    processors: torch.nn.ModuleList
    cmc_gate: CMCGate


def _load_adapter_bundle(cfg: dict[str, Any], device: torch.device, dtype: torch.dtype) -> _AdapterBundle:
    """Assemble the evaluation pipeline: SDXL + BioCLIP text/image encoders + three-stream projection + decoupled UNet processors + CMCGate (weights will be injected by checkpoint)."""
    from diffusers import StableDiffusionXLPipeline

    models_cfg = cfg.get("models") or {}
    train_cfg = cfg.get("training") or {}
    pipe = StableDiffusionXLPipeline.from_pretrained(
        models_cfg.get("sdxl_model_id"),
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.to(device)
    tax = BioCLIPTaxonEncoder(models_cfg.get("bioclip_text_hub")).to(device)
    ref = BioCLIP2VisualEncoder(models_cfg.get("bioclip2_image_hub")).to(device)
    sem = SDXLSemanticTextEncoder(pipe)
    proj = TaxonRefProjectionBundle(
        tax.embed_dim, ref.embed_dim, int(train_cfg.get("sdxl_hidden_size", 2048))
    ).to(device=device, dtype=dtype)
    conditioning = TripleStreamConditioning(proj).to(device=device, dtype=dtype)
    cmc_gate = CMCGate(in_dim=4).to(device=device, dtype=dtype)
    # 将 UNet 中 cross-attn 替换为可区分「文本 / tax / ref」三路的 processor；与训练脚本 attach 方式需一致。
    processors = attach_decoupled_cross_attention(pipe.unet)
    ref_param = next(pipe.unet.parameters())
    processors.to(device=ref_param.device, dtype=ref_param.dtype)
    conditioning.eval()
    cmc_gate.eval()
    tax.eval()
    ref.eval()
    for enc in iter_sdxl_text_encoders(pipe):
        enc.eval()
    pipe.unet.eval()
    return _AdapterBundle(
        pipe=pipe,
        tax_encoder=tax,
        ref_encoder=ref,
        semantic_encoder=sem,
        conditioning=conditioning,
        processors=processors,
        cmc_gate=cmc_gate,
    )


def _load_checkpoint_into_bundle(
    bundle: _AdapterBundle,
    ckpt_path: Path,
    *,
    strict_processors: bool = False,
    load_conditioning: bool = True,
    load_processors: bool = True,
    load_cmc_gate: bool = True,
) -> None:
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    if load_conditioning:
        bundle.conditioning.load_state_dict(payload["conditioning_state_dict"], strict=True)
    if load_processors:
        sd = payload.get("decoupled_attn_processors_state_dict")
        if sd is not None:
            bundle.processors.load_state_dict(sd, strict=strict_processors)
    if load_cmc_gate:
        cmc_sd = payload.get("cmc_gate_state_dict")
        if cmc_sd is None:
            raise ValueError(
                f"Checkpoint {ckpt_path} is missing cmc_gate_state_dict. "
                "Current pipeline requires CMCGate; please retrain with the new training code."
            )
        bundle.cmc_gate.load_state_dict(cmc_sd, strict=False)


def _validate_checkpoint_mode_compat(ckpt_path: Path, mode: GenerationMode) -> None:
    """Fail fast if checkpoint was trained with a different reference-condition mode."""
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    trainer_cfg = payload.get("trainer_config") or {}
    if "use_reference_condition" not in trainer_cfg:
        return
    trained_ref = bool(trainer_cfg.get("use_reference_condition"))
    expected_ref = _mode_expected_use_reference(mode)
    if trained_ref != expected_ref:
        raise ValueError(
            f"Checkpoint mode mismatch: mode={mode} expects use_reference_condition={expected_ref}, "
            f"but checkpoint {ckpt_path} has use_reference_condition={trained_ref}."
        )


def _maybe_build_retriever(
    cfg: dict[str, Any], device: torch.device, tax_enc: BioCLIPTaxonEncoder
) -> PrecomputedVisualBioRetriever | None:
    retrieval_cfg = cfg.get("retrieval") or {}
    if not bool(retrieval_cfg.get("enabled", False)):
        return None
    paths = resolve_paths(cfg)
    index_name = str(retrieval_cfg.get("index_name", "species_index"))
    embedding_name = str(retrieval_cfg.get("embedding_name", f"{index_name}_image_embeddings"))
    return PrecomputedVisualBioRetriever.from_precomputed(
        index_dir=paths.index_dir,
        index_name=index_name,
        embedding_name=embedding_name,
        taxon_encoder=tax_enc,
    )


def _run_generation_benchmark_single(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Single ``evaluation.generation.mode`` generation evaluation: save images and manifest/metrics in ``outputs/eval/<run_id>/<mode>/``.

    Mode semantic summary:
    - ``b0_sdxl``: standard ``StableDiffusionXLPipeline``, only use text prompt (with optional negative), **not** load adapter weights.
    - ``b1_taxonomy_only``: load ``conditioning`` + decoupled cross-attn processors from checkpoint; **not** use retrieval / CMC dynamic gate (``use_reference_condition=False``, ``lam_r=0``).
    - ``ravti``: on top of B1, ``use_reference_condition=True``; if ``retrieval.enabled`` is configured, use species retrieval reference embedding, and use **CMCGate** to estimate the confidence during inference, dynamically scale ``lambda_tax`` / ``lambda_ref`` (the same function as the "reliability weighting" during training, but the cmc value may come from the gate rather than the fixed hyper-parameters).

    If ``evaluation.generation.adapter_weights`` is ``random_init``, B1/RAVTI **skip checkpoint**, for P0 (random adapter vs trained weights) comparison; at this time, ``adapter_checkpoint`` is not required.

    If the generated image appears to have strong geometric artifacts or pure color noise, you can first check the training resolution/steps and fixed.height/width/steps, whether the checkpoint is consistent with the ``use_reference_condition`` of this mode, and whether the taxon/ref negative branch is consistent with the training when ``guidance_scale>1``.
    """
    from diffusers import StableDiffusionXLPipeline

    root = project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_runtime_dtype(cfg, device)

    ev = cfg.get("evaluation") or {}
    gen = ev.get("generation") or {}
    mode = normalize_generation_mode(str(gen.get("mode", "b0_sdxl")))
    adapter_weights_cfg = normalize_adapter_weights(gen.get("adapter_weights"))
    adapter_weights = adapter_weights_cfg if mode != "b0_sdxl" else "b0_sdxl"
    load_adapter_checkpoint = mode != "b0_sdxl" and adapter_weights_cfg == "trained"
    load_conditioning, load_processors, load_cmc_gate = _normalize_ckpt_components(
        gen.get("checkpoint_components")
    )
    fixed = gen.get("fixed") or {}
    seed = int(fixed.get("seed", cfg.get("seed", 42)))
    steps = int(fixed.get("num_inference_steps", 30))
    # Note: if training.image_size is inconsistent with height/width here, it is a distribution shift, and the adapter often crashes into noise or stripes at higher resolutions.
    height = int(fixed.get("height", 512))
    width = int(fixed.get("width", 512))
    guidance_scale = float(fixed.get("guidance_scale", 1.0))
    processor_alpha = _resolve_processor_alpha(gen, mode)
    # cmc in the configuration: used to get a pair of static lambda in the non-RAVTI path; RAVTI main loop will be covered by runtime CMC.
    cmc_cfg = float(fixed.get("cmc", (cfg.get("training") or {}).get("cmc_train_default", 0.5)))
    train_cfg = cfg.get("training") or {}
    lambda_tax_base = float(train_cfg.get("lambda_tax_base", 0.35))
    lambda_ref_base = float(train_cfg.get("lambda_ref_base", 0.25))
    lambda_tax, lambda_ref = reliability_lambdas(cmc_cfg, lambda_tax_base, lambda_ref_base)
    fixed_lambda_tax = fixed.get("lambda_tax")
    fixed_lambda_ref = fixed.get("lambda_ref")
    use_fixed_lambdas = fixed_lambda_tax is not None or fixed_lambda_ref is not None

    neg_prompt = str(gen.get("negative_prompt", ""))
    neg_taxon = str(gen.get("negative_taxon", ""))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    run_id = gen.get("run_id")
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = str(run_id)
    out_root = Path(str(gen.get("output_dir", "outputs/eval")))
    if not out_root.is_absolute():
        out_root = (root / out_root).resolve()
    method_dir = out_root / run_id / mode
    method_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = _resolve_mode_checkpoint(cfg, mode)
    if mode in ("b1_taxonomy_only", "ravti") and load_adapter_checkpoint:
        if ckpt_path is None or not ckpt_path.is_file():
            raise FileNotFoundError(
                f"No checkpoint found for mode={mode}. Set evaluation.generation.adapter_checkpoint "
                f"or adapter_checkpoint_by_mode.{mode}, or ensure training.output_dir has {_mode_ckpt_prefix(mode)}_best_*.pt."
            )
        _validate_checkpoint_mode_compat(ckpt_path, mode)

    checklist = load_checklist(cfg)
    if not checklist:
        raise ValueError("evaluation checklist is empty")

    models_cfg = cfg.get("models") or {}
    prompt_tmpl = str(
        train_cfg.get(
            "prompt_template",
            "wildlife photograph of {species}, natural lighting, sharp focus",
        )
    )

    manifest_path = method_dir / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    metric_switches = _resolve_metric_switches(cfg)
    need_cas = metric_switches["cas_at_1"] or metric_switches["cas_at_5"]
    need_clip = metric_switches["clip_common_name"]
    need_bioclip = metric_switches["bioclip_taxonomic"]

    cas_metric = ClassificationAccuracyMetric.from_config(ev, device) if need_cas else None
    clip_metric = None
    bioclip_metric = None
    if need_clip or need_bioclip:
        clip_metric, bioclip_metric = build_t2i_metrics_from_config(cfg)

    per_sample: list[dict[str, Any]] = []
    clip_common_name_scores: list[float] = []
    bioclip_taxonomic_scores: list[float] = []
    cas_at_1_scores: list[float] = []
    cas_at_5_scores: list[float] = []

    if mode == "b0_sdxl":
        # Baseline: no TaxaAdapter / RAVTI weights; condition is only string prompt (fish species SDXL itself is often weak).
        pipe = StableDiffusionXLPipeline.from_pretrained(
            models_cfg.get("sdxl_model_id"),
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        for row in tqdm(checklist, desc=f"eval:{mode}", dynamic_ncols=True):
            sid = str(row.get("sample_id", "sample"))
            file_stem = _safe_image_stem(sid)
            species = str(row["species"])
            taxon_line = str(row.get("taxonomy_line", species))
            common_name = str(row.get("common_name") or common_name_from_meta(row, species))
            prompt = str(row.get("prompt") or prompt_tmpl.format(species=species, taxonomy=taxon_line))
            with torch.inference_mode():
                # Different from the adapter path: here we use the built-in dual text encoder + default UNet attention of diffusers.
                out = pipe(
                    prompt=prompt,
                    negative_prompt=neg_prompt or None,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            img = out.images[0]
            img_path = method_dir / f"{file_stem}.png"
            img.save(img_path)
            ref_src = "n/a"
            meta = {
                "sample_id": sid,
                "species": species,
                "common_name": common_name,
                "taxonomy_line": taxon_line,
                "prompt": prompt,
                "ref_source": ref_src,
                "generation_mode": mode,
                "adapter_weights": adapter_weights,
                "adapter_checkpoint": None,
                "image_path": str(img_path.relative_to(out_root) if img_path.is_relative_to(out_root) else img_path),
            }
            _append_jsonl(manifest_path, meta)
            row_metrics: dict[str, Any] = {"sample_id": sid}
            if need_clip:
                assert clip_metric is not None
                clip_s = float(clip_metric.score(img, common_name, device))
                clip_common_name_scores.append(clip_s)
                row_metrics["clip_common_name_score"] = clip_s
            if need_bioclip:
                assert bioclip_metric is not None
                bio_s = float(bioclip_metric.score(img, taxon_line, device))
                bioclip_taxonomic_scores.append(bio_s)
                row_metrics["bioclip_taxonomic_score"] = bio_s
            if need_cas:
                cas_t = _pil_to_model_input(img)
                cidx = _parse_optional_class_index(row)
                if metric_switches["cas_at_1"]:
                    c1 = float(cas_metric.cas_at_k(cas_t, cidx, 1)) if cas_metric is not None else 0.0
                    cas_at_1_scores.append(c1)
                    row_metrics["cas_at_1"] = c1
                if metric_switches["cas_at_5"]:
                    c5 = float(cas_metric.cas_at_k(cas_t, cidx, 5)) if cas_metric is not None else 0.0
                    cas_at_5_scores.append(c5)
                    row_metrics["cas_at_5"] = c5
            per_sample.append(row_metrics)
    else:
        # ---------- B1 (only taxonomy condition) or RAVTI (taxonomy + reference) ----------
        use_reference_condition = mode == "ravti"
        # Build the SDXL with decoupled attention: the UNet body comes from sdxl_model_id;
        # processors / conditioning / cmc_gate are randomly initialized first, then overridden by the checkpoint (adapter_weights=trained).
        bundle = _load_adapter_bundle(cfg, device, dtype)
        bundle.pipe.set_progress_bar_config(disable=True)
        if load_adapter_checkpoint:
            assert ckpt_path is not None
            _load_checkpoint_into_bundle(
                bundle,
                ckpt_path,
                strict_processors=False,
                load_conditioning=load_conditioning,
                load_processors=load_processors,
                load_cmc_gate=load_cmc_gate,
            )
        retriever = _maybe_build_retriever(cfg, device, bundle.tax_encoder) if use_reference_condition else None
        retrieval_cfg = cfg.get("retrieval") or {}

        for row in tqdm(checklist, desc=f"eval:{mode}", dynamic_ncols=True):
            sid = str(row.get("sample_id", "sample"))
            file_stem = _safe_image_stem(sid)
            species = str(row["species"])
            taxon_line = str(row.get("taxonomy_line", species))
            common_name = str(row.get("common_name") or common_name_from_meta(row, species))
            prompt = str(row.get("prompt") or prompt_tmpl.format(species=species, taxonomy=taxon_line))
            # Retrieval exclusion set: default to the same sample id, avoid the "evaluation image" being the same as the id in the retrieval library (leakage); also can be explicitly specified by the checklist.
            exclude = row.get("exclude_retrieval_ids")
            if exclude is None:
                excl = {sid}
            else:
                excl = set(str(x) for x in exclude) if isinstance(exclude, (list, tuple, set)) else {str(exclude)}

            ref_vector: torch.Tensor | None = None
            pil_ref: Image.Image | None = None
            if use_reference_condition and retriever is not None:
                # Species name / taxonomic string → the nearest reference image embedding mean in the precomputed index (specific distance see PrecomputedVisualBioRetriever).
                ref_vector, _hits = retriever.retrieve_embedding(
                    species_query=species,
                    k=int(retrieval_cfg.get("k_default", 3)),
                    device=device,
                    fallback_queries=[taxon_line] if taxon_line != species else None,
                    exclude_ids=excl,
                )
            else:
                # B1 or no retrieval: no hits, the gate will still run in RAVTI (has_retrieved_ref=False).
                _hits = []
            ref_img_path = row.get("reference_image")
            if use_reference_condition and ref_vector is None and ref_img_path:
                rp = Path(str(ref_img_path))
                if not rp.is_absolute():
                    rp = (root / rp).resolve()
                if rp.is_file():
                    pil_ref = Image.open(rp).convert("RGB")

            ref_src = _ref_source_label(use_reference_condition, retriever, ref_vector, pil_ref)

            # Only RAVTI: use CMCGate to see the retrieval quality feature, output scalar cmc∈[0,1] (implementation may slightly exceed the boundary, depending on the gate training); then map to lam_t/lam_r.
            if use_fixed_lambdas:
                # Debug/ablation: fix lambda, close the dynamic mapping of runtime CMC, to isolate the influence of retrieval and gate.
                cmc_runtime = None
                lam_t = float(fixed_lambda_tax if fixed_lambda_tax is not None else lambda_tax)
                lam_r = float(fixed_lambda_ref if fixed_lambda_ref is not None else lambda_ref)
            else:
                cmc_runtime = _runtime_cmc_for_eval(
                    species=species,
                    taxon_line=taxon_line,
                    hits=_hits,
                    has_retrieved_ref=ref_vector is not None,
                    use_reference_condition=use_reference_condition,
                    cmc_gate=bundle.cmc_gate if use_reference_condition else None,
                    device=device,
                    dtype=dtype,
                )
                lam_t, lam_r = reliability_lambdas(cmc_runtime, lambda_tax_base, lambda_ref_base)
            if not use_reference_condition:
                # B1: align with the training of "no reference token", force close the strength of the reference branch, avoid using non-zero lambda_ref.
                lam_r = 0.0

            pos_pe, pos_pooled = _build_conditioned_embeds(
                sem_encoder=bundle.semantic_encoder,
                conditioning=bundle.conditioning,
                tax_enc=bundle.tax_encoder,
                ref_enc=bundle.ref_encoder,
                prompt=prompt,
                taxon_line=taxon_line,
                ref_vector=ref_vector,
                pil_ref=pil_ref,
                use_reference_condition=use_reference_condition,
                lambda_tax=lam_t,
                lambda_ref=lam_r,
                device=device,
                dtype=dtype,
            )

            # Pass the DecoupledStreamAttnProcessor to the UNet: the coupling of the three paths of "semantic / tax / ref" in the attention by sample.
            ca_kw = {
                "lambda_tax": float(lam_t),
                "lambda_ref": float(lam_r),
                "processor_alpha": float(processor_alpha),
                "use_reference_condition": bool(use_reference_condition),
            }

            with torch.inference_mode():
                if guidance_scale <= 1.0:
                    # Training often uses CFG=1 or equivalent single branch; inference also keeps 1 to avoid the additional risk of "negative taxon_line is empty / inconsistent with training".
                    out = bundle.pipe(
                        prompt_embeds=pos_pe,
                        pooled_prompt_embeds=pos_pooled,
                        negative_prompt=None,
                        num_inference_steps=steps,
                        height=height,
                        width=width,
                        guidance_scale=1.0,
                        generator=generator,
                        cross_attention_kwargs=ca_kw,
                    )
                else:
                    # CFG>1: the negative branch needs to construct neg prompt_embeds / pooled; the reference branch uses a zero vector to occupy, so that the "unconditional" does not carry the retrieval semantics.
                    rdtype = next(bundle.ref_encoder.parameters()).dtype
                    neg_pe, neg_pooled = _build_conditioned_embeds(
                        sem_encoder=bundle.semantic_encoder,
                        conditioning=bundle.conditioning,
                        tax_enc=bundle.tax_encoder,
                        ref_enc=bundle.ref_encoder,
                        prompt=neg_prompt or " ",
                        taxon_line=neg_taxon,
                        ref_vector=(
                            torch.zeros(1, bundle.ref_encoder.embed_dim, device=device, dtype=rdtype)
                            if use_reference_condition
                            else None
                        ),
                        pil_ref=None,
                        use_reference_condition=use_reference_condition,
                        lambda_tax=lam_t,
                        lambda_ref=lam_r,
                        device=device,
                        dtype=dtype,
                    )
                    out = bundle.pipe(
                        prompt_embeds=pos_pe,
                        negative_prompt_embeds=neg_pe,
                        pooled_prompt_embeds=pos_pooled,
                        negative_pooled_prompt_embeds=neg_pooled,
                        num_inference_steps=steps,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        cross_attention_kwargs=ca_kw,
                    )

            img = out.images[0]
            img_path = method_dir / f"{file_stem}.png"
            img.save(img_path)
            meta = {
                "sample_id": sid,
                "species": species,
                "common_name": common_name,
                "taxonomy_line": taxon_line,
                "prompt": prompt,
                "ref_source": ref_src,
                "generation_mode": mode,
                "adapter_weights": adapter_weights,
                "adapter_checkpoint": str(ckpt_path) if ckpt_path and load_adapter_checkpoint else None,
                "image_path": str(img_path.relative_to(out_root) if img_path.is_relative_to(out_root) else img_path),
            }
            _append_jsonl(manifest_path, meta)
            row_metrics = {"sample_id": sid}
            if need_clip:
                assert clip_metric is not None
                clip_s = float(clip_metric.score(img, common_name, device))
                clip_common_name_scores.append(clip_s)
                row_metrics["clip_common_name_score"] = clip_s
            if need_bioclip:
                assert bioclip_metric is not None
                bio_s = float(bioclip_metric.score(img, taxon_line, device))
                bioclip_taxonomic_scores.append(bio_s)
                row_metrics["bioclip_taxonomic_score"] = bio_s
            if need_cas:
                cas_t = _pil_to_model_input(img)
                cidx = _parse_optional_class_index(row)
                if metric_switches["cas_at_1"]:
                    c1 = float(cas_metric.cas_at_k(cas_t, cidx, 1)) if cas_metric is not None else 0.0
                    cas_at_1_scores.append(c1)
                    row_metrics["cas_at_1"] = c1
                if metric_switches["cas_at_5"]:
                    c5 = float(cas_metric.cas_at_k(cas_t, cidx, 5)) if cas_metric is not None else 0.0
                    cas_at_5_scores.append(c5)
                    row_metrics["cas_at_5"] = c5
            per_sample.append(row_metrics)

    metric_notes: dict[str, str] = {}
    if need_clip:
        metric_notes["mean_clip_common_name_score"] = (
            "TaxaAdapter-style CLIP: OpenCLIP cosine (generated image vs species common name)."
        )
    if need_bioclip:
        metric_notes["mean_bioclip_taxonomic_score"] = (
            "TaxaAdapter-style BioCLIP: joint BioCLIP iNat cosine (image vs taxonomic string / taxonomy_line)."
        )
    if metric_switches["cas_at_1"]:
        metric_notes["mean_cas_at_1"] = (
            "CAS@1: top-1 class match vs checklist class_index; requires cas_classifier."
        )
    if metric_switches["cas_at_5"]:
        metric_notes["mean_cas_at_5"] = (
            "CAS@5: target class in top-5 logits; requires cas_classifier."
        )

    metrics = {
        "generation_mode": mode,
        "adapter_weights": adapter_weights,
        "run_id": run_id,
        "n_samples": len(checklist),
        "images_dir": str(method_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "metric_switches": metric_switches,
        "mean_clip_common_name_score": (
            float(sum(clip_common_name_scores) / max(len(clip_common_name_scores), 1)) if need_clip else None
        ),
        "mean_bioclip_taxonomic_score": (
            float(sum(bioclip_taxonomic_scores) / max(len(bioclip_taxonomic_scores), 1)) if need_bioclip else None
        ),
        "mean_cas_at_1": (
            float(sum(cas_at_1_scores) / max(len(cas_at_1_scores), 1)) if metric_switches["cas_at_1"] else None
        ),
        "mean_cas_at_5": (
            float(sum(cas_at_5_scores) / max(len(cas_at_5_scores), 1)) if metric_switches["cas_at_5"] else None
        ),
        "per_sample": per_sample,
        "metric_notes": metric_notes,
        "fixed": {
            "seed": seed,
            "num_inference_steps": steps,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "processor_alpha": processor_alpha,
            "cmc": cmc_cfg,
            "cmc_runtime": "fixed_lambda" if use_fixed_lambdas else "retrieval_dynamic",
            "lambda_tax": float(fixed_lambda_tax) if fixed_lambda_tax is not None else None,
            "lambda_ref": float(fixed_lambda_ref) if fixed_lambda_ref is not None else None,
        },
        "provenance": {
            "project_root": str(root),
            "git_commit": _git_commit_short(root),
            "adapter_checkpoint": str(ckpt_path) if ckpt_path and load_adapter_checkpoint else None,
            "adapter_sha256": _sha256_file(ckpt_path) if ckpt_path and load_adapter_checkpoint else None,
            "checkpoint_components": {
                "conditioning": bool(load_conditioning) if load_adapter_checkpoint else False,
                "processors": bool(load_processors) if load_adapter_checkpoint else False,
                "cmc_gate": bool(load_cmc_gate) if load_adapter_checkpoint else False,
            },
            "config_path": str(gen.get("_resolved_config_path", "")),
        },
    }
    metrics_path = method_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return metrics


def run_generation_benchmark(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    If ``evaluation.generation.mode`` is a string, run once; if a YAML list (e.g., three methods in parallel),
    use the same ``run_id`` and write to subdirectories, then aggregate scores in ``<run_id>/metrics_by_mode.json``.
    """
    # Get the evaluation configuration
    ev = cfg.get("evaluation") or {}
    gen = ev.get("generation") or {}
    raw_mode = gen.get("mode", "b0_sdxl")
    if isinstance(raw_mode, list):
        if not raw_mode:
            raise ValueError("evaluation.generation.mode is a non-empty list")
        modes = [normalize_generation_mode(str(m)) for m in raw_mode]
        base_run = gen.get("run_id") or datetime.now().strftime("%Y%m%d%H%M%S")
        base_run = str(base_run)
        root = project_root()
        out_root = Path(str(gen.get("output_dir", "outputs/eval")))
        if not out_root.is_absolute():
            out_root = (root / out_root).resolve()
        by_mode: dict[str, Any] = {}
        for m in modes:
            c = copy.deepcopy(cfg)
            sub = c.setdefault("evaluation", {}).setdefault("generation", {})
            sub["mode"] = m
            sub["run_id"] = base_run
            by_mode[m] = _run_generation_benchmark_single(c)
        summary: dict[str, Any] = {
            "run_id": base_run,
            "modes_order": modes,
            "mean_scores_by_mode": {
                m: {
                    "mean_clip_common_name_score": by_mode[m].get("mean_clip_common_name_score"),
                    "mean_bioclip_taxonomic_score": by_mode[m].get("mean_bioclip_taxonomic_score"),
                    "mean_cas_at_1": by_mode[m].get("mean_cas_at_1"),
                    "mean_cas_at_5": by_mode[m].get("mean_cas_at_5"),
                    "n_samples": by_mode[m]["n_samples"],
                }
                for m in modes
            },
            "metrics_json_paths": {
                m: str((out_root / base_run / m / "metrics.json").resolve()) for m in modes
            },
        }
        combined = out_root / base_run / "metrics_by_mode.json"
        combined.parent.mkdir(parents=True, exist_ok=True)
        with combined.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        summary["metrics_by_mode_path"] = str(combined.resolve())
        return summary
    return _run_generation_benchmark_single(cfg)


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
