# RAVTI — Research Codebase

**RAVTI** (Reliability-Aware Visual-Taxonomic Injection) augments **Stable Diffusion XL** with triple-stream conditioning: semantic prompts, BioCLIP taxonomic text, and retrieved references encoded by BioCLIP-2. Trainable components are limited to **projection layers**, **decoupled cross-attention processors**, and a small **CMC reliability gate**; SDXL and the encoders stay frozen.

This repository is intended for **reproducing the FishNet-oriented experiments** described in the accompanying paper. Configuration lives under `configs/`.

---

## Authors
Yuwei Ye, Wancheng Lin

## Requirements

- **Python** ≥ 3.10  
- **NVIDIA GPU** with CUDA (SDXL training and image generation are not practical on CPU)  
- **Disk / VRAM** sufficient for SDXL and CLIP-family models (first-time Hub downloads are large)  
- **Network** for Hugging Face model weights; set `HF_TOKEN` if you use gated assets

---

## Installation

From the repository root:

```bash
pip install -e .
```

Install runtime dependencies (the package `pyproject.toml` does not pin them):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install diffusers accelerate transformers safetensors open-clip-torch faiss-cpu pyyaml pillow
```

Adjust the PyTorch index URL to match your CUDA version.

---

## Configuration


| File                       | Purpose                                                                                                        |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `configs/fishnet.yaml`     | **Primary** setting for the FishNet 10-species curated split, training, retrieval index paths, and evaluation. |
| `configs/inaturalist.yaml` | Optional; iNaturalist-oriented paths (not required for the main FishNet results).                              |


All paths in YAML are relative to the project root unless absolute. Override the project root with `RAVTI_PROJECT_ROOT` if you run scripts from another working directory.

---

## Data layout

1. **FishNet images**
  Place the dataset so that paths in your **train/eval JSONL manifests** resolve correctly (see `dataset.fishnet` in `configs/fishnet.yaml` and the `image_path` fields in those manifests).
2. **Retrieval index** (FAISS + metadata + optional precomputed embeddings)
  - Index files and JSONL metadata live under `data/indices/fishnet/` by default (`retrieval.index_dir`, `retrieval.index_name`, etc.).  
  - Build or refresh the index with `scripts/build_retrieval_index.py` once you have a suitable gallery manifest (species / paths as expected by the script).
3. **CAS classifier (optional, for CAS@1 / CAS@5)**
  Point `evaluation.cas_classifier` in the YAML to a **ResNet-50** (or compatible) checkpoint trained for your label space. Without a valid checkpoint, CAS metrics are not meaningful.

---

## Training

Use the FishNet config and the training entry point:

```bash
python scripts/train_ravti.py --config configs/fishnet.yaml
```

Notable training fields in `configs/fishnet.yaml` include `training.learning_rate`, `training.max_train_epochs`, `training.mixed_precision` (e.g. `bf16`), `training.lambda_tax_base` / `lambda_ref_base`, and `training.use_reference_condition` (taxonomy-only vs full RAVTI-style reference branch).

Checkpoints and logs are written under `training.output_dir` (default `outputs/train_runs/`).

**Synthetic smoke data (no real images):** set `RAVTI_SYNTHETIC_DATA=1` to exercise the pipeline with a tiny fake dataset (for CI or import checks only).

---

## Evaluation

Generation and metrics (CLIP common-name, BioCLIP taxonomic, CAS@1 / @5) are driven by `evaluation` and `evaluation.generation` in the same YAML used for training:

```bash
python scripts/eval_ravti.py --config configs/fishnet.yaml
```

Modes (`b0_sdxl`, `b1_taxonomy_only`, `ravti`) and checkpoints are configured under `evaluation.generation`. Per-mode outputs and `metrics_by_mode` aggregates are written under `evaluation.generation.output_dir` (see `configs/fishnet.yaml`).

---

## Repository layout (short)


| Path         | Role                                                                  |
| ------------ | --------------------------------------------------------------------- |
| `configs/`   | Experiment YAML                                                       |
| `src/ravti/` | Library code (data, encoders, retrieval, models, training, eval)      |
| `scripts/`   | `train_ravti.py`, `eval_ravti.py`, `build_retrieval_index.py`, etc.   |
| `data/`      | Local indices, metadata, caches (large artifacts; usually gitignored) |
| `outputs/`   | Training logs and checkpoints (recommended local output; gitignored)  |


---

## Citation

Cite **Stable Diffusion XL**, **diffusers**, **BioCLIP / BioCLIP-2** (Imageomics), **FAISS**, **OpenCLIP**, and other third-party components according to their licenses and citation guidelines.