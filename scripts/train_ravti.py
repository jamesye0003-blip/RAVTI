"""Train RAVTI adapter weights using the dataset selected in YAML (see ``dataset.provider``)."""

from __future__ import annotations

import argparse

import torch

from ravti.config import load_yaml_config
from ravti.experiments.smoke import run_train_smoke, set_seed
from ravti.paths import project_root
from ravti.training.train_loop import train_loop_from_config


def resolve_runtime_dtype(cfg: dict, device: torch.device) -> torch.dtype:
    """Map training.mixed_precision to real model/runtime dtype."""
    if device.type != "cuda":
        return torch.float32
    mp = str((cfg.get("training") or {}).get("mixed_precision", "fp16")).lower()
    if mp in ("fp32", "float32", "no"):
        return torch.float32
    if mp in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(project_root() / "configs" / "default.yaml"))
    parser.add_argument(
        "--smoke-step",
        action="store_true",
        help="Only run a single optimization step on the first dataloader batch (SDXL still loads).",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_yaml_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_runtime_dtype(cfg, device)

    # Run training loop
    if args.smoke_step:
        run_train_smoke(cfg, device, dtype)
    else:
        train_loop_from_config(cfg, device, dtype)


if __name__ == "__main__":
    main()
