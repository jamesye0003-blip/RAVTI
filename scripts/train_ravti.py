"""Train RAVTI adapter weights using the dataset selected in YAML (see ``dataset.provider``)."""

from __future__ import annotations

import argparse

import torch

from ravti.config import load_yaml_config, resolve_runtime_dtype
from ravti.paths import project_root
from ravti.training.train_loop import train_loop_from_config
from ravti.utils.seed import set_seed


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(project_root() / "configs" / "inaturalist.yaml"))
    args = parser.parse_args()

    # Load configuration
    cfg = load_yaml_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_runtime_dtype(cfg, device)

    # Run the training loop
    train_loop_from_config(cfg, device, dtype)


if __name__ == "__main__":
    main()
