"""Configuration utilities for RAVTI.

This module provides utilities for loading and merging configuration files,
resolving runtime paths, and mapping mixed precision settings to model dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from ravti.paths import project_root


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data)} from {p}")
    return data


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two dictionaries, with override taking precedence."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def resolve_runtime_dtype(cfg: dict[str, Any], device: torch.device) -> torch.dtype:
    """Map ``training.mixed_precision`` to model/runtime dtype (CUDA-aware)."""
    if device.type != "cuda":
        return torch.float32
    mp = str((cfg.get("training") or {}).get("mixed_precision", "fp16")).lower()
    if mp in ("fp32", "float32", "no"):
        return torch.float32
    if mp in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


@dataclass
class RuntimePaths:
    """Encapsulates runtime paths for the project."""
    data_root: Path
    cache_dir: Path
    index_dir: Path
    metadata_db: Path


def resolve_paths(cfg: dict[str, Any]) -> RuntimePaths:
    """Resolve runtime paths from the configuration."""
    root = project_root()
    paths = cfg.get("paths") or {}
    data_root = Path(paths.get("data_root", "data"))

    if not data_root.is_absolute():  # If the data root is not absolute, resolve it relative to the project root
        data_root = (root / data_root).resolve()
    cache_dir = Path(paths.get("cache_dir", "data/cache"))
    if not cache_dir.is_absolute():  # If the cache directory is not absolute, resolve it relative to the project root
        cache_dir = (root / cache_dir).resolve()
    index_dir = Path(paths.get("index_dir", "data/indices"))
    if not index_dir.is_absolute():  # If the index directory is not absolute, resolve it relative to the project root
        index_dir = (root / index_dir).resolve()
    metadata_db = Path(paths.get("metadata_db", "data/metadata/ravti.sqlite"))
    if not metadata_db.is_absolute():  # If the metadata database is not absolute, resolve it relative to the project root
        metadata_db = (root / metadata_db).resolve()
    
    # Return a RuntimePaths object encapsulating the resolved paths
    return RuntimePaths(
        data_root=data_root,
        cache_dir=cache_dir,
        index_dir=index_dir,
        metadata_db=metadata_db,
    )
