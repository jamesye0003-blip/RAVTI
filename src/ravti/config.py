from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from ravti.paths import project_root


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data)} from {p}")
    return data


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class RuntimePaths:
    data_root: Path
    cache_dir: Path
    index_dir: Path
    metadata_db: Path


def resolve_paths(cfg: dict[str, Any]) -> RuntimePaths:
    root = project_root()
    paths = cfg.get("paths") or {}
    data_root = Path(paths.get("data_root", "data"))
    if not data_root.is_absolute():
        data_root = (root / data_root).resolve()
    cache_dir = Path(paths.get("cache_dir", "data/cache"))
    if not cache_dir.is_absolute():
        cache_dir = (root / cache_dir).resolve()
    index_dir = Path(paths.get("index_dir", "data/indices"))
    if not index_dir.is_absolute():
        index_dir = (root / index_dir).resolve()
    metadata_db = Path(paths.get("metadata_db", "data/metadata/ravti.sqlite"))
    if not metadata_db.is_absolute():
        metadata_db = (root / metadata_db).resolve()
    return RuntimePaths(
        data_root=data_root,
        cache_dir=cache_dir,
        index_dir=index_dir,
        metadata_db=metadata_db,
    )
