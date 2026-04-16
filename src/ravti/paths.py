from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Resolve repository root (directory containing `configs/` and `data/`)."""
    env = os.environ.get("RAVTI_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    # src/ravti/paths.py -> parents[2] == repo root when installed editable from repo
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
