from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Resolve repository root (directory containing `configs/` and `data/`)."""
    env = os.environ.get("RAVTI_PROJECT_ROOT")  # Environment variable to override the repository root
    if env:  # If the environment variable is set, use it to resolve the repository root
        return Path(env).resolve()
    # src/ravti/paths.py -> parents[2] == repo root when installed editable from repo
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it and its parents if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path
