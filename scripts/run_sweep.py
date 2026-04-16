"""
Grid runner for ablation / sweep sections in the RAVTI plan (CMC cliff + k-NN size).

This script does not re-train automatically; it emits a run matrix you can feed to
`ravti.experiments.smoke` extensions or Slurm. Keeps orchestration explicit.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from ravti.config import load_yaml_config
from ravti.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(project_root() / "configs" / "default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/sweep_matrix.json"))
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    sweeps = cfg.get("sweeps") or {}
    cmc_vals = list(sweeps.get("cmc_values", [0.0, 0.5, 1.0]))
    k_vals = list(sweeps.get("k_values", [1, 3, 5]))
    runs = [{"cmc": c, "k": k} for c, k in product(cmc_vals, k_vals)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    print("Wrote", args.out, "with", len(runs), "combinations")


if __name__ == "__main__":
    main()
