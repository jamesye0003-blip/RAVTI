"""
RAVTI performance evaluation: 
switch between Base model (SDXL Zero-Shot) / TaxaAdapter (Taxonomy-only) / RAVTI (Taxonomy + Retrieval) three generation modes 
through ``evaluation.generation.mode`` in YAML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ravti.config import load_yaml_config
from ravti.eval.benchmark import run_generation_benchmark
from ravti.paths import project_root
from ravti.utils.seed import set_seed


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAVTI eval: configurable generation mode + metrics")
    parser.add_argument("--config", type=str, default=str(project_root() / "configs" / "inaturalist.yaml"))
    args = parser.parse_args()

    # Load configuration
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml_config(cfg_path)
    set_seed(int(cfg.get("seed", 42)))
    ev = cfg.setdefault("evaluation", {})
    gen = ev.setdefault("generation", {})
    gen["_resolved_config_path"] = str(cfg_path)

    # Run the generation benchmark
    metrics = run_generation_benchmark(cfg)

    # Print the metrics
    if "mean_scores_by_mode" in metrics:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        paths = metrics.get("metrics_json_paths") or {}
        if isinstance(paths, dict):
            print("Image outputs by mode:")
            for m, p in paths.items():
                try:
                    mp = Path(str(p))
                    print(f"  {m}: {(mp.parent).resolve()}")
                except Exception:
                    print(f"  {m}: {p}")
    else:
        print(
            json.dumps(
                {
                    k: metrics[k]
                    for k in (
                        "generation_mode",
                        "run_id",
                        "mean_clip_common_name_score",
                        "mean_bioclip_taxonomic_score",
                        "mean_cas_at_1",
                        "mean_cas_at_5",
                        "n_samples",
                    )
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if metrics.get("images_dir"):
            print("images_dir:", metrics["images_dir"])


if __name__ == "__main__":
    main()
