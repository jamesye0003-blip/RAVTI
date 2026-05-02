"""
Export a JSONL gallery manifest for `build_retrieval_index.py` from iNaturalist mini.

Uses the same torchvision `INaturalist` index order as training (`inaturalist:{i}` sample_id),
so train-time `exclude_ids` matches gallery rows.

Example:
  conda activate ai_full
  pip install -e .
  python scripts/export_inaturalist_gallery_jsonl.py --config configs/inaturalist.yaml
  python scripts/build_retrieval_index.py --manifest data/metadata/inat_mini_gallery.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from torchvision.datasets import INaturalist

from ravti.config import load_yaml_config, resolve_paths
from ravti.data.providers.inaturalist import parse_inaturalist_2021_folder_name
from ravti.paths import project_root as repo_root


def _inat_parent_root(cfg: dict) -> Path:
    ds_cfg = cfg.get("dataset") or {}
    icfg = ds_cfg.get("inaturalist") or {}
    paths = resolve_paths(cfg)
    raw_root = icfg.get("root")
    if raw_root is None:
        return (paths.data_root / "datasets" / "inaturalist").resolve()
    root = Path(str(raw_root))
    if not root.is_absolute():
        root = (repo_root() / root).resolve()
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(repo_root() / "configs" / "default.yaml"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: <repo>/data/metadata/inat_mini_gallery.jsonl)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, only export the first N rows (for smoke tests). 0 = full dataset.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Pass download=True into torchvision INaturalist (if data missing).",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    icfg = (cfg.get("dataset") or {}).get("inaturalist") or {}
    version = str(icfg.get("version", "2021_train_mini"))
    parent = _inat_parent_root(cfg)

    out = args.out
    if out is None:
        meta_dir = repo_root() / "data" / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        out = meta_dir / "inat_mini_gallery.jsonl"
    else:
        if not out.is_absolute():
            out = (repo_root() / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

    ds = INaturalist(
        root=str(parent),
        version=version,
        target_type="full",
        transform=None,
        download=bool(args.download or icfg.get("download", False)),
    )

    n = len(ds)
    if args.max_rows > 0:
        n = min(n, int(args.max_rows))

    with out.open("w", encoding="utf-8") as f:
        for i in tqdm(range(n), desc="export gallery", dynamic_ncols=True):
            cat_id, fname = ds.index[i]
            image_path = (Path(ds.root) / ds.all_categories[cat_id] / fname).resolve()
            folder = ds.all_categories[cat_id]
            species, taxonomy_line = parse_inaturalist_2021_folder_name(folder)
            rec = {
                "sample_id": f"inaturalist:{i}",
                "species": species,
                "image_path": str(image_path),
                "taxonomy_line": taxonomy_line,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Wrote", out, "rows=", n)


if __name__ == "__main__":
    main()
