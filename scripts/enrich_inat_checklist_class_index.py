"""
Enrich evaluation checklist with iNaturalist class_index for CAS@1 / CAS@5.

Priority for reverse lookup:
1) sample_id like "inaturalist:{idx}" -> direct dataset index -> class id
2) species text -> class id (only when species maps to a single class)

Usage examples:
  python scripts/enrich_inat_checklist_class_index.py --config configs/inaturalist.yaml
  python scripts/enrich_inat_checklist_class_index.py --config configs/inaturalist.yaml --input data/metadata/eval_500.yaml --output data/metadata/eval_500_with_cls.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from torchvision.datasets import INaturalist

from ravti.config import load_yaml_config, resolve_paths
from ravti.data.providers.inaturalist import parse_inaturalist_2021_folder_name
from ravti.eval.species_checklist import build_species_balanced_checklist_from_index
from ravti.paths import project_root


def _load_yaml_list_or_mapping(path: Path) -> Any:
    """Load a YAML file as a list or mapping."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _inat_parent_root(cfg: dict[str, Any]) -> Path:
    """Get the parent root of the iNaturalist dataset"""
    ds_cfg = cfg.get("dataset") or {}
    icfg = ds_cfg.get("inaturalist") or {}
    paths = resolve_paths(cfg)
    raw_root = icfg.get("root")
    if raw_root is None:
        return (paths.data_root / "datasets" / "inaturalist").resolve()
    root = Path(str(raw_root))
    if not root.is_absolute():
        root = (project_root() / root).resolve()
    return root


def _load_checklist(cfg: dict[str, Any], input_path: Path | None) -> tuple[list[dict[str, Any]], Path | None]:
    """Load the checklist from the configuration"""
    gen = ((cfg.get("evaluation") or {}).get("generation") or {})
    source_path: Path | None = None
    if input_path is not None:
        source_path = input_path
        if not source_path.is_absolute():
            source_path = (project_root() / source_path).resolve()
    elif gen.get("prompts_file"):
        source_path = Path(str(gen.get("prompts_file")))
        if not source_path.is_absolute():
            source_path = (project_root() / source_path).resolve()

    if source_path is not None:
        data = _load_yaml_list_or_mapping(source_path)
        if isinstance(data, list):
            return [dict(x) for x in data], source_path
        if isinstance(data, dict) and isinstance(data.get("checklist"), list):
            return [dict(x) for x in data["checklist"]], source_path
        raise ValueError(f"Unsupported checklist file format: {source_path}")

    checklist = gen.get("checklist") or []
    if not isinstance(checklist, list):
        raise ValueError("evaluation.generation.checklist must be a list")
    return [dict(x) for x in checklist], None


def _build_species_to_class(ds: INaturalist) -> dict[str, set[int]]:
    """Build the mapping from species to class index"""
    mapping: dict[str, set[int]] = {}
    for cid, folder in enumerate(ds.all_categories):
        species, _tax = parse_inaturalist_2021_folder_name(folder)
        mapping.setdefault(species, set()).add(int(cid))
    return mapping


def _extract_inat_index(sample_id: Any) -> int | None:
    """Extract the iNaturalist index from the sample ID"""
    if sample_id is None:
        return None
    s = str(sample_id)
    if not s.startswith("inaturalist:"):
        return None
    tail = s.split(":", 1)[1].strip()
    if not tail.isdigit():
        return None
    return int(tail)


def _write_output(path: Path, rows: list[dict[str, Any]], source_was_mapping: bool) -> None:
    """Write the output to a YAML file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if source_was_mapping:
        payload = {"checklist": rows}
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    else:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(rows, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(project_root() / "configs" / "inaturalist.yaml"))
    parser.add_argument("--input", type=Path, default=None, help="Optional checklist YAML path")
    parser.add_argument("--output", type=Path, default=None, help="Output YAML path")
    parser.add_argument("--download", action="store_true", help="Download iNat when missing")
    args = parser.parse_args()

    # Load configuration
    cfg = load_yaml_config(args.config)

    # Load the checklist
    checklist, source_path = _load_checklist(cfg, args.input)
    if not checklist:
        # Support auto-generated checklist from evaluation.generation.balanced_checklist.
        gen = ((cfg.get("evaluation") or {}).get("generation") or {})
        balanced = gen.get("balanced_checklist") or {}
        if bool(balanced.get("enabled", False)):
            n = int(balanced.get("num_species", balanced.get("num_samples", 500)))
            seed = int(balanced.get("seed", cfg.get("seed", 42)))
            checklist = build_species_balanced_checklist_from_index(cfg, num_species=n, seed=seed)
        if not checklist:
            raise ValueError("Checklist is empty; nothing to enrich.")

    # Load the iNaturalist dataset
    icfg = ((cfg.get("dataset") or {}).get("inaturalist") or {})
    version = str(icfg.get("version", "2021_train_mini"))
    parent = _inat_parent_root(cfg)
    ds = INaturalist(
        root=str(parent),
        version=version,
        target_type="full",
        transform=None,
        download=bool(args.download or icfg.get("download", False)),
    )

    # Build the mapping from species to class index
    species_to_class = _build_species_to_class(ds)

    # Enrich the checklist with the class index
    filled = 0
    kept = 0
    unresolved = 0
    out_rows: list[dict[str, Any]] = []
    for row in checklist:  # Iterate over the checklist
        rec = dict(row)
        if rec.get("class_index") is not None:  # If the class index is already filled, skip
            kept += 1
            out_rows.append(rec)
            continue

        class_idx: int | None = None
        ds_index = _extract_inat_index(rec.get("sample_id"))
        if ds_index is not None and 0 <= ds_index < len(ds):
            cat_id, _fname = ds.index[ds_index]
            class_idx = int(cat_id)
        else:
            species = str(rec.get("species") or "").strip()
            if species:
                cands = species_to_class.get(species, set())
                if len(cands) == 1:
                    class_idx = int(next(iter(cands)))

        if class_idx is None:
            unresolved += 1
        else:
            rec["class_index"] = int(class_idx)
            filled += 1
        out_rows.append(rec)  # Add the enriched row to the output list

    if args.output is not None:  # If the output path is provided, use it
        out = args.output
        if not out.is_absolute():
            out = (project_root() / out).resolve()
    elif source_path is not None:  # If the source path is provided, use it
        out = source_path.with_name(f"{source_path.stem}_with_class_index{source_path.suffix}")
    else:  # If no output path is provided, use the default path
        out = (project_root() / "data" / "metadata" / "eval_checklist_with_class_index.yaml").resolve()

    # Check if the source path is a mapping
    source_was_mapping = False  
    if source_path is not None:
        parsed = _load_yaml_list_or_mapping(source_path)
        source_was_mapping = isinstance(parsed, dict)
    
    # Write the output to a YAML file
    _write_output(out, out_rows, source_was_mapping=source_was_mapping)

    # Print the summary
    summary = {
        "output": str(out),
        "total": len(out_rows),
        "filled_new": filled,
        "kept_existing": kept,
        "unresolved": unresolved,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))  # Print the summary
    if unresolved:
        print("Hint: unresolved rows usually miss sample_id (inaturalist:{idx}) or have ambiguous species->class mapping.")


if __name__ == "__main__":
    main()

