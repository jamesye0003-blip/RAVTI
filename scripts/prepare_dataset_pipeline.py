#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from torchvision.datasets import INaturalist

from ravti.config import load_yaml_config, resolve_paths
from ravti.data.providers.inaturalist import parse_inaturalist_2021_folder_name
from ravti.paths import project_root
from ravti.taxonomy_text import RANKS, canonical_taxonomy_line, taxonomy_line_from_mapping


def _resolve_path(raw: str | None, *, default: Path | None = None) -> Path:
    if raw is None:
        if default is None:
            raise ValueError("Path is required but got None")
        return default
    p = Path(str(raw))
    if not p.is_absolute():
        p = (project_root() / p).resolve()
    return p


def _taxonomy_from_line(taxonomy_line: str, species_name: str) -> dict[str, str | None]:
    parts = [x.strip() for x in taxonomy_line.split(">") if x.strip()]
    out: dict[str, str | None] = {k: None for k in RANKS}
    for i, key in enumerate(RANKS[:-1]):
        out[key] = parts[i] if i < len(parts) else None
    out["species"] = species_name.strip() or (parts[-1] if parts else None)
    return out


def _taxonomy_line_from_taxonomy(taxonomy: dict[str, Any], species_name: str) -> str:
    return taxonomy_line_from_mapping(taxonomy, species_name)


def _build_inat_manifest(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    ds_cfg = cfg.get("dataset") or {}
    icfg = ds_cfg.get("inaturalist") or {}
    root = _resolve_path(icfg.get("root"), default=project_root() / "data" / "datasets" / "inaturalist")
    version = str(icfg.get("version", "2021_train_mini"))
    base = INaturalist(
        root=str(root),
        version=version,
        target_type="full",
        transform=None,
        download=bool(icfg.get("download", False)),
    )
    rows: list[dict[str, Any]] = []
    for idx, (cat_id, rel_path) in enumerate(base.index):
        full_path = Path(base.root) / rel_path
        folder = base.category_name("full", int(cat_id))
        species, taxonomy_line = parse_inaturalist_2021_folder_name(folder)
        taxonomy_line = canonical_taxonomy_line(taxonomy_line, species)
        taxonomy = _taxonomy_from_line(taxonomy_line, species)
        rows.append(
            {
                "dataset": "inaturalist",
                "sample_id": f"inaturalist:{idx}",
                "image_path": str(full_path.resolve()),
                "species": species,
                "taxonomy_line": taxonomy_line,
                "taxonomy": taxonomy,
            }
        )
    return rows


def _read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".jl"):
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
        return out
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(x) for x in data if isinstance(x, dict)]
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return [dict(x) for x in data["rows"] if isinstance(x, dict)]
        raise ValueError(f"Unsupported JSON table format: {path}")
    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        # Use utf-8-sig to tolerate BOM-prefixed headers (common on Windows CSV exports).
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows: list[dict[str, Any]] = []
            for r in csv.DictReader(f, delimiter=delim):
                rr: dict[str, Any] = {}
                for k, v in r.items():
                    kk = str(k).replace("\ufeff", "").strip() if k is not None else ""
                    rr[kk] = v
                rows.append(rr)
            return rows
    raise ValueError(f"Unsupported annotation table type: {path}")


def _build_fishnet_manifest(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    ds_cfg = cfg.get("dataset") or {}
    fcfg = ds_cfg.get("fishnet") or {}
    root = _resolve_path(fcfg.get("root"), default=project_root() / "data" / "datasets" / "fishnet")
    table_path = fcfg.get("annotation_table") or fcfg.get("manifest_csv")
    if not table_path:
        raise ValueError("dataset.fishnet.annotation_table (or manifest_csv) is required for FishNet pipeline.")
    table = _read_table(_resolve_path(table_path))
    image_col = str(fcfg.get("image_column", "image_path"))
    species_col = str(fcfg.get("species_column", "species"))
    sample_id_col = str(fcfg.get("sample_id_column", "sample_id"))
    tax_line_col = fcfg.get("taxonomy_column")
    rank_cols = fcfg.get("taxonomy_rank_columns") or {}

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(table):
        raw_img = row.get(image_col)
        if not raw_img:
            continue
        p = Path(str(raw_img))
        if not p.is_absolute():
            p = (root / p).resolve()
        species = str(row.get(species_col) or "").strip()
        if not species:
            continue
        taxonomy: dict[str, Any] = {k: None for k in RANKS}
        for k in RANKS:
            col = rank_cols.get(k, k) if isinstance(rank_cols, dict) else k
            v = row.get(col)
            taxonomy[k] = str(v).strip() if v is not None and str(v).strip() else None
        taxonomy["species"] = taxonomy.get("species") or species
        tax_line = None
        if tax_line_col and row.get(str(tax_line_col)):
            tax_line = str(row[str(tax_line_col)]).strip()
        if not tax_line:
            tax_line = _taxonomy_line_from_taxonomy(taxonomy, species)
        tax_line = canonical_taxonomy_line(tax_line, species)
        sample_id = str(row.get(sample_id_col) or f"fishnet:{i}")
        rows.append(
            {
                "dataset": "fishnet",
                "sample_id": sample_id,
                "image_path": str(p),
                "species": species,
                "taxonomy_line": tax_line,
                "taxonomy": taxonomy,
            }
        )
    return rows


def _assign_class_index(records: list[dict[str, Any]]) -> dict[str, int]:
    species_sorted = sorted({str(r["species"]).strip() for r in records})
    mapping = {sp: i for i, sp in enumerate(species_sorted)}
    for r in records:
        r["class_index"] = mapping[str(r["species"]).strip()]
    return mapping


def _species_count_map(records: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in records:
        out[str(r["species"]).strip()] += 1
    return dict(out)


def _count_bucket(n_images: int) -> str:
    n = int(n_images)
    if n <= 2:
        return "2"
    if n <= 5:
        return "3_5"
    if n <= 10:
        return "6_10"
    return "11_plus"


def _select_species_subset(
    records: list[dict[str, Any]],
    *,
    subset_num_species: int | None,
    subset_min_images_per_species: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    count_map = _species_count_map(records)
    if subset_num_species is None or int(subset_num_species) <= 0:
        chosen_all = sorted(count_map.keys())
        meta_all = {
            "subset_enabled": False,
            "subset_num_species_requested": 0,
            "subset_num_species_selected": len(chosen_all),
            "subset_min_images_per_species": int(subset_min_images_per_species),
            "subset_bucket_hist_species": {
                "2": sum(1 for sp in chosen_all if _count_bucket(count_map[sp]) == "2"),
                "3_5": sum(1 for sp in chosen_all if _count_bucket(count_map[sp]) == "3_5"),
                "6_10": sum(1 for sp in chosen_all if _count_bucket(count_map[sp]) == "6_10"),
                "11_plus": sum(1 for sp in chosen_all if _count_bucket(count_map[sp]) == "11_plus"),
            },
        }
        return records, chosen_all, meta_all

    eligible = [sp for sp, n in count_map.items() if int(n) >= int(subset_min_images_per_species)]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if subset_num_species is None or subset_num_species <= 0:
        chosen = sorted(eligible)
    else:
        if len(eligible) < int(subset_num_species):
            raise ValueError(
                f"Requested subset_num_species={subset_num_species}, "
                f"but only {len(eligible)} species meet min_images={subset_min_images_per_species}."
            )
        # Stratified by species image-count buckets to keep long-tail structure.
        bucket_to_species: dict[str, list[str]] = {"2": [], "3_5": [], "6_10": [], "11_plus": []}
        for sp in eligible:
            b = _count_bucket(count_map[sp])
            bucket_to_species.setdefault(b, []).append(sp)
        for arr in bucket_to_species.values():
            rng.shuffle(arr)
        total_eligible = len(eligible)
        want = int(subset_num_species)
        alloc: dict[str, int] = {}
        # Proportional base allocation
        for b, arr in bucket_to_species.items():
            alloc[b] = min(len(arr), int(round(want * (len(arr) / max(total_eligible, 1)))))
        # Ensure non-empty buckets (if possible) get at least one sample.
        for b, arr in bucket_to_species.items():
            if len(arr) > 0 and alloc[b] == 0 and sum(alloc.values()) < want:
                alloc[b] = 1
        # Trim/expand to exact target
        while sum(alloc.values()) > want:
            cands = [b for b in alloc if alloc[b] > 0]
            if not cands:
                break
            b = rng.choice(cands)
            alloc[b] -= 1
        while sum(alloc.values()) < want:
            cands = [b for b, arr in bucket_to_species.items() if alloc[b] < len(arr)]
            if not cands:
                break
            b = rng.choice(cands)
            alloc[b] += 1
        chosen = []
        for b, arr in bucket_to_species.items():
            chosen.extend(arr[: alloc[b]])
        if len(chosen) < want:
            seen = set(chosen)
            fill = [sp for sp in eligible if sp not in seen]
            rng.shuffle(fill)
            chosen.extend(fill[: (want - len(chosen))])
        chosen = sorted(chosen[:want])

    chosen_set = set(chosen)
    subset_records = [r for r in records if str(r["species"]).strip() in chosen_set]
    bucket_hist: dict[str, int] = {"2": 0, "3_5": 0, "6_10": 0, "11_plus": 0}
    for sp in chosen:
        bucket_hist[_count_bucket(count_map[sp])] += 1
    meta = {
        "subset_enabled": subset_num_species is not None and int(subset_num_species) > 0,
        "subset_num_species_requested": int(subset_num_species or 0),
        "subset_num_species_selected": len(chosen),
        "subset_min_images_per_species": int(subset_min_images_per_species),
        "subset_bucket_hist_species": bucket_hist,
    }
    return subset_records, chosen, meta


def _split_records(
    records: list[dict[str, Any]],
    *,
    eval_per_species: int,
    max_train_per_species: int,
    min_samples_per_species: int,
    exposed: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_species[str(r["species"]).strip()].append(r)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    ignored_species = 0
    ignored_samples = 0
    exposed_eval_samples = 0
    eval_take = max(1, int(eval_per_species))
    train_cap = int(max_train_per_species)
    min_required = max(2, int(min_samples_per_species))

    for sp, rows in by_species.items():
        rng.shuffle(rows)
        n = len(rows)

        # Too few samples to make train/eval disjoint: skip by default; optionally expose the eval sample to train.
        if n < min_required:
            if exposed and n > 0:
                chosen_eval = rows[: min(eval_take, n)]
                eval_rows.extend(chosen_eval)
                train_rows.extend(chosen_eval)
                exposed_eval_samples += len(chosen_eval)
            else:
                ignored_species += 1
                ignored_samples += n
            continue

        eval_part = rows[: min(eval_take, n - 1)]
        train_part = rows if exposed else rows[len(eval_part) :]
        if train_cap > 0:
            train_part = train_part[:train_cap]
        eval_rows.extend(eval_part)
        train_rows.extend(train_part)
        if exposed:
            exposed_eval_samples += len(eval_part)

    stats = {
        "total_species": len(by_species),
        "ignored_species_lt_min_samples": ignored_species,
        "ignored_samples_lt_min_samples": ignored_samples,
        "exposed_eval_samples_in_train": exposed_eval_samples,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "eval_per_species": eval_take,
        "max_train_per_species": train_cap,
        "min_samples_per_species": min_required,
        "exposed": bool(exposed),
        "seed": int(seed),
    }
    return train_rows, eval_rows, stats


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified manifest and train/eval split for iNat mini or FishNet.")
    parser.add_argument("--config", type=str, default=str(project_root() / "configs" / "inaturalist.yaml"))
    parser.add_argument("--provider", type=str, default=None, choices=["inaturalist_mini", "fishnet"])
    parser.add_argument("--train-num", type=int, default=0, help="Legacy alias of --max-train-per-species.")
    parser.add_argument("--max-train-per-species", type=int, default=0, help="0 means keep all remaining train samples.")
    parser.add_argument("--eval-per-species", type=int, default=1, help="Number of eval samples per selected species.")
    parser.add_argument(
        "--min-samples-per-species",
        type=int,
        default=2,
        help="Species with fewer samples are ignored unless --exposed is set.",
    )
    parser.add_argument(
        "--exposed",
        action="store_true",
        help="Allow eval samples to also appear in train (for very low-shot species or overlap ablation).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-name", type=str, default="default")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--subset-num-species", type=int, default=0, help="0 disables subset; e.g. 1000 for 1000-species subset.")
    parser.add_argument(
        "--subset-min-images-per-species",
        type=int,
        default=2,
        help="Only species with at least this many images are eligible for subset selection.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = resolve_paths(cfg)
    provider = (args.provider or str((cfg.get("dataset") or {}).get("provider", "inaturalist_mini"))).lower()
    out_dir = _resolve_path(args.out_dir, default=paths.data_root / "metadata" / "splits" / provider / args.split_name)

    if provider in ("inaturalist_mini", "inaturalist", "inat_mini", "inat2021_mini"):
        records = _build_inat_manifest(cfg)
        provider = "inaturalist_mini"
    elif provider == "fishnet":
        records = _build_fishnet_manifest(cfg)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    if not records:
        raise ValueError(f"No usable records found for provider={provider}")

    records, subset_species, subset_meta = _select_species_subset(
        records,
        subset_num_species=int(args.subset_num_species),
        subset_min_images_per_species=int(args.subset_min_images_per_species),
        seed=int(args.seed),
    )
    if not records:
        raise ValueError("Records empty after subset filtering; relax subset constraints.")

    species_to_class = _assign_class_index(records)
    max_train_per_species = (
        int(args.max_train_per_species) if int(args.max_train_per_species) > 0 else int(args.train_num)
    )
    train_rows, eval_rows, stats = _split_records(
        records,
        eval_per_species=int(args.eval_per_species),
        max_train_per_species=max_train_per_species,
        min_samples_per_species=int(args.min_samples_per_species),
        exposed=bool(args.exposed),
        seed=int(args.seed),
    )

    all_path = out_dir / "all_manifest.jsonl"
    train_path = out_dir / "train_manifest.jsonl"
    eval_path = out_dir / "eval_manifest.jsonl"
    class_map_path = out_dir / "class_index_map.json"
    subset_species_path = out_dir / "subset_species.txt"
    subset_species_json = out_dir / "subset_species.json"
    meta_path = out_dir / "split_meta.json"
    _write_jsonl(all_path, records)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    class_map_path.write_text(json.dumps(species_to_class, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subset_species_path.write_text("\n".join(subset_species) + ("\n" if subset_species else ""), encoding="utf-8")
    subset_counts = _species_count_map(records)
    subset_species_json.write_text(
        json.dumps(
            [{"species": sp, "n_images": int(subset_counts.get(sp, 0))} for sp in subset_species],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    split_meta = {
        "provider": provider,
        "split_name": args.split_name,
        "all_manifest_jsonl": str(all_path),
        "train_manifest_jsonl": str(train_path),
        "eval_manifest_jsonl": str(eval_path),
        "class_index_map_json": str(class_map_path),
        "subset_species_txt": str(subset_species_path),
        "subset_species_json": str(subset_species_json),
        **subset_meta,
        **stats,
    }
    meta_path.write_text(json.dumps(split_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(split_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
