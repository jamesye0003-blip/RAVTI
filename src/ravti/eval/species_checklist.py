"""
Species-first evaluation checklist construction from retrieval index metadata.

- One checklist row per species (single gallery row / ``reference_image`` anchor).
- Optional kingdom balancing (Animalia vs Plantae) to avoid all-animal or all-plant bias.
"""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any

from ravti.config import resolve_paths
from ravti.paths import project_root
from ravti.taxonomy_text import canonical_taxonomy_line


def taxonomy_line_from_meta(row: dict[str, Any], species: str) -> str:
    """Extract the taxonomy line from the metadata"""
    for k in ("taxonomy_line", "taxonomy", "taxonomy_path"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return canonical_taxonomy_line(v, species)
    return canonical_taxonomy_line("", species)


def common_name_from_meta(row: dict[str, Any], species: str) -> str:
    """Extract the common name from the metadata"""
    for k in ("common_name", "vernacular_name", "name_common", "common", "english_name"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return species.strip()


def infer_kingdom_bucket(row: dict[str, Any], taxonomy_line: str, species: str) -> str:
    """Infer the kingdom bucket from the metadata"""
    for key in ("kingdom", "kingdom_name"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            low = v.strip().lower()
            if "animal" in low:
                return "animalia"
            if "plant" in low:
                return "plantae"
            if "fungi" in low:
                return "fungi"
            return low
    blob = f"{taxonomy_line} {species}".lower()
    if "animalia" in blob:
        return "animalia"
    if "plantae" in blob or "chlorophyta" in blob or "streptophyta" in blob:
        return "plantae"
    if "fungi" in blob:
        return "fungi"
    return "other"


def _resolve_image_path(raw: str | None) -> str | None:
    """Resolve the image path from the metadata"""
    if not raw or not str(raw).strip():
        return None
    p = Path(str(raw))
    if p.is_absolute():
        return str(p) if p.is_file() else str(p)
    cand = (project_root() / p).resolve()
    return str(cand) if cand.is_file() else str(cand)


def _rows_by_species_from_jsonl(meta_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Group the rows by species from the metadata"""
    by_species: dict[str, list[dict[str, Any]]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            species = str(row.get("species") or row.get("scientific_name") or "").strip()
            if not species:
                continue
            by_species.setdefault(species, []).append(dict(row))
    return by_species


def _rows_by_species_from_manifest(manifest_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Group the rows by species from the manifest"""
    by_species: dict[str, list[dict[str, Any]]] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            species = str(row.get("species") or row.get("scientific_name") or "").strip()
            if not species:
                continue
            by_species.setdefault(species, []).append(dict(row))
    return by_species


def build_species_balanced_checklist_from_index(
    cfg: dict[str, Any],
    *,
    num_species: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Pick ``num_species`` distinct species with at most one checklist row each.

    Rows include ``species`` (scientific), ``taxonomy_line``, ``common_name`` (for CLIP),
    ``sample_id``, optional ``class_index``, and ``reference_image`` when ``image_path`` exists
    in index metadata (for future image-quality metrics).
    """
    if num_species < 1:
        raise ValueError("num_species must be >= 1")

    # Resolve the paths
    paths = resolve_paths(cfg)

    # Get the retrieval configuration
    retrieval_cfg = cfg.get("retrieval") or {}
    index_name = str(retrieval_cfg.get("index_name", "species_index"))
    meta_path = paths.index_dir / f"{index_name}.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Species checklist requires retrieval metadata jsonl: {meta_path}")

    # Get the balanced checklist configuration
    gen = (cfg.get("evaluation") or {}).get("generation") or {}
    balanced = gen.get("balanced_checklist") or {}
    kb = balanced.get("kingdom_balance") or {}
    balance_kingdoms = bool(kb.get("enabled", True))
    animalia_fraction = float(kb.get("animalia_fraction", 0.5))
    animalia_fraction = min(max(animalia_fraction, 0.0), 1.0)

    # Group the rows by species from the metadata
    by_species = _rows_by_species_from_jsonl(meta_path)
    if not by_species:
        raise ValueError(f"No usable species rows found in {meta_path}")

    rng = random.Random(seed)

    # One representative gallery row per species (random choice for variety).
    species_repr: dict[str, dict[str, Any]] = {}
    species_kingdom: dict[str, str] = {}
    for sp, rows in by_species.items():
        if not rows:
            continue
        pick = dict(rng.choice(rows))
        species_repr[sp] = pick
        tl = taxonomy_line_from_meta(pick, sp)
        species_kingdom[sp] = infer_kingdom_bucket(pick, tl, sp)

    # Get all the species
    all_species = list(species_repr.keys())
    if len(all_species) < num_species:
        raise ValueError(
            f"num_species={num_species} but index only has {len(all_species)} distinct species in {meta_path}"
        )

    # Choose the species
    chosen: list[str] = []
    if balance_kingdoms:
        # Choose the species from the animalia pool
        animalia_pool = [s for s in all_species if species_kingdom.get(s) == "animalia"]
        # Choose the species from the plantae pool
        plantae_pool = [s for s in all_species if species_kingdom.get(s) == "plantae"]
        # Choose the species from the other pool
        other_pool = [s for s in all_species if species_kingdom.get(s) not in ("animalia", "plantae")]
        rng.shuffle(animalia_pool)
        rng.shuffle(plantae_pool)
        rng.shuffle(other_pool)
        n_anim_want = int(round(num_species * animalia_fraction))
        n_plant_want = num_species - n_anim_want
        n_anim = min(len(animalia_pool), n_anim_want)
        n_plant = min(len(plantae_pool), n_plant_want)
        chosen.extend(animalia_pool[:n_anim])
        chosen.extend(plantae_pool[:n_plant])
        deficit = num_species - len(chosen)
        if deficit > 0:
            fill = [s for s in animalia_pool[n_anim:] + plantae_pool[n_plant:] + other_pool if s not in chosen]
            chosen.extend(fill[:deficit])
        if len(chosen) < num_species:
            raise ValueError(
                f"Could only sample {len(chosen)} of {num_species} species with kingdom_balance; "
                "try kingdom_balance.enabled: false or lower num_species."
            )
        chosen = chosen[:num_species]
        # Shuffle the chosen species
        rng.shuffle(chosen)
        # Count the number of animalia and plantae species
        n_a = sum(1 for s in chosen if species_kingdom.get(s) == "animalia")
        n_p = sum(1 for s in chosen if species_kingdom.get(s) == "plantae")
        # If there are no animalia or plantae species, warn the user
        if n_a == 0 or n_p == 0:
            warnings.warn(
                f"Kingdom balance still yielded a single-kingdom slice (animalia={n_a}, plantae={n_p}); "
                "check taxonomy_line / kingdom metadata if you need both branches.",
                UserWarning,
                stacklevel=2,
            )
    else:
        rng.shuffle(all_species)
        chosen = all_species[:num_species]

    # Build the checklist
    out: list[dict[str, Any]] = []
    # Iterate over the chosen species
    for sp in chosen:
        row = species_repr[sp]
        species = str(row.get("species") or row.get("scientific_name") or sp).strip()  # Species name
        taxonomy_line = taxonomy_line_from_meta(row, species)  # Taxonomy line
        common_name = common_name_from_meta(row, species)  # Common name
        sample_id = str(row.get("sample_id") or row.get("id") or "").strip() or f"eval_{species.replace(' ', '_')}"  # Sample ID
        ref = row.get("image_path") or row.get("path")
        ref_resolved = _resolve_image_path(str(ref)) if ref else None
        entry: dict[str, Any] = {
            "sample_id": sample_id,
            "species": species,
            "taxonomy_line": taxonomy_line,
            "common_name": common_name,
        }
        # Get the class index
        for k in ("class_index", "class_id", "label"):
            if k in row and row[k] is not None:
                entry["class_index"] = row[k]
                break
        if ref_resolved:
            entry["reference_image"] = ref_resolved
        out.append(entry)
    return out


def build_species_balanced_checklist_from_manifest(
    manifest_path: Path,
    *,
    num_species: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build the species balanced checklist from the manifest"""
    if num_species < 1:
        raise ValueError("num_species must be >= 1")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Eval manifest not found: {manifest_path}")

    # Group the rows by species from the manifest
    by_species = _rows_by_species_from_manifest(manifest_path)
    if not by_species:
        raise ValueError(f"No usable rows found in eval manifest: {manifest_path}")

    # Choose the species
    rng = random.Random(seed)  # Random number generator
    species_repr: dict[str, dict[str, Any]] = {}
    species_kingdom: dict[str, str] = {}
    for sp, rows in by_species.items():
        pick = dict(rng.choice(rows))
        species_repr[sp] = pick
        tl = taxonomy_line_from_meta(pick, sp)
        species_kingdom[sp] = infer_kingdom_bucket(pick, tl, sp)

    # Get all the species
    all_species = list(species_repr.keys())
    if len(all_species) < num_species:
        raise ValueError(f"num_species={num_species}, but only {len(all_species)} species in {manifest_path}")

    chosen = all_species[:]
    rng.shuffle(chosen)
    chosen = chosen[:num_species]

    out: list[dict[str, Any]] = []
    for sp in chosen:
        row = species_repr[sp]
        species = str(row.get("species") or row.get("scientific_name") or sp).strip()
        taxonomy_line = taxonomy_line_from_meta(row, species)
        common_name = common_name_from_meta(row, species)
        sample_id = str(row.get("sample_id") or "").strip() or f"eval_{species.replace(' ', '_')}"
        ref = row.get("image_path") or row.get("path")
        ref_resolved = _resolve_image_path(str(ref)) if ref else None
        entry: dict[str, Any] = {
            "sample_id": sample_id,
            "species": species,
            "taxonomy_line": taxonomy_line,
            "common_name": common_name,
        }
        for k in ("class_index", "class_id", "label"):
            if k in row and row[k] is not None:
                entry["class_index"] = row[k]
                break
        if ref_resolved:
            entry["reference_image"] = ref_resolved
        out.append(entry)
    return out
