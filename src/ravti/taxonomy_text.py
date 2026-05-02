from __future__ import annotations

from typing import Any

RANKS: tuple[str, ...] = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


def canonical_taxonomy_line(raw: Any, species_fallback: str) -> str:
    """
    Normalize taxonomy text for BioCLIP text encoding.

    - Accept common separators: ';', '|', '/', ',' and normalize to ' > '
    - Collapse extra spaces and drop empty segments
    - Ensure species fallback is used when taxonomy text is empty
    """
    s = str(raw or "").strip()
    if not s:
        return str(species_fallback).strip()
    for sep in (";", "|", "/", ","):
        s = s.replace(sep, " > ")
    parts = [p.strip() for p in s.split(">") if p.strip()]
    if not parts:
        return str(species_fallback).strip()
    return " > ".join(parts)


def taxonomy_line_from_mapping(taxonomy: dict[str, Any], species_fallback: str) -> str:
    vals: list[str] = []
    for key in RANKS:
        v = taxonomy.get(key)
        if v is None:
            continue
        sv = str(v).strip()
        if sv:
            vals.append(sv)
    if not vals:
        return str(species_fallback).strip()
    return canonical_taxonomy_line(" > ".join(vals), species_fallback)
