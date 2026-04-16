from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


class MetadataStore:
    """Lightweight SQLite cache for taxonomy strings and optional embedding paths."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT UNIQUE,
                species_name TEXT,
                taxonomy_json TEXT,
                bioclip_text_npy_offset INTEGER,
                extra_json TEXT
            );
            """
        )
        self._conn.commit()

    def upsert_sample(
        self,
        external_id: str,
        species_name: str,
        taxonomy: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO samples (external_id, species_name, taxonomy_json, extra_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(external_id) DO UPDATE SET
                species_name=excluded.species_name,
                taxonomy_json=excluded.taxonomy_json,
                extra_json=excluded.extra_json;
            """,
            (
                external_id,
                species_name,
                json.dumps(taxonomy or {}, ensure_ascii=False),
                json.dumps(extra or {}, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
