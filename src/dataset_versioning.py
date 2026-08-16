"""Processed dataset versioning and metadata management."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import METADATA_DIR, PROCESSED_DATA_DIR

LOGGER = logging.getLogger(__name__)
METADATA_FILE = METADATA_DIR / "metadata.json"


def load_metadata() -> dict[str, Any]:
    """Load dataset version history metadata."""
    if not METADATA_FILE.exists():
        return {"versions": []}
    return json.loads(METADATA_FILE.read_text(encoding="utf-8"))


def save_metadata(metadata: dict[str, Any]) -> None:
    """Write dataset metadata to disk."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def next_dataset_version() -> int:
    """Return next processed dataset version number."""
    metadata = load_metadata()
    versions = [entry.get("version", 0) for entry in metadata.get("versions", [])]
    return int(max(versions, default=0)) + 1


def save_dataset_version(
    data: pd.DataFrame,
    statistics: dict[str, Any],
    column_mappings: dict[str, str],
    quality_score: float,
) -> Path:
    """Save processed dataset version and append metadata history."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    version = next_dataset_version()
    file_name = f"dataset_v{version}.csv"
    output_path = PROCESSED_DATA_DIR / file_name
    data.to_csv(output_path, index=False)

    metadata = load_metadata()
    metadata.setdefault("versions", []).append(
        {
            "version": version,
            "file_name": file_name,
            "path": str(output_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rows": int(data.shape[0]),
            "columns": int(data.shape[1]),
            "quality_score": quality_score,
            "statistics": statistics,
            "column_mappings": column_mappings,
        }
    )
    save_metadata(metadata)
    LOGGER.info("Saved processed dataset version: %s", output_path)
    return output_path


def list_processed_versions() -> list[Path]:
    """List processed dataset version files."""
    if not PROCESSED_DATA_DIR.exists():
        return []
    return sorted(PROCESSED_DATA_DIR.glob("dataset_v*.csv"))
