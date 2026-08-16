"""Dataset loading utilities for CSV, XLSX, JSON, and multi-file import."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".json"}


@dataclass(frozen=True)
class LoadedDataset:
    """Loaded dataset and source metadata."""

    name: str
    source_path: Path
    data: pd.DataFrame


def load_dataset(path: str | Path) -> LoadedDataset:
    """Load dataset from CSV, XLSX, or JSON file."""
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    try:
        if suffix == ".csv":
            data = pd.read_csv(source_path)
        elif suffix == ".xlsx":
            data = pd.read_excel(source_path)
        else:
            data = pd.read_json(source_path)
    except Exception as exc:
        LOGGER.exception("Failed to load dataset: %s", source_path)
        raise ValueError(f"Failed to load {source_path.name}: {exc}") from exc

    LOGGER.info("Loaded dataset %s with shape %s", source_path.name, data.shape)
    return LoadedDataset(name=source_path.name, source_path=source_path, data=data)


def load_multiple_datasets(paths: list[str | Path]) -> list[LoadedDataset]:
    """Load multiple datasets and skip none; raise first file-level error."""
    return [load_dataset(path) for path in paths]


def save_uploaded_file(uploaded_file: Any, destination_dir: str | Path) -> Path:
    """Persist uploaded Streamlit file-like object and return saved path."""
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    file_name = Path(getattr(uploaded_file, "name", "uploaded_dataset")).name
    destination_path = destination / file_name
    destination_path.write_bytes(uploaded_file.getvalue())
    LOGGER.info("Saved uploaded dataset: %s", destination_path)
    return destination_path
