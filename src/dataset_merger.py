"""Dataset merging utilities."""

from __future__ import annotations

import logging

import pandas as pd

from src.dataset_loader import LoadedDataset
from src.dataset_standardizer import standardize_schema

LOGGER = logging.getLogger(__name__)


def merge_datasets(datasets: list[LoadedDataset]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Standardize and concatenate multiple loaded datasets."""
    frames: list[pd.DataFrame] = []
    mappings: dict[str, str] = {}

    for dataset in datasets:
        result = standardize_schema(dataset.data)
        frame = result.data.copy()
        frame["source_file"] = dataset.name
        frames.append(frame)
        for source_column, canonical_column in result.column_mappings.items():
            mappings[f"{dataset.name}:{source_column}"] = canonical_column

    if not frames:
        raise ValueError("No datasets provided for merging.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    LOGGER.info("Merged %s datasets into shape %s", len(frames), merged.shape)
    return merged, mappings
