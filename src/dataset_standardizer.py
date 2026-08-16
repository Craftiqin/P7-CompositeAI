"""Schema standardization for composite laminate datasets."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

LOGGER = logging.getLogger(__name__)

CANONICAL_COLUMN_SYNONYMS: dict[str, list[str]] = {
    "material": ["material", "material system", "matl", "composite material"],
    "fiber": ["fiber", "fibre", "fiber type", "fibre type", "reinforcement"],
    "matrix": ["matrix", "resin", "polymer", "matrix type"],
    "fiber_volume_fraction": [
        "fiber volume fraction",
        "fibre volume fraction",
        "fiber vol frac",
        "fibre vol frac",
        "vf",
        "v_f",
        "fiber %",
        "fibre %",
    ],
    "density_g_cm3": [
        "density",
        "density g cm3",
        "density g/cm3",
        "rho",
        "specific density",
    ],
    "ply_orientation": [
        "ply orientation",
        "orientation",
        "angle",
        "ply angle",
        "theta",
        "ply_angle",
    ],
    "stacking_sequence": [
        "stacking sequence",
        "layup",
        "lay-up",
        "stacking",
        "laminate sequence",
        "sequence",
    ],
    "thickness_mm": ["thickness", "thickness mm", "ply thickness", "laminate thickness"],
    "tensile_strength_mpa": [
        "tensile strength",
        "tensile strength mpa",
        "uts",
        "ultimate tensile strength",
    ],
    "compressive_strength_mpa": [
        "compressive strength",
        "compressive strength mpa",
        "compression strength",
    ],
    "flexural_strength_mpa": [
        "flexural strength",
        "flexural strength mpa",
        "bending strength",
    ],
    "shear_strength_mpa": ["shear strength", "shear strength mpa", "ilss"],
    "elastic_modulus_gpa": [
        "elastic modulus",
        "modulus",
        "youngs modulus",
        "young modulus",
        "e modulus",
    ],
    "poisson_ratio": ["poisson ratio", "poissons ratio", "nu"],
    "failure_strain_percent": [
        "failure strain",
        "strain at failure",
        "elongation",
        "failure strain %",
    ],
    "temperature_c": ["temperature", "temperature c", "test temperature", "temp"],
    "source": ["source", "dataset source", "origin"],
}


@dataclass(frozen=True)
class StandardizationResult:
    """Standardized dataframe plus source-to-canonical column mappings."""

    data: pd.DataFrame
    column_mappings: dict[str, str]


def normalize_column_name(column_name: str) -> str:
    """Normalize column names across case, spaces, underscores, and symbols."""
    normalized = column_name.lower().strip()
    normalized = normalized.replace("%", " percent ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def build_column_lookup() -> dict[str, str]:
    """Build normalized synonym to canonical column lookup."""
    lookup: dict[str, str] = {}
    for canonical_name, synonyms in CANONICAL_COLUMN_SYNONYMS.items():
        lookup[normalize_column_name(canonical_name)] = canonical_name
        for synonym in synonyms:
            lookup[normalize_column_name(synonym)] = canonical_name
    return lookup


def standardize_schema(data: pd.DataFrame) -> StandardizationResult:
    """Map synonymous columns to canonical names and log all mappings."""
    lookup = build_column_lookup()
    renamed_columns: dict[str, str] = {}
    new_columns: list[str] = []
    used_counts: dict[str, int] = {}

    for index, column in enumerate(data.columns):
        normalized = normalize_column_name(str(column))
        canonical = lookup.get(normalized)
        target_name = canonical or normalized.replace(" ", "_")

        used_counts[target_name] = used_counts.get(target_name, 0) + 1
        if used_counts[target_name] > 1:
            target_name = f"{target_name}_{used_counts[target_name]}"

        mapping_key = str(column)
        if mapping_key in renamed_columns:
            mapping_key = f"{column}__column_{index}"
        renamed_columns[mapping_key] = target_name
        new_columns.append(target_name)
        LOGGER.info("Column mapped: %s -> %s", column, target_name)

    standardized = data.copy()
    standardized.columns = new_columns
    return StandardizationResult(
        data=standardized,
        column_mappings=renamed_columns,
    )
