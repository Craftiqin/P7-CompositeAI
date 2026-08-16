"""Composite laminate feature engineering utilities."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
ORIENTATIONS = [0, 90, 45, -45]


def engineer_laminate_features(
    data: pd.DataFrame,
    enabled_features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create selected engineering features from laminate columns."""
    enabled = set(enabled_features or available_engineered_features())
    result = data.copy()
    angles = _angle_lists(result)

    if "stacking_sequence_string" in enabled:
        result["stacking_sequence_string"] = _stacking_sequence_strings(result, angles)
    if "stacking_symmetry_flag" in enabled:
        result["stacking_symmetry_flag"] = angles.apply(_is_symmetric).astype(int)
    if "balanced_laminate_flag" in enabled:
        result["balanced_laminate_flag"] = angles.apply(_is_balanced).astype(int)
    if "average_ply_angle" in enabled:
        result["average_ply_angle"] = angles.apply(_safe_mean)
    if "angle_variance" in enabled:
        result["angle_variance"] = angles.apply(_safe_variance)
    if "angle_entropy" in enabled:
        result["angle_entropy"] = angles.apply(_angle_entropy)
    if "layup_complexity_score" in enabled:
        result["layup_complexity_score"] = angles.apply(_layup_complexity)
    if "total_thickness" in enabled:
        result["total_thickness"] = _total_thickness(result, angles)
    if "thickness_per_layer" in enabled:
        result["thickness_per_layer"] = _thickness_per_layer(result, angles)
    if "strength_to_weight_ratio" in enabled:
        result["strength_to_weight_ratio"] = _strength_to_weight_ratio(result)
    if "material_family" in enabled:
        result["material_family"] = _material_family(result)
    if "resin_family" in enabled:
        result["resin_family"] = _resin_family(result)

    for orientation in ORIENTATIONS:
        count_name = f"num_{_orientation_label(orientation)}_plies"
        pct_name = f"pct_{_orientation_label(orientation)}_plies"
        if count_name in enabled:
            result[count_name] = angles.apply(lambda values, angle=orientation: values.count(angle))
        if pct_name in enabled:
            result[pct_name] = angles.apply(
                lambda values, angle=orientation: _orientation_percent(values, angle)
            )

    if "feature_interaction_terms" in enabled:
        result = _add_interaction_terms(result)

    LOGGER.info("Engineered features added: %s", sorted(set(result.columns) - set(data.columns)))
    return result


def available_engineered_features() -> list[str]:
    """Return supported engineered feature names."""
    features = [
        "stacking_sequence_string",
        "stacking_symmetry_flag",
        "balanced_laminate_flag",
        "total_thickness",
        "average_ply_angle",
        "angle_variance",
        "angle_entropy",
        "thickness_per_layer",
        "strength_to_weight_ratio",
        "material_family",
        "resin_family",
        "layup_complexity_score",
        "feature_interaction_terms",
    ]
    for orientation in ORIENTATIONS:
        label = _orientation_label(orientation)
        features.append(f"num_{label}_plies")
        features.append(f"pct_{label}_plies")
    return features


def parse_stacking_sequence(value: object) -> list[int]:
    """Extract ply angles from layup string or scalar orientation."""
    if pd.isna(value):
        return []
    if isinstance(value, (int, float)):
        return [int(round(float(value)))]
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", str(value))
    return [int(round(float(number))) for number in numbers]


def _angle_lists(data: pd.DataFrame) -> pd.Series:
    """Return per-row ply angle lists from available laminate columns."""
    if "stacking_sequence" in data.columns:
        return data["stacking_sequence"].apply(parse_stacking_sequence)
    if "ply_orientation" in data.columns:
        return data["ply_orientation"].apply(parse_stacking_sequence)
    return pd.Series([[] for _ in range(len(data))], index=data.index)


def _stacking_sequence_strings(data: pd.DataFrame, angles: pd.Series) -> pd.Series:
    """Return normalized stacking sequence strings."""
    if "stacking_sequence" in data.columns:
        return data["stacking_sequence"].astype(str)
    return angles.apply(lambda values: "[" + "/".join(str(value) for value in values) + "]")


def _is_symmetric(values: list[int]) -> bool:
    """Return True when ply sequence is symmetric."""
    return bool(values) and values == list(reversed(values))


def _is_balanced(values: list[int]) -> bool:
    """Return True when +45 and -45 ply counts match."""
    return values.count(45) == values.count(-45)


def _safe_mean(values: list[int]) -> float:
    """Return mean angle or NaN."""
    return float(np.mean(values)) if values else np.nan


def _safe_variance(values: list[int]) -> float:
    """Return angle variance or NaN."""
    return float(np.var(values)) if values else np.nan


def _angle_entropy(values: list[int]) -> float:
    """Return Shannon entropy of orientation distribution."""
    if not values:
        return np.nan
    counts = Counter(values)
    total = len(values)
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def _layup_complexity(values: list[int]) -> float:
    """Return simple complexity score from unique orientations and transitions."""
    if not values:
        return np.nan
    transitions = sum(1 for left, right in zip(values, values[1:]) if left != right)
    return float(len(set(values)) + transitions / max(len(values), 1))


def _total_thickness(data: pd.DataFrame, angles: pd.Series) -> pd.Series:
    """Return total thickness from available thickness fields."""
    if "total_thickness" in data.columns:
        return pd.to_numeric(data["total_thickness"], errors="coerce")
    if "thickness_mm" in data.columns:
        thickness = pd.to_numeric(data["thickness_mm"], errors="coerce")
        ply_counts = angles.apply(lambda values: max(len(values), 1))
        return thickness * ply_counts
    return pd.Series(np.nan, index=data.index)


def _thickness_per_layer(data: pd.DataFrame, angles: pd.Series) -> pd.Series:
    """Return thickness per ply layer."""
    if "thickness_mm" in data.columns:
        thickness = pd.to_numeric(data["thickness_mm"], errors="coerce")
        ply_counts = angles.apply(lambda values: max(len(values), 1))
        return thickness / ply_counts
    return pd.Series(np.nan, index=data.index)


def _strength_to_weight_ratio(data: pd.DataFrame) -> pd.Series:
    """Return strength-to-density ratio when available."""
    strength_columns = [
        "tensile_strength_mpa",
        "compressive_strength_mpa",
        "flexural_strength_mpa",
        "shear_strength_mpa",
    ]
    strength = None
    for column in strength_columns:
        if column in data.columns:
            strength = pd.to_numeric(data[column], errors="coerce")
            break
    if strength is None or "density_g_cm3" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    density = pd.to_numeric(data["density_g_cm3"], errors="coerce")
    return strength / density.replace(0, np.nan)


def _material_family(data: pd.DataFrame) -> pd.Series:
    """Map material or fiber names to broad material family."""
    source = data.get("material", data.get("fiber", pd.Series("", index=data.index)))
    return source.astype(str).str.lower().apply(_classify_material)


def _resin_family(data: pd.DataFrame) -> pd.Series:
    """Map matrix names to broad resin family."""
    source = data.get("matrix", data.get("material", pd.Series("", index=data.index)))
    return source.astype(str).str.lower().apply(_classify_resin)


def _classify_material(value: str) -> str:
    """Classify broad material family from text."""
    if "carbon" in value or "cfrp" in value:
        return "carbon"
    if "glass" in value or "gfrp" in value:
        return "glass"
    if "kevlar" in value or "aramid" in value:
        return "aramid"
    if "hybrid" in value:
        return "hybrid"
    return "unknown"


def _classify_resin(value: str) -> str:
    """Classify broad resin family from text."""
    if "epoxy" in value:
        return "epoxy"
    if "polyester" in value:
        return "polyester"
    if "vinyl" in value:
        return "vinyl_ester"
    if "peek" in value or "thermoplastic" in value:
        return "thermoplastic"
    return "unknown"


def _orientation_label(orientation: int) -> str:
    """Return safe label for orientation feature names."""
    if orientation < 0:
        return f"minus_{abs(orientation)}"
    if orientation > 0:
        return f"plus_{orientation}"
    return "zero"


def _orientation_percent(values: list[int], orientation: int) -> float:
    """Return percentage of selected ply orientation."""
    if not values:
        return np.nan
    return values.count(orientation) / len(values) * 100


def _add_interaction_terms(data: pd.DataFrame) -> pd.DataFrame:
    """Add simple interaction terms from common numeric fields."""
    result = data.copy()
    interactions = [
        ("fiber_volume_fraction", "total_thickness"),
        ("density_g_cm3", "total_thickness"),
        ("average_ply_angle", "angle_variance"),
    ]
    for left, right in interactions:
        if left in result.columns and right in result.columns:
            left_values = pd.to_numeric(result[left], errors="coerce")
            right_values = pd.to_numeric(result[right], errors="coerce")
            result[f"{left}_x_{right}"] = left_values * right_values
    return result
