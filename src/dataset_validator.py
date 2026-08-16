"""Dataset validation and quality scoring."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

NUMERIC_COLUMNS = {
    "fiber_volume_fraction",
    "density_g_cm3",
    "ply_orientation",
    "thickness_mm",
    "tensile_strength_mpa",
    "compressive_strength_mpa",
    "flexural_strength_mpa",
    "shear_strength_mpa",
    "elastic_modulus_gpa",
    "poisson_ratio",
    "failure_strain_percent",
    "temperature_c",
}

MECHANICAL_PROPERTY_COLUMNS = {
    "tensile_strength_mpa",
    "compressive_strength_mpa",
    "flexural_strength_mpa",
    "shear_strength_mpa",
    "elastic_modulus_gpa",
}

DEFAULT_SCORE_WEIGHTS = {
    "missing_values": 25,
    "duplicates": 15,
    "numeric_ranges": 25,
    "schema": 20,
    "data_types": 15,
}


@dataclass(frozen=True)
class ValidationIssue:
    """Single dataset validation issue."""

    category: str
    column: str
    severity: str
    message: str
    count: int


@dataclass(frozen=True)
class ValidationResult:
    """Dataset validation issues and quality score."""

    issues: list[ValidationIssue]
    quality_score: float
    metrics: dict[str, Any]


def validate_dataset(
    data: pd.DataFrame,
    score_weights: dict[str, int] | None = None,
) -> ValidationResult:
    """Validate data quality and compute weighted 0-100 score."""
    weights = score_weights or DEFAULT_SCORE_WEIGHTS
    issues: list[ValidationIssue] = []
    row_count = max(len(data), 1)
    cell_count = max(data.shape[0] * data.shape[1], 1)

    missing_count = int(data.isna().sum().sum())
    duplicate_count = int(data.duplicated().sum())
    type_mismatch_count = _count_type_mismatches(data, issues)
    numeric_range_count = _count_numeric_range_issues(data, issues)

    if missing_count:
        issues.append(
            ValidationIssue(
                "missing_values",
                "*",
                "warning",
                "Dataset contains missing values.",
                missing_count,
            )
        )
    if duplicate_count:
        issues.append(
            ValidationIssue(
                "duplicates",
                "*",
                "warning",
                "Dataset contains duplicate rows.",
                duplicate_count,
            )
        )

    schema_score = _schema_completeness_score(data)
    score = 100.0
    score -= weights["missing_values"] * min(missing_count / cell_count, 1.0)
    score -= weights["duplicates"] * min(duplicate_count / row_count, 1.0)
    score -= weights["numeric_ranges"] * min(numeric_range_count / row_count, 1.0)
    score -= weights["data_types"] * min(type_mismatch_count / cell_count, 1.0)
    score -= weights["schema"] * (1.0 - schema_score)
    quality_score = round(max(score, 0.0), 2)

    metrics = {
        "rows": len(data),
        "columns": len(data.columns),
        "missing_values": missing_count,
        "duplicate_rows": duplicate_count,
        "type_mismatches": type_mismatch_count,
        "numeric_range_issues": numeric_range_count,
        "schema_completeness": round(schema_score, 3),
    }
    LOGGER.info("Dataset validation complete: score=%s metrics=%s", quality_score, metrics)
    return ValidationResult(issues=issues, quality_score=quality_score, metrics=metrics)


def _count_type_mismatches(
    data: pd.DataFrame,
    issues: list[ValidationIssue],
) -> int:
    """Count non-numeric values in canonical numeric columns."""
    mismatch_count = 0
    for column in NUMERIC_COLUMNS.intersection(data.columns):
        numeric = pd.to_numeric(data[column], errors="coerce")
        invalid_mask = data[column].notna() & numeric.isna()
        count = int(invalid_mask.sum())
        mismatch_count += count
        if count:
            issues.append(
                ValidationIssue(
                    "data_types",
                    column,
                    "error",
                    "Numeric column contains non-numeric values.",
                    count,
                )
            )
    return mismatch_count


def _count_numeric_range_issues(
    data: pd.DataFrame,
    issues: list[ValidationIssue],
) -> int:
    """Count invalid physical ranges in canonical columns."""
    invalid_total = 0
    checks = {
        "density_g_cm3": lambda values: (values <= 0) | (values > 25),
        "thickness_mm": lambda values: values <= 0,
        "poisson_ratio": lambda values: (values < 0) | (values >= 0.5),
        "failure_strain_percent": lambda values: values < 0,
    }

    for column in MECHANICAL_PROPERTY_COLUMNS.intersection(data.columns):
        invalid_total += _record_invalid_numeric(
            data,
            column,
            lambda values: values < 0,
            "Mechanical property contains negative values.",
            issues,
        )

    for column, check_fn in checks.items():
        if column in data.columns:
            invalid_total += _record_invalid_numeric(
                data,
                column,
                check_fn,
                "Column contains physically invalid values.",
                issues,
            )

    if "fiber_volume_fraction" in data.columns:
        invalid_total += _record_invalid_numeric(
            data,
            "fiber_volume_fraction",
            lambda values: (values <= 0) | (values > 100),
            "Fibre volume fraction must be positive and not exceed 100 percent.",
            issues,
        )

    if "ply_orientation" in data.columns:
        invalid_count = _count_invalid_orientations(data["ply_orientation"])
        invalid_total += invalid_count
        if invalid_count:
            issues.append(
                ValidationIssue(
                    "numeric_ranges",
                    "ply_orientation",
                    "error",
                    "Ply orientation values must be between -90 and 90 degrees.",
                    invalid_count,
                )
            )

    return invalid_total


def _record_invalid_numeric(
    data: pd.DataFrame,
    column: str,
    check_fn: Any,
    message: str,
    issues: list[ValidationIssue],
) -> int:
    """Convert numeric column, apply range check, and record issues."""
    numeric = pd.to_numeric(data[column], errors="coerce")
    invalid = check_fn(numeric) & numeric.notna()
    count = int(invalid.sum())
    if count:
        issues.append(
            ValidationIssue(
                "numeric_ranges",
                column,
                "error",
                message,
                count,
            )
        )
    return count


def _count_invalid_orientations(series: pd.Series) -> int:
    """Count values containing ply angles outside -90 to 90 degrees."""
    invalid_count = 0
    for value in series.dropna():
        angles = _extract_angles(value)
        if not angles or any(angle < -90 or angle > 90 for angle in angles):
            invalid_count += 1
    return invalid_count


def _extract_angles(value: object) -> list[float]:
    """Extract one or more orientation angles from scalar or sequence string."""
    if isinstance(value, (int, float)):
        return [float(value)]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    return [float(number) for number in numbers]


def _schema_completeness_score(data: pd.DataFrame) -> float:
    """Estimate availability of core laminate fields."""
    core_columns = {
        "fiber_volume_fraction",
        "density_g_cm3",
        "stacking_sequence",
        "tensile_strength_mpa",
    }
    present = len(core_columns.intersection(data.columns))
    return present / len(core_columns)
