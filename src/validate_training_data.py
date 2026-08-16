"""Validate locked candidate dataset for baseline strength prediction.

This script performs dataset checks only. It does not train models, clean rows,
or infer stacking-sequence data.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_ROOT / "data" / "kaggle" / "2" / "composite_material_strength.csv"
LOCKED_DATASET = PROJECT_ROOT / "data" / "training" / "composite_strength_training.csv"
TARGET_COLUMN = "tensile_strength_mpa"
EXPECTED_COLUMNS = [
    "fiber_type",
    "resin_type",
    "density_g_cm3",
    "layer_count",
    "curing_temperature_c",
    "fiber_volume_fraction",
    "void_content_pct",
    "tensile_strength_mpa",
]
STACKING_KEYWORDS = ["stack", "sequence", "angle", "orientation", "ply", "layup"]


@dataclass(frozen=True)
class ValidationResult:
    """Validation result with status and findings."""

    status: str
    confirmed_problems: list[str]
    warnings: list[str]
    report: dict[str, Any]


def load_training_candidate(path: Path = SOURCE_DATASET) -> pd.DataFrame:
    """Load candidate training CSV."""
    return pd.read_csv(path)


def validate_training_data(data: pd.DataFrame) -> ValidationResult:
    """Run dataset validation checks and return PASS/WARN/FAIL status."""
    confirmed_problems: list[str] = []
    warnings: list[str] = []

    missing_expected = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    extra_columns = [column for column in data.columns if column not in EXPECTED_COLUMNS]
    if missing_expected:
        confirmed_problems.append(f"Missing expected columns: {missing_expected}")
    if extra_columns:
        warnings.append(f"Unexpected extra columns: {extra_columns}")
    if TARGET_COLUMN not in data.columns:
        confirmed_problems.append(f"Missing target column: {TARGET_COLUMN}")

    report = build_validation_report(data)
    target = data[TARGET_COLUMN] if TARGET_COLUMN in data.columns else pd.Series(dtype=float)

    negative_numeric = {
        column: int((data[column] < 0).sum())
        for column in data.select_dtypes(include="number").columns
    }
    if any(count > 0 for count in negative_numeric.values()):
        confirmed_problems.append(f"Negative numeric values found: {negative_numeric}")

    if "fiber_volume_fraction" in data.columns:
        invalid = int(((data["fiber_volume_fraction"] <= 0) | (data["fiber_volume_fraction"] > 1)).sum())
        if invalid:
            confirmed_problems.append(f"Invalid fiber_volume_fraction rows: {invalid}")

    if "void_content_pct" in data.columns:
        invalid = int(((data["void_content_pct"] < 0) | (data["void_content_pct"] > 100)).sum())
        if invalid:
            confirmed_problems.append(f"Invalid void_content_pct rows: {invalid}")

    if "layer_count" in data.columns:
        invalid = int(((data["layer_count"] <= 0) | ((data["layer_count"] % 1) != 0)).sum())
        if invalid:
            confirmed_problems.append(f"Invalid layer_count rows: {invalid}")

    if "curing_temperature_c" in data.columns:
        temp_min = float(data["curing_temperature_c"].min())
        temp_max = float(data["curing_temperature_c"].max())
        if temp_min < 0 or temp_max > 400:
            warnings.append(f"Curing temperature range needs domain review: {temp_min} to {temp_max}")

    if not target.empty:
        if int(target.isna().sum()):
            confirmed_problems.append(f"Missing target values: {int(target.isna().sum())}")
        if int((target <= 0).sum()):
            confirmed_problems.append(f"Non-positive target values: {int((target <= 0).sum())}")
        if report["target"]["iqr_outlier_count"]:
            warnings.append(
                "Target has IQR outliers requiring domain review: "
                f"{report['target']['iqr_outlier_count']}"
            )

    stacking_columns = report["stacking_sequence_columns"]
    if not stacking_columns:
        warnings.append("No stacking-sequence, ply-orientation, angle, or layup columns found.")

    status = "FAIL" if confirmed_problems else "WARN" if warnings else "PASS"
    return ValidationResult(status, confirmed_problems, warnings, report)


def build_validation_report(data: pd.DataFrame) -> dict[str, Any]:
    """Build detailed validation report."""
    categorical = data.select_dtypes(exclude="number")
    numeric = data.select_dtypes(include="number")
    missing = pd.DataFrame(
        {
            "missing_count": data.isna().sum(),
            "missing_percentage": (data.isna().mean() * 100).round(3),
        }
    )
    duplicate_count = int(data.duplicated().sum())
    target = data[TARGET_COLUMN] if TARGET_COLUMN in data.columns else pd.Series(dtype=float)

    return {
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "columns": list(data.columns),
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "head": data.head(5),
        "tail": data.tail(5),
        "categorical_unique_values": {
            column: sorted(categorical[column].dropna().unique().tolist())
            for column in categorical.columns
        },
        "numeric_ranges": {
            column: {
                "min": float(numeric[column].min()),
                "max": float(numeric[column].max()),
                "mean": float(numeric[column].mean()),
                "median": float(numeric[column].median()),
                "std": float(numeric[column].std()),
            }
            for column in numeric.columns
        },
        "missing_values": missing,
        "duplicates": {
            "count": duplicate_count,
            "percentage": round(float(data.duplicated().mean() * 100), 3),
        },
        "target": target_report(target),
        "data_quality": data_quality_report(data),
        "stacking_sequence_columns": [
            column
            for column in data.columns
            if any(keyword in column.lower() for keyword in STACKING_KEYWORDS)
        ],
    }


def target_report(target: pd.Series) -> dict[str, Any]:
    """Return target statistics and suspicious-value counts."""
    if target.empty:
        return {}
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (target < lower_bound) | (target > upper_bound)
    return {
        "min": float(target.min()),
        "max": float(target.max()),
        "mean": float(target.mean()),
        "median": float(target.median()),
        "std": float(target.std()),
        "missing": int(target.isna().sum()),
        "zero": int((target == 0).sum()),
        "negative": int((target < 0).sum()),
        "iqr_lower_bound": float(lower_bound),
        "iqr_upper_bound": float(upper_bound),
        "iqr_outlier_count": int(outlier_mask.sum()),
    }


def data_quality_report(data: pd.DataFrame) -> dict[str, Any]:
    """Return confirmed quality checks and domain-review concerns."""
    numeric = data.select_dtypes(include="number")
    return {
        "negative_numeric_counts": {
            column: int((numeric[column] < 0).sum())
            for column in numeric.columns
        },
        "invalid_percentages": {
            "fiber_volume_fraction_le_0": int((data["fiber_volume_fraction"] <= 0).sum()),
            "fiber_volume_fraction_gt_1": int((data["fiber_volume_fraction"] > 1).sum()),
            "void_content_pct_lt_0": int((data["void_content_pct"] < 0).sum()),
            "void_content_pct_gt_100": int((data["void_content_pct"] > 100).sum()),
        },
        "invalid_layer_counts": {
            "layer_count_le_0": int((data["layer_count"] <= 0).sum()),
            "layer_count_non_integer": int(((data["layer_count"] % 1) != 0).sum()),
        },
        "temperature_range_c": {
            "min": float(data["curing_temperature_c"].min()),
            "max": float(data["curing_temperature_c"].max()),
        },
        "categorical_values": {
            "fiber_type": sorted(data["fiber_type"].dropna().unique().tolist()),
            "resin_type": sorted(data["resin_type"].dropna().unique().tolist()),
        },
    }


def lock_training_dataset(source_path: Path = SOURCE_DATASET, output_path: Path = LOCKED_DATASET) -> None:
    """Create locked training copy without modifying source Kaggle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)


def print_report(result: ValidationResult) -> None:
    """Print concise validation report."""
    report = result.report
    print("=== CompositeAI Training Dataset Validation ===")
    print(f"Status: {result.status}")
    print(f"Rows: {report['shape']['rows']}")
    print(f"Columns: {report['shape']['columns']}")
    print(f"Column names: {report['columns']}")
    print(f"Data types: {report['dtypes']}")
    print("\nFirst 5 rows:")
    print(report["head"].to_string(index=False))
    print("\nLast 5 rows:")
    print(report["tail"].to_string(index=False))
    print(f"\nCategorical unique values: {report['categorical_unique_values']}")
    print(f"\nNumerical ranges: {report['numeric_ranges']}")
    print("\nMissing values:")
    print(report["missing_values"].to_string())
    print(f"\nDuplicates: {report['duplicates']}")
    print(f"\nTarget report ({TARGET_COLUMN}): {report['target']}")
    print(f"\nData-quality report: {report['data_quality']}")
    print(f"\nStacking-sequence columns: {report['stacking_sequence_columns']}")
    print(f"\nConfirmed problems: {result.confirmed_problems or 'None'}")
    print(f"Warnings: {result.warnings or 'None'}")
    print(
        "\nStacking note: The current dataset can be used for tensile-strength "
        "prediction but is insufficient by itself for true stacking-sequence optimization."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Validate CompositeAI training dataset.")
    parser.add_argument("--csv", type=Path, default=SOURCE_DATASET, help="Candidate CSV path.")
    parser.add_argument("--lock", action="store_true", help="Create locked training copy if not FAIL.")
    return parser.parse_args()


def main() -> int:
    """Run validation script."""
    args = parse_args()
    data = load_training_candidate(args.csv)
    result = validate_training_data(data)
    print_report(result)
    if args.lock and result.status != "FAIL":
        lock_training_dataset(args.csv, LOCKED_DATASET)
        print(f"\nLocked training dataset: {LOCKED_DATASET}")
    elif args.lock:
        print("\nLocked training dataset: not created because validation failed.")
    return 1 if result.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
