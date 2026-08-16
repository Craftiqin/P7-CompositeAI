"""Reusable preprocessing pipeline for baseline strength prediction.

Step 2 creates pipeline structure only. Final fitting must happen on the
training split in Step 4 to prevent leakage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "training" / "composite_strength_training.csv"
PREPROCESSING_CONFIG_PATH = PROJECT_ROOT / "data" / "training" / "preprocessing_config.json"
TARGET_COLUMN = "tensile_strength_mpa"
STACKING_KEYWORDS = ("stack", "sequence", "angle", "orientation", "ply", "layup")


@dataclass(frozen=True)
class DatasetInspection:
    """Locked dataset structure and quality facts."""

    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    categorical_features: list[str]
    numerical_features: list[str]
    target: str
    missing_values: dict[str, int]
    infinite_values: dict[str, int]
    duplicate_rows: int
    categorical_unique_values: dict[str, list[str]]
    numerical_ranges: dict[str, dict[str, float]]
    target_outlier_count: int
    target_outlier_range: dict[str, float | None]
    stacking_sequence_columns: list[str]


def load_training_data(path: str | Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """Load locked training dataset."""
    return pd.read_csv(path)


def split_features_target(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X features and y target."""
    if target_column not in data.columns:
        raise ValueError(f"Target column missing: {target_column}")
    return data.drop(columns=[target_column]).copy(), data[target_column].copy()


def infer_feature_columns(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[list[str], list[str]]:
    """Infer categorical and numerical feature columns from actual data."""
    features, _ = split_features_target(data, target_column)
    categorical_features = list(features.select_dtypes(exclude="number").columns)
    numerical_features = list(features.select_dtypes(include="number").columns)
    return categorical_features, numerical_features


def inspect_training_dataset(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> DatasetInspection:
    """Inspect locked dataset for Step 2 preprocessing decisions."""
    categorical_features, numerical_features = infer_feature_columns(data, target_column)
    numeric = data.select_dtypes(include="number")
    target = data[target_column]
    outlier_mask = target_iqr_outliers(target)
    outlier_values = target.loc[outlier_mask]
    stacking_columns = [
        column
        for column in data.columns
        if any(keyword in column.lower() for keyword in STACKING_KEYWORDS)
    ]
    return DatasetInspection(
        rows=int(data.shape[0]),
        columns=int(data.shape[1]),
        column_names=list(data.columns),
        dtypes={column: str(dtype) for column, dtype in data.dtypes.items()},
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target=target_column,
        missing_values=data.isna().sum().astype(int).to_dict(),
        infinite_values={
            column: int(np.isinf(numeric[column]).sum())
            for column in numeric.columns
        },
        duplicate_rows=int(data.duplicated().sum()),
        categorical_unique_values={
            column: sorted(data[column].dropna().astype(str).unique().tolist())
            for column in categorical_features
        },
        numerical_ranges={
            column: {
                "min": float(data[column].min()),
                "max": float(data[column].max()),
                "zero_count": int((data[column] == 0).sum()),
                "negative_count": int((data[column] < 0).sum()),
            }
            for column in numerical_features + [target_column]
        },
        target_outlier_count=int(outlier_mask.sum()),
        target_outlier_range={
            "min": float(outlier_values.min()) if not outlier_values.empty else None,
            "max": float(outlier_values.max()) if not outlier_values.empty else None,
        },
        stacking_sequence_columns=stacking_columns,
    )


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build unfitted sklearn preprocessing transformer."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_one_hot_encoder() -> OneHotEncoder:
    """Build OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor_from_data(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> ColumnTransformer:
    """Infer columns and build unfitted preprocessor from dataframe."""
    categorical_features, numerical_features = infer_feature_columns(data, target_column)
    return build_preprocessor(numerical_features, categorical_features)


def target_iqr_outliers(target: pd.Series) -> pd.Series:
    """Return mask for target IQR outliers."""
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (target < lower_bound) | (target > upper_bound)


def check_for_infinite_values(data: pd.DataFrame) -> dict[str, int]:
    """Return +inf/-inf count per numeric column."""
    numeric = data.select_dtypes(include="number")
    return {column: int(np.isinf(numeric[column]).sum()) for column in numeric.columns}


def create_preprocessing_config(
    inspection: DatasetInspection,
    source_dataset: str = "data/training/composite_strength_training.csv",
) -> dict[str, Any]:
    """Create project-relative preprocessing metadata."""
    return {
        "project_stage": "Step 2 - Data Cleaning and Preprocessing",
        "source_dataset": source_dataset,
        "target": inspection.target,
        "categorical_features": inspection.categorical_features,
        "numerical_features": inspection.numerical_features,
        "encoder": 'OneHotEncoder(handle_unknown="ignore")',
        "scaler": "StandardScaler for numerical features",
        "missing_value_policy": {
            "current_missing_values": inspection.missing_values,
            "pipeline_numeric_strategy": "median",
            "pipeline_categorical_strategy": "most_frequent",
        },
        "outlier_policy": {
            "target_iqr_outliers": inspection.target_outlier_count,
            "target_outlier_range": inspection.target_outlier_range,
            "action": "retain; domain review required",
        },
        "leakage_prevention": (
            "Build pipeline now, but fit only on training split during Step 4."
        ),
        "stacking_sequence_data_available": bool(inspection.stacking_sequence_columns),
        "version_date": "2026-08-13",
    }


def write_preprocessing_config(
    config: dict[str, Any],
    output_path: str | Path = PREPROCESSING_CONFIG_PATH,
) -> Path:
    """Write preprocessing metadata JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def smoke_test_preprocessor(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[tuple[int, int], list[str]]:
    """Fit-transform small sample only to validate structure; not model training."""
    sample = data.head(20).copy()
    features, _ = split_features_target(sample, target_column)
    categorical_features, numerical_features = infer_feature_columns(sample, target_column)
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    transformed = preprocessor.fit_transform(features)
    feature_names = list(preprocessor.get_feature_names_out())
    if np.isnan(transformed).any():
        raise ValueError("Smoke test produced NaN values.")
    return transformed.shape, feature_names


def save_preprocessor(
    preprocessor: ColumnTransformer,
    output_path: str | Path,
) -> Path:
    """Serialize sklearn preprocessing object if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    return path


def clean_laminate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Legacy helper: remove duplicates and missing rows when explicitly requested."""
    return data.drop_duplicates().dropna().reset_index(drop=True)


def main() -> int:
    """Inspect dataset, write config, and run preprocessing smoke test."""
    data = load_training_data()
    inspection = inspect_training_dataset(data)
    config = create_preprocessing_config(inspection)
    config_path = write_preprocessing_config(config)
    transformed_shape, feature_names = smoke_test_preprocessor(data)

    print("=== CompositeAI Step 2 Preprocessing Validation ===")
    print(f"Dataset: {TRAINING_DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {inspection.rows}")
    print(f"Columns: {inspection.columns}")
    print(f"Target: {inspection.target}")
    print(f"Categorical features: {inspection.categorical_features}")
    print(f"Numerical features: {inspection.numerical_features}")
    print(f"Missing values: {inspection.missing_values}")
    print(f"Infinite values: {inspection.infinite_values}")
    print(f"Duplicate rows: {inspection.duplicate_rows}")
    print(f"Target IQR outliers retained: {inspection.target_outlier_count}")
    print(f"Target outlier range: {inspection.target_outlier_range}")
    print(f"Stacking-sequence columns: {inspection.stacking_sequence_columns}")
    print(f"Smoke transformed shape: {transformed_shape}")
    print(f"Smoke feature names: {feature_names}")
    print(f"Config written: {config_path.relative_to(PROJECT_ROOT)}")
    print("Status: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
