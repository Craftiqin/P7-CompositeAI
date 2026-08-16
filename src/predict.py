"""Production inference helpers for tensile-strength prediction.

This module loads the validated Step 5 sklearn Pipeline and performs inference
only. It must not train or refit models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_strength_model.joblib"
METADATA_PATH = PROJECT_ROOT / "saved_models" / "model_metadata.json"
FEATURE_SPEC_PATH = PROJECT_ROOT / "data" / "training" / "feature_specification.json"
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "training" / "ml_ready_features.csv"

TARGET_COLUMN = "tensile_strength_mpa"
REQUIRED_FEATURES = [
    "fiber_type",
    "resin_type",
    "density_g_cm3",
    "layer_count",
    "curing_temperature_c",
    "fiber_volume_fraction",
    "void_content_pct",
]
CATEGORICAL_FEATURES = ["fiber_type", "resin_type"]
NUMERICAL_FEATURES = [
    "density_g_cm3",
    "layer_count",
    "curing_temperature_c",
    "fiber_volume_fraction",
    "void_content_pct",
]


class PredictionInputError(ValueError):
    """Raised when prediction input fails validation."""


@dataclass(frozen=True)
class PredictionResult:
    """Prediction result with warnings and validation metrics."""

    predictions: list[float]
    warnings: list[str]
    model_name: str
    metrics: dict[str, float]

    @property
    def predicted_tensile_strength_mpa(self) -> float:
        """Return first prediction for single-row callers."""
        return self.predictions[0]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result for Streamlit and tests."""
        return {
            "predicted_tensile_strength_mpa": self.predicted_tensile_strength_mpa,
            "predictions": self.predictions,
            "warnings": self.warnings,
            "model_name": self.model_name,
            "metrics": self.metrics,
        }


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    """Load validated sklearn Pipeline once."""
    model = joblib.load(MODEL_PATH)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(model).__name__}")
    if not hasattr(model, "predict"):
        raise TypeError("Loaded model artifact does not expose predict().")
    return model


@lru_cache(maxsize=1)
def load_model_metadata() -> dict[str, Any]:
    """Load Step 4/5 model metadata."""
    return load_json(METADATA_PATH)


@lru_cache(maxsize=1)
def load_feature_specification() -> dict[str, Any]:
    """Load Step 3 feature specification."""
    return load_json(FEATURE_SPEC_PATH)


@lru_cache(maxsize=1)
def load_training_reference() -> pd.DataFrame:
    """Load locked ML-ready training dataset for category/range checks."""
    return pd.read_csv(TRAINING_DATA_PATH)


def inspect_model_artifact() -> dict[str, Any]:
    """Return saved model structure and expected input information."""
    model = load_model()
    metadata = load_model_metadata()
    data = load_training_reference()

    expected_columns = list(getattr(model, "feature_names_in_", REQUIRED_FEATURES))
    named_steps = list(getattr(model, "named_steps", {}).keys())
    model_step = getattr(model, "named_steps", {}).get("model")

    return {
        "artifact_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "loads_correctly": True,
        "is_sklearn_pipeline": isinstance(model, Pipeline),
        "preprocessing_included": "preprocessing" in named_steps,
        "pipeline_steps": named_steps,
        "model_type": type(model_step).__name__ if model_step is not None else type(model).__name__,
        "expected_input_columns": expected_columns,
        "expected_feature_ordering": expected_columns,
        "supported_categories": supported_categories(data),
        "dataset_supported_ranges": dataset_supported_ranges(data),
        "model_name": metadata.get("model_name", "Unknown"),
    }


def supported_categories(data: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Return categorical values observed in training data."""
    reference = data if data is not None else load_training_reference()
    return {
        column: sorted(str(value) for value in reference[column].dropna().unique())
        for column in CATEGORICAL_FEATURES
    }


def dataset_supported_ranges(data: pd.DataFrame | None = None) -> dict[str, dict[str, float]]:
    """Return min/max numerical ranges observed in training data."""
    reference = data if data is not None else load_training_reference()
    return {
        column: {
            "min": float(reference[column].min()),
            "max": float(reference[column].max()),
        }
        for column in NUMERICAL_FEATURES
    }


def normalize_prediction_input(input_data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Convert dict/DataFrame input to ordered prediction DataFrame."""
    if isinstance(input_data, pd.DataFrame):
        frame = input_data.copy()
    elif isinstance(input_data, dict):
        frame = pd.DataFrame([input_data])
    else:
        raise PredictionInputError("Input must be a dict or pandas DataFrame.")

    missing = [column for column in REQUIRED_FEATURES if column not in frame.columns]
    if missing:
        raise PredictionInputError(f"Missing required input field(s): {', '.join(missing)}")

    return frame.loc[:, REQUIRED_FEATURES].copy()


def validate_prediction_input(input_data: dict[str, Any] | pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate prediction input and return ordered frame plus warnings."""
    frame = normalize_prediction_input(input_data)
    categories = supported_categories()
    ranges = dataset_supported_ranges()
    warnings: list[str] = []

    for column in CATEGORICAL_FEATURES:
        values = frame[column].astype(str)
        invalid = sorted(set(values) - set(categories[column]))
        if invalid:
            allowed = ", ".join(categories[column])
            raise PredictionInputError(
                f"Unsupported {column}: {', '.join(invalid)}. Supported values: {allowed}"
            )
        frame[column] = values

    for column in NUMERICAL_FEATURES:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid_mask = numeric.isna() | np.isinf(numeric.to_numpy())
        if invalid_mask.any():
            raise PredictionInputError(f"{column} must contain finite numeric value(s).")

        if column == "layer_count":
            non_integer = (numeric % 1 != 0).any()
            if non_integer:
                raise PredictionInputError("layer_count must be an integer count.")
            if (numeric <= 0).any():
                raise PredictionInputError("layer_count must be greater than zero.")

        if column == "density_g_cm3" and (numeric <= 0).any():
            raise PredictionInputError("density_g_cm3 must be greater than zero.")

        if column == "fiber_volume_fraction" and ((numeric < 0) | (numeric > 1)).any():
            raise PredictionInputError("fiber_volume_fraction must be a 0-1 fraction.")

        if column == "void_content_pct" and (numeric < 0).any():
            raise PredictionInputError("void_content_pct must be non-negative.")

        lower = ranges[column]["min"]
        upper = ranges[column]["max"]
        out_of_range = (numeric < lower) | (numeric > upper)
        if out_of_range.any():
            warnings.append(
                f"{column} outside observed training-data range "
                f"[{lower:g}, {upper:g}]. Prediction may involve extrapolation."
            )
        frame[column] = numeric

    return frame, warnings


def predict_strength(input_data: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    """Predict tensile strength from validated laminate/material inputs."""
    model = load_model()
    metadata = load_model_metadata()
    frame, warnings = validate_prediction_input(input_data)
    predictions = model.predict(frame)

    if not np.issubdtype(predictions.dtype, np.number):
        raise RuntimeError("Model returned non-numeric predictions.")
    if np.isnan(predictions).any() or np.isinf(predictions).any():
        raise RuntimeError("Model returned NaN or infinite predictions.")

    result = PredictionResult(
        predictions=[float(value) for value in predictions],
        warnings=warnings,
        model_name=str(metadata.get("model_name", "ANN/MLP")),
        metrics={
            "mae": float(metadata["metrics"]["mae"]),
            "rmse": float(metadata["metrics"]["rmse"]),
            "r2": float(metadata["metrics"]["r2"]),
        },
    )
    LOGGER.info("Generated %s tensile-strength prediction(s).", len(result.predictions))
    return result.to_dict()
