"""Reusable preprocessing pipeline for Step 3."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    mutual_info_regression,
)
from sklearn.impute import KNNImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.encoding import encode_categorical
from src.feature_engineering import engineer_laminate_features
from src.scaling import scale_numeric

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing pipeline configuration."""

    target_column: str | None = None
    enabled_engineered_features: list[str] | None = None
    numeric_imputer: str = "Median"
    categorical_imputer: str = "Mode"
    encoding_method: str = "One-Hot Encoding"
    scaling_method: str = "StandardScaler"
    feature_selection_method: str = "None"
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.95
    top_k_features: int = 20


class LaminateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible laminate feature engineering transformer."""

    def __init__(self, enabled_features: list[str] | None = None) -> None:
        self.enabled_features = enabled_features

    def fit(self, x_data: pd.DataFrame, y_data: Any = None) -> "LaminateFeatureEngineer":
        """Fit no-op transformer."""
        return self

    def transform(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """Add engineering features."""
        return engineer_laminate_features(x_data, self.enabled_features)


class DataFramePreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible dataframe preprocessor."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.metadata_: dict[str, Any] = {}

    def fit(self, x_data: pd.DataFrame, y_data: pd.Series | None = None) -> "DataFramePreprocessor":
        """Fit preprocessing metadata."""
        self.feature_columns_ = list(x_data.columns)
        return self

    def transform(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation, encoding, scaling, and feature selection."""
        data = x_data.copy()
        data = impute_missing_values(
            data,
            numeric_strategy=self.config.numeric_imputer,
            categorical_strategy=self.config.categorical_imputer,
        )
        data, encoders = encode_categorical(data, self.config.encoding_method)
        data, scaler = scale_numeric(data, self.config.scaling_method)
        data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.metadata_["encoders"] = list(encoders.keys())
        self.metadata_["scaler"] = type(scaler).__name__ if scaler else None
        return data


def build_preprocessing_pipeline(config: PreprocessingConfig) -> Pipeline:
    """Build sklearn Pipeline with feature engineering and preprocessing stages."""
    return Pipeline(
        steps=[
            ("feature_engineering", LaminateFeatureEngineer(config.enabled_engineered_features)),
            ("preprocessing", DataFramePreprocessor(config)),
        ]
    )


def run_preprocessing_pipeline(
    data: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, Pipeline, dict[str, Any]]:
    """Execute preprocessing pipeline and optional feature selection."""
    target = None
    features = data.copy()
    if config.target_column and config.target_column in features.columns:
        target = features[config.target_column]
        features = features.drop(columns=[config.target_column])

    pipeline = build_preprocessing_pipeline(config)
    processed_features = pipeline.fit_transform(features, target)
    ranking = rank_features(processed_features, target, config)
    selected_features = select_features(processed_features, target, config)

    if target is not None:
        selected_features[config.target_column] = target.reset_index(drop=True)

    metadata = {
        "config": asdict(config),
        "input_shape": data.shape,
        "output_shape": selected_features.shape,
        "feature_ranking": ranking.to_dict(orient="records"),
    }
    LOGGER.info("Preprocessing pipeline complete: %s -> %s", data.shape, selected_features.shape)
    return selected_features, pipeline, metadata


def impute_missing_values(
    data: pd.DataFrame,
    numeric_strategy: str = "Median",
    categorical_strategy: str = "Mode",
) -> pd.DataFrame:
    """Impute missing values using selected strategies."""
    result = data.copy()
    numeric_columns = list(result.select_dtypes(include="number").columns)
    categorical_columns = list(result.select_dtypes(exclude="number").columns)

    if numeric_columns:
        result[numeric_columns] = _impute_numeric(result[numeric_columns], numeric_strategy)
    if categorical_columns:
        result[categorical_columns] = _impute_categorical(
            result[categorical_columns],
            categorical_strategy,
        )
    return result


def rank_features(
    features: pd.DataFrame,
    target: pd.Series | None,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Rank features for selection display."""
    numeric = features.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame(columns=["feature", "score", "method"])

    rows = []
    variances = numeric.var(numeric_only=True).sort_values(ascending=False)
    for feature, score in variances.items():
        rows.append({"feature": feature, "score": float(score), "method": "variance"})

    if target is not None and pd.api.types.is_numeric_dtype(target):
        aligned_target = pd.to_numeric(target, errors="coerce").fillna(target.median())
        filled = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)
        mi_scores = mutual_info_regression(filled, aligned_target, random_state=42)
        for feature, score in zip(filled.columns, mi_scores):
            rows.append({"feature": feature, "score": float(score), "method": "mutual_information"})

        try:
            model = ExtraTreesRegressor(n_estimators=40, random_state=42)
            model.fit(filled, aligned_target)
            for feature, score in zip(filled.columns, model.feature_importances_):
                rows.append({"feature": feature, "score": float(score), "method": "tree_importance"})
        except Exception as exc:
            LOGGER.warning("Tree feature importance failed: %s", exc)

    return pd.DataFrame(rows).sort_values(["method", "score"], ascending=[True, False])


def select_features(
    features: pd.DataFrame,
    target: pd.Series | None,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Apply configured feature selection method."""
    method = config.feature_selection_method
    numeric = features.select_dtypes(include="number")
    non_numeric = features.drop(columns=list(numeric.columns), errors="ignore")
    if method == "None" or numeric.empty:
        return features.reset_index(drop=True)

    filled = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)
    selected_columns = list(filled.columns)

    if method == "Variance Threshold":
        selector = VarianceThreshold(threshold=config.variance_threshold)
        selector.fit(filled)
        selected_columns = list(filled.columns[selector.get_support()])
    elif method == "Correlation Threshold":
        selected_columns = _correlation_selected_columns(filled, config.correlation_threshold)
    elif (
        method == "Mutual Information"
        and target is not None
        and pd.api.types.is_numeric_dtype(target)
    ):
        selector = SelectKBest(mutual_info_regression, k=min(config.top_k_features, filled.shape[1]))
        selector.fit(filled, target)
        selected_columns = list(filled.columns[selector.get_support()])
    elif (
        method == "Recursive Feature Elimination"
        and target is not None
        and pd.api.types.is_numeric_dtype(target)
    ):
        estimator = RandomForestRegressor(n_estimators=30, random_state=42)
        selector = RFE(estimator, n_features_to_select=min(config.top_k_features, filled.shape[1]))
        selector.fit(filled, target)
        selected_columns = list(filled.columns[selector.get_support()])
    elif (
        method == "Tree Feature Importance"
        and target is not None
        and pd.api.types.is_numeric_dtype(target)
    ):
        estimator = ExtraTreesRegressor(n_estimators=50, random_state=42)
        estimator.fit(filled, target)
        scores = pd.Series(estimator.feature_importances_, index=filled.columns)
        selected_columns = list(scores.sort_values(ascending=False).head(config.top_k_features).index)

    selected = pd.concat([filled[selected_columns], non_numeric.reset_index(drop=True)], axis=1)
    return selected.reset_index(drop=True)


def save_pipeline_artifacts(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    output_dir: str | Path,
    stem: str = "preprocessing_pipeline",
) -> dict[str, Path]:
    """Save preprocessing pipeline and metadata."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pipeline_path = directory / f"{stem}.joblib"
    metadata_path = directory / f"{stem}_metadata.json"
    joblib.dump(pipeline, pipeline_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"pipeline": pipeline_path, "metadata": metadata_path}


def _impute_numeric(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Impute numeric dataframe."""
    if strategy == "Mean":
        return data.fillna(data.mean(numeric_only=True))
    if strategy == "Median":
        return data.fillna(data.median(numeric_only=True))
    if strategy == "Mode":
        return data.fillna(data.mode().iloc[0])
    if strategy == "KNN Imputer":
        imputer = KNNImputer(n_neighbors=min(5, max(1, len(data) - 1)))
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
    if strategy == "Forward Fill":
        return data.ffill().bfill()
    if strategy == "Backward Fill":
        return data.bfill().ffill()
    raise ValueError(f"Unsupported numeric imputer: {strategy}")


def _impute_categorical(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Impute categorical dataframe."""
    if strategy == "Mode":
        modes = data.mode(dropna=True)
        if modes.empty:
            return data.fillna("missing")
        return data.fillna(modes.iloc[0])
    if strategy == "Forward Fill":
        return data.ffill().bfill()
    if strategy == "Backward Fill":
        return data.bfill().ffill()
    return data.fillna("missing")


def _correlation_selected_columns(data: pd.DataFrame, threshold: float) -> list[str]:
    """Drop highly correlated duplicate features."""
    correlation = data.corr(numeric_only=True).abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [column for column in data.columns if column not in to_drop]
