"""Outlier detection utilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

LOGGER = logging.getLogger(__name__)


def detect_outliers(
    data: pd.DataFrame,
    method: str = "IQR",
    columns: list[str] | None = None,
    contamination: float = 0.05,
    z_threshold: float = 3.0,
) -> pd.Series:
    """Return boolean mask where True marks outlier rows."""
    numeric = _numeric_subset(data, columns)
    if numeric.empty:
        return pd.Series(False, index=data.index)

    filled = numeric.fillna(numeric.median(numeric_only=True))
    method_key = method.lower().replace("-", "_").replace(" ", "_")
    if method_key == "iqr":
        mask = _iqr_mask(filled)
    elif method_key in {"z_score", "zscore"}:
        mask = _z_score_mask(filled, z_threshold)
    elif method_key == "isolation_forest":
        detector = IsolationForest(contamination=contamination, random_state=42)
        mask = pd.Series(detector.fit_predict(filled) == -1, index=data.index)
    elif method_key == "local_outlier_factor":
        neighbors = min(20, max(2, len(filled) - 1))
        detector = LocalOutlierFactor(n_neighbors=neighbors, contamination=contamination)
        mask = pd.Series(detector.fit_predict(filled) == -1, index=data.index)
    else:
        raise ValueError(f"Unsupported outlier method: {method}")

    LOGGER.info("Outliers detected with %s: %s", method, int(mask.sum()))
    return mask


def remove_outliers(data: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame:
    """Return dataset with outlier rows removed."""
    return data.loc[~outlier_mask].reset_index(drop=True)


def _numeric_subset(data: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    """Return numeric columns selected for outlier detection."""
    selected = data[columns] if columns else data
    return selected.select_dtypes(include="number")


def _iqr_mask(data: pd.DataFrame) -> pd.Series:
    """Return IQR outlier mask."""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((data < lower) | (data > upper)).any(axis=1)


def _z_score_mask(data: pd.DataFrame, threshold: float) -> pd.Series:
    """Return Z-score outlier mask."""
    std = data.std(ddof=0).replace(0, np.nan)
    z_scores = ((data - data.mean()) / std).abs()
    return z_scores.gt(threshold).any(axis=1).fillna(False)
