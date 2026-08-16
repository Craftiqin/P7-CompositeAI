"""Numerical scaling helpers."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler

LOGGER = logging.getLogger(__name__)

SCALERS = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "Normalizer": Normalizer,
}


def numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric feature columns."""
    return list(data.select_dtypes(include="number").columns)


def scale_numeric(
    data: pd.DataFrame,
    method: str = "StandardScaler",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, object | None]:
    """Scale numeric columns and return fitted scaler."""
    selected_columns = columns or numeric_columns(data)
    selected_columns = [column for column in selected_columns if column in data.columns]
    if not selected_columns or method == "None":
        return data.copy(), None
    if method not in SCALERS:
        raise ValueError(f"Unsupported scaler: {method}")

    result = data.copy()
    numeric_values = result[selected_columns].replace([float("inf"), float("-inf")], pd.NA)
    numeric_values = numeric_values.fillna(numeric_values.median(numeric_only=True)).fillna(0)
    scaler = SCALERS[method]()
    result[selected_columns] = scaler.fit_transform(numeric_values)
    LOGGER.info("Scaled numeric columns using %s: %s", method, selected_columns)
    return result, scaler
