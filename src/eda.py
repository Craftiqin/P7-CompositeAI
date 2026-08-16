"""Exploratory data analysis helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def dataset_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Return high-level dataset summary."""
    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": int(data.isna().sum().sum()),
        "memory_mb": round(float(data.memory_usage(deep=True).sum() / 1_000_000), 3),
    }


def feature_types(data: pd.DataFrame) -> pd.DataFrame:
    """Return feature type summary."""
    rows = []
    for column in data.columns:
        rows.append(
            {
                "column": column,
                "dtype": str(data[column].dtype),
                "non_null": int(data[column].notna().sum()),
                "missing": int(data[column].isna().sum()),
                "unique": int(data[column].nunique(dropna=True)),
                "kind": "numeric" if pd.api.types.is_numeric_dtype(data[column]) else "categorical",
            }
        )
    return pd.DataFrame(rows)


def duplicate_report(data: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate row report."""
    duplicate_mask = data.duplicated(keep=False)
    duplicates = data.loc[duplicate_mask].copy()
    return duplicates.reset_index(names="original_index")


def numerical_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Return numerical summary with skewness and kurtosis."""
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary["skewness"] = numeric.skew(numeric_only=True)
    summary["kurtosis"] = numeric.kurtosis(numeric_only=True)
    return summary.reset_index(names="column")


def pairwise_correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise correlations sorted by absolute value."""
    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation"])
    correlation = numeric.corr(numeric_only=True)
    rows = []
    columns = list(correlation.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            value = correlation.loc[left, right]
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "correlation": round(float(value), 4),
                    "abs_correlation": round(abs(float(value)), 4),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def target_distribution(data: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Return target distribution metadata."""
    if target_column not in data.columns:
        return {}
    target = data[target_column]
    result: dict[str, Any] = {
        "target": target_column,
        "missing": int(target.isna().sum()),
        "unique": int(target.nunique(dropna=True)),
        "dtype": str(target.dtype),
    }
    if pd.api.types.is_numeric_dtype(target):
        result["summary"] = target.describe().round(3).to_dict()
    else:
        result["class_balance"] = target.value_counts(dropna=False).to_dict()
    return result


def categorical_balance(data: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return class counts for categorical columns."""
    categorical = data.select_dtypes(exclude="number")
    return {
        column: categorical[column].value_counts(dropna=False).head(20).astype(int).to_dict()
        for column in categorical.columns
    }
