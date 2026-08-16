"""Dataset profiling and statistics helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """Return compact dataset profile for UI, metadata, and Gemini context."""
    numeric_data = data.select_dtypes(include="number")
    missing_by_column = data.isna().sum().sort_values(ascending=False)
    profile: dict[str, Any] = {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": int(data.isna().sum().sum()),
        "missing_by_column": missing_by_column.astype(int).to_dict(),
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "numeric_columns": list(numeric_data.columns),
        "categorical_columns": list(data.select_dtypes(exclude="number").columns),
    }

    if not numeric_data.empty:
        profile["numeric_summary"] = numeric_data.describe().round(3).to_dict()
        profile["correlation"] = numeric_data.corr(numeric_only=True).round(3).fillna(0).to_dict()
    else:
        profile["numeric_summary"] = {}
        profile["correlation"] = {}

    return profile


def dataset_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe-level statistics for display."""
    stats = {
        "Rows": data.shape[0],
        "Columns": data.shape[1],
        "Duplicate Rows": int(data.duplicated().sum()),
        "Missing Values": int(data.isna().sum().sum()),
        "Numeric Columns": len(data.select_dtypes(include="number").columns),
        "Categorical Columns": len(data.select_dtypes(exclude="number").columns),
    }
    return pd.DataFrame(stats.items(), columns=["Metric", "Value"])


def missing_values_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages by column."""
    missing = data.isna().sum()
    missing_percent = (missing / max(len(data), 1) * 100).round(2)
    return pd.DataFrame(
        {
            "column": missing.index,
            "missing_count": missing.astype(int).values,
            "missing_percent": missing_percent.values,
        }
    ).sort_values("missing_count", ascending=False)
