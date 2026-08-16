"""Categorical encoding helpers."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

LOGGER = logging.getLogger(__name__)


def categorical_columns(data: pd.DataFrame) -> list[str]:
    """Return categorical feature columns."""
    return list(data.select_dtypes(exclude="number").columns)


def encode_categorical(
    data: pd.DataFrame,
    method: str = "One-Hot Encoding",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Encode categorical columns and return encoders metadata."""
    selected_columns = columns or categorical_columns(data)
    selected_columns = [column for column in selected_columns if column in data.columns]
    if not selected_columns:
        return data.copy(), {}

    method_key = method.lower().replace("-", "_").replace(" ", "_")
    result = data.copy()
    encoders: dict[str, object] = {}

    if method_key == "one_hot_encoding":
        result = pd.get_dummies(result, columns=selected_columns, dummy_na=True)
        encoders["method"] = "one_hot_encoding"
    elif method_key == "ordinal_encoding":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        result[selected_columns] = encoder.fit_transform(result[selected_columns].astype(str))
        encoders["ordinal"] = encoder
    elif method_key == "label_encoding":
        for column in selected_columns:
            encoder = LabelEncoder()
            result[column] = encoder.fit_transform(result[column].astype(str))
            encoders[column] = encoder
    else:
        raise ValueError(f"Unsupported encoding method: {method}")

    LOGGER.info("Encoded categorical columns using %s: %s", method, selected_columns)
    return result, encoders
