"""Tests for Step 2 preprocessing pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    infer_feature_columns,
    inspect_training_dataset,
    load_training_data,
    save_preprocessor,
    split_features_target,
)


class PreprocessingPipelineTest(unittest.TestCase):
    """Validate preprocessing structure without training models."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load locked dataset once."""
        cls.data = load_training_data()

    def test_expected_columns_exist(self) -> None:
        """Dataset contains expected model columns."""
        expected = {
            "fiber_type",
            "resin_type",
            "density_g_cm3",
            "layer_count",
            "curing_temperature_c",
            "fiber_volume_fraction",
            "void_content_pct",
            "tensile_strength_mpa",
        }
        self.assertEqual(set(self.data.columns), expected)

    def test_target_excluded_from_features(self) -> None:
        """X excludes target and y uses target."""
        features, target = split_features_target(self.data)
        self.assertNotIn(TARGET_COLUMN, features.columns)
        self.assertEqual(target.name, TARGET_COLUMN)

    def test_feature_type_inference(self) -> None:
        """Categorical and numerical columns are inferred from real data."""
        categorical, numerical = infer_feature_columns(self.data)
        self.assertEqual(categorical, ["fiber_type", "resin_type"])
        self.assertEqual(
            numerical,
            [
                "density_g_cm3",
                "layer_count",
                "curing_temperature_c",
                "fiber_volume_fraction",
                "void_content_pct",
            ],
        )

    def test_unknown_categories_do_not_crash(self) -> None:
        """OneHotEncoder ignores unseen categories during inference."""
        categorical, numerical = infer_feature_columns(self.data)
        preprocessor = build_preprocessor(numerical, categorical)
        features, _ = split_features_target(self.data)
        train_sample = features.head(50)
        preprocessor.fit(train_sample)

        inference_sample = train_sample.head(2).copy()
        inference_sample.loc[inference_sample.index[0], "fiber_type"] = "NewFiber"
        inference_sample.loc[inference_sample.index[1], "resin_type"] = "NewResin"
        transformed = preprocessor.transform(inference_sample)

        self.assertEqual(transformed.shape[0], 2)
        self.assertFalse(np.isnan(transformed).any())

    def test_numerical_processing_outputs_no_nan(self) -> None:
        """Preprocessor produces numeric output without NaNs for valid input."""
        categorical, numerical = infer_feature_columns(self.data)
        preprocessor = build_preprocessor(numerical, categorical)
        features, _ = split_features_target(self.data)
        transformed = preprocessor.fit_transform(features.head(20))
        self.assertFalse(np.isnan(transformed).any())

    def test_inspection_records_outlier_policy_facts(self) -> None:
        """Inspection captures retained target outliers."""
        inspection = inspect_training_dataset(self.data)
        self.assertEqual(inspection.target_outlier_count, 168)
        self.assertEqual(inspection.stacking_sequence_columns, [])

    def test_pipeline_can_be_serialized(self) -> None:
        """Unfitted preprocessing object can be serialized."""
        categorical, numerical = infer_feature_columns(self.data)
        preprocessor = build_preprocessor(numerical, categorical)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "preprocessor.joblib"
            saved_path = save_preprocessor(preprocessor, output_path)
            loaded = joblib.load(saved_path)
        self.assertEqual(type(loaded).__name__, type(preprocessor).__name__)


if __name__ == "__main__":
    unittest.main()
