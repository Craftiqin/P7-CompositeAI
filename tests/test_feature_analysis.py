"""Tests for Step 3 feature analysis artifacts."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.feature_analysis import (
    FEATURE_SPEC_PATH,
    ML_READY_DATA_PATH,
    baseline_feature_set,
    create_feature_specification,
    leakage_review,
    load_training_data,
)
from src.preprocessing import TARGET_COLUMN, split_features_target


class FeatureAnalysisTest(unittest.TestCase):
    """Validate ML-ready feature specification without training models."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load locked dataset."""
        cls.data = load_training_data()

    def test_feature_spec_matches_dataset_columns(self) -> None:
        """Spec X + y matches actual dataset columns."""
        spec = create_feature_specification(self.data)
        expected_columns = set(spec["baseline_features"] + [spec["target"]])
        self.assertEqual(expected_columns, set(self.data.columns))

    def test_target_excluded_from_x(self) -> None:
        """Target never enters X columns."""
        features, target = split_features_target(self.data)
        self.assertNotIn(TARGET_COLUMN, features.columns)
        self.assertEqual(target.name, TARGET_COLUMN)

    def test_baseline_feature_lists_are_correct(self) -> None:
        """Categorical/numerical baseline lists match locked data."""
        feature_set = baseline_feature_set(self.data)
        self.assertEqual(feature_set["categorical"], ["fiber_type", "resin_type"])
        self.assertEqual(
            feature_set["numerical"],
            [
                "density_g_cm3",
                "layer_count",
                "curing_temperature_c",
                "fiber_volume_fraction",
                "void_content_pct",
            ],
        )

    def test_no_target_leakage(self) -> None:
        """Leakage review passes for baseline feature set."""
        review = leakage_review(self.data)
        self.assertEqual(review["status"], "PASS")
        self.assertFalse(review["target_in_features"])
        self.assertEqual(review["suspicious_feature_names"], [])

    def test_ml_ready_artifact_shape_is_deterministic(self) -> None:
        """ML-ready artifact has locked shape when present."""
        ml_ready = pd.read_csv(ML_READY_DATA_PATH)
        self.assertEqual(ml_ready.shape, (10000, 8))
        self.assertEqual(list(ml_ready.columns), baseline_feature_set(self.data)["x_columns"] + [TARGET_COLUMN])

    def test_feature_spec_artifact_is_ready(self) -> None:
        """Feature specification exists and records no engineered features."""
        spec = json.loads(FEATURE_SPEC_PATH.read_text(encoding="utf-8"))
        self.assertEqual(spec["target"], TARGET_COLUMN)
        self.assertEqual(spec["engineered_features"], [])
        self.assertEqual(spec["leakage_decisions"]["status"], "PASS")

    def test_feature_engineering_does_not_add_nan_or_inf(self) -> None:
        """No Step 3 engineered features are added; baseline has finite numeric values."""
        features, _ = split_features_target(self.data)
        numeric = features.select_dtypes(include="number")
        self.assertEqual(int(numeric.isna().sum().sum()), 0)
        self.assertEqual(int(np.isinf(numeric).sum().sum()), 0)


if __name__ == "__main__":
    unittest.main()
