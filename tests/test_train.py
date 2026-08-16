"""Tests for Step 4 model training artifacts."""

from __future__ import annotations

import json
import unittest

import joblib
import pandas as pd

from src.train import (
    BEST_MODEL_PATH,
    COMPARISON_PATH,
    DATASET_PATH,
    FEATURE_SPEC_PATH,
    METADATA_PATH,
    OUTLIER_EXPERIMENT_PATH,
    split_data,
    validate_dataset_against_spec,
)


class Step4TrainingArtifactTest(unittest.TestCase):
    """Validate Step 4 training outputs without retraining models."""

    def test_dataset_matches_feature_spec(self) -> None:
        """ML-ready dataset must match Step 3 specification."""
        data = pd.read_csv(DATASET_PATH)
        spec = json.loads(FEATURE_SPEC_PATH.read_text(encoding="utf-8"))
        validate_dataset_against_spec(data, spec)
        self.assertEqual(data.shape, (10000, 8))

    def test_train_test_split_is_reproducible(self) -> None:
        """Split must use 80/20 deterministic rows."""
        data = pd.read_csv(DATASET_PATH)
        x_train, x_test, y_train, y_test = split_data(data)
        self.assertEqual(len(x_train), 8000)
        self.assertEqual(len(x_test), 2000)
        self.assertEqual(len(y_train), 8000)
        self.assertEqual(len(y_test), 2000)

    def test_model_comparison_has_required_metrics(self) -> None:
        """Comparison CSV contains regression metrics."""
        comparison = pd.read_csv(COMPARISON_PATH)
        required = {
            "model",
            "train_mae",
            "test_mae",
            "train_rmse",
            "test_rmse",
            "train_r2",
            "test_r2",
        }
        self.assertTrue(required.issubset(comparison.columns))
        self.assertGreaterEqual(len(comparison), 3)
        self.assertTrue((comparison["test_rmse"] > 0).all())

    def test_best_model_artifact_loads(self) -> None:
        """Serialized trained pipeline loads successfully."""
        model = joblib.load(BEST_MODEL_PATH)
        self.assertEqual([name for name, _ in model.steps], ["preprocessing", "model"])

    def test_metadata_matches_best_model(self) -> None:
        """Metadata records selected model and metrics."""
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        comparison = pd.read_csv(COMPARISON_PATH).sort_values("test_rmse")
        self.assertEqual(metadata["model_name"], comparison.iloc[0]["model"])
        self.assertEqual(metadata["train_test_split"]["train_rows"], 8000)
        self.assertEqual(metadata["train_test_split"]["test_rows"], 2000)

    def test_outlier_experiment_exists(self) -> None:
        """Secondary outlier experiment result is recorded."""
        experiment = pd.read_csv(OUTLIER_EXPERIMENT_PATH)
        self.assertEqual(len(experiment), 2)
        self.assertIn("outlier_filtered_training_only", set(experiment["experiment"]))


if __name__ == "__main__":
    unittest.main()
