"""Tests for Step 5 model validation artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model_validation import (
    VALIDATION_PLOT_FILENAMES,
    VALIDATION_PLOT_DIR,
    VALIDATION_REPORT_PATH,
    get_validation_plot_path,
    load_validation_context,
    prediction_plausibility,
    run_model_validation,
    verify_saved_model,
)
from src.train import BEST_MODEL_PATH


class Step5ModelValidationTest(unittest.TestCase):
    """Validate selected model and Step 5 report."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load model/context/report once."""
        cls.context = load_validation_context()
        cls.model = joblib.load(BEST_MODEL_PATH)
        cls.report = json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))

    def test_saved_model_predicts_finite_values(self) -> None:
        """Saved model returns finite numeric predictions."""
        validation = verify_saved_model(self.context, self.model)
        self.assertEqual(validation["status"], "PASS")
        predictions = self.model.predict(self.context.x_test)
        self.assertFalse(np.isnan(predictions).any())
        self.assertFalse(np.isinf(predictions).any())

    def test_target_is_excluded_from_features(self) -> None:
        """Target must not be present in X."""
        self.assertNotIn("tensile_strength_mpa", self.context.x_train.columns)
        self.assertNotIn("tensile_strength_mpa", self.context.x_test.columns)

    def test_report_generation_ready(self) -> None:
        """Validation report exists and contains decision."""
        self.assertIn(self.report["final_decision"]["status"], {"READY", "NEEDS IMPROVEMENT", "NOT RELIABLE"})
        self.assertEqual(self.report["leakage_assessment"]["status"], "PASS")

    def test_reproducible_split_sizes(self) -> None:
        """Train/test split sizes remain 80/20."""
        self.assertEqual(len(self.context.x_train), 8000)
        self.assertEqual(len(self.context.x_test), 2000)

    def test_prediction_plausibility_counts(self) -> None:
        """Prediction plausibility report has no negative predictions."""
        predictions = self.model.predict(self.context.x_test)
        plausibility = prediction_plausibility(self.context, predictions)
        self.assertEqual(plausibility["negative_predictions"], 0)

    def test_subgroup_metrics_present(self) -> None:
        """Fiber/resin subgroup metrics are recorded."""
        subgroup = self.report["subgroup_validation"]
        self.assertIn("fiber_type", subgroup)
        self.assertIn("resin_type", subgroup)
        self.assertGreaterEqual(len(subgroup["fiber_type"]), 4)
        self.assertGreaterEqual(len(subgroup["resin_type"]), 4)

    def test_validation_plot_path_existing_file(self) -> None:
        """Plot helper finds existing validation plot in canonical report dir."""
        run_model_validation()
        for filename in VALIDATION_PLOT_FILENAMES:
            path = get_validation_plot_path(filename)
            self.assertIsNotNone(path)
            self.assertTrue(path.exists())

    def test_validation_plot_path_missing_file(self) -> None:
        """Plot helper returns None for valid filename missing from runtime dirs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = get_validation_plot_path(
                "actual_vs_predicted.html",
                project_root=Path(tmp_dir),
            )
        self.assertIsNone(path)

    def test_validation_plots_regenerated(self) -> None:
        """Step 5 validation regenerates missing plot artifacts."""
        target = VALIDATION_PLOT_DIR / "residual_distribution.html"
        if target.exists():
            target.unlink()
        self.assertFalse(target.exists())

        report = run_model_validation()

        self.assertIn("plots", report)
        for filename in VALIDATION_PLOT_FILENAMES:
            path = get_validation_plot_path(filename)
            self.assertIsNotNone(path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
