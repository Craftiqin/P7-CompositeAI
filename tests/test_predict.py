"""Tests for Step 6 real prediction service."""

from __future__ import annotations

import math
import unittest

import pandas as pd
from sklearn.pipeline import Pipeline

from src.predict import (
    REQUIRED_FEATURES,
    PredictionInputError,
    inspect_model_artifact,
    load_model,
    predict_strength,
)
from src.train import DATASET_PATH


class Step6PredictionServiceTest(unittest.TestCase):
    """Validate production prediction module without model training."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load fixed valid samples from locked ML-ready data."""
        cls.data = pd.read_csv(DATASET_PATH)
        cls.valid_row = cls.data.loc[0, REQUIRED_FEATURES].to_dict()
        cls.valid_batch = cls.data.loc[:4, REQUIRED_FEATURES]

    def test_saved_model_loads_as_pipeline(self) -> None:
        """Saved artifact loads as sklearn Pipeline with preprocessing."""
        model = load_model()
        artifact = inspect_model_artifact()
        self.assertIsInstance(model, Pipeline)
        self.assertTrue(artifact["preprocessing_included"])
        self.assertEqual(artifact["expected_input_columns"], REQUIRED_FEATURES)

    def test_valid_single_prediction_is_finite(self) -> None:
        """One valid row returns one finite numeric prediction."""
        result = predict_strength(self.valid_row)
        prediction = result["predicted_tensile_strength_mpa"]
        self.assertIsInstance(prediction, float)
        self.assertTrue(math.isfinite(prediction))
        self.assertEqual(len(result["predictions"]), 1)
        self.assertNotIn("confidence", result)

    def test_missing_input_field_fails(self) -> None:
        """Missing required fields raise useful input error."""
        bad_input = dict(self.valid_row)
        bad_input.pop("fiber_type")
        with self.assertRaisesRegex(PredictionInputError, "Missing required input"):
            predict_strength(bad_input)

    def test_invalid_categorical_input_fails(self) -> None:
        """Unknown category is rejected before prediction."""
        bad_input = dict(self.valid_row)
        bad_input["fiber_type"] = "Unobtanium"
        with self.assertRaisesRegex(PredictionInputError, "Unsupported fiber_type"):
            predict_strength(bad_input)

    def test_invalid_numerical_input_fails(self) -> None:
        """NaN/inf/non-numeric values are rejected."""
        bad_input = dict(self.valid_row)
        bad_input["density_g_cm3"] = float("nan")
        with self.assertRaisesRegex(PredictionInputError, "finite numeric"):
            predict_strength(bad_input)

        bad_input = dict(self.valid_row)
        bad_input["curing_temperature_c"] = float("inf")
        with self.assertRaisesRegex(PredictionInputError, "finite numeric"):
            predict_strength(bad_input)

        bad_input = dict(self.valid_row)
        bad_input["void_content_pct"] = "not numeric"
        with self.assertRaisesRegex(PredictionInputError, "finite numeric"):
            predict_strength(bad_input)

    def test_batch_prediction_preserves_count(self) -> None:
        """Multiple input rows produce same number of predictions."""
        result = predict_strength(self.valid_batch)
        self.assertEqual(len(result["predictions"]), 5)
        self.assertTrue(all(math.isfinite(value) for value in result["predictions"]))

    def test_out_of_range_input_warns(self) -> None:
        """Out-of-range but numeric values produce extrapolation warning."""
        edge_input = dict(self.valid_row)
        edge_input["curing_temperature_c"] = 200.0
        result = predict_strength(edge_input)
        self.assertTrue(any("outside observed training-data range" in warning for warning in result["warnings"]))

    def test_streamlit_prediction_imports(self) -> None:
        """Streamlit app can import prediction module."""
        import app

        self.assertTrue(callable(app.render_strength_prediction))


if __name__ == "__main__":
    unittest.main()
