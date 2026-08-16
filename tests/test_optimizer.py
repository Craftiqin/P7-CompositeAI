"""Tests for Step 10 CLT optimization demonstrator."""

from __future__ import annotations

import unittest

from src.clt import LaminateLoadCase
from src.optimizer import (
    OptimizationConfig,
    default_demo_optimization,
    load_tu_delft_demo_case,
    optimize_stacking_sequence,
    optimizer_result_is_valid,
    validate_reference_case,
)
from src.stacking_sequence import DEFAULT_ALLOWED_ANGLES, SequenceConfig, is_balanced, is_symmetric


class Step10OptimizerTest(unittest.TestCase):
    """Validate transparent CLT optimizer without ANN coupling."""

    def test_tu_delft_reference_case(self) -> None:
        """Reference lambda_cs is reproduced within documented tolerance."""
        validation = validate_reference_case()
        self.assertEqual(validation["validation_status"], "pass")
        self.assertAlmostEqual(validation["reference_lambda_cs"], 10394.81)
        self.assertLess(abs(validation["difference_pct"]), 1.0)

    def test_valid_sequence_returned(self) -> None:
        """Optimizer returns constraint-valid sequence."""
        result = default_demo_optimization(max_candidates=30)
        self.assertTrue(optimizer_result_is_valid(result))
        self.assertEqual(len(result["best_sequence"]), 48)

    def test_symmetry_balance_allowed_angles(self) -> None:
        """Best sequence satisfies symmetry, balance, and allowed angles."""
        result = default_demo_optimization(max_candidates=30)
        sequence = result["best_sequence"]
        self.assertTrue(is_symmetric(sequence))
        self.assertTrue(is_balanced(sequence, DEFAULT_ALLOWED_ANGLES))
        self.assertTrue(set(sequence).issubset(set(DEFAULT_ALLOWED_ANGLES)))

    def test_fixed_ply_count(self) -> None:
        """Configured ply count is preserved."""
        case = load_tu_delft_demo_case()
        result = optimize_stacking_sequence(
            material=case["material"],
            load_case=case["load_case"],
            config=OptimizationConfig(
                sequence_config=SequenceConfig(
                    allowed_angles=DEFAULT_ALLOWED_ANGLES,
                    require_symmetric=True,
                    require_balanced=True,
                    expected_ply_count=8,
                ),
                max_candidates=20,
            ),
            allowables=case["allowables"],
        )
        self.assertEqual(result["ply_count"], 8)
        self.assertEqual(len(result["best_sequence"]), 8)

    def test_objective_improves_or_matches_baseline(self) -> None:
        """Best found sequence is at least as good as evaluated baseline pool member."""
        result = default_demo_optimization(max_candidates=30)
        self.assertGreaterEqual(result["best_lambda_cs"], result["baseline"]["lambda_cs"])

    def test_no_duplicate_candidates_in_top_results(self) -> None:
        """Top candidate list contains no duplicate sequences."""
        result = default_demo_optimization(max_candidates=50)
        top_sequences = [tuple(row["sequence"]) for row in result["top_candidates"]]
        self.assertEqual(len(top_sequences), len(set(top_sequences)))

    def test_invalid_load_for_source_failure_route(self) -> None:
        """Source-compatible lambda_cs rejects Nxy for current route."""
        case = load_tu_delft_demo_case()
        with self.assertRaisesRegex(ValueError, "Nx/Ny"):
            optimize_stacking_sequence(
                material=case["material"],
                load_case=LaminateLoadCase(nx_n_per_m=-100.0, nxy_n_per_m=10.0),
                config=OptimizationConfig(
                    sequence_config=SequenceConfig(
                        allowed_angles=DEFAULT_ALLOWED_ANGLES,
                        require_symmetric=True,
                        require_balanced=True,
                        expected_ply_count=8,
                    ),
                    max_candidates=5,
                ),
                allowables=case["allowables"],
            )

    def test_streamlit_pages_import(self) -> None:
        """Prediction and optimizer pages import successfully."""
        import app

        self.assertTrue(callable(app.render_strength_prediction))
        self.assertTrue(callable(app.render_stacking_optimizer))


if __name__ == "__main__":
    unittest.main()
