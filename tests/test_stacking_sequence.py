"""Tests for Step 7 stacking-sequence representation."""

from __future__ import annotations

import unittest

from src.stacking_sequence import (
    DEFAULT_ALLOWED_ANGLES,
    LaminateSequence,
    SequenceConfig,
    estimate_search_space_size,
    format_laminate_sequence,
    generate_candidate_sequences,
    is_balanced,
    is_symmetric,
    validate_sequence,
)


class Step7StackingSequenceTest(unittest.TestCase):
    """Validate sequence representation without strength evaluation."""

    def test_valid_sequence(self) -> None:
        """Baseline sequence validates and preserves order."""
        sequence = [0, 45, -45, 90]
        result = validate_sequence(sequence, expected_ply_count=4)
        laminate = LaminateSequence(tuple(sequence), DEFAULT_ALLOWED_ANGLES)
        self.assertTrue(result.valid)
        self.assertEqual(laminate.sequence, tuple(sequence))
        self.assertEqual(laminate.ply_count, 4)

    def test_invalid_angle(self) -> None:
        """Angles outside allowed set fail validation."""
        result = validate_sequence([0, 30, 90])
        self.assertFalse(result.valid)
        self.assertIn("angle(s) not allowed", result.reasons[0])

    def test_invalid_ply_count(self) -> None:
        """Unexpected ply count fails validation."""
        result = validate_sequence([0, 45, -45], expected_ply_count=4)
        self.assertFalse(result.valid)
        self.assertIn("expected 4", result.reasons[0])

    def test_invalid_numeric_angle(self) -> None:
        """NaN, infinity, and non-numeric angles fail."""
        for sequence in ([0, float("nan")], [0, float("inf")], [0, "bad"]):
            result = validate_sequence(sequence)
            self.assertFalse(result.valid)

    def test_symmetric_sequence(self) -> None:
        """Mirrored sequence is symmetric."""
        self.assertTrue(is_symmetric([0, 45, -45, 90, 90, -45, 45, 0]))

    def test_non_symmetric_sequence(self) -> None:
        """Non-mirrored sequence is not symmetric."""
        self.assertFalse(is_symmetric([0, 45, -45, 90]))

    def test_balanced_sequence(self) -> None:
        """Equal +45/-45 counts are balanced."""
        self.assertTrue(is_balanced([0, 45, -45, 90]))

    def test_unbalanced_sequence(self) -> None:
        """Unequal +45/-45 counts are unbalanced."""
        self.assertFalse(is_balanced([0, 45, 45, 90]))

    def test_combined_validation(self) -> None:
        """Combined symmetric/balanced constraints return clear result."""
        result = validate_sequence(
            [0, 45, -45, 90],
            require_symmetric=True,
            require_balanced=True,
            expected_ply_count=4,
        )
        self.assertFalse(result.valid)
        self.assertIn("sequence is not symmetric", result.reasons)

    def test_candidate_generation(self) -> None:
        """Generator produces only valid constrained candidates."""
        config = SequenceConfig(
            allowed_angles=DEFAULT_ALLOWED_ANGLES,
            require_symmetric=True,
            require_balanced=True,
            expected_ply_count=8,
        )
        candidates = generate_candidate_sequences(config)
        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            result = validate_sequence(
                candidate.sequence,
                require_symmetric=True,
                require_balanced=True,
                expected_ply_count=8,
            )
            self.assertTrue(result.valid)

    def test_duplicate_candidates(self) -> None:
        """Generated candidates contain no duplicate sequences."""
        config = SequenceConfig(require_symmetric=True, expected_ply_count=6)
        candidates = generate_candidate_sequences(config)
        unique = {candidate.sequence for candidate in candidates}
        self.assertEqual(len(candidates), len(unique))

    def test_search_space_calculation(self) -> None:
        """Search-space estimate handles full and symmetric cases."""
        self.assertEqual(estimate_search_space_size(8, DEFAULT_ALLOWED_ANGLES), 4**8)
        self.assertEqual(
            estimate_search_space_size(8, DEFAULT_ALLOWED_ANGLES, require_symmetric=True),
            4**4,
        )

    def test_large_search_space_guard(self) -> None:
        """Huge exhaustive generation is rejected."""
        config = SequenceConfig(expected_ply_count=20, max_search_space=100)
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            generate_candidate_sequences(config)

    def test_visual_format(self) -> None:
        """Formatter returns mid-plane marker without evaluating strength."""
        text = format_laminate_sequence([0, 45, -45, 90])
        self.assertIn("Mid-plane", text)
        self.assertIn("+45°", text)


if __name__ == "__main__":
    unittest.main()
