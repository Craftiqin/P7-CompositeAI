"""Tests for reference-material benchmark utilities."""

from __future__ import annotations

import unittest

from src.material_benchmark import (
    DEFAULT_REFERENCE_DATABASE_PATH,
    calculate_density_ratio,
    calculate_specific_strength,
    calculate_strength_ratio,
    generate_engineering_insights,
    load_reference_materials,
    rank_materials,
)


class MaterialBenchmarkTest(unittest.TestCase):
    """Validate aerospace metals benchmark calculations."""

    def test_database_loads(self) -> None:
        """Reference database exists and exposes expected records."""
        self.assertTrue(DEFAULT_REFERENCE_DATABASE_PATH.exists())
        materials = load_reference_materials()
        self.assertEqual(len(materials), 5)
        self.assertEqual(materials[0]["material"], "Aluminum 2024-T3")

    def test_ratio_calculations(self) -> None:
        """Strength, density, and specific-strength math is stable."""
        self.assertAlmostEqual(calculate_strength_ratio(1644.0, 572.0), 2.8741258741)
        self.assertAlmostEqual(calculate_density_ratio(1.55, 2.81), 0.5516014234)
        self.assertAlmostEqual(calculate_specific_strength(1644.0, 1.55), 1060.64516129)
        self.assertIsNone(calculate_density_ratio(None, 2.81))
        self.assertIsNone(calculate_specific_strength(1644.0, None))

    def test_rank_materials(self) -> None:
        """Ranking exposes ordered reference-material rows."""
        ranked = rank_materials(1644.0, 1.55)
        self.assertEqual(len(ranked), 5)
        self.assertEqual(ranked[0]["material"], "Inconel 718")
        aluminum = next(item for item in ranked if item["material"] == "Aluminum 7075-T6")
        self.assertAlmostEqual(aluminum["strength_ratio"], 1644.0 / 572.0)
        self.assertEqual(aluminum["strength_rank"], 3)
        self.assertGreater(aluminum["specific_strength_ratio"], 1.0)

    def test_engineering_insights(self) -> None:
        """Insight helper returns factual comparison text."""
        insights = generate_engineering_insights(1644.0, 1.55)
        self.assertTrue(any("Aluminum 7075-T6" in insight for insight in insights))
        self.assertTrue(any("specific strength" in insight.casefold() for insight in insights))

    def test_empty_input_handling(self) -> None:
        """Invalid composite strength fails clearly."""
        with self.assertRaisesRegex(ValueError, "composite_strength_mpa"):
            rank_materials(0.0, 1.55)


if __name__ == "__main__":
    unittest.main()
