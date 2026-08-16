"""Tests for Step 11 AI-vs-CLT comparison backend."""

from __future__ import annotations

import unittest

from src.ai_clt_comparison import (
    LOAD_CAPACITY_MODE,
    STRESS_MODE,
    ComparisonCase,
    calculate_percentage_difference,
    compare_ai_and_clt,
    convert_to_common_quantity,
)
from src.clt import LaminaMaterial, LaminateLoadCase, PA_PER_GPA


def test_material() -> LaminaMaterial:
    """Return deterministic stiffness-only lamina for comparison tests."""
    return LaminaMaterial(
        name="Verified UnitTest Carbon/Epoxy",
        e1_pa=127.55 * PA_PER_GPA,
        e2_pa=13.03 * PA_PER_GPA,
        g12_pa=6.41 * PA_PER_GPA,
        nu12=0.3,
        ply_thickness_m=0.000127,
    )


def ann_input() -> dict[str, object]:
    """Return valid ANN feature payload shape."""
    return {
        "fiber_type": "Carbon",
        "resin_type": "Epoxy",
        "density_g_cm3": 1.6,
        "layer_count": 4,
        "curing_temperature_c": 120.0,
        "fiber_volume_fraction": 0.55,
        "void_content_pct": 1.0,
    }


def compatible_case(**overrides: object) -> ComparisonCase:
    """Build compatible default case with explicit material evidence."""
    values = {
        "ann_input": ann_input(),
        "stacking_sequence": [0, 90, 90, 0],
        "material_card": test_material(),
        "load_case": LaminateLoadCase(nx_n_per_m=1_000.0),
        "comparison_quantity": STRESS_MODE,
        "material_equivalence_verified": True,
        "material_equivalence_evidence": "Unit test verified same material card and ANN sample.",
    }
    values.update(overrides)
    return ComparisonCase(**values)


class Step11AiCltComparisonTest(unittest.TestCase):
    """Validate comparison guardrails and valid conversions."""

    def test_compatible_comparison(self) -> None:
        """Compatible material and tensile load produce comparable result."""
        result = compare_ai_and_clt(
            compatible_case(),
            ann_result={"predicted_tensile_strength_mpa": 500.0},
        )
        self.assertTrue(result.comparable)
        self.assertEqual(result.material_compatibility["status"], "COMPATIBLE")
        self.assertEqual(result.common_quantity, STRESS_MODE)
        self.assertEqual(result.common_unit, "MPa")

    def test_incompatible_material(self) -> None:
        """Unverified ANN/CLT material equivalence blocks comparison."""
        result = compare_ai_and_clt(
            compatible_case(
                material_equivalence_verified=False,
                material_equivalence_evidence=None,
            ),
            ann_result={"predicted_tensile_strength_mpa": 500.0},
        )
        self.assertFalse(result.comparable)
        self.assertIn("equivalence to CLT material card is unproven", result.reason)
        self.assertIsNone(result.absolute_difference)

    def test_invalid_load_units(self) -> None:
        """Non-CLT base load units block comparison."""
        result = compare_ai_and_clt(
            compatible_case(base_load_unit="MPa"),
            ann_result={"predicted_tensile_strength_mpa": 500.0},
        )
        self.assertFalse(result.comparable)
        self.assertIn("Invalid load units", result.reason)
        self.assertIsNone(result.clt_common_value)

    def test_missing_base_load(self) -> None:
        """Missing CLT load case blocks comparison."""
        result = compare_ai_and_clt(
            compatible_case(load_case=None),
            ann_result={"predicted_tensile_strength_mpa": 500.0},
        )
        self.assertFalse(result.comparable)
        self.assertIn("Missing base load", result.reason)

    def test_missing_thickness(self) -> None:
        """Missing CLT material card blocks thickness-based conversion."""
        result = compare_ai_and_clt(
            compatible_case(material_card=None),
            ann_result={"predicted_tensile_strength_mpa": 500.0},
        )
        self.assertFalse(result.comparable)
        self.assertIn("Missing thickness", result.reason)

    def test_valid_conversion(self) -> None:
        """Known lambda and load convert to stress and load capacity."""
        stress_trace: list[dict[str, object]] = []
        stress = convert_to_common_quantity(
            compatible_case(),
            ann_tensile_strength_mpa=500.0,
            clt_lambda_cs=10.0,
            calculation_trace=stress_trace,
        )
        self.assertEqual(stress["common_unit"], "MPa")
        self.assertAlmostEqual(stress["ann_common_value"], 500.0)
        self.assertAlmostEqual(stress["clt_common_value"], 19.685039370078737)
        self.assertTrue(any(step["step"] == "CLT failure Nx" for step in stress_trace))

        load_capacity = convert_to_common_quantity(
            compatible_case(comparison_quantity=LOAD_CAPACITY_MODE),
            ann_tensile_strength_mpa=500.0,
            clt_lambda_cs=10.0,
        )
        self.assertEqual(load_capacity["common_unit"], "N/m")
        self.assertAlmostEqual(load_capacity["ann_common_value"], 254_000.0)
        self.assertAlmostEqual(load_capacity["clt_common_value"], 10_000.0)

    def test_invalid_conversion(self) -> None:
        """Compressive CLT load cannot compare to ANN tensile strength."""
        result = compare_ai_and_clt(
            compatible_case(load_case=LaminateLoadCase(nx_n_per_m=-1_000.0)),
            ann_result={"predicted_tensile_strength_mpa": 500.0},
        )
        self.assertFalse(result.comparable)
        self.assertIn("uniaxial tensile Nx", result.reason)

    def test_same_unit_comparison(self) -> None:
        """Stress mode compares MPa to MPa only."""
        result = compare_ai_and_clt(
            compatible_case(comparison_quantity=STRESS_MODE),
            ann_result={"predicted_tensile_strength_mpa": 500.0},
        )
        self.assertTrue(result.comparable)
        self.assertEqual(result.ann_unit, "MPa")
        self.assertEqual(result.common_unit, "MPa")
        self.assertIsNotNone(result.ann_common_value)
        self.assertIsNotNone(result.clt_common_value)

    def test_percentage_difference(self) -> None:
        """Relative difference uses CLT as reference."""
        self.assertAlmostEqual(calculate_percentage_difference(110.0, 100.0), 10.0)
        with self.assertRaisesRegex(ValueError, "non-zero"):
            calculate_percentage_difference(110.0, 0.0)

    def test_no_fabricated_comparison(self) -> None:
        """Incompatible case keeps raw outputs separate and omits differences."""
        result = compare_ai_and_clt(
            compatible_case(material_equivalence_verified=False),
            ann_result={"predicted_tensile_strength_mpa": 500.0},
        )
        self.assertFalse(result.comparable)
        self.assertIsNotNone(result.ann_value)
        self.assertIsNotNone(result.clt_value)
        self.assertIsNone(result.ann_common_value)
        self.assertIsNone(result.clt_common_value)
        self.assertIsNone(result.percentage_difference)


if __name__ == "__main__":
    unittest.main()
