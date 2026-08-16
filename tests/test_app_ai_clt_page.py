"""Streamlit-facing tests for Step 12 AI-vs-CLT page wiring."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.ai_clt_comparison import compare_ai_and_clt
from src.optimizer import validate_reference_case

import app


class Step12AiCltStreamlitIntegrationTest(unittest.TestCase):
    """Validate AI-vs-CLT Streamlit integration without browser rendering."""

    def test_page_imports(self) -> None:
        """AI-vs-CLT page renderer imports as callable."""
        self.assertTrue(callable(app.render_ai_vs_clt_comparison))
        self.assertTrue(callable(app.render_strength_prediction))
        self.assertTrue(callable(app.render_stacking_optimizer))

    def test_navigation_under_composite_engineering(self) -> None:
        """Navigation contains one AI-vs-CLT entry under engineering."""
        engineering = app.MENU_SECTIONS["ENGINEERING ANALYSIS"]
        self.assertEqual(
            engineering,
            ["CLT Analysis", "Stacking Optimizer", "AI vs CLT Comparison", "Composite vs Aerospace Metals"],
        )
        self.assertEqual(app.MENU_ITEMS.count("AI vs CLT Comparison"), 1)
        self.assertNotIn("Model Training", app.MENU_ITEMS)

    def test_sidebar_click_updates_active_entry_immediately(self) -> None:
        """Clicking a later page updates sidebar active state in same interaction."""
        app_test = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app_test.run(timeout=120)

        next(button for button in app_test.sidebar.button if button.label == "● About Project").click()
        app_test.run(timeout=120)

        self.assertEqual(app_test.session_state["selected_page"], "About Project")
        active_entries = [
            markdown.value
            for markdown in app_test.sidebar.markdown
            if "sidebar-active" in markdown.value
        ]
        self.assertEqual(active_entries, ['<div class="sidebar-active">● About Project</div>'])

    def test_invalid_input_state(self) -> None:
        """Invalid stacking sequence is caught before comparison."""
        sequence, error = app.parse_stacking_sequence_input("0, 30, 90")
        self.assertIsNone(sequence)
        self.assertIn("not allowed", error)

    def test_incompatible_material_state(self) -> None:
        """Default state is not directly comparable."""
        sequence, error = app.parse_stacking_sequence_input("0, 90, 90, 0")
        self.assertIsNone(error)
        case = app.build_ai_clt_comparison_case(
            ann_input=app.default_ai_clt_ann_input(),
            sequence=sequence,
            nx_value=1_000.0,
            ny_value=0.0,
            nxy_value=0.0,
            material_equivalence_verified=False,
            material_equivalence_evidence="",
            comparison_quantity=app.STRESS_MODE,
        )
        result = compare_ai_and_clt(
            case,
            ann_result={"predicted_tensile_strength_mpa": 1827.77},
        )
        self.assertFalse(result.comparable)
        self.assertIsNone(result.percentage_difference)
        self.assertIn("equivalence to CLT material card is unproven", result.reason)

    def test_valid_comparison_state(self) -> None:
        """Explicit evidence and tensile Nx allow backend comparison."""
        sequence, error = app.parse_stacking_sequence_input("0, 90, 90, 0")
        self.assertIsNone(error)
        case = app.build_ai_clt_comparison_case(
            ann_input=app.default_ai_clt_ann_input(),
            sequence=sequence,
            nx_value=1_000.0,
            ny_value=0.0,
            nxy_value=0.0,
            material_equivalence_verified=True,
            material_equivalence_evidence="Verified same material card for unit test.",
            comparison_quantity=app.STRESS_MODE,
        )
        result = compare_ai_and_clt(
            case,
            ann_result={"predicted_tensile_strength_mpa": 1827.77},
        )
        self.assertTrue(result.comparable)
        self.assertEqual(result.common_unit, "MPa")
        self.assertIsNotNone(result.percentage_difference)

    def test_no_direct_mpa_vs_lambda_comparison(self) -> None:
        """Incompatible state keeps ANN MPa and CLT lambda separate."""
        sequence, error = app.parse_stacking_sequence_input("0, 90, 90, 0")
        self.assertIsNone(error)
        case = app.build_ai_clt_comparison_case(
            ann_input=app.default_ai_clt_ann_input(),
            sequence=sequence,
            nx_value=1_000.0,
            ny_value=0.0,
            nxy_value=0.0,
            material_equivalence_verified=False,
            material_equivalence_evidence="",
            comparison_quantity=app.STRESS_MODE,
        )
        result = compare_ai_and_clt(
            case,
            ann_result={"predicted_tensile_strength_mpa": 1827.77},
        )
        self.assertEqual(result.ann_unit, "MPa")
        self.assertEqual(result.clt_unit, "dimensionless load factor")
        self.assertIsNone(result.absolute_difference)

    def test_clt_validation_display_data(self) -> None:
        """Reference validation values remain available for UI display."""
        validation = validate_reference_case()
        self.assertAlmostEqual(validation["reference_lambda_cs"], 10394.81)
        self.assertAlmostEqual(validation["our_lambda_cs"], 10319.4275, places=3)
        self.assertEqual(validation["validation_status"], "pass")


if __name__ == "__main__":
    unittest.main()
