"""Tests for professional 3D engineering visualizations."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

import app
from src.engineering_interpretation import (
    ENGINEERING_EXPLANATION,
    INTERPRETATION_SECTIONS,
    SIMPLE_EXPLANATION,
    build_engineering_interpretation,
)
from src.engineering_visualizations import (
    build_material_benchmark_3d,
    build_optimization_landscape,
    build_ply_failure_map,
    build_strength_response_surface,
)
from src.report_generator import build_report_csv_payload, build_report_html_payload


class EngineeringVisualizationsTest(unittest.TestCase):
    """Validate real-data Plotly 3D visualization builders and pages."""

    def test_navigation_section_contains_visualization_pages(self) -> None:
        """Dedicated sidebar section exposes all engineering visualization pages."""
        self.assertEqual(
            app.MENU_SECTIONS["ENGINEERING VISUALIZATIONS"],
            [
                "Strength Surface",
                "Optimization Landscape",
                "Ply Failure Map",
                "Material Benchmark",
            ],
        )

    def test_strength_surface_uses_model_predictions(self) -> None:
        """Strength surface uses ANN prediction grid, not placeholder values."""
        figure, metadata = build_strength_response_surface(grid_size=8)
        surface = figure.data[0]
        self.assertEqual(surface.type, "surface")
        self.assertEqual(metadata["grid_points"], 64)
        self.assertGreater(metadata["max_predicted_strength_mpa"], metadata["min_predicted_strength_mpa"])
        self.assertIn("ml_ready_features.csv", metadata["dataset_path"])
        self.assertIn("Predicted Strength", surface.hovertemplate)

    def test_strength_surface_interpretation_changes_with_chart_data(self) -> None:
        """Strength interpretation is generated from displayed chart values."""
        figure_small, metadata_small = build_strength_response_surface(grid_size=6)
        figure_large, metadata_large = build_strength_response_surface(grid_size=8)
        small_text = build_engineering_interpretation(
            "strength_surface",
            figure_small,
            metadata_small,
            ENGINEERING_EXPLANATION,
        )["sections"]["What This Graph Shows"]
        large_text = build_engineering_interpretation(
            "strength_surface",
            figure_large,
            metadata_large,
            ENGINEERING_EXPLANATION,
        )["sections"]["What This Graph Shows"]
        self.assertIn("36 grid points", small_text)
        self.assertIn("64 grid points", large_text)
        self.assertNotEqual(small_text, large_text)

    def test_optimization_landscape_uses_evaluated_candidates(self) -> None:
        """Optimization landscape uses actual optimizer candidate history."""
        figure, metadata = build_optimization_landscape(max_candidates=250)
        self.assertGreaterEqual(len(figure.data), 3)
        self.assertEqual(metadata["candidates_evaluated"], 250)
        self.assertAlmostEqual(metadata["baseline_lambda_cs"], 10319.427536738505)
        self.assertAlmostEqual(metadata["optimized_lambda_cs"], 21506.235235969612)
        self.assertAlmostEqual(metadata["improvement_pct"], 108.4053127889567)
        self.assertIn("λ_cs", figure.data[0].hovertemplate)
        interpretation = build_engineering_interpretation(
            "optimization_landscape",
            figure,
            metadata,
            ENGINEERING_EXPLANATION,
        )
        self.assertEqual(list(interpretation["sections"].keys()), INTERPRETATION_SECTIONS)
        self.assertIn("10319", interpretation["sections"]["Executive Summary"])
        self.assertIn("21506", interpretation["sections"]["Executive Summary"])
        self.assertIn("convergence cannot be proven", interpretation["sections"]["Limitations"])

    def test_ply_failure_map_uses_clt_results(self) -> None:
        """Ply failure map uses source-backed CLT failure rows."""
        figure, metadata = build_ply_failure_map()
        self.assertEqual(len(figure.data), 2)
        self.assertEqual(metadata["ply_count"], 48)
        self.assertEqual(metadata["critical_ply"], 1)
        self.assertGreater(metadata["max_failure_index"], 0)
        self.assertIn("Failure Index", figure.data[0].hovertemplate)
        simple = build_engineering_interpretation(
            "ply_failure_map",
            figure,
            metadata,
            SIMPLE_EXPLANATION,
        )
        engineering = build_engineering_interpretation(
            "ply_failure_map",
            figure,
            metadata,
            ENGINEERING_EXPLANATION,
        )
        self.assertNotEqual(simple["sections"]["Executive Summary"], engineering["sections"]["Executive Summary"])
        self.assertIn("Ply 1", engineering["sections"]["Executive Summary"])

    def test_material_benchmark_uses_reference_and_training_data(self) -> None:
        """Material benchmark combines reference DB metals and ML-ready composite aggregates."""
        figure, metadata = build_material_benchmark_3d()
        self.assertEqual(len(figure.data), 1)
        self.assertIn("material_benchmark_database.json", metadata["reference_database"])
        self.assertIn("ml_ready_features.csv", metadata["training_dataset"])
        self.assertIn("Carbon Fiber Composite", metadata["materials_plotted"])
        self.assertIn("Glass Fiber Composite", metadata["materials_plotted"])
        self.assertIn("Aramid Composite", metadata["materials_plotted"])
        self.assertIn("Stainless Steel 316L", metadata["missing_requested_materials"])
        self.assertIn("Specific Strength", figure.data[0].hovertemplate)
        interpretation = build_engineering_interpretation(
            "material_benchmark",
            figure,
            metadata,
            ENGINEERING_EXPLANATION,
        )
        self.assertIn("maximum tensile-strength", interpretation["sections"]["Executive Summary"])
        self.assertIn("not fabricated", interpretation["sections"]["Limitations"])

    def test_visualization_pages_render(self) -> None:
        """Each visualization page renders without empty chart failures."""
        app_file = Path(__file__).resolve().parents[1] / "app.py"
        expected_titles = {
            "Strength Surface": "Composite Strength Response Surface",
            "Optimization Landscape": "Optimization Search Landscape",
            "Ply Failure Map": "Ply Failure Distribution",
            "Material Benchmark": "Material Performance Benchmark",
        }
        for page_name, expected_title in expected_titles.items():
            with self.subTest(page=page_name):
                app_test = AppTest.from_file(str(app_file))
                app_test.session_state["selected_page"] = page_name
                app_test.run(timeout=120)
                self.assertEqual(app_test.exception, [])
                self.assertTrue(any(expected_title in item.value for item in app_test.markdown))
                self.assertTrue(any("Engineering Interpretation" in item.value for item in app_test.markdown))

    def test_report_exports_include_visualization_interpretations(self) -> None:
        """HTML and CSV report exports include generated visualization interpretation text."""
        html_name, html_text = build_report_html_payload()
        csv_name, csv_bytes = build_report_csv_payload()
        self.assertTrue(html_name.endswith(".html"))
        self.assertIn("Engineering Visualizations", html_text)
        self.assertIn("Composite Strength Response Surface", html_text)
        self.assertTrue(csv_name.endswith(".csv"))
        self.assertIn(b"engineering_visualization_interpretation", csv_bytes)


if __name__ == "__main__":
    unittest.main()
