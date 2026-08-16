"""Tests for Step 3 engineering benchmark Streamlit integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

import app
from src.material_comparison import (
    DEFAULT_HTML_REPORT_PATH,
    generate_material_benchmark_html_report,
    generate_material_benchmark_pdf_report,
)


class Step3BenchmarkUiTest(unittest.TestCase):
    """Validate benchmark page, exports, and final-report integration."""

    def test_page_is_registered_in_sidebar(self) -> None:
        """Engineering benchmark appears as dedicated sidebar section."""
        self.assertIn("ENGINEERING ANALYSIS", app.MENU_SECTIONS)
        self.assertEqual(
            app.MENU_SECTIONS["ENGINEERING ANALYSIS"],
            ["CLT Analysis", "Stacking Optimizer", "AI vs CLT Comparison", "Composite vs Aerospace Metals"],
        )
        self.assertIn("Composite vs Aerospace Metals", app.MENU_ITEMS)
        self.assertTrue(callable(app.render_engineering_benchmark))

    def test_benchmark_page_runs_with_prediction_context(self) -> None:
        """AppTest can render benchmark page from saved prediction context."""
        app_test = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app_test.session_state["selected_page"] = "Composite vs Aerospace Metals"
        app_test.session_state["prediction_context"] = {
            "predicted_tensile_strength_mpa": 1827.77,
            "input_features": {"density_g_cm3": 1.6},
        }
        app_test.run(timeout=120)

        next(button for button in app_test.button if button.label == "Run Benchmark").click()
        app_test.run(timeout=120)

        self.assertIn("benchmark_context", app_test.session_state)
        self.assertTrue(
            any(
                "Engineering Benchmark" in markdown.value
                or "Composite vs Aerospace Metals" in markdown.value
                for markdown in app_test.markdown
            )
        )

    def test_html_report_is_generated(self) -> None:
        """HTML export writes benchmark artifact with dataset source label."""
        with tempfile.TemporaryDirectory() as tmp:
            html_text = generate_material_benchmark_html_report(
                1827.77,
                1.6,
                output_path=Path(tmp) / "material_benchmark_report.html",
            )
        self.assertIn("Composite vs Aerospace Metals Benchmark", html_text)
        self.assertIn("Reference source", html_text)
        self.assertIn("Titanium Ti-6Al-4V", html_text)

    def test_pdf_report_is_generated(self) -> None:
        """Benchmark PDF export returns PDF bytes."""
        pdf_bytes = generate_material_benchmark_pdf_report(1827.77, 1.6)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_bytes), 1000)

    def test_default_html_report_path_is_project_reports_dir(self) -> None:
        """Configured HTML report artifact path is stable."""
        self.assertEqual(DEFAULT_HTML_REPORT_PATH.name, "material_benchmark_report.html")


if __name__ == "__main__":
    unittest.main()
