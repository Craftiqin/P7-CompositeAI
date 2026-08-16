"""Tests for Optimization Impact summary card and helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.optimization_impact import (
    calculate_improvement_pct,
    calculate_improvement_ratio,
    load_optimization_impact,
)
from src.report_generator import generate_final_report_pdf


class OptimizationImpactTest(unittest.TestCase):
    """Validate optimization impact calculations and rendering."""

    def test_improvement_calculation(self) -> None:
        """Expected stored improvement remains stable."""
        improvement = calculate_improvement_pct(10319.4275, 21506.2352)
        ratio = calculate_improvement_ratio(10319.4275, 21506.2352)
        self.assertAlmostEqual(improvement, 108.4053133009, places=6)
        self.assertAlmostEqual(ratio, 2.0840531330, places=6)

    def test_missing_baseline_handling(self) -> None:
        """Missing baseline yields unavailable improvement."""
        self.assertIsNone(calculate_improvement_pct(None, 21506.2352))
        self.assertIsNone(calculate_improvement_ratio(None, 21506.2352))

    def test_missing_optimized_handling(self) -> None:
        """Missing optimized yields unavailable improvement."""
        self.assertIsNone(calculate_improvement_pct(10319.4275, None))
        self.assertIsNone(calculate_improvement_ratio(10319.4275, None))

    def test_report_rendering(self) -> None:
        """PDF includes optimization impact summary values."""
        pdf_bytes = generate_final_report_pdf(selected_sections=["Optimization Results"])
        self.assertIn(b"Optimization Results Summary", pdf_bytes)
        self.assertIn(b"10319.4275", pdf_bytes)
        self.assertIn(b"21506.2352", pdf_bytes)
        self.assertIn(b"108.4053", pdf_bytes)
        self.assertIn(b"2.08", pdf_bytes)

    def test_dashboard_rendering(self) -> None:
        """Dashboard shows optimization impact tile and section."""
        app_test = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app_test.session_state["selected_page"] = "Dashboard"
        app_test.run(timeout=120)
        markdown_values = [item.value for item in app_test.markdown]
        self.assertTrue(any("Best Optimization Gain" in value for value in markdown_values))
        subheaders = [item.value for item in app_test.subheader]
        self.assertIn("🚀 Optimization Impact", subheaders)

    def test_optimizer_page_rendering(self) -> None:
        """Optimizer page shows impact card from stored validation data."""
        app_test = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app_test.session_state["selected_page"] = "Stacking Optimizer"
        app_test.run(timeout=120)
        subheaders = [item.value for item in app_test.subheader]
        self.assertIn("🚀 Optimization Impact", subheaders)

    def test_load_from_stored_validation(self) -> None:
        """Stored validation report remains fallback source."""
        impact = load_optimization_impact()
        self.assertAlmostEqual(impact["baseline_lambda_cs"], 10319.427536738505)
        self.assertAlmostEqual(impact["optimized_lambda_cs"], 21506.235235969612)


if __name__ == "__main__":
    unittest.main()
