"""Tests for benchmark comparison exports and specification."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.material_comparison import (
    DEFAULT_HTML_REPORT_PATH,
    DEFAULT_SPEC_PATH,
    build_comparison_specification,
    build_material_benchmark_report,
    compare_against_materials,
    export_comparison_csv,
    export_comparison_excel,
    generate_material_benchmark_html_report,
    generate_material_benchmark_pdf_report,
    summarize_material_comparison,
)


class MaterialComparisonTest(unittest.TestCase):
    """Validate benchmark report helpers."""

    def test_compare_against_materials(self) -> None:
        """Comparison includes requested aerospace reference families."""
        rows = compare_against_materials(1644.0, 1.55)
        materials = [row["material"] for row in rows]
        self.assertEqual(
            materials,
            [
                "Inconel 718",
                "Titanium Ti-6Al-4V",
                "Aluminum 7075-T6",
                "Stainless Steel 304",
                "Aluminum 2024-T3",
            ],
        )

    def test_summary_generates_rankings(self) -> None:
        """Summary exposes best-specific-strength and closest-match sections."""
        summary = summarize_material_comparison(1644.0, 1.55)
        self.assertEqual(summary["strongest_material"]["material"], "Inconel 718")
        self.assertEqual(summary["best_specific_strength"]["material"], "Titanium Ti-6Al-4V")
        self.assertIsNotNone(summary["closest_strength_match"])

    def test_report_contains_disclaimers(self) -> None:
        """Report context labels the data source correctly."""
        report = build_material_benchmark_report(1644.0, 1.55)
        self.assertIn("source_database", report)
        self.assertEqual(len(report["disclaimers"]), 2)
        self.assertIn("engineering benchmark values", report["disclaimers"][0].casefold())

    def test_export_generation(self) -> None:
        """CSV, Excel, HTML, and PDF exports are generated."""
        csv_bytes = export_comparison_csv(1644.0, 1.55)
        excel_bytes = export_comparison_excel(1644.0, 1.55)
        pdf_bytes = generate_material_benchmark_pdf_report(1644.0, 1.55)
        self.assertIn(b"Aluminum 7075-T6", csv_bytes)
        self.assertTrue(excel_bytes.startswith(b"PK"))
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        with tempfile.TemporaryDirectory() as tmp:
            html_text = generate_material_benchmark_html_report(
                1644.0,
                1.55,
                output_path=Path(tmp) / "material_benchmark_report.html",
            )
        self.assertIn("Composite vs Aerospace Metals Benchmark", html_text)

    def test_specification_file_matches_builder(self) -> None:
        """Stored specification matches current builder output."""
        self.assertEqual(DEFAULT_HTML_REPORT_PATH.name, "material_benchmark_report.html")
        stored = json.loads(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))
        self.assertEqual(stored, build_comparison_specification())

    def test_empty_input_handling(self) -> None:
        """Invalid strength is rejected before export."""
        with self.assertRaisesRegex(ValueError, "predicted_strength_mpa|composite_strength_mpa"):
            export_comparison_csv(-1.0, 1.55)


if __name__ == "__main__":
    unittest.main()
