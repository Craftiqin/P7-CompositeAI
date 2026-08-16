"""Tests for final Report Generator PDF output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.report_generator import (
    PDF_FILE_NAME,
    ProjectDetails,
    build_report_download_payload,
    collect_report_data,
    generate_final_report_pdf,
    save_final_report,
)


class FinalReportGeneratorTest(unittest.TestCase):
    """Validate report generation without retraining or fabricating data."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate once for byte-level assertions."""
        cls.pdf_bytes = generate_final_report_pdf(
            ProjectDetails(student_1="Student A"),
        )

    def test_pdf_generation(self) -> None:
        """Report generator returns PDF bytes."""
        self.assertIsInstance(self.pdf_bytes, bytes)

    def test_pdf_is_non_empty(self) -> None:
        """Generated report has substantial content."""
        self.assertGreater(len(self.pdf_bytes), 20_000)

    def test_pdf_has_valid_header(self) -> None:
        """Generated report starts with PDF signature."""
        self.assertTrue(self.pdf_bytes.startswith(b"%PDF-"))

    def test_report_includes_project_title(self) -> None:
        """PDF includes CompositeAI title text."""
        self.assertIn(b"CompositeAI", self.pdf_bytes)

    def test_report_includes_ann_metrics(self) -> None:
        """PDF includes actual ANN metric values from artifacts."""
        self.assertIn(b"ANN/MLP", self.pdf_bytes)
        self.assertIn(b"0.9952", self.pdf_bytes)
        self.assertIn(b"32.5122", self.pdf_bytes)
        self.assertIn(b"43.3776", self.pdf_bytes)

    def test_report_includes_clt_validation(self) -> None:
        """PDF includes CLT validation result."""
        self.assertIn(b"CLT Validation", self.pdf_bytes)
        self.assertIn(b"10319.4275", self.pdf_bytes)
        self.assertIn(b"0.7252", self.pdf_bytes)

    def test_report_includes_optimization_result(self) -> None:
        """PDF includes bounded random-search optimization result."""
        self.assertIn(b"21506.2352", self.pdf_bytes)
        self.assertIn(b"108.4053", self.pdf_bytes)
        self.assertIn(b"Optimization Results Summary", self.pdf_bytes)
        self.assertIn(b"2.08", self.pdf_bytes)

    def test_report_includes_engineering_visualization_interpretations(self) -> None:
        """PDF includes automatic graph interpretations."""
        self.assertIn(b"Engineering Visualizations", self.pdf_bytes)
        self.assertIn(b"Composite Strength Response Surface", self.pdf_bytes)
        self.assertIn(b"Executive Summary", self.pdf_bytes)

    def test_report_includes_ai_vs_clt_limitation(self) -> None:
        """PDF states AI-vs-CLT is not directly comparable."""
        self.assertIn(b"NOT DIRECTLY COMPARABLE", self.pdf_bytes)
        self.assertIn(b"material equivalence", self.pdf_bytes)

    def test_report_handles_missing_optional_artifacts(self) -> None:
        """Missing artifacts produce unavailable labels, not fake values."""
        with tempfile.TemporaryDirectory() as tmp:
            pdf_bytes = generate_final_report_pdf(project_root=Path(tmp))
        self.assertIn(b"Data unavailable in current project artifacts.", pdf_bytes)
        self.assertNotIn(b"0.9952", pdf_bytes)

    def test_collect_report_data_marks_missing_artifacts(self) -> None:
        """Artifact loader records unavailable files explicitly."""
        with tempfile.TemporaryDirectory() as tmp:
            data = collect_report_data(Path(tmp))
        self.assertFalse(data["availability"]["model_metadata"])
        self.assertEqual(
            data["dataset_stats"]["rows"],
            "Data unavailable in current project artifacts.",
        )

    def test_streamlit_report_page_is_registered(self) -> None:
        """Streamlit app exposes Report Generator page."""
        import app

        self.assertIn("Report Generator", app.MENU_ITEMS)
        self.assertTrue(callable(app.render_report_generator))

    def test_download_button_receives_pdf_bytes(self) -> None:
        """Download payload returns expected filename and PDF bytes."""
        file_name, pdf_bytes = build_report_download_payload()
        self.assertEqual(file_name, PDF_FILE_NAME)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_save_report_writes_pdf_file(self) -> None:
        """Generated PDF can be saved for local demo use."""
        with tempfile.TemporaryDirectory() as tmp:
            output = save_final_report(self.pdf_bytes, Path(tmp))
            self.assertEqual(output.name, PDF_FILE_NAME)
            self.assertEqual(output.read_bytes()[:5], b"%PDF-")


if __name__ == "__main__":
    unittest.main()
