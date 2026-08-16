"""Application stability tests for Streamlit runtime pages and shared state."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import app
from src.material_comparison import (
    build_material_benchmark_report,
    export_comparison_csv,
    export_comparison_excel,
    generate_material_benchmark_pdf_report,
)
from src.report_generator import (
    build_report_csv_payload,
    build_report_download_payload,
    build_report_html_payload,
)
from src.state_manager import DEFAULT_SESSION_STATE, initialize_session_state


REQUESTED_PAGE_MAP = {
    "Dashboard": "Dashboard",
    "Dataset Analysis": "Dataset Explorer",
    "Data Preparation": "Preprocessing",
    "Model Training": "Model Performance",
    "Validation Results": "Model Performance",
    "Strength Prediction": "Strength Prediction",
    "CLT Analysis": "CLT Analysis",
    "Stacking Optimizer": "Stacking Optimizer",
    "Composite vs Aerospace Metals": "Composite vs Aerospace Metals",
    "Reports": "Report Generator",
    "Engineering Assistant": "Gemini Assistant",
    "About": "About Project",
}


class ApplicationStabilityTest(unittest.TestCase):
    """Verify no first-open page crashes or missing core state."""

    def test_initialize_session_state_populates_required_keys(self) -> None:
        """State initializer creates all required app keys."""
        state: dict[str, object] = {}
        initialize_session_state(state)
        for key in DEFAULT_SESSION_STATE:
            self.assertIn(key, state)
        self.assertIn("prediction_result", state)
        self.assertIn("optimization_result", state)
        self.assertIn("benchmark_result", state)
        self.assertIn("app_mode", state)
        self.assertIn("report_data", state)

    def test_runtime_path_registry_is_non_crashing(self) -> None:
        """Runtime path registry exposes required directories for diagnostics."""
        self.assertIn("saved_models", app.REQUIRED_RUNTIME_PATHS)
        self.assertIn("reports", app.REQUIRED_RUNTIME_PATHS)
        self.assertIn("data", app.REQUIRED_RUNTIME_PATHS)
        self.assertIn("reference_materials", app.REQUIRED_RUNTIME_PATHS)

    def test_dashboard_kpis_use_ml_ready_dataset(self) -> None:
        """Dashboard KPIs come from ML-ready dataset, not empty session state."""
        kpis = app.load_dashboard_kpi_data()
        self.assertTrue(kpis["dataset_path"].endswith("data/training/ml_ready_features.csv"))
        self.assertEqual(kpis["total_samples"], 10_000)
        self.assertEqual(kpis["total_features"], 7)
        self.assertEqual(kpis["rows"], 10_000)
        self.assertEqual(kpis["columns"], 8)
        self.assertTrue(kpis["profile_exists"])

    def test_dashboard_engineering_mode_displays_dataset_kpis(self) -> None:
        """Dashboard renders non-zero ML-ready KPI values in Engineering Mode."""
        app_file = Path(__file__).resolve().parents[1] / "app.py"
        app_test = AppTest.from_file(str(app_file))
        app_test.session_state["selected_page"] = "Dashboard"
        app_test.session_state["app_mode"] = app.APP_MODE_ENGINEERING
        app_test.run(timeout=120)

        markdown_values = [item.value for item in app_test.markdown]
        captions = [item.value for item in app_test.caption]
        self.assertEqual(app_test.exception, [])
        self.assertTrue(any("Total Samples" in value and "10000" in value for value in markdown_values))
        self.assertTrue(any("Features" in value and "7" in value for value in markdown_values))
        self.assertIn("Rows: 10000", captions)
        self.assertIn("Columns: 8", captions)

    def test_dataset_explorer_loads_local_dataset_without_processed_session(self) -> None:
        """Dataset pages load bundled ML-ready dataset after fresh clone."""
        app_file = Path(__file__).resolve().parents[1] / "app.py"
        app_test = AppTest.from_file(str(app_file))
        app_test.session_state["selected_page"] = "Dataset Explorer"
        app_test.run(timeout=120)

        self.assertEqual(app_test.exception, [])
        self.assertIn("processed_dataset", app_test.session_state)
        self.assertEqual(len(app_test.session_state["processed_dataset"]), 10_000)
        self.assertFalse(
            any("No processed dataset available" in item.value for item in app_test.info),
        )

    def test_dashboard_kpi_missing_dataset_raises_actual_exception(self) -> None:
        """Dashboard KPI loader raises clear file error instead of returning zero."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spec_path = tmp_path / "feature_specification.json"
            profile_path = tmp_path / "feature_analysis_report.json"
            spec_path.write_text(
                json.dumps({"baseline_features": ["fiber_type"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "ML-ready dataset not found"):
                app.load_dashboard_kpi_data(
                    dataset_path=tmp_path / "missing.csv",
                    feature_spec_path=spec_path,
                    profile_path=profile_path,
                )

    def test_dashboard_kpi_reads_shape_and_input_features(self) -> None:
        """Dashboard KPI loader reads CSV shape and counts model input features."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "ml_ready_features.csv"
            spec_path = tmp_path / "feature_specification.json"
            profile_path = tmp_path / "feature_analysis_report.json"
            pd.DataFrame(
                [
                    {"fiber_type": "Carbon", "density_g_cm3": 1.5, "tensile_strength_mpa": 1000.0},
                    {"fiber_type": "Glass", "density_g_cm3": 1.7, "tensile_strength_mpa": 900.0},
                ]
            ).to_csv(dataset_path, index=False)
            spec_path.write_text(
                json.dumps({"baseline_features": ["fiber_type", "density_g_cm3"]}),
                encoding="utf-8",
            )
            profile_path.write_text("{}", encoding="utf-8")

            kpis = app.load_dashboard_kpi_data(dataset_path, spec_path, profile_path)

        self.assertEqual(kpis["total_samples"], 2)
        self.assertEqual(kpis["total_features"], 2)
        self.assertEqual(kpis["rows"], 2)
        self.assertEqual(kpis["columns"], 3)
        self.assertTrue(kpis["profile_exists"])

    def test_requested_pages_render_first_open(self) -> None:
        """Every requested evaluation page renders without a Streamlit exception."""
        app_file = Path(__file__).resolve().parents[1] / "app.py"
        for requested_name, actual_page in REQUESTED_PAGE_MAP.items():
            with self.subTest(page=requested_name):
                app_test = AppTest.from_file(str(app_file))
                app_test.session_state["selected_page"] = actual_page
                app_test.run(timeout=120)
                self.assertEqual(app_test.exception, [])

    def test_strength_prediction_button_has_initialized_metrics(self) -> None:
        """Prediction click path does not reference metrics before assignment."""
        app_file = Path(__file__).resolve().parents[1] / "app.py"
        app_test = AppTest.from_file(str(app_file))
        app_test.session_state["selected_page"] = "Strength Prediction"
        app_test.run(timeout=120)
        next(button for button in app_test.button if button.label == "Predict Strength").click()
        app_test.run(timeout=120)
        self.assertEqual(app_test.exception, [])
        self.assertIn("prediction_context", app_test.session_state)
        self.assertIn("validation_metrics", app_test.session_state["prediction_context"])

    def test_report_exports_generate_payloads(self) -> None:
        """PDF, HTML, and CSV report export payloads are available."""
        pdf_name, pdf_bytes = build_report_download_payload()
        html_name, html_text = build_report_html_payload()
        csv_name, csv_bytes = build_report_csv_payload()
        self.assertTrue(pdf_name.endswith(".pdf"))
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertTrue(html_name.endswith(".html"))
        self.assertIn("CompositeAI", html_text)
        self.assertTrue(csv_name.endswith(".csv"))
        self.assertIn(b"ai_model", csv_bytes)

    def test_benchmark_example_exports(self) -> None:
        """Benchmark example produces ranking and export files."""
        report = build_material_benchmark_report(1644.0, 1.55)
        rows = report["comparison_rows"]
        self.assertEqual(rows[0]["material"], "Inconel 718")
        self.assertGreater(rows[0]["strength_ratio"], 1.0)
        self.assertIn(b"Aluminum 7075-T6", export_comparison_csv(1644.0, 1.55))
        self.assertGreater(len(export_comparison_excel(1644.0, 1.55)), 1000)
        self.assertTrue(generate_material_benchmark_pdf_report(1644.0, 1.55).startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
