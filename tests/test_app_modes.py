"""Tests for CompositeAI simple/engineering mode behavior."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

import app


class AppModeIntegrationTest(unittest.TestCase):
    """Validate mode switching, persistence, and page-level visibility."""

    @staticmethod
    def _app_test() -> AppTest:
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))

    def test_default_mode_is_simple(self) -> None:
        """App defaults to simple mode for first-time users."""
        app_test = self._app_test()
        app_test.run(timeout=120)
        self.assertEqual(app_test.session_state["app_mode"], app.APP_MODE_SIMPLE)

    def test_mode_switch_persists_across_pages(self) -> None:
        """Sidebar mode switch survives page navigation."""
        app_test = self._app_test()
        app_test.run(timeout=120)

        app_test.sidebar.radio[0].set_value(app.APP_MODE_ENGINEERING)
        app_test.run(timeout=120)
        next(button for button in app_test.sidebar.button if button.label == "● About Project").click()
        app_test.run(timeout=120)

        self.assertEqual(app_test.session_state["app_mode"], app.APP_MODE_ENGINEERING)
        self.assertEqual(app_test.session_state["selected_page"], "About Project")

    def test_simple_mode_hides_engineering_sections(self) -> None:
        """Simple mode hides residual/cv/seed sections on model page."""
        app_test = self._app_test()
        app_test.session_state["selected_page"] = "Model Performance"
        app_test.session_state["app_mode"] = app.APP_MODE_SIMPLE
        app_test.run(timeout=120)

        subheaders = [item.value for item in app_test.subheader]
        self.assertNotIn("Cross-Validation", subheaders)
        self.assertNotIn("Random-Seed Robustness", subheaders)
        self.assertNotIn("Residual Analysis and Validation Plots", subheaders)
        self.assertIn("What does this mean?", subheaders)

    def test_engineering_mode_shows_engineering_sections(self) -> None:
        """Engineering mode exposes detailed validation sections."""
        app_test = self._app_test()
        app_test.session_state["selected_page"] = "Model Performance"
        app_test.session_state["app_mode"] = app.APP_MODE_ENGINEERING
        app_test.run(timeout=120)

        subheaders = [item.value for item in app_test.subheader]
        self.assertIn("Cross-Validation", subheaders)
        self.assertIn("Random-Seed Robustness", subheaders)
        self.assertIn("Residual Analysis and Validation Plots", subheaders)


if __name__ == "__main__":
    unittest.main()
