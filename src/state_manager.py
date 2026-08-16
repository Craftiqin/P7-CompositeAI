"""Streamlit session-state initialization for CompositeAI."""

from __future__ import annotations

from typing import Any

from src.ui_text import DEFAULT_APP_MODE

DEFAULT_SESSION_STATE: dict[str, Any] = {
    "selected_page": "Dashboard",
    "app_mode": DEFAULT_APP_MODE,
    "processed_dataset": None,
    "validation_result": None,
    "dataset_profile": {},
    "column_mappings": {},
    "version_path": None,
    "eda_summary": {},
    "engineered_dataset": None,
    "clean_dataset": None,
    "preprocessing_pipeline": None,
    "preprocessing_metadata": {},
    "pipeline_paths": {},
    "feature_ranking": None,
    "prediction_result": None,
    "optimization_result": None,
    "benchmark_result": None,
    "report_data": {},
    "prediction_context": {},
    "benchmark_context": {},
}


def initialize_session_state(session_state: Any) -> None:
    """Populate all required keys once per app session."""
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in session_state:
            session_state[key] = value.copy() if isinstance(value, dict) else value
