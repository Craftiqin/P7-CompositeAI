"""Streamlit UI for CompositeAI."""

from __future__ import annotations

import json
import logging
import html
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.config import (
    APP_TITLE,
    DATA_DIR,
    DATASET_SOURCE_DIRS,
    MERGED_DATA_DIR,
    PROJECT_ROOT,
    REPORT_DIR,
    SAVED_MODEL_DIR,
    UPLOADED_DATA_DIR,
)
from src.eda import (
    categorical_balance,
    dataset_summary,
    duplicate_report,
    feature_types,
    numerical_statistics,
    pairwise_correlation_table,
    target_distribution,
)
from src.dataset_loader import (
    SUPPORTED_EXTENSIONS,
    load_multiple_datasets,
    save_uploaded_file,
)
from src.dataset_merger import merge_datasets
from src.dataset_profiler import (
    dataset_statistics,
    missing_values_frame,
    profile_dataset,
)
from src.dataset_validator import ValidationIssue, validate_dataset
from src.dataset_versioning import list_processed_versions, save_dataset_version
from src.engineering_visualizations import (
    build_material_benchmark_3d,
    build_optimization_landscape,
    build_ply_failure_map,
    build_strength_response_surface,
)
from src.engineering_interpretation import (
    ENGINEERING_EXPLANATION,
    INTERPRETATION_SECTIONS,
    SIMPLE_EXPLANATION,
    build_engineering_interpretation,
)
from src.feature_engineering import (
    available_engineered_features,
    engineer_laminate_features,
)
from src.gemini_service import GeminiResult, GeminiService
from src.model_validation import (
    VALIDATION_PLOT_FILENAMES,
    get_validation_plot_path,
    run_model_validation,
)
from src.outlier_detection import detect_outliers, remove_outliers
from src.predict import (
    PredictionInputError,
    inspect_model_artifact,
    predict_strength,
)
from src.ai_clt_comparison import (
    LOAD_CAPACITY_MODE,
    STRESS_MODE,
    ComparisonCase,
    compare_ai_and_clt,
)
from src.material_comparison import (
    DEFAULT_HTML_REPORT_PATH,
    build_material_benchmark_report,
    export_comparison_csv,
    export_comparison_excel,
    generate_material_benchmark_html_report,
    generate_material_benchmark_pdf_report,
)
from src.navigation import get_available_pages, render_page, validate_navigation
from src.clt import LaminateLoadCase
from src.optimizer import (
    OptimizationConfig,
    load_tu_delft_demo_case,
    optimize_stacking_sequence,
    validate_reference_case,
)
from src.optimization_impact import load_optimization_impact
from src.preprocessing_pipeline import (
    PreprocessingConfig,
    run_preprocessing_pipeline,
    save_pipeline_artifacts,
)
from src.report_generator import (
    ARTIFACT_PATHS,
    PDF_FILE_NAME,
    ProjectDetails,
    SECTION_LABELS,
    build_report_download_payload,
    build_report_html_payload,
    build_report_csv_payload,
    save_final_report,
    save_report_text_artifact,
)
from src.state_manager import initialize_session_state
from src.ui_text import (
    APP_MODE_ENGINEERING,
    APP_MODE_SIMPLE,
    executive_summary,
    get_app_mode,
    metadata_version,
    prediction_distribution,
    strength_status,
    term_details,
    term_help,
    ui_term,
)
from src.visualization import (
    boxplot,
    category_counts,
    correlation_heatmap,
    distribution_plot,
    missing_value_matrix,
    outlier_scatter,
    scatter_matrix,
    scatterplot,
    target_correlation,
    violin_plot,
)
from src.stacking_sequence import (
    DEFAULT_ALLOWED_ANGLES,
    SequenceConfig,
    format_laminate_sequence,
    validate_sequence,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

MENU_SECTIONS = get_available_pages()

MENU_ITEMS = [
    item
    for section_items in MENU_SECTIONS.values()
    for item in section_items
]

REQUIRED_RUNTIME_PATHS = {
    "saved_models": SAVED_MODEL_DIR,
    "reports": REPORT_DIR,
    "data": DATA_DIR,
    "reference_materials": DATA_DIR / "reference_materials",
}
DASHBOARD_KPI_DATASET_PATH = PROJECT_ROOT / "data" / "training" / "ml_ready_features.csv"
DASHBOARD_KPI_FEATURE_SPEC_PATH = PROJECT_ROOT / "data" / "training" / "feature_specification.json"
DASHBOARD_KPI_PROFILE_PATH = PROJECT_ROOT / "data" / "training" / "feature_analysis_report.json"
LOCAL_DATASET_FALLBACK_PATH = PROJECT_ROOT / "data" / "training" / "ml_ready_features.csv"

WORKFLOW_STEPS = [
    (
        "EDA",
        "Inspect distributions, correlations, missingness, skewness, and targets.",
    ),
    (
        "Feature Engineering",
        "Create laminate-specific layup, angle, material, and interaction features.",
    ),
    (
        "Preprocessing",
        "Impute, encode, scale, remove outliers, and select model-ready features.",
    ),
    (
        "Export",
        "Save clean datasets, metadata, and preprocessing pipelines for Step 4.",
    ),
]

PAGE_ALIASES = {
    "About": "About Project",
    "Dataset Import": "Dataset Explorer",
    "Exploratory Data Analysis": "EDA",
    "Data Preprocessing": "Preprocessing",
    "Model Training": "Model Performance",
    "Physics Verification": "AI vs CLT Comparison",
    "CLT": "CLT Analysis",
    "Engineering Benchmark": "Composite vs Aerospace Metals",
    "Engineering Assistant": "Gemini Assistant",
    "Gemini Engineering Assistant": "Gemini Assistant",
}


def normalize_page_name(page_name: str) -> str:
    """Map legacy page names to current IA labels."""
    return PAGE_ALIASES.get(page_name, page_name)


def configure_page() -> None:
    """Set page metadata and global styles."""
    st.set_page_config(
        page_title="CompositeAI",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            :root {
                --carbon: #101820;
                --graphite: #1b2733;
                --skyline: #3da5d9;
                --plasma: #f2b705;
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(61, 165, 217, 0.14), transparent 34rem),
                    linear-gradient(180deg, #f7fbff 0%, #edf2f7 100%);
                color: var(--carbon);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, var(--carbon), var(--graphite));
            }

            [data-testid="stSidebar"] * {
                color: #edf6ff;
            }

            [data-testid="stSidebar"] .stButton > button {
                background: transparent;
                border: 1px solid rgba(237, 246, 255, 0.16);
                border-radius: 8px;
                color: #edf6ff;
                justify-content: flex-start;
                min-height: 2.25rem;
                padding: 0.35rem 0.65rem;
                width: 100%;
            }

            [data-testid="stSidebar"] .stButton > button:hover {
                background: rgba(61, 165, 217, 0.18);
                border-color: rgba(61, 165, 217, 0.42);
                color: #ffffff;
            }

            .sidebar-title {
                color: #ffffff;
                font-size: 1.25rem;
                font-weight: 850;
                letter-spacing: 0.08em;
                margin-bottom: 0.1rem;
            }

            .sidebar-caption {
                color: #b9d7ea;
                font-size: 0.82rem;
                margin-bottom: 1rem;
            }

            .sidebar-section {
                color: #f2b705;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                margin: 1rem 0 0.35rem;
                white-space: nowrap;
            }

            .sidebar-active {
                background: rgba(61, 165, 217, 0.24);
                border: 1px solid rgba(61, 165, 217, 0.62);
                border-radius: 8px;
                color: #ffffff;
                font-weight: 750;
                margin-bottom: 0.35rem;
                padding: 0.45rem 0.65rem;
            }

            .main-title {
                color: var(--carbon);
                font-size: 2.35rem;
                font-weight: 800;
                line-height: 1.12;
                margin-bottom: 0.5rem;
            }

            .section-copy {
                color: #465666;
                font-size: 1rem;
                max-width: 980px;
            }

            .kpi-card,
            .workflow-card,
            .assistant-panel {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(16, 24, 32, 0.08);
                border-radius: 8px;
                box-shadow: 0 16px 34px rgba(16, 24, 32, 0.08);
                padding: 1rem;
            }

            .kpi-label {
                color: #5b6875;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }

            .kpi-value {
                color: var(--carbon);
                font-size: 1.8rem;
                font-weight: 800;
                margin-top: 0.3rem;
            }

            .workflow-title {
                color: var(--carbon);
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }

            .workflow-copy {
                color: #526170;
                font-size: 0.92rem;
            }

            .prediction-value {
                color: var(--carbon);
                font-size: 2.6rem;
                font-weight: 850;
                letter-spacing: -0.03em;
                text-align: center;
            }

            .footer {
                border-top: 1px solid rgba(16, 24, 32, 0.1);
                color: #627181;
                font-size: 0.85rem;
                margin-top: 2.5rem;
                padding-top: 1rem;
            }

            @media (max-width: 768px) {
                .main-title {
                    font-size: 1.75rem;
                }

                .kpi-card,
                .workflow-card,
                .assistant-panel {
                    padding: 0.85rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def is_simple_mode() -> bool:
    """Return True when UI is in simple mode."""
    return get_app_mode(st.session_state) == APP_MODE_SIMPLE


def page_title(title: str) -> str:
    """Return mode-aware page title."""
    replacements = {
        "Physics Verification": "AI vs Physics Verification" if is_simple_mode() else "Physics Verification",
        "Engineering Assistant": "AI Engineering Assistant" if is_simple_mode() else "Engineering Assistant",
        "Composite vs Aerospace Metals": (
            "Engineering Benchmark" if is_simple_mode() else "Composite vs Aerospace Metals"
        ),
    }
    return replacements.get(title, title)


def render_mode_toggle() -> None:
    """Render persistent app-mode switcher."""
    st.sidebar.markdown("**Application Mode**")
    st.sidebar.radio(
        "Application Mode",
        options=[APP_MODE_SIMPLE, APP_MODE_ENGINEERING],
        format_func=lambda value: "Simple Mode" if value == APP_MODE_SIMPLE else "Engineering Mode",
        key="app_mode",
        label_visibility="collapsed",
    )


def render_term_expander(terms: list[str]) -> None:
    """Render glossary help for current page terms."""
    unique_terms = [term for term in dict.fromkeys(terms) if term_details(term)]
    if not unique_terms:
        return
    with st.expander("Terms and explanations", expanded=False):
        for term in unique_terms:
            details = term_details(term)
            if not details:
                continue
            st.markdown(f"**{ui_term(term, get_app_mode(st.session_state))}**")
            st.caption(details["tooltip"])
            st.write(details["explanation"])
            st.write(f"Example: {details['example']}")


def render_workflow_visual(mode: str) -> None:
    """Render project workflow as visual blocks."""
    steps = [
        "Dataset Collection",
        "Data Validation",
        "Data Analysis",
        "Feature Engineering",
        "Model Training",
        "Model Validation",
        "Strength Prediction",
        "Physics Verification",
        "Stacking Optimization",
        "Engineering Report",
    ]
    icons = ["🗂️", "✅", "📊", "🧩", "🤖", "📈", "💪", "📐", "🛠️", "📝"]
    cols = st.columns(5)
    for index, step in enumerate(steps):
        with cols[index % 5]:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-title">{icons[index]} {step}</div>
                    <div class="workflow-copy">
                        {"Simple project flow for non-engineers." if mode == APP_MODE_SIMPLE else "Validated project pipeline stage."}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_prediction_summary_sections(prediction_value: float) -> None:
    """Render simple/engineering prediction interpretation sections."""
    mode = get_app_mode(st.session_state)
    status = strength_status(prediction_value)
    distribution = prediction_distribution()
    badge_map = {
        "Excellent": "🟢 Excellent",
        "Good": "🟡 Good",
        "Moderate": "🟠 Moderate",
        "Weak": "🔴 Weak",
    }
    st.subheader("Prediction Summary")
    left, right = st.columns([2, 1])
    with left:
        st.metric(ui_term("Tensile Strength", mode), f"{prediction_value:,.2f} MPa")
    with right:
        st.metric("Status", badge_map[status["badge"]])

    if mode == APP_MODE_SIMPLE:
        st.subheader("What does this mean?")
        st.info(
            f"This laminate can withstand approximately {prediction_value:,.0f} MPa of pulling stress before failure. Higher values generally indicate stronger composite structures."
        )
        st.subheader("Recommended Applications")
        if prediction_value >= distribution["q3"]:
            st.write("Suitable for higher-performance aerospace composite studies where lightweight strength matters most.")
        elif prediction_value >= distribution["median"]:
            st.write("Suitable for general structural composite studies and balanced aerospace design scenarios.")
        else:
            st.write("Better suited for lower-demand studies, secondary structures, or early design exploration.")

    st.subheader("Engineering Interpretation")
    if prediction_value >= distribution["q3"]:
        level = "high"
    elif prediction_value >= distribution["median"]:
        level = "above average"
    elif prediction_value >= distribution["q1"]:
        level = "average to low"
    else:
        level = "low"
    st.write(
        f"This prediction is {level} relative to the locked training dataset distribution. {status['interpretation']}"
    )


def benchmark_category_note(report: dict[str, Any]) -> None:
    """Render reference-database disclaimer block."""
    for disclaimer in report.get("disclaimers", []):
        st.warning(disclaimer)


def render_workflow_page() -> None:
    """Render workflow overview page."""
    mode = get_app_mode(st.session_state)
    st.markdown(f'<div class="main-title">{page_title("Workflow")}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class="section-copy">
            {"Understand full project pipeline in under two minutes." if mode == APP_MODE_SIMPLE else "End-to-end CompositeAI pipeline from data ingestion to reporting."}
        </p>
        """,
        unsafe_allow_html=True,
    )
    render_workflow_visual(mode)
    st.markdown("---")
    rows = [
        ("Dataset Collection", "Collect composite material/process data."),
        ("Data Validation", "Check ranges, missing values, duplicates, leakage risks."),
        ("Data Analysis", "Inspect distributions and relationships."),
        ("Feature Engineering", "Prepare model-ready inputs."),
        ("Model Development", "Compare regression models; lock ANN/MLP winner."),
        ("Model Validation", "Evaluate R², MAE, RMSE, residual behavior, seed robustness."),
        ("Strength Prediction", "Predict tensile strength from user inputs."),
        ("AI vs CLT Comparison", "Use CLT backend for compatibility and mechanical checks."),
        ("Stacking Optimization", "Search valid laminate sequences under constraints."),
        ("Engineering Report", "Generate PDF/HTML/CSV deliverables."),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Stage", "Purpose"]), use_container_width=True, hide_index=True)
    render_footer()


def render_model_training_page() -> None:
    """Render read-only model training summary page."""
    st.markdown(f'<div class="main-title">{page_title("Model Training")}</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="section-copy">
            Read-only summary of how CompositeAI trained and selected its final ANN model.
            No retraining controls are exposed here.
        </p>
        """,
        unsafe_allow_html=True,
    )

    metadata_path = PROJECT_ROOT / "saved_models" / "model_metadata.json"
    comparison_path = PROJECT_ROOT / "data" / "training" / "model_comparison.csv"
    validation_path = PROJECT_ROOT / "data" / "training" / "model_validation_report.json"
    if not metadata_path.exists() or not comparison_path.exists():
        st.warning("Training artifacts are unavailable.")
        render_footer()
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(comparison_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}

    st.subheader("Training Pipeline")
    pipeline_rows = [
        ("Dataset", metadata.get("training_dataset", "Unavailable")),
        ("Split", "80/20 train-test"),
        ("Preprocessing", "Impute -> Encode -> Scale inside sklearn pipeline"),
        ("Models Evaluated", ", ".join(comparison["model"].tolist())),
        ("Selected Model", metadata.get("model_name", "Unavailable")),
        ("Random State", metadata.get("random_state", "Unavailable")),
    ]
    st.dataframe(build_display_frame(pipeline_rows, ["Step", "Value"]), use_container_width=True, hide_index=True)

    st.subheader("Compared Models")
    render_model_comparison(comparison)
    st.subheader("Selected ANN Configuration")
    ann_rows = [
        ("Hidden layers", "64, 32"),
        ("Activation", "relu"),
        ("Solver", "adam"),
        ("Max iterations", "600"),
        ("Early stopping", "True"),
        ("Project version", metadata_version()),
    ]
    st.dataframe(build_display_frame(ann_rows, ["Field", "Value"]), use_container_width=True, hide_index=True)
    if validation:
        render_cross_validation(validation)
    render_footer()
def render_sidebar() -> str:
    """Render sidebar navigation and return selected page."""
    initialize_session_state(st.session_state)
    selected_page = normalize_page_name(st.session_state.get("selected_page", "Dashboard"))
    if selected_page not in MENU_ITEMS:
        selected_page = "Dashboard"

    st.sidebar.markdown('<div class="sidebar-title">COMPOSITEAI</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="sidebar-caption">Aerospace laminate intelligence</div>',
        unsafe_allow_html=True,
    )
    render_mode_toggle()
    render_runtime_status()

    for section, items in MENU_SECTIONS.items():
        st.sidebar.markdown(
            f'<div class="sidebar-section">── {section} ──</div>',
            unsafe_allow_html=True,
        )
        for item in items:
            if item == selected_page:
                st.sidebar.markdown(
                    f'<div class="sidebar-active">● {item}</div>',
                    unsafe_allow_html=True,
                )
                continue
            if st.sidebar.button(f"● {item}", key=f"nav_{item}"):
                selected_page = item
                st.session_state["selected_page"] = item
                st.rerun()

    st.session_state["selected_page"] = selected_page
    return selected_page


def render_runtime_status() -> None:
    """Show non-crashing diagnostics for required runtime directories."""
    missing = [
        label
        for label, path in REQUIRED_RUNTIME_PATHS.items()
        if not path.exists()
    ]
    if not missing:
        return
    with st.sidebar.expander("Runtime files"):
        st.warning("Some project folders are missing. Related pages will show recovery guidance instead of crashing.")
        st.caption(", ".join(missing))


def render_dashboard() -> None:
    """Render main dashboard page."""
    mode = get_app_mode(st.session_state)
    summary = executive_summary(mode)
    st.markdown(f'<div class="main-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class="section-copy">
            {summary["headline"]}
        </p>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(4)
    card_items = [
        ("What It Does", summary["what_it_does"]),
        ("Why It Matters", summary["why_it_matters"]),
        ("AI Role", summary["ai_role"]),
        ("Physics Role", summary["physics_role"]),
    ]
    for col, (title, copy) in zip(cards, card_items):
        with col:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Project Workflow")
    render_workflow_visual(mode)

    st.subheader("What Makes This Project Unique")
    st.info(summary["uniqueness"])

    render_optimization_impact_highlight()
    render_optimization_impact_card(mode)

    if mode == APP_MODE_ENGINEERING:
        st.subheader("Project KPIs")
        render_kpis()
        st.subheader("Analytics Preview")
        render_placeholder_charts()
    else:
        st.subheader("What does this mean?")
        st.info(
            "CompositeAI combines artificial intelligence prediction, composite laminate structural analysis, optimization, benchmarking, and reporting in one guided workflow."
        )
    render_footer()


def render_kpis() -> None:
    """Render dashboard KPI cards from locked ML-ready dataset."""
    try:
        kpi_data = load_dashboard_kpi_data()
        total_samples = str(kpi_data["total_samples"])
        total_features = str(kpi_data["total_features"])
        kpi_error = None
    except Exception as exc:
        LOGGER.exception("Dashboard KPI loading failed")
        kpi_data = {
            "dataset_path": str(DASHBOARD_KPI_DATASET_PATH),
            "rows": "Error",
            "columns": "Error",
            "profile_path": str(DASHBOARD_KPI_PROFILE_PATH),
            "profile_exists": DASHBOARD_KPI_PROFILE_PATH.exists(),
        }
        total_samples = "Error"
        total_features = "Error"
        kpi_error = exc

    metrics = [
        ("Total Samples", total_samples),
        ("Features", total_features),
        ("Models Trained", "4"),
        ("Best Optimization Gain", _optimization_gain_label()),
    ]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption(f"Dataset path: {kpi_data['dataset_path']}")
    st.caption(f"Rows: {kpi_data['rows']}")
    st.caption(f"Columns: {kpi_data['columns']}")
    if not kpi_data["profile_exists"]:
        st.warning(f"Dataset profile JSON not found: {kpi_data['profile_path']}")
    if kpi_error is not None:
        st.error(f"Dashboard KPI loading failed: {type(kpi_error).__name__}: {kpi_error}")


def load_dashboard_kpi_data(
    dataset_path: Path = DASHBOARD_KPI_DATASET_PATH,
    feature_spec_path: Path = DASHBOARD_KPI_FEATURE_SPEC_PATH,
    profile_path: Path = DASHBOARD_KPI_PROFILE_PATH,
) -> dict[str, Any]:
    """Load dashboard KPIs from ML-ready dataset and feature specification."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"ML-ready dataset not found: {dataset_path}")
    if not feature_spec_path.exists():
        raise FileNotFoundError(f"Feature specification not found: {feature_spec_path}")

    dataset = pd.read_csv(dataset_path)
    feature_spec = json.loads(feature_spec_path.read_text(encoding="utf-8"))
    input_features = feature_spec.get("baseline_features")
    if not isinstance(input_features, list) or not input_features:
        raise ValueError(f"Feature specification missing non-empty baseline_features: {feature_spec_path}")

    missing_features = [feature for feature in input_features if feature not in dataset.columns]
    if missing_features:
        raise KeyError(f"Feature(s) missing from ML-ready dataset: {missing_features}")

    return {
        "dataset_path": str(dataset_path),
        "rows": int(dataset.shape[0]),
        "columns": int(dataset.shape[1]),
        "total_samples": int(len(dataset)),
        "total_features": len(input_features),
        "input_features": input_features,
        "profile_path": str(profile_path),
        "profile_exists": profile_path.exists(),
    }


def render_workflow_cards() -> None:
    """Render project workflow cards."""
    columns = st.columns(4)
    for column, (title, copy) in zip(columns, WORKFLOW_STEPS):
        with column:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _optimization_gain_label() -> str:
    """Return formatted optimization gain label."""
    impact = load_optimization_impact(st.session_state.get("optimization_result"))
    value = impact.get("improvement_pct")
    return "Unavailable" if value is None else f"+{value:.1f}%"


def render_optimization_impact_highlight() -> None:
    """Render dashboard optimization gain anchor tile."""
    gain = _optimization_gain_label()
    st.markdown(
        f"""
        <a href="#optimization-impact" style="text-decoration:none;">
            <div class="kpi-card">
                <div class="kpi-label">Best Optimization Gain</div>
                <div class="kpi-value">{gain}</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )


def render_optimization_impact_card(mode: str) -> None:
    """Render high-visibility optimization impact card."""
    impact = load_optimization_impact(st.session_state.get("optimization_result"))
    baseline = impact.get("baseline_lambda_cs")
    optimized = impact.get("optimized_lambda_cs")
    improvement = impact.get("improvement_pct")
    ratio = impact.get("improvement_ratio")
    if baseline is None or optimized is None:
        st.warning("Optimization Impact unavailable. Missing baseline or optimized result.")
        return

    st.markdown('<div id="optimization-impact"></div>', unsafe_allow_html=True)
    st.subheader("🚀 Optimization Impact")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Baseline", f"{baseline:,.0f}")
    metric_cols[1].metric("Optimized", f"{optimized:,.0f}")
    metric_cols[2].metric("Improvement", f"+{improvement:.1f}%")

    if mode == APP_MODE_SIMPLE:
        st.info(
            f"The optimized laminate can withstand approximately {ratio:.2f}× higher load before failure compared to the baseline design."
        )
    else:
        detail_rows = [
            ("Baseline λcs", f"{baseline:.4f}"),
            ("Optimized λcs", f"{optimized:.4f}"),
            ("Improvement %", f"{improvement:.4f}%"),
            ("Improvement Ratio", f"{ratio:.2f}×" if ratio is not None else "Unavailable"),
            ("Optimization constraints", json.dumps(impact.get("constraints", {}))),
            ("Number of candidates evaluated", impact.get("candidates_evaluated", "Unavailable")),
            ("Best sequence", str(impact.get("best_sequence", "Unavailable"))),
        ]
        st.dataframe(build_display_frame(detail_rows, ["Field", "Value"]), use_container_width=True, hide_index=True)

    chart_frame = pd.DataFrame(
        [
            {"Case": "Baseline", "Failure Load Factor": baseline},
            {"Case": "Optimized", "Failure Load Factor": optimized},
        ]
    )
    fig = px.bar(
        chart_frame,
        x="Case",
        y="Failure Load Factor",
        color="Case",
        title="Baseline vs Optimized",
        text="Failure Load Factor",
    )
    fig.add_annotation(
        x=1,
        y=optimized,
        text=f"+{improvement:.1f}%",
        showarrow=False,
        yshift=18,
    )
    fig.update_layout(template="plotly_white", height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_placeholder_charts() -> None:
    """Render confirmed Step 4/5 model and dataset charts."""
    comparison_path = Path("data/training/model_comparison.csv")
    training_path = Path("data/training/ml_ready_features.csv")

    left, right = st.columns(2)
    with left:
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path).sort_values("test_rmse")
            fig = px.bar(
                comparison,
                x="model",
                y="test_rmse",
                color="test_r2",
                color_continuous_scale="Blues",
                title="Validated Model Comparison",
                labels={
                    "model": "Model",
                    "test_rmse": "Test RMSE (MPa)",
                    "test_r2": "Test R²",
                },
            )
            fig.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model comparison artifact not found yet.")

    with right:
        if training_path.exists():
            training_data = pd.read_csv(training_path)
            fig = px.box(
                training_data,
                x="fiber_type",
                y="tensile_strength_mpa",
                color="fiber_type",
                title="Training Target Distribution by Fiber Type",
                labels={
                    "fiber_type": "Fiber Type",
                    "tensile_strength_mpa": "Tensile Strength (MPa)",
                },
            )
            fig.update_layout(template="plotly_white", height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Training dataset artifact not found yet.")


def render_dataset_import() -> None:
    """Render dataset ingestion, validation, profiling, and versioning page."""
    st.markdown('<div class="main-title">Dataset Import</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="section-copy">
            Import CSV, XLSX, or JSON files, merge multiple datasets, standardize
            schema, validate data quality, and save processed versions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload datasets",
        type=["csv", "xlsx", "json"],
        accept_multiple_files=True,
    )
    source_name = st.selectbox("Existing source folder", list(DATASET_SOURCE_DIRS))
    existing_files = find_dataset_files(DATASET_SOURCE_DIRS[source_name])
    selected_existing = st.multiselect(
        "Select existing files",
        existing_files,
        format_func=lambda path: str(path.relative_to(DATASET_SOURCE_DIRS[source_name])),
    )

    if st.button("Process Dataset", type="primary"):
        process_dataset_import(uploaded_files, selected_existing)

    current_data = get_current_dataset()
    if current_data is not None:
        st.subheader("Processed Preview")
        st.dataframe(current_data, use_container_width=True, height=320)
        render_validation_summary(st.session_state.get("validation_result"))
        render_download_button(current_data, "processed_dataset.csv")
        render_dataset_gemini_assistant()

    render_footer()


def process_dataset_import(
    uploaded_files: list[Any],
    selected_existing: list[Path],
) -> None:
    """Load, merge, validate, profile, and version selected datasets."""
    try:
        uploaded_paths = [
            save_uploaded_file(uploaded_file, UPLOADED_DATA_DIR)
            for uploaded_file in uploaded_files
        ]
        paths = [*uploaded_paths, *selected_existing]
        if not paths:
            st.warning("Upload or select at least one dataset file.")
            return

        loaded = load_multiple_datasets(paths)
        merged, mappings = merge_datasets(loaded)
        validation = validate_dataset(merged)
        profile = profile_dataset(merged)
        MERGED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_csv(MERGED_DATA_DIR / "merged_latest.csv", index=False)
        version_path = save_dataset_version(
            data=merged,
            statistics=profile,
            column_mappings=mappings,
            quality_score=validation.quality_score,
        )

        st.session_state["processed_dataset"] = merged
        st.session_state["validation_result"] = validation
        st.session_state["dataset_profile"] = profile
        st.session_state["column_mappings"] = mappings
        st.session_state["version_path"] = version_path
        st.success(f"Processed dataset saved: {version_path.name}")
        LOGGER.info("Dataset import completed: %s", version_path)
    except Exception as exc:
        LOGGER.exception("Dataset import failed")
        st.error(f"Dataset import failed: {exc}")


def render_dataset_explorer() -> None:
    """Render interactive dataframe explorer."""
    st.markdown('<div class="main-title">Dataset Explorer</div>', unsafe_allow_html=True)
    data = select_dataset_version()
    if data is None:
        st.info("No processed dataset available. Load a saved processed dataset first.")
        render_footer()
        return

    st.subheader("Interactive Preview")
    row_count = st.slider("Rows to preview", 5, min(max(len(data), 5), 500), 25)
    st.dataframe(data.head(row_count), use_container_width=True, height=420)

    st.subheader("Dataset Statistics")
    st.dataframe(dataset_statistics(data), use_container_width=True, hide_index=True)
    render_download_button(data, "processed_dataset.csv")
    render_footer()


def render_dataset_profile() -> None:
    """Render dataset profile charts and validation details."""
    st.markdown('<div class="main-title">Dataset Profile</div>', unsafe_allow_html=True)
    data = select_dataset_version()
    if data is None:
        st.info("No processed dataset available. Load a saved processed dataset first.")
        render_footer()
        return

    validation = validate_dataset(data)
    profile = profile_dataset(data)
    st.session_state["processed_dataset"] = data
    st.session_state["validation_result"] = validation
    st.session_state["dataset_profile"] = profile

    st.subheader("Quality Score")
    st.metric("Dataset Quality", f"{validation.quality_score}/100")
    render_validation_summary(validation)

    st.subheader("Missing Values")
    missing_frame = missing_values_frame(data)
    st.dataframe(missing_frame, use_container_width=True, hide_index=True)
    missing_fig = px.bar(
        missing_frame,
        x="column",
        y="missing_percent",
        title="Missing Values by Column",
        labels={"missing_percent": "Missing (%)", "column": "Column"},
    )
    missing_fig.update_layout(template="plotly_white", height=360)
    st.plotly_chart(missing_fig, use_container_width=True)

    numeric_data = data.select_dtypes(include="number")
    if numeric_data.empty:
        st.info("No numeric columns available for correlation or distributions.")
    else:
        render_correlation_matrix(numeric_data)
        render_feature_distribution(numeric_data)

    render_dataset_gemini_assistant()
    render_footer()


def render_correlation_matrix(numeric_data: pd.DataFrame) -> None:
    """Render numeric correlation matrix heatmap."""
    st.subheader("Correlation Matrix")
    correlation = numeric_data.corr(numeric_only=True).fillna(0)
    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Numeric Feature Correlation",
    )
    fig.update_layout(template="plotly_white", height=520)
    st.plotly_chart(fig, use_container_width=True)


def render_feature_distribution(numeric_data: pd.DataFrame) -> None:
    """Render selected numeric feature distribution."""
    st.subheader("Feature Distributions")
    selected_column = st.selectbox("Numeric feature", list(numeric_data.columns))
    fig = px.histogram(
        numeric_data,
        x=selected_column,
        marginal="box",
        title=f"Distribution: {selected_column}",
    )
    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)


def render_eda_page() -> None:
    """Render complete EDA page."""
    st.markdown('<div class="main-title">EDA</div>', unsafe_allow_html=True)
    data = select_dataset_version()
    if data is None:
        st.info("No processed dataset available. Load a saved processed dataset first.")
        render_footer()
        return

    target_column = st.selectbox("Target variable", ["None", *list(data.columns)])
    target = None if target_column == "None" else target_column
    summary = dataset_summary(data)
    st.session_state["eda_summary"] = summary

    st.subheader("Dataset Summary")
    render_summary_metrics(summary)

    left, right = st.columns(2)
    with left:
        st.subheader("Feature Types")
        st.dataframe(feature_types(data), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Duplicate Report")
        duplicates = duplicate_report(data)
        st.metric("Duplicate Rows", len(duplicates))
        st.dataframe(duplicates.head(100), use_container_width=True, height=260)

    st.subheader("Missing Value Matrix")
    st.plotly_chart(missing_value_matrix(data), use_container_width=True)

    numeric = data.select_dtypes(include="number")
    categorical = data.select_dtypes(exclude="number")
    if not numeric.empty:
        st.subheader("Numerical Statistics")
        st.dataframe(numerical_statistics(data), use_container_width=True, hide_index=True)

        st.subheader("Correlation Heatmap")
        st.plotly_chart(correlation_heatmap(data), use_container_width=True)

        st.subheader("Pairwise Correlation Table")
        st.dataframe(pairwise_correlation_table(data), use_container_width=True, hide_index=True)

        if target and target in numeric.columns:
            st.subheader("Target Correlation")
            st.plotly_chart(target_correlation(data, target), use_container_width=True)

        render_eda_numeric_charts(data, numeric)

    if not categorical.empty:
        st.subheader("Category Counts")
        category_column = st.selectbox("Categorical feature", list(categorical.columns))
        st.plotly_chart(category_counts(data, category_column), use_container_width=True)
        st.subheader("Class Balance")
        st.json(categorical_balance(data))

    if target:
        st.subheader("Target Distribution")
        st.json(target_distribution(data, target))
        if pd.api.types.is_numeric_dtype(data[target]):
            st.plotly_chart(distribution_plot(data, target), use_container_width=True)
        else:
            st.plotly_chart(category_counts(data, target), use_container_width=True)

    render_preprocessing_gemini_advisor(
        "EDA Gemini Advisor",
        {"page": "eda", "target": target, "summary": summary},
    )
    render_footer()


def render_eda_numeric_charts(data: pd.DataFrame, numeric: pd.DataFrame) -> None:
    """Render numeric EDA chart controls."""
    st.subheader("Feature Distribution")
    selected = st.selectbox("Distribution feature", list(numeric.columns), key="eda_dist")
    chart_type = st.radio(
        "Distribution chart",
        ["Histogram", "Boxplot", "Violin Plot"],
        horizontal=True,
    )
    if chart_type == "Histogram":
        st.plotly_chart(distribution_plot(data, selected), use_container_width=True)
    elif chart_type == "Boxplot":
        st.plotly_chart(boxplot(data, selected), use_container_width=True)
    else:
        st.plotly_chart(violin_plot(data, selected), use_container_width=True)

    if len(numeric.columns) >= 2:
        st.subheader("Scatterplots")
        x_column = st.selectbox("X feature", list(numeric.columns), key="eda_x")
        y_column = st.selectbox("Y feature", list(numeric.columns), index=1, key="eda_y")
        st.plotly_chart(scatterplot(data, x_column, y_column), use_container_width=True)

        st.subheader("Scatter Matrix / Pair Plot")
        selected_columns = st.multiselect(
            "Matrix features",
            list(numeric.columns),
            default=list(numeric.columns[: min(4, len(numeric.columns))]),
        )
        if len(selected_columns) >= 2:
            st.plotly_chart(scatter_matrix(data, selected_columns), use_container_width=True)


def render_feature_engineering_page() -> None:
    """Render feature engineering page."""
    st.markdown('<div class="main-title">Feature Engineering</div>', unsafe_allow_html=True)
    data = select_dataset_version()
    if data is None:
        st.info("No processed dataset available. Load a saved processed dataset first.")
        render_footer()
        return

    st.subheader("Engineering Feature Controls")
    all_features = available_engineered_features()
    enabled = st.multiselect(
        "Enable engineered features",
        all_features,
        default=all_features,
    )

    engineered = engineer_laminate_features(data, enabled)
    added_columns = [column for column in engineered.columns if column not in data.columns]
    st.session_state["engineered_dataset"] = engineered
    st.session_state["processed_dataset"] = engineered

    st.metric("Engineered Features Added", len(added_columns))
    st.dataframe(pd.DataFrame({"engineered_feature": added_columns}), use_container_width=True)
    st.subheader("Engineered Dataset Preview")
    st.dataframe(engineered.head(100), use_container_width=True, height=420)
    render_download_button(engineered, "engineered_dataset.csv")

    render_preprocessing_gemini_advisor(
        "Feature Engineering Gemini Advisor",
        {
            "page": "feature_engineering",
            "enabled_features": enabled,
            "added_columns": added_columns,
            "input_shape": data.shape,
            "output_shape": engineered.shape,
        },
    )
    render_footer()


def render_preprocessing_page() -> None:
    """Render preprocessing pipeline page."""
    st.markdown('<div class="main-title">Preprocessing</div>', unsafe_allow_html=True)
    data = select_dataset_version()
    if data is None:
        st.info("No processed dataset available. Load a saved processed dataset first.")
        render_footer()
        return

    target_column = st.selectbox("Target variable", ["None", *list(data.columns)], key="prep_target")
    target = None if target_column == "None" else target_column
    numeric_columns = list(data.select_dtypes(include="number").columns)

    left, right = st.columns(2)
    with left:
        numeric_imputer = st.selectbox(
            "Numeric missing values",
            ["Mean", "Median", "Mode", "KNN Imputer", "Forward Fill", "Backward Fill"],
            index=1,
        )
        encoding_method = st.selectbox(
            "Categorical encoding",
            ["One-Hot Encoding", "Ordinal Encoding", "Label Encoding"],
        )
        scaling_method = st.selectbox(
            "Scaling",
            ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer", "None"],
        )
    with right:
        categorical_imputer = st.selectbox(
            "Categorical missing values",
            ["Mode", "Forward Fill", "Backward Fill"],
        )
        feature_selection = st.selectbox(
            "Feature selection",
            [
                "None",
                "Variance Threshold",
                "Correlation Threshold",
                "Mutual Information",
                "Recursive Feature Elimination",
                "Tree Feature Importance",
            ],
        )
        top_k = st.number_input("Top K features", min_value=1, max_value=200, value=20)

    st.subheader("Outlier Detection")
    outlier_method = st.selectbox(
        "Outlier method",
        ["IQR", "Z-score", "Isolation Forest", "Local Outlier Factor"],
    )
    outlier_columns = st.multiselect("Outlier feature columns", numeric_columns, default=numeric_columns[:4])
    remove_flag = st.checkbox("Remove detected outliers")
    outlier_mask = detect_outliers(data, outlier_method, outlier_columns)
    st.metric("Detected Outlier Rows", int(outlier_mask.sum()))
    st.dataframe(data.loc[outlier_mask].head(100), use_container_width=True, height=280)

    if len(outlier_columns) >= 2:
        st.plotly_chart(
            outlier_scatter(data, outlier_columns[0], outlier_columns[1], outlier_mask),
            use_container_width=True,
        )

    pipeline_input = remove_outliers(data, outlier_mask) if remove_flag else data.copy()
    config = PreprocessingConfig(
        target_column=target,
        enabled_engineered_features=available_engineered_features(),
        numeric_imputer=numeric_imputer,
        categorical_imputer=categorical_imputer,
        encoding_method=encoding_method,
        scaling_method=scaling_method,
        feature_selection_method=feature_selection,
        top_k_features=int(top_k),
    )

    if st.button("Run Preprocessing Pipeline", type="primary"):
        run_preprocessing_for_ui(pipeline_input, config, outlier_mask, remove_flag)

    processed = st.session_state.get("clean_dataset")
    if isinstance(processed, pd.DataFrame):
        st.subheader("Clean Dataset")
        st.dataframe(processed.head(150), use_container_width=True, height=420)
        render_export_controls(processed)

        ranking = st.session_state.get("feature_ranking")
        if isinstance(ranking, pd.DataFrame) and not ranking.empty:
            st.subheader("Feature Selection Ranking")
            st.dataframe(ranking, use_container_width=True, hide_index=True)
            ranking_head = ranking.head(30)
            fig = px.bar(
                ranking_head,
                x="feature",
                y="score",
                color="method",
                title="Feature Ranking",
            )
            fig.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig, use_container_width=True)

    render_preprocessing_gemini_advisor(
        "Preprocessing Gemini Advisor",
        {
            "page": "preprocessing",
            "target": target,
            "outlier_method": outlier_method,
            "outlier_count": int(outlier_mask.sum()),
            "config": config.__dict__,
            "input_shape": data.shape,
        },
    )
    render_footer()


def run_preprocessing_for_ui(
    data: pd.DataFrame,
    config: PreprocessingConfig,
    outlier_mask: pd.Series,
    remove_flag: bool,
) -> None:
    """Execute preprocessing pipeline for Streamlit UI."""
    try:
        clean_data, pipeline, metadata = run_preprocessing_pipeline(data, config)
        metadata["outliers_removed"] = int(outlier_mask.sum()) if remove_flag else 0
        artifact_paths = save_pipeline_artifacts(pipeline, metadata, REPORT_DIR)
        st.session_state["clean_dataset"] = clean_data
        st.session_state["preprocessing_pipeline"] = pipeline
        st.session_state["preprocessing_metadata"] = metadata
        st.session_state["pipeline_paths"] = artifact_paths
        st.session_state["feature_ranking"] = pd.DataFrame(metadata["feature_ranking"])
        st.success("Preprocessing pipeline executed and saved.")
        LOGGER.info("Preprocessing artifacts saved: %s", artifact_paths)
    except Exception as exc:
        LOGGER.exception("Preprocessing pipeline failed")
        st.error(f"Preprocessing failed: {exc}")


def render_export_controls(data: pd.DataFrame) -> None:
    """Render CSV, Excel, Joblib, and metadata export controls."""
    st.subheader("Dataset Export")
    csv_bytes = data.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", csv_bytes, "clean_dataset.csv", "text/csv")

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="clean_dataset")
    st.download_button(
        "Export Excel",
        excel_buffer.getvalue(),
        "clean_dataset.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    metadata = st.session_state.get("preprocessing_metadata", {})
    st.download_button(
        "Export Metadata JSON",
        json.dumps(metadata, indent=2, default=str).encode("utf-8"),
        "preprocessing_metadata.json",
        "application/json",
    )

    pipeline = st.session_state.get("preprocessing_pipeline")
    if pipeline is not None:
        joblib_buffer = BytesIO()
        joblib.dump(pipeline, joblib_buffer)
        st.download_button(
            "Export Joblib Preprocessing Pipeline",
            joblib_buffer.getvalue(),
            "preprocessing_pipeline.joblib",
            "application/octet-stream",
        )


def render_summary_metrics(summary: dict[str, Any]) -> None:
    """Render summary metrics as cards."""
    columns = st.columns(len(summary))
    for column, (label, value) in zip(columns, summary.items()):
        with column:
            st.metric(label.replace("_", " ").title(), value)


def render_preprocessing_gemini_advisor(title: str, context: dict[str, Any]) -> None:
    """Render Gemini advisory button for Step 3 only."""
    st.subheader(title)
    st.caption("Advisory only. Gemini never edits data or fabricates values.")
    if st.button(f"Ask Gemini: {title}", key=f"gemini_{title}"):
        service = GeminiService()
        if not service.is_configured:
            render_gemini_failure(
                "Gemini preprocessing advice is unavailable right now.",
                service.generate_text(""),
            )
            return

        eda_summary = st.session_state.get("eda_summary", {})
        with st.spinner("Generating advisory analysis..."):
            result = service.preprocessing_advice(title, eda_summary, context)
            if not result.success:
                render_gemini_failure(
                    "Gemini preprocessing advice is unavailable right now.",
                    result,
                )
                return
        render_gemini_text_response(
            title=title,
            text=result.text or "",
            caption="Gemini is advisory only. Deterministic CompositeAI outputs remain authoritative.",
        )


def render_strength_prediction() -> None:
    """Render real tensile-strength prediction workflow."""
    mode = get_app_mode(st.session_state)
    result: dict[str, Any] | None = None
    metrics: dict[str, float] = {}
    st.markdown(
        f'<div class="main-title">{page_title("Strength Prediction")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <p class="section-copy">
            {"Predict how much pulling stress this laminate can resist." if mode == APP_MODE_SIMPLE else "Predict tensile strength using the validated ANN/MLP sklearn Pipeline. Inputs are checked against supported categories and observed training-data ranges. No stacking optimization is performed here."}
        </p>
        """,
        unsafe_allow_html=True,
    )

    try:
        artifact = inspect_model_artifact()
    except Exception as exc:
        LOGGER.exception("Could not load prediction model")
        st.error(f"Prediction model could not be loaded: {exc}")
        render_footer()
        return

    categories = artifact["supported_categories"]
    ranges = artifact["dataset_supported_ranges"]

    left, right = st.columns(2)
    with left:
        st.subheader("Material")
        fiber_type = st.selectbox("Fiber Type", categories["fiber_type"], help="Composite reinforcement family.")
        resin_type = st.selectbox("Resin Type", categories["resin_type"], help="Polymer matrix family.")

        st.subheader("Material Properties")
        density = st.number_input(
            "Density (g/cm³)",
            value=1.55,
            step=0.01,
            format="%.3f",
            help="Material mass per unit volume.",
        )
        layer_count = st.number_input(
            "Layer Count",
            min_value=1,
            value=8,
            step=1,
            help="Number of laminate layers.",
        )

    with right:
        st.subheader("Processing")
        curing_temperature = st.number_input(
            "Curing Temperature (°C)",
            value=120.0,
            step=1.0,
            format="%.1f",
        )
        fiber_volume_fraction = st.number_input(
            "Fiber Volume Fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.01,
            format="%.3f",
            help=term_help("Fiber Volume Fraction"),
        )
        void_content = st.number_input(
            "Void Content (%)",
            min_value=0.0,
            value=2.0,
            step=0.1,
            format="%.2f",
            help=term_help("Void Content"),
        )

        with st.expander("Observed training-data ranges"):
            range_frame = pd.DataFrame(ranges).T.reset_index()
            range_frame.columns = ["Feature", "Minimum", "Maximum"]
            st.dataframe(range_frame, use_container_width=True, hide_index=True)

    input_data = {
        "fiber_type": fiber_type,
        "resin_type": resin_type,
        "density_g_cm3": density,
        "layer_count": int(layer_count),
        "curing_temperature_c": curing_temperature,
        "fiber_volume_fraction": fiber_volume_fraction,
        "void_content_pct": void_content,
    }

    if st.button("Predict Strength", type="primary"):
        try:
            result = predict_strength(input_data)
        except PredictionInputError as exc:
            st.error(str(exc))
            render_footer()
            return
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            st.error(f"Prediction failed: {exc}")
            render_footer()
            return

        metrics = result.get("metrics", {})
        st.session_state["prediction_result"] = result
        for warning in result["warnings"]:
            st.warning(warning)

        st.markdown("---")
        st.subheader("Predicted Tensile Strength")
        st.markdown(
            f'<div class="prediction-value">{result["predicted_tensile_strength_mpa"]:,.2f} MPa</div>',
            unsafe_allow_html=True,
        )

        if mode == APP_MODE_ENGINEERING:
            st.subheader("Model Performance")
            st.caption("Validation metrics from held-out Step 5 model evaluation. Not prediction confidence.")
            metric_columns = st.columns(4)
            metric_columns[0].metric("Model", result["model_name"])
            metric_columns[1].metric("R²", f"{metrics.get('r2', 0.0):.4f}")
            metric_columns[2].metric("MAE", f"{metrics.get('mae', 0.0):.2f} MPa")
            metric_columns[3].metric("RMSE", f"{metrics.get('rmse', 0.0):.2f} MPa")
        st.session_state["prediction_context"] = {
            "input_features": input_data,
            "predicted_tensile_strength_mpa": result["predicted_tensile_strength_mpa"],
            "model_name": result["model_name"],
            "validation_metrics": metrics or result.get("metrics", {}),
            "warnings": result["warnings"],
            "notes": "Step 6 prediction only; no stacking sequence or optimizer used.",
        }

    prediction_context = st.session_state.get("prediction_context")
    if prediction_context and prediction_context.get("predicted_tensile_strength_mpa") is not None:
        predicted_value = float(prediction_context["predicted_tensile_strength_mpa"])
        st.markdown("---")
        st.subheader(ui_term("Tensile Strength", mode))
        st.markdown(
            f'<div class="prediction-value">{predicted_value:,.2f} MPa</div>',
            unsafe_allow_html=True,
        )
        render_prediction_summary_sections(predicted_value)
        if mode == APP_MODE_ENGINEERING:
            render_term_expander(["ANN/MLP", "Tensile Strength", "Fiber Volume Fraction", "Void Content", "R² Score"])

    render_footer()


def render_model_performance() -> None:
    """Render saved model training and validation results."""
    st.markdown('<div class="main-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="section-copy">
            Validated machine-learning performance for composite tensile-strength prediction.
        </p>
        """,
        unsafe_allow_html=True,
    )

    model_path = PROJECT_ROOT / "saved_models" / "best_strength_model.joblib"
    metadata_path = PROJECT_ROOT / "saved_models" / "model_metadata.json"
    comparison_path = PROJECT_ROOT / "data" / "training" / "model_comparison.csv"
    validation_path = PROJECT_ROOT / "data" / "training" / "model_validation_report.json"

    if not model_path.exists() or not metadata_path.exists():
        st.warning("Trained model artifacts are unavailable. Model performance results cannot be displayed.")
        render_footer()
        return

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validation_report = (
            json.loads(validation_path.read_text(encoding="utf-8"))
            if validation_path.exists()
            else {}
        )
        comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
        saved_model = joblib.load(model_path)
    except Exception as exc:
        LOGGER.exception("Failed to load model performance artifacts")
        st.warning(f"Trained model artifacts are unavailable. Model performance results cannot be displayed. {exc}")
        render_footer()
        return

    mode = get_app_mode(st.session_state)
    if mode == APP_MODE_SIMPLE:
        metrics = metadata.get("metrics", {})
        st.subheader("Artificial Intelligence Prediction Model")
        cards = st.columns(4)
        cards[0].metric("Prediction Method", "Artificial Intelligence Prediction Model")
        cards[1].metric("Model Accuracy Score", f"{float(metrics.get('r2', 0.0)):.4f}")
        cards[2].metric("Average Prediction Error", f"{float(metrics.get('mae', 0.0)):.2f} MPa")
        cards[3].metric("Typical Prediction Error", f"{float(metrics.get('rmse', 0.0)):.2f} MPa")
        st.subheader("What does this mean?")
        st.info(
            "This prediction model performs strongly on locked validation data. It is reliable for project demonstration, but it does not replace experimental testing or certification."
        )
        st.subheader("Recommended Use")
        st.write(
            "Use this page to understand overall prediction quality at high level. Use Engineering Mode for residual analysis, cross-validation, seed robustness, and full training evidence."
        )
        render_footer()
        return

    render_primary_model_result(metadata)
    render_model_comparison(comparison)
    render_best_model_summary(comparison)
    render_cross_validation(validation_report)
    render_seed_robustness(validation_report)
    render_validation_status(validation_report)
    render_validation_plots(validation_report)
    render_model_information(saved_model, metadata)
    render_data_information(metadata)
    render_model_interpretation()


def render_engineering_benchmark() -> None:
    """Render composite-vs-material engineering benchmark page."""
    mode = get_app_mode(st.session_state)
    st.markdown(
        f'<div class="main-title">{page_title("Composite vs Aerospace Metals")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <p class="section-copy">
            {"Compare a predicted composite laminate against aerospace metal reference values." if mode == APP_MODE_SIMPLE else "Compare CompositeAI tensile-strength predictions against a separate engineering reference database of aerospace metals and alloys. This benchmark is advisory only and is not used for machine-learning training."}
        </p>
        """,
        unsafe_allow_html=True,
    )

    prediction_context = st.session_state.get("prediction_context", {})
    predicted_default = float(prediction_context.get("predicted_tensile_strength_mpa", 0.0) or 0.0)
    density_default = float(prediction_context.get("input_features", {}).get("density_g_cm3", 0.0) or 0.0)

    st.info(
        "Reference material properties are engineering benchmark values and are not part of the machine-learning training dataset."
    )
    left, right = st.columns(2)
    with left:
        predicted_strength = st.number_input(
            "Predicted Composite Strength (MPa)",
            min_value=0.0,
            value=predicted_default,
            step=1.0,
            format="%.2f",
        )
    with right:
        composite_density_value = st.number_input(
            "Composite Density (g/cm³, optional)",
            min_value=0.0,
            value=density_default,
            step=0.01,
            format="%.3f",
        )
    composite_density = composite_density_value if composite_density_value > 0 else None

    if st.button("Run Benchmark", type="primary"):
        if predicted_strength <= 0:
            st.error("Predicted composite strength must be greater than zero.")
            render_footer()
            return
        try:
            report = build_material_benchmark_report(predicted_strength, composite_density)
            html_report = generate_material_benchmark_html_report(predicted_strength, composite_density)
            pdf_bytes = generate_material_benchmark_pdf_report(predicted_strength, composite_density)
            csv_bytes = export_comparison_csv(predicted_strength, composite_density)
            excel_bytes = export_comparison_excel(predicted_strength, composite_density)
        except Exception as exc:
            LOGGER.exception("Engineering benchmark failed")
            st.error(f"Engineering benchmark failed: {exc}")
            render_footer()
            return

        st.session_state["benchmark_context"] = {
            "report": report,
            "html_report": html_report,
            "pdf_bytes": pdf_bytes,
            "csv_bytes": csv_bytes,
            "excel_bytes": excel_bytes,
            "html_report_path": str(DEFAULT_HTML_REPORT_PATH),
        }
        st.session_state["benchmark_result"] = report

    benchmark_context = st.session_state.get("benchmark_context")
    if not benchmark_context:
        st.info("Run Strength Prediction first or enter benchmark inputs, then select Run Benchmark.")
        render_footer()
        return

    if not isinstance(benchmark_context, dict) or "report" not in benchmark_context:
        st.info("Run Strength Prediction first or enter benchmark inputs, then select Run Benchmark.")
        render_footer()
        return

    report = benchmark_context["report"]
    benchmark_category_note(report)
    comparison_frame = pd.DataFrame(report["comparison_rows"]).rename(
        columns={
            "material": "Material",
            "category": "Category",
            "application": "Application",
            "tensile_strength_mpa": "Strength (MPa)",
            "density_g_cm3": "Density (g/cm³)",
            "specific_strength": "Specific Strength",
            "difference_vs_composite_mpa": "Difference vs Composite (MPa)",
            "strength_ratio": "Strength Ratio",
            "density_ratio": "Density Ratio",
            "specific_strength_ratio": "Specific Strength Ratio",
            "strength_rank": "Strength Rank",
            "density_rank": "Density Rank",
            "specific_strength_rank": "Specific Strength Rank",
        }
    )

    st.subheader("Composite Prediction Card")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Predicted Strength", f'{report["predicted_strength_mpa"]:,.2f} MPa')
    metric_columns[1].metric(
        "Composite Density",
        f'{report["composite_density_g_cm3"]:.3f} g/cm³' if report["composite_density_g_cm3"] is not None else "Unavailable",
    )
    metric_columns[2].metric(
        "Specific Strength",
        f'{report["composite_specific_strength"]:.2f}' if report["composite_specific_strength"] is not None else "Unavailable",
    )

    st.subheader("Comparison Table")
    if mode == APP_MODE_SIMPLE:
        simple_columns = [
            "Material",
            "Category",
            "Strength (MPa)",
            "Density (g/cm³)",
            "Specific Strength",
            "Difference vs Composite (MPa)",
        ]
        st.dataframe(comparison_frame[simple_columns], use_container_width=True, hide_index=True)
    else:
        st.dataframe(comparison_frame, use_container_width=True, hide_index=True)

    st.subheader("Rankings")
    ranking_columns = st.columns(3)
    ranking_columns[0].dataframe(
        comparison_frame[["Material", "Strength (MPa)", "Strength Rank"]].sort_values("Strength Rank"),
        use_container_width=True,
        hide_index=True,
    )
    ranking_columns[1].dataframe(
        comparison_frame[["Material", "Density (g/cm³)", "Density Rank"]].sort_values("Density Rank"),
        use_container_width=True,
        hide_index=True,
    )
    ranking_columns[2].dataframe(
        comparison_frame[["Material", "Specific Strength", "Specific Strength Rank"]].sort_values("Specific Strength Rank"),
        use_container_width=True,
        hide_index=True,
    )

    best_specific = report["summary"].get("best_specific_strength")
    if best_specific:
        st.success(
            f'Best reference strength-to-weight ratio: {best_specific["material"]} '
            f'({best_specific["specific_strength"]:.2f} MPa per g/cm³).'
        )

    st.subheader("Charts")
    strength_fig = px.bar(
        comparison_frame,
        x="Material",
        y="Strength (MPa)",
        color="Category",
        title="Strength Comparison",
    )
    strength_fig.update_layout(template="plotly_white", height=360)
    st.plotly_chart(strength_fig, use_container_width=True)

    density_frame = comparison_frame.dropna(subset=["Density (g/cm³)"])
    if density_frame.empty:
        st.warning("Density comparison unavailable because dataset density values are missing.")
    else:
        density_fig = px.bar(
            density_frame,
            x="Material",
            y="Density (g/cm³)",
            color="Category",
            title="Density Comparison",
        )
        density_fig.update_layout(template="plotly_white", height=360)
        st.plotly_chart(density_fig, use_container_width=True)

    specific_frame = comparison_frame.dropna(subset=["Specific Strength"])
    if specific_frame.empty:
        st.warning("Specific strength comparison unavailable because density values are missing.")
    else:
        specific_fig = px.bar(
            specific_frame,
            x="Material",
            y="Specific Strength",
            color="Category",
            title="Specific Strength Comparison",
        )
        specific_fig.update_layout(template="plotly_white", height=360)
        st.plotly_chart(specific_fig, use_container_width=True)

    ranking_frame = comparison_frame[["Material", "Strength Rank"]].copy()
    ranking_frame["Composite"] = "Reference Materials"
    ranking_fig = px.scatter(
        ranking_frame,
        x="Strength Rank",
        y="Material",
        title="Composite Position Ranking Context",
        color="Composite",
        size=[18] * len(ranking_frame),
    )
    ranking_fig.update_layout(template="plotly_white", height=360)
    st.plotly_chart(ranking_fig, use_container_width=True)

    st.subheader("Engineering Insights")
    for insight in report["insights"]:
        st.markdown(f"- {insight}")

    if mode == APP_MODE_ENGINEERING:
        st.subheader("Engineering Mode")
        equation_rows = [
            ("Strength Ratio", report["equations"]["strength_ratio"]),
            ("Specific Strength", report["equations"]["specific_strength"]),
            ("Density Ratio", report["equations"]["density_ratio"]),
        ]
        st.dataframe(
            build_display_frame(equation_rows, ["Metric", "Equation"]),
            use_container_width=True,
            hide_index=True,
        )
        raw_rows = [
            ("Closest Match", report["summary"].get("closest_strength_match", {}).get("material", "Unavailable") if isinstance(report["summary"].get("closest_strength_match"), dict) else "Unavailable"),
            ("Best Strength-To-Weight", report["summary"].get("best_specific_strength", {}).get("material", "Unavailable") if isinstance(report["summary"].get("best_specific_strength"), dict) else "Unavailable"),
            ("Strongest Reference", report["summary"].get("strongest_material", {}).get("material", "Unavailable") if isinstance(report["summary"].get("strongest_material"), dict) else "Unavailable"),
            ("Lightest Reference", report["summary"].get("lightest_material", {}).get("material", "Unavailable") if isinstance(report["summary"].get("lightest_material"), dict) else "Unavailable"),
            ("Reference Database", report["source_database"]),
        ]
        st.dataframe(build_display_frame(raw_rows, ["Field", "Value"]), use_container_width=True, hide_index=True)
        render_term_expander(["Tensile Strength"])

    st.subheader("Export")
    st.caption(f'HTML report saved to: {benchmark_context["html_report_path"]}')
    st.download_button(
        label="Download PDF",
        data=benchmark_context["pdf_bytes"],
        file_name="material_benchmark_report.pdf",
        mime="application/pdf",
    )
    st.download_button(
        label="Download CSV",
        data=benchmark_context["csv_bytes"],
        file_name="material_benchmark_comparison.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download Excel",
        data=benchmark_context["excel_bytes"],
        file_name="material_benchmark_comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    render_footer()


def render_primary_model_result(metadata: dict[str, Any]) -> None:
    """Render primary saved model metrics."""
    metrics = metadata.get("metrics", {})
    st.subheader("Primary Model Result")
    columns = st.columns(4)
    columns[0].metric("Model", metadata.get("model_name", "Unavailable"))
    columns[1].metric("R² Score", format_float(metrics.get("r2"), 4))
    columns[2].metric("MAE", f"{format_float(metrics.get('mae'), 2)} MPa")
    columns[3].metric("RMSE", f"{format_float(metrics.get('rmse'), 2)} MPa")


def render_model_comparison(comparison: pd.DataFrame) -> None:
    """Render saved model comparison table."""
    st.subheader("Model Comparison")
    if comparison.empty:
        st.warning("Model comparison file is unavailable.")
        return

    display = comparison.rename(
        columns={
            "model": "Model",
            "train_mae": "Train MAE",
            "test_mae": "Test MAE",
            "train_rmse": "Train RMSE",
            "test_rmse": "Test RMSE",
            "train_r2": "Train R²",
            "test_r2": "Test R²",
        }
    )
    numeric_columns = [column for column in display.columns if column != "Model"]
    for column in numeric_columns:
        display[column] = display[column].map(lambda value: round(float(value), 4))
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_best_model_summary(comparison: pd.DataFrame) -> None:
    """Render best validated model selected from comparison CSV."""
    if comparison.empty or "test_rmse" not in comparison.columns:
        return

    best = comparison.sort_values(["test_rmse", "test_mae", "test_r2"], ascending=[True, True, False]).iloc[0]
    st.success(f"Best validated model: {best['model']}")
    columns = st.columns(3)
    columns[0].metric("Test R²", f"{float(best['test_r2']):.4f}")
    columns[1].metric("Test MAE", f"{float(best['test_mae']):.2f} MPa")
    columns[2].metric("Test RMSE", f"{float(best['test_rmse']):.2f} MPa")


def render_cross_validation(validation_report: dict[str, Any]) -> None:
    """Render cross-validation metrics from Step 5 report."""
    st.subheader("Cross-Validation")
    cross_validation = validation_report.get("cross_validation", {})
    if not cross_validation:
        st.warning("Cross-validation report is unavailable.")
        return

    rows = []
    for model_name, metrics in cross_validation.items():
        rows.append(
            {
                "Model": model_name,
                "MAE": f"{metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f} MPa",
                "RMSE": f"{metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f} MPa",
                "R²": f"{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_seed_robustness(validation_report: dict[str, Any]) -> None:
    """Render random-seed robustness metrics."""
    st.subheader("Random-Seed Robustness")
    robustness = validation_report.get("seed_robustness", {})
    rows = robustness.get("rows", [])
    summary = robustness.get("summary", {})
    if not rows or not summary:
        st.warning("Random-seed robustness report is unavailable.")
        return

    seeds = ", ".join(str(row["seed"]) for row in rows)
    st.caption(f"Seeds tested: {seeds}")
    columns = st.columns(3)
    columns[0].metric("Mean MAE", f"{summary['mae_mean']:.4f} ± {summary['mae_std']:.4f} MPa")
    columns[1].metric("Mean RMSE", f"{summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f} MPa")
    columns[2].metric("Mean R²", f"{summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_validation_status(validation_report: dict[str, Any]) -> None:
    """Render validation status indicators from report."""
    st.subheader("Validation Status")
    leakage = validation_report.get("leakage_assessment", {})
    data_validation = validation_report.get("data_validation", {})
    saved_model = validation_report.get("saved_model_validation", {})
    overfitting = validation_report.get("overfitting_assessment", {})
    final_decision = validation_report.get("final_decision", {})
    cross_validation = validation_report.get("cross_validation", {})
    seed_robustness = validation_report.get("seed_robustness", {})

    rows = [
        {"Check": "Leakage Check", "Status": leakage.get("status", "Unavailable")},
        {
            "Check": "Test Set Untouched",
            "Status": "YES" if data_validation.get("test_set_untouched") else "Unavailable",
        },
        {
            "Check": "Overfitting",
            "Status": overfitting.get("finding", "Unavailable"),
        },
        {"Check": "Saved Model", "Status": saved_model.get("status", "Unavailable")},
        {"Check": "Cross-Validation", "Status": "PASS" if cross_validation else "Unavailable"},
        {"Check": "Seed Robustness", "Status": "PASS" if seed_robustness else "Unavailable"},
        {"Check": "Final Decision", "Status": final_decision.get("status", "Unavailable")},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_validation_plots(validation_report: dict[str, Any]) -> None:
    """Render Step 5 validation plots from HTML artifacts."""
    st.subheader("Residual Analysis and Validation Plots")
    labels = {
        "actual_vs_predicted": "Actual vs Predicted Tensile Strength",
        "residual_vs_predicted": "Residual vs Predicted",
        "residual_distribution": "Residual Distribution",
        "cv_comparison": "Cross-Validation Comparison",
        "seed_robustness": "Seed Robustness",
    }
    filenames = {
        key: f"{key}.html"
        for key in labels
        if f"{key}.html" in VALIDATION_PLOT_FILENAMES
    }

    if st.button("Generate Validation Report", key="generate_validation_report"):
        with st.spinner("Generating validation report and plots..."):
            try:
                validation_report = run_model_validation()
            except Exception as exc:
                LOGGER.exception("Validation report generation failed")
                st.error(f"Validation report generation failed: {exc}")
                return
        st.success("Validation report generated.")

    missing_before_regeneration = [
        filename
        for filename in filenames.values()
        if get_validation_plot_path(filename) is None
    ]
    if missing_before_regeneration:
        with st.spinner("Validation plots missing; regenerating report artifacts..."):
            try:
                validation_report = run_model_validation()
            except Exception as exc:
                LOGGER.exception("Automatic validation plot regeneration failed")
                st.warning(f"Validation plot regeneration failed: {exc}")

    missing_plots: list[str] = []
    for key, label in labels.items():
        filename = filenames[key]
        plot_path = get_validation_plot_path(filename)
        with st.expander(label, expanded=(key == "actual_vs_predicted")):
            if plot_path is not None:
                components.html(plot_path.read_text(encoding="utf-8"), height=520, scrolling=True)
            else:
                st.info("Validation plot not found. Click Generate Validation Report.")
                missing_plots.append(label)

    if missing_plots:
        st.caption(
            "Missing plots: " + ", ".join(missing_plots)
        )


def render_model_information(saved_model: Any, metadata: dict[str, Any]) -> None:
    """Render confirmed model configuration."""
    st.subheader("Model Information")
    model_step = getattr(saved_model, "named_steps", {}).get("model")
    info = {
        "Algorithm": metadata.get("model_name", "Unavailable"),
        "Architecture": " → ".join(str(value) for value in getattr(model_step, "hidden_layer_sizes", []))
        if model_step is not None
        else "Unavailable",
        "Activation": getattr(model_step, "activation", "Unavailable"),
        "Solver": getattr(model_step, "solver", "Unavailable"),
        "Early stopping": getattr(model_step, "early_stopping", "Unavailable"),
        "Maximum iterations": getattr(model_step, "max_iter", "Unavailable"),
        "Validation fraction": getattr(model_step, "validation_fraction", "Unavailable"),
        "Random state": getattr(model_step, "random_state", metadata.get("random_state", "Unavailable")),
    }
    st.dataframe(build_display_frame(list(info.items()), ["Field", "Value"]), use_container_width=True, hide_index=True)


def render_data_information(metadata: dict[str, Any]) -> None:
    """Render dataset information from metadata and artifact."""
    st.subheader("Data Information")
    dataset_path = metadata.get("training_dataset", "data/training/ml_ready_features.csv")
    dataset_file = PROJECT_ROOT / dataset_path
    rows = "Unavailable"
    if dataset_file.exists():
        try:
            rows = f"{len(pd.read_csv(dataset_file)):,}"
        except Exception:
            rows = "Unavailable"
    split = metadata.get("train_test_split", {})
    features = metadata.get("feature_list", [])
    info = {
        "Dataset": dataset_path,
        "Rows": rows,
        "Training": f"{split.get('train_rows', 'Unavailable'):,}"
        if isinstance(split.get("train_rows"), int)
        else split.get("train_rows", "Unavailable"),
        "Testing": f"{split.get('test_rows', 'Unavailable'):,}"
        if isinstance(split.get("test_rows"), int)
        else split.get("test_rows", "Unavailable"),
        "Target": metadata.get("target", "Unavailable"),
        "Input features": len(features),
    }
    st.dataframe(build_display_frame(list(info.items()), ["Field", "Value"]), use_container_width=True, hide_index=True)
    with st.expander("Input feature list"):
        st.write(features)


def render_model_interpretation() -> None:
    """Render engineering interpretation and navigation guidance."""
    st.subheader("Interpretation")
    st.info(
        "The ANN/MLP provides strong predictive performance on the held-out test set "
        "and remains stable under cross-validation and multiple random seeds. The model "
        "predicts tensile strength from material and processing parameters."
    )
    st.warning(
        "These metrics represent model performance on the available dataset and do not "
        "constitute experimental certification of a physical aerospace laminate."
    )
    st.markdown("**Ready to make a prediction?** Open the **Strength Prediction** page.")
    st.markdown(
        "Stacking-sequence optimization is evaluated separately using the CLT-based "
        "mechanics module on the **Stacking Optimizer** page."
    )


def format_float(value: Any, digits: int) -> str:
    """Format numeric value or return unavailable marker."""
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "Unavailable"


def render_stacking_optimizer() -> None:
    """Render CLT-based stacking-sequence optimization demonstrator."""
    mode = get_app_mode(st.session_state)
    st.markdown('<div class="main-title">Stacking Optimizer</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class="section-copy">
            {"Find best layer arrangement for current laminate case." if mode == APP_MODE_SIMPLE else "Searches valid stacking sequences and evaluates each candidate using source-backed CLT strain failure logic. This page does not use the ANN strength predictor."}
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "Research/educational computational demonstrator only. Results depend on "
        "material properties, load case, assumptions, and failure model; they are "
        "not experimentally validated for a physical aerospace laminate."
    )
    render_optimization_impact_card(mode)

    try:
        case = load_tu_delft_demo_case()
        reference = validate_reference_case()
    except Exception as exc:
        LOGGER.exception("Failed to load TU Delft optimization demo case")
        st.error(f"Could not load TU Delft source-backed case: {exc}")
        render_footer()
        return

    material = case["material"]
    default_load = case["load_case"]
    baseline_sequence = case["baseline_sequence"]
    allowables = case["allowables"]

    st.subheader("Material")
    st.selectbox(
        "Verified material card",
        [f"{material.name} | TU Delft/Zenodo 10.5281/zenodo.15864524"],
        disabled=True,
    )
    if mode == APP_MODE_ENGINEERING:
        material_cols = st.columns(5)
        material_cols[0].metric("E1", f"{material.e1_pa / 1e9:.2f} GPa")
        material_cols[1].metric("E2", f"{material.e2_pa / 1e9:.2f} GPa")
        material_cols[2].metric("G12", f"{material.g12_pa / 1e9:.2f} GPa")
        material_cols[3].metric("ν12", f"{material.nu12:.2f}")
        material_cols[4].metric("Ply t", f"{material.ply_thickness_m:.6f} m")
    else:
        st.caption("Reference material card loaded from validated source case.")

    st.subheader("Laminate Constraints")
    left, right = st.columns(2)
    with left:
        ply_count = st.selectbox("Ply count", [8, 16, 24, 48], index=3)
        selected_angles = st.multiselect(
            "Allowed angles (degrees)",
            list(DEFAULT_ALLOWED_ANGLES),
            default=list(DEFAULT_ALLOWED_ANGLES),
        )
        require_symmetric = st.checkbox("Require symmetric laminate", value=True)
        require_balanced = st.checkbox("Require balanced laminate", value=True)
    with right:
        st.subheader("Load Case")
        nx_value = st.number_input(
            "Nxx (N/m)",
            value=float(default_load.nx_n_per_m),
            step=10.0,
            format="%.6f",
        )
        ny_value = st.number_input(
            "Nyy (N/m)",
            value=float(default_load.ny_n_per_m),
            step=10.0,
            format="%.6f",
        )
        nxy_value = st.number_input("Nxy (N/m)", value=0.0, step=10.0, format="%.6f")
        max_candidates = st.slider("Max candidates evaluated", 25, 500, 150, 25)

    if mode == APP_MODE_ENGINEERING:
        st.caption(
            "Failure model: source-compatible strain allowables "
            f"ε1={allowables.epsilon_1_allowable}, ε2={allowables.epsilon_2_allowable}, "
            f"γ12={allowables.gamma_12_allowable}. Objective: maximize λ_cs."
        )
        st.caption(
            "Reference validation: "
            f"source λ_cs={reference['reference_lambda_cs']:.2f}, "
            f"ours={reference['our_lambda_cs']:.2f}, "
            f"diff={reference['difference_pct']:.2f}%."
        )

    if st.button("Optimize Stacking Sequence", type="primary"):
        if not selected_angles:
            st.error("Select at least one allowed angle.")
            render_footer()
            return
        try:
            load_case = LaminateLoadCase(
                nx_n_per_m=nx_value,
                ny_n_per_m=ny_value,
                nxy_n_per_m=nxy_value,
            )
            config = OptimizationConfig(
                sequence_config=SequenceConfig(
                    allowed_angles=tuple(int(angle) for angle in selected_angles),
                    require_symmetric=require_symmetric,
                    require_balanced=require_balanced,
                    expected_ply_count=int(ply_count),
                ),
                max_candidates=max_candidates,
            )
            result = optimize_stacking_sequence(
                material=material,
                load_case=load_case,
                config=config,
                baseline_sequence=(
                    baseline_sequence if len(baseline_sequence) == int(ply_count) else None
                ),
                allowables=allowables,
            )
        except Exception as exc:
            LOGGER.exception("Stacking optimization failed")
            st.error(f"Optimization failed: {exc}")
            render_footer()
            return
        st.session_state["optimization_result"] = result

        st.subheader("Optimized Stacking Sequence")
        st.code(str(result["best_sequence"]))
        render_optimization_impact_card(mode)
        metric_cols = st.columns(4)
        if mode == APP_MODE_SIMPLE:
            metric_cols[0].metric("Failure Load Capacity Factor", f"{result['best_lambda_cs']:.2f}")
            metric_cols[1].metric("Best Sequence", str(len(result["best_sequence"])))
            metric_cols[2].metric("Candidates Checked", result["candidates_evaluated"])
            metric_cols[3].metric("Improvement", f"{result['improvement_pct']:.2f}%")
        else:
            metric_cols[0].metric("λ_cs", f"{result['best_lambda_cs']:.2f}")
            metric_cols[1].metric("Critical ply", result["critical_ply"])
            metric_cols[2].metric("Failure mode", result["failure_mode"])
            metric_cols[3].metric("Candidates", result["candidates_evaluated"])

            st.subheader("Constraints")
            st.json(result["constraints"])
            st.caption(
                f"Search method: {result['search_method']} | theoretical search space: "
                f"{result['theoretical_search_space']:,} | runtime: {result['runtime_seconds']:.3f}s"
            )

        if result["baseline"]:
            baseline = result["baseline"]
            st.subheader("Baseline Comparison")
            comparison = pd.DataFrame(
                [
                    {"Case": "Baseline", "λ_cs": baseline["lambda_cs"]},
                    {"Case": "Best found", "λ_cs": result["best_lambda_cs"]},
                ]
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.metric("Improvement", f"{result['improvement_pct']:.2f}%")

        st.subheader("Laminate Visualization")
        st.text(format_laminate_sequence(result["best_sequence"]))

        if mode == APP_MODE_SIMPLE:
            st.subheader("What does this mean?")
            st.info(
                "Optimizer found layer arrangement with higher load-carrying capacity than baseline sequence under current constraints."
            )
        else:
            st.subheader("Top Candidates")
            top_rows = [
                {
                    "Rank": index + 1,
                    "Sequence": row["sequence"],
                    "λ_cs": row["lambda_cs"],
                    "Critical Ply": row["critical_ply"],
                    "Failure Mode": row["failure_mode"],
                }
                for index, row in enumerate(result["top_candidates"])
            ]
            st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

            st.subheader("Ply Failure Table")
            ply_rows = result["top_candidates"][0]["evaluation"]["ply_results"]
            ply_frame = pd.DataFrame(
                [
                    {
                        "Ply": row["ply_number"],
                        "Angle": row["angle_deg"],
                        "Failure Index @ λ=1": row["failure_index_at_lambda_1"],
                        "Failure Mode": row["failure_mode"],
                        "ε1": row["local_strains_at_lambda_1"]["epsilon_1"],
                        "ε2": row["local_strains_at_lambda_1"]["epsilon_2"],
                        "γ12": row["local_strains_at_lambda_1"]["gamma_12"],
                    }
                    for row in ply_rows
                ]
            )
            st.dataframe(ply_frame, use_container_width=True, hide_index=True)

        st.info(
            "Best sequence found within configured search space and constraints. "
            "No buckling optimization, no ANN sequence evaluation, no experimental certification."
        )

    render_footer()


def default_ai_clt_ann_input() -> dict[str, Any]:
    """Return deterministic ANN case for Step 12 demo defaults."""
    return {
        "fiber_type": "Aramid",
        "resin_type": "Phenolic",
        "density_g_cm3": 2.16,
        "layer_count": 16,
        "curing_temperature_c": 138.0,
        "fiber_volume_fraction": 0.56,
        "void_content_pct": 1.84,
    }


def parse_stacking_sequence_input(raw_sequence: str) -> tuple[list[int] | None, str | None]:
    """Parse comma/slash/space separated stacking-sequence input."""
    cleaned = raw_sequence.strip().replace("[", "").replace("]", "").replace("/", ",")
    if not cleaned:
        return None, "Stacking sequence is required."
    parts = [part.strip() for part in cleaned.replace(" ", ",").split(",") if part.strip()]
    try:
        sequence = [int(float(part.replace("+", ""))) for part in parts]
    except ValueError:
        return None, "Stacking sequence must contain numeric ply angles only."
    validation = validate_sequence(sequence, allowed_angles=DEFAULT_ALLOWED_ANGLES)
    if not validation.valid:
        return None, "; ".join(validation.reasons)
    return sequence, None


def build_ai_clt_comparison_case(
    ann_input: dict[str, Any],
    sequence: list[int],
    nx_value: float,
    ny_value: float,
    nxy_value: float,
    material_equivalence_verified: bool,
    material_equivalence_evidence: str,
    comparison_quantity: str,
) -> ComparisonCase:
    """Build Step 12 comparison case from supported UI fields."""
    case = load_tu_delft_demo_case()
    material = case["material"]
    return ComparisonCase(
        ann_input=ann_input,
        stacking_sequence=sequence,
        material_card=material,
        load_case=LaminateLoadCase(
            nx_n_per_m=nx_value,
            ny_n_per_m=ny_value,
            nxy_n_per_m=nxy_value,
        ),
        comparison_quantity=comparison_quantity,
        material_equivalence_verified=material_equivalence_verified,
        material_equivalence_evidence=(
            material_equivalence_evidence.strip() if material_equivalence_evidence.strip() else None
        ),
        allowables=case["allowables"],
        assumptions=[
            "CLT material card is parsed from preserved TU Delft/Zenodo source files.",
            "Default state intentionally does not assert equivalence between ANN dataset and TU Delft card.",
        ],
    )


def render_ai_clt_method_cards() -> None:
    """Render method explanation cards."""
    left, right = st.columns(2)
    with left:
        st.markdown("### AI / ANN" if not is_simple_mode() else "### Artificial Intelligence Prediction Model")
        st.markdown(
            """
            <div class="assistant-panel">
                Material and process parameters<br>
                ↓<br>
                ANN/MLP<br>
                ↓<br>
                Tensile strength prediction<br><br>
                <strong>Current model:</strong> ANN/MLP<br>
                <strong>Output:</strong> tensile_strength_mpa
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### CLT / Engineering" if not is_simple_mode() else "### Composite Laminate Structural Analysis")
        st.markdown(
            """
            <div class="assistant-panel">
                Material card + stacking sequence + load case<br>
                ↓<br>
                Classical Laminate Theory<br>
                ↓<br>
                Ply stress/strain<br>
                ↓<br>
                Failure evaluation<br>
                ↓<br>
                lambda_cs
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "CLT output lambda_cs is a dimensionless load factor and is not "
            "directly equivalent to tensile strength in MPa."
        )


def render_ai_clt_inputs() -> tuple[dict[str, Any], list[int] | None, dict[str, Any]]:
    """Render supported Step 12 inputs and return parsed values."""
    defaults = default_ai_clt_ann_input()
    case = load_tu_delft_demo_case()
    material = case["material"]
    default_load = case["load_case"]
    default_sequence = case["baseline_sequence"]

    st.subheader("Input Case")
    left, right = st.columns(2)
    with left:
        st.markdown("### ANN Case")
        fiber_options = ["Carbon", "Glass", "Aramid", "Basalt"]
        resin_options = ["Epoxy", "Phenolic", "Polyester", "Vinyl Ester"]
        fiber_type = st.selectbox(
            "Fiber Type",
            fiber_options,
            index=fiber_options.index(str(defaults["fiber_type"])),
        )
        resin_type = st.selectbox(
            "Resin Type",
            resin_options,
            index=resin_options.index(str(defaults["resin_type"])),
        )
        density = st.number_input("Density (g/cm³)", value=float(defaults["density_g_cm3"]), step=0.01, format="%.3f")
        layer_count = st.number_input("Layer Count", min_value=1, value=int(defaults["layer_count"]), step=1)
        curing_temperature = st.number_input(
            "Curing Temperature (°C)",
            value=float(defaults["curing_temperature_c"]),
            step=1.0,
            format="%.1f",
        )
        fiber_volume_fraction = st.number_input(
            "Fiber Volume Fraction",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["fiber_volume_fraction"]),
            step=0.01,
            format="%.3f",
        )
        void_content = st.number_input(
            "Void Content (%)",
            min_value=0.0,
            value=float(defaults["void_content_pct"]),
            step=0.1,
            format="%.2f",
        )

    with right:
        st.markdown("### CLT Case")
        st.selectbox(
            "Material Card",
            [f"{material.name} | TU Delft/Zenodo 10.5281/zenodo.15864524"],
            disabled=True,
        )
        material_cols = st.columns(5)
        material_cols[0].metric("E1", f"{material.e1_pa / 1e9:.2f} GPa")
        material_cols[1].metric("E2", f"{material.e2_pa / 1e9:.2f} GPa")
        material_cols[2].metric("G12", f"{material.g12_pa / 1e9:.2f} GPa")
        material_cols[3].metric("ν12", f"{material.nu12:.2f}")
        material_cols[4].metric("Ply t", f"{material.ply_thickness_m:.6f} m")
        sequence_text = st.text_input(
            "Stacking Sequence",
            value=", ".join(str(angle) for angle in default_sequence),
        )
        sequence, sequence_error = parse_stacking_sequence_input(sequence_text)
        if sequence_error:
            st.error(sequence_error)
        else:
            st.caption(f"Valid sequence with {len(sequence)} plies.")
        ply_thickness = st.number_input(
            "Ply Thickness (m)",
            value=float(material.ply_thickness_m),
            disabled=True,
            format="%.6f",
        )
        st.caption(f"Ply thickness is source-backed and locked at {ply_thickness:.6f} m.")
        st.markdown("#### Load Case")
        nx_value = st.number_input("Nxx (N/m)", value=float(default_load.nx_n_per_m), step=10.0, format="%.6f")
        ny_value = st.number_input("Nyy (N/m)", value=float(default_load.ny_n_per_m), step=10.0, format="%.6f")
        nxy_value = st.number_input("Nxy (N/m)", value=0.0, step=10.0, format="%.6f")

    st.subheader("Material Equivalence")
    material_equivalence_verified = st.checkbox(
        "I have verified ANN case and CLT material card represent the same physical material",
        value=False,
    )
    material_equivalence_evidence = st.text_input(
        "Material equivalence evidence",
        value="",
        placeholder="Example: same test coupon, same E1/E2/G12/nu12/thickness source",
    )
    comparison_quantity = st.selectbox(
        "Common Quantity",
        [STRESS_MODE, LOAD_CAPACITY_MODE],
        format_func=lambda value: {
            STRESS_MODE: "Equivalent laminate tensile stress (MPa)",
            LOAD_CAPACITY_MODE: "Tensile load capacity (N/m)",
        }[value],
    )

    ann_input = {
        "fiber_type": fiber_type,
        "resin_type": resin_type,
        "density_g_cm3": density,
        "layer_count": int(layer_count),
        "curing_temperature_c": curing_temperature,
        "fiber_volume_fraction": fiber_volume_fraction,
        "void_content_pct": void_content,
    }
    controls = {
        "nx_value": nx_value,
        "ny_value": ny_value,
        "nxy_value": nxy_value,
        "material_equivalence_verified": material_equivalence_verified,
        "material_equivalence_evidence": material_equivalence_evidence,
        "comparison_quantity": comparison_quantity,
        "sequence_error": sequence_error,
    }
    return ann_input, sequence, controls


def render_ai_clt_result(result: Any) -> None:
    """Render backend comparison result state."""
    mode = get_app_mode(st.session_state)
    st.subheader("Compatibility Status")
    compatibility = result.material_compatibility
    status_label = "Not Comparable"
    if compatibility["compatible"] and result.comparable:
        status_label = "Consistent"
        st.success("✓ Consistent")
    elif compatibility["compatible"] and not result.comparable:
        status_label = "Partial Agreement"
        st.warning("⚠ Partial Agreement")
    else:
        st.warning("✗ Not Comparable")
    st.write(compatibility["reason"])

    st.markdown("---")
    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "ANN tensile strength" if mode == APP_MODE_ENGINEERING else "Predicted Strength",
        f"{result.ann_value:,.2f} {result.ann_unit}",
    )
    metric_cols[1].metric(
        "CLT lambda_cs" if mode == APP_MODE_ENGINEERING else "Failure Load Capacity Factor",
        f"{result.clt_value:,.2f}" if result.clt_value is not None else "Unavailable",
    )
    metric_cols[2].metric("Compatibility Status", status_label)

    st.subheader("Comparison Result")
    if result.comparable:
        st.success("Valid comparison")
        result_cols = st.columns(4)
        result_cols[0].metric("AI result", f"{result.ann_common_value:,.2f} {result.common_unit}")
        result_cols[1].metric("CLT result", f"{result.clt_common_value:,.2f} {result.common_unit}")
        result_cols[2].metric("Common quantity", result.common_quantity or "Unavailable")
        result_cols[3].metric("Relative difference from CLT", f"{result.percentage_difference:.2f}%")
    else:
        st.info("Comparison unavailable")
        st.write(
            "The current ANN dataset cannot be mapped directly to the TU Delft CLT "
            "material card. A numerical comparison is therefore disabled."
        )
        st.write(result.reason)
        st.warning(
            "ANN tensile strength (MPa) and CLT lambda_cs (dimensionless) are "
            "different physical quantities and must not be compared numerically."
        )

    for warning in result.warnings:
        st.warning(warning)

    if mode == APP_MODE_ENGINEERING:
        with st.expander("Show calculation trace"):
            if result.calculation_trace:
                st.dataframe(pd.DataFrame(result.calculation_trace), use_container_width=True, hide_index=True)
            else:
                st.write("No calculation trace available for this input state.")


def render_clt_reference_validation() -> None:
    """Render independent TU Delft-vs-local CLT validation."""
    st.subheader("CLT Reference Validation")
    try:
        reference = validate_reference_case()
    except Exception as exc:
        LOGGER.exception("CLT reference validation failed")
        st.error(f"CLT reference validation unavailable: {exc}")
        return

    columns = st.columns(4)
    columns[0].metric("TU Delft reference λ_cs", f"{reference['reference_lambda_cs']:.2f}")
    columns[1].metric("Our CLT λ_cs", f"{reference['our_lambda_cs']:.2f}")
    columns[2].metric("Relative difference", f"{abs(reference['difference_pct']):.4f}%")
    columns[3].metric("Status", reference["validation_status"].upper())
    if reference["validation_status"] == "pass":
        st.success(
            "This validates the CLT implementation against the source reference case. "
            "It is NOT an AI-vs-CLT comparison."
        )
    else:
        st.warning("CLT reference validation did not pass documented tolerance.")


def render_clt_analysis() -> None:
    """Render standalone CLT analysis summary page."""
    mode = get_app_mode(st.session_state)
    st.markdown('<div class="main-title">CLT Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class="section-copy">
            {"Understand laminate structural analysis in plain language." if mode == APP_MODE_SIMPLE else "View validated Classical Laminate Theory material card, allowables, laminate assumptions, and reference-case verification."}
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "This page reports existing CLT implementation details only. It does not "
        "change equations, material card values, or optimizer logic."
    )
    try:
        case = load_tu_delft_demo_case()
    except Exception as exc:
        LOGGER.exception("Failed to load CLT demo case")
        st.error(f"CLT analysis unavailable: {exc}")
        render_footer()
        return

    material = case["material"]
    allowables = case["allowables"]
    load_case = case["load_case"]

    left, right = st.columns(2)
    with left:
        st.subheader("Material Card")
        material_rows = build_display_frame(
            [
                ("E1", f"{material.e1_pa / 1e9:.2f} GPa"),
                ("E2", f"{material.e2_pa / 1e9:.2f} GPa"),
                ("G12", f"{material.g12_pa / 1e9:.2f} GPa"),
                ("ν12", f"{material.nu12:.4f}"),
                ("Ply thickness", f"{material.ply_thickness_m:.6f} m"),
            ],
            ["Property", "Value"],
        )
        st.dataframe(material_rows, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Failure Allowables")
        allowable_rows = build_display_frame(
            [
                ("ε1", allowables.epsilon_1_allowable),
                ("ε2", allowables.epsilon_2_allowable),
                ("γ12", allowables.gamma_12_allowable),
                ("Default Nxx", f"{load_case.nx_n_per_m:.2f} N/m"),
                ("Default Nyy", f"{load_case.ny_n_per_m:.2f} N/m"),
            ],
            ["Field", "Value"],
        )
        st.dataframe(allowable_rows, use_container_width=True, hide_index=True)

    if mode == APP_MODE_SIMPLE:
        st.subheader("What does this mean?")
        st.info(
            "This module checks how composite laminate responds to load using structural analysis equations. It helps verify whether layup is mechanically reasonable before failure."
        )
    else:
        st.subheader("Implemented CLT Outputs")
        implemented_rows = pd.DataFrame(
            [
                ("Reduced stiffness matrix", "Q"),
                ("Transformed stiffness matrix", "Q̄"),
                ("Laminate extensional matrix", "A"),
                ("Laminate coupling matrix", "B"),
                ("Laminate bending matrix", "D"),
                ("Ply strain evaluation", "Implemented"),
                ("Ply stress evaluation", "Implemented"),
                ("Failure analysis", "Source-compatible strain allowable route"),
            ],
            columns=["Component", "Status"],
        )
        st.dataframe(implemented_rows, use_container_width=True, hide_index=True)
    render_clt_reference_validation()
    render_footer()


def render_ai_vs_clt_comparison() -> None:
    """Render Step 12 AI-vs-CLT comparison page."""
    mode = get_app_mode(st.session_state)
    st.markdown('<div class="main-title">AI vs CLT Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class="section-copy">
            {"See whether AI prediction can be supported by engineering physics for current case." if mode == APP_MODE_SIMPLE else "Compare data-driven strength prediction with mechanics-based laminate analysis when a physically compatible case is available."}
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "The ANN and CLT systems evaluate different quantities and material representations. "
        "A numerical comparison is performed only when material equivalence, loading, "
        "geometry, and units are explicitly compatible."
    )
    st.warning(
        "This comparison is a research/educational computational analysis. A numerical "
        "AI-vs-CLT comparison is shown only when the material, loading, geometry, and "
        "physical quantities are compatible. Results are not experimental certification."
    )

    render_ai_clt_method_cards()
    if mode == APP_MODE_ENGINEERING:
        render_term_expander(["ANN/MLP", "CLT", "Lambda_cs", "ABD Matrix", "Q Matrix"])
    try:
        ann_input, sequence, controls = render_ai_clt_inputs()
    except Exception as exc:
        LOGGER.exception("Failed to render AI-vs-CLT inputs")
        st.error(f"AI-vs-CLT page setup failed: {exc}")
        render_footer()
        return

    if controls["sequence_error"] or sequence is None:
        st.subheader("Comparison Result")
        st.error("Invalid input")
        st.write(controls["sequence_error"])
        render_clt_reference_validation()
        render_footer()
        return

    try:
        comparison_case = build_ai_clt_comparison_case(
            ann_input=ann_input,
            sequence=sequence,
            nx_value=controls["nx_value"],
            ny_value=controls["ny_value"],
            nxy_value=controls["nxy_value"],
            material_equivalence_verified=controls["material_equivalence_verified"],
            material_equivalence_evidence=controls["material_equivalence_evidence"],
            comparison_quantity=controls["comparison_quantity"],
        )
        ann_result = predict_strength(ann_input)
        result = compare_ai_and_clt(comparison_case, ann_result=ann_result)
    except PredictionInputError as exc:
        st.subheader("Comparison Result")
        st.error("Invalid input")
        st.write(str(exc))
        render_clt_reference_validation()
        render_footer()
        return
    except Exception as exc:
        LOGGER.exception("AI-vs-CLT comparison failed")
        st.subheader("Comparison Result")
        st.error(f"Comparison failed: {exc}")
        render_clt_reference_validation()
        render_footer()
        return

    render_ai_clt_result(result)
    render_clt_reference_validation()
    render_footer()


def render_strength_surface() -> None:
    """Render ANN strength response surface."""
    st.markdown('<div class="main-title">Composite Strength Response Surface</div>', unsafe_allow_html=True)
    st.caption("Real ANN inference over valid ML-ready dataset ranges. No synthetic target values.")
    try:
        figure, metadata = build_strength_response_surface(st.session_state.get("prediction_context"))
    except Exception as exc:
        LOGGER.exception("Strength response surface failed")
        st.error(f"Strength response surface failed: {type(exc).__name__}: {exc}")
        render_footer()
        return
    columns = st.columns(3)
    columns[0].metric("Minimum Predicted Strength", f"{metadata['min_predicted_strength_mpa']:.2f} MPa")
    columns[1].metric("Maximum Predicted Strength", f"{metadata['max_predicted_strength_mpa']:.2f} MPa")
    columns[2].metric("Grid Points", metadata["grid_points"])
    if not metadata["current_marker_available"]:
        st.info("Run Strength Prediction to add current user prediction marker.")
    st.plotly_chart(figure, use_container_width=True)
    render_engineering_interpretation("strength_surface", figure, metadata)
    st.caption(f"Dataset source: {metadata['dataset_path']}")
    render_footer()


def render_optimization_landscape() -> None:
    """Render optimizer search landscape."""
    st.markdown('<div class="main-title">Optimization Search Landscape</div>', unsafe_allow_html=True)
    st.caption("Real evaluated candidates from TU Delft source-backed optimizer.")
    try:
        figure, metadata = build_optimization_landscape()
    except Exception as exc:
        LOGGER.exception("Optimization landscape failed")
        st.error(f"Optimization landscape failed: {type(exc).__name__}: {exc}")
        render_footer()
        return
    columns = st.columns(4)
    columns[0].metric("Candidates", metadata["candidates_evaluated"])
    columns[1].metric("Baseline λ_cs", f"{metadata['baseline_lambda_cs']:.0f}")
    columns[2].metric("Optimized λ_cs", f"{metadata['optimized_lambda_cs']:.0f}")
    columns[3].metric("Improvement", f"+{metadata['improvement_pct']:.1f}%")
    st.plotly_chart(figure, use_container_width=True)
    render_engineering_interpretation("optimization_landscape", figure, metadata)
    st.caption(f"Source: {metadata['source']}")
    render_footer()


def render_ply_failure_map() -> None:
    """Render CLT ply failure distribution."""
    st.markdown('<div class="main-title">Ply Failure Distribution</div>', unsafe_allow_html=True)
    st.caption("CLT strain-allowable failure-index distribution across source-backed laminate plies.")
    try:
        figure, metadata = build_ply_failure_map()
    except Exception as exc:
        LOGGER.exception("Ply failure map failed")
        st.error(f"Ply failure map failed: {type(exc).__name__}: {exc}")
        render_footer()
        return
    columns = st.columns(3)
    columns[0].metric("Ply Count", metadata["ply_count"])
    columns[1].metric("Critical Ply", metadata["critical_ply"])
    columns[2].metric("Max Failure Index", f"{metadata['max_failure_index']:.6f}")
    st.plotly_chart(figure, use_container_width=True)
    render_engineering_interpretation("ply_failure_map", figure, metadata)
    st.caption(f"Source: {metadata['source']} | λ_cs={metadata['lambda_cs']:.2f}")
    render_footer()


def render_material_benchmark_3d() -> None:
    """Render 3D material performance benchmark."""
    st.markdown('<div class="main-title">Material Performance Benchmark</div>', unsafe_allow_html=True)
    st.caption("Uses project reference database plus ML-ready composite dataset aggregates. Missing materials are not fabricated.")
    try:
        figure, metadata = build_material_benchmark_3d()
    except Exception as exc:
        LOGGER.exception("Material benchmark 3D failed")
        st.error(f"Material benchmark 3D failed: {type(exc).__name__}: {exc}")
        render_footer()
        return
    columns = st.columns(2)
    columns[0].metric("Materials Plotted", metadata["row_count"])
    columns[1].metric("Missing Requested Materials", len(metadata["missing_requested_materials"]))
    if metadata["missing_requested_materials"]:
        st.warning("Missing from project reference data: " + ", ".join(metadata["missing_requested_materials"]))
    st.plotly_chart(figure, use_container_width=True)
    render_engineering_interpretation("material_benchmark", figure, metadata)
    st.caption(f"Reference DB: {metadata['reference_database']}")
    st.caption(f"Composite source: {metadata['training_dataset']}")
    render_footer()


def render_engineering_interpretation(
    graph_name: str,
    figure: Any,
    metadata: dict[str, Any],
) -> None:
    """Render deterministic chart interpretation derived from visible chart data."""
    label = st.radio(
        "Explanation Detail",
        ["Simple Explanation", "Engineering Explanation"],
        horizontal=True,
        key=f"{graph_name}_explanation_detail",
    )
    mode = SIMPLE_EXPLANATION if label == "Simple Explanation" else ENGINEERING_EXPLANATION
    interpretation = build_engineering_interpretation(graph_name, figure, metadata, mode)
    sections = interpretation["sections"]
    blocks = [
        "<div style='border:1px solid rgba(148,163,184,.35);border-radius:14px;padding:18px;margin-top:16px;background:rgba(15,23,42,.35);'>",
        "<h3 style='margin-top:0;'>Engineering Interpretation</h3>",
    ]
    for section in INTERPRETATION_SECTIONS:
        blocks.append(f"<h4>{html.escape(section)}</h4>")
        blocks.append(f"<p>{html.escape(sections[section])}</p>")
    blocks.append(f"<p><strong>Data source:</strong> {html.escape(interpretation['data_source'])}</p>")
    blocks.append("</div>")
    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_validation_summary(validation: Any) -> None:
    """Render validation metrics and issue table."""
    if validation is None:
        return

    st.subheader("Validation Summary")
    st.metric("Quality Score", f"{validation.quality_score}/100")
    st.dataframe(
        build_display_frame(list(validation.metrics.items()), ["Metric", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

    if validation.issues:
        issue_rows = [issue_to_dict(issue) for issue in validation.issues]
        st.dataframe(pd.DataFrame(issue_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No validation issues found.")

    mappings = st.session_state.get("column_mappings")
    if mappings:
        with st.expander("Column mappings log"):
            mapping_frame = pd.DataFrame(
                mappings.items(),
                columns=["Source Column", "Canonical Column"],
            )
            st.dataframe(mapping_frame, use_container_width=True, hide_index=True)


def render_gemini_status(
    service: GeminiService,
    result: GeminiResult | None = None,
) -> None:
    """Render Gemini status without over-claiming availability."""
    if not service.is_configured:
        st.info("Gemini status: Not Configured")
    elif result is None:
        st.info("Gemini status: Configured")
    elif result.success:
        if result.fallback_used:
            st.success("Gemini: Response generated using fallback model")
        else:
            st.success("Gemini: Response generated")
        if result.model_used:
            st.caption(f"Model used: {result.model_used}")
    else:
        st.warning("Gemini: Currently unavailable")


def render_gemini_failure(message: str, result: GeminiResult) -> None:
    """Render Gemini failure safely without fake fallback content."""
    st.warning("Assistant unavailable. Core CompositeAI features remain functional.")
    if result.error_message:
        st.caption(result.error_message)
    if result.technical_details:
        with st.expander("Technical details"):
            st.write(result.technical_details)
    if result.attempted_models:
        with st.expander("Models attempted"):
            st.write("\n".join(result.attempted_models))


def render_gemini_text_response(title: str, text: str, caption: str) -> None:
    """Render validated Gemini text in a dedicated section."""
    st.markdown(f"## {title}")
    st.markdown(text.strip())
    st.caption(caption)


def format_metric_number(value: Any, digits: int, suffix: str = "") -> str:
    """Format metric number or return unavailable marker."""
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "Unavailable"


def get_dataset_context() -> dict[str, Any]:
    """Build deterministic dataset context for Gemini interpretation."""
    data = st.session_state.get("processed_dataset")
    profile = st.session_state.get("dataset_profile", {})
    validation = st.session_state.get("validation_result")
    metrics = getattr(validation, "metrics", {}) or {}
    target = "tensile_strength_mpa" if isinstance(data, pd.DataFrame) and (
        "tensile_strength_mpa" in data.columns
    ) else "Not selected"
    rows = metrics.get("rows")
    columns = metrics.get("columns")
    if isinstance(data, pd.DataFrame):
        rows = rows if rows is not None else int(data.shape[0])
        columns = columns if columns is not None else int(data.shape[1])
        feature_names = [column for column in data.columns if column != target]
    else:
        feature_names = []

    issues = [issue_to_dict(issue) for issue in getattr(validation, "issues", [])]
    validation_status = "WARN" if issues else "PASS"
    return {
        "dataset": {
            "rows": rows,
            "columns": columns,
            "features": feature_names,
            "feature_count": len(feature_names),
            "target": target,
        },
        "validation": {
            "status": validation_status,
            "quality_score": getattr(validation, "quality_score", None),
            "metrics": metrics,
            "issues": issues,
        },
        "profile": profile,
    }


def render_dataset_context_cards(context: dict[str, Any]) -> None:
    """Render deterministic dataset metrics for Gemini page."""
    dataset = context["dataset"]
    validation = context["validation"]
    metrics = validation["metrics"]
    cards = [
        ("Dataset", f"{dataset.get('rows', 'Unavailable'):,} rows" if dataset.get("rows") is not None else "Unavailable"),
        ("Features", dataset.get("feature_count", "Unavailable")),
        ("Target", dataset.get("target", "Unavailable")),
        ("Missing Values", metrics.get("missing_values", "Unavailable")),
        ("Duplicates", metrics.get("duplicate_rows", "Unavailable")),
        ("Validation Status", validation.get("status", "Unavailable")),
        ("Quality Score", validation.get("quality_score", "Unavailable")),
    ]
    columns = st.columns(4)
    for index, (label, value) in enumerate(cards):
        columns[index % 4].metric(label, value)


def render_dataset_gemini_assistant() -> None:
    """Render Gemini assistant for dataset analysis only."""
    st.subheader("Gemini Dataset Analysis Assistant")
    st.caption("AI-assisted interpretation of dataset quality, structure, and ML readiness.")
    st.caption("Read-only assistant. It never edits or fabricates dataset values.")

    context = get_dataset_context()
    render_dataset_context_cards(context)

    st.markdown("## Dataset Quality Summary")
    validation = context["validation"]
    st.dataframe(
        build_display_frame(list(validation["metrics"].items()), ["Metric", "Value"]),
        use_container_width=True,
        hide_index=True,
    )
    if validation["issues"]:
        st.dataframe(pd.DataFrame(validation["issues"]), use_container_width=True, hide_index=True)
    else:
        st.success("No deterministic validation issues found.")

    service = GeminiService()
    render_gemini_status(service)
    prompt = st.text_area(
        "Dataset question for Gemini",
        value=(
            "Summarize this dataset's ML readiness using only the displayed "
            "deterministic validation results. Recommend preprocessing actions, "
            "identify risks, and do not modify, invent, or reinterpret numerical "
            "dataset values."
        ),
        height=140,
        key="dataset_gemini_prompt",
    )

    if st.button("Summarize Dataset Quality with Gemini"):
        if not service.is_configured:
            render_gemini_status(service)
            render_gemini_failure(
                "Gemini dataset analysis is currently unavailable.",
                service.generate_text(""),
            )
            return

        with st.spinner("Analyzing dataset quality..."):
            result = service.dataset_interpretation(prompt, context)
            render_gemini_status(service, result)
            if not result.success:
                render_gemini_failure(
                    "Gemini dataset analysis is currently unavailable.",
                    result,
                )
                return
        render_gemini_text_response(
            title="Gemini Dataset Interpretation",
            text=result.text or "",
            caption=(
                "Gemini is interpretation only. Dataset values and validation "
                "metrics come from deterministic CompositeAI pipeline outputs."
            ),
        )


def get_prediction_context() -> dict[str, object]:
    """Return latest real prediction context for assistant prompts."""
    context = st.session_state.get("prediction_context")
    if isinstance(context, dict):
        return context
    return {
        "predicted_tensile_strength_mpa": None,
        "model_name": "ANN/MLP",
        "validation_metrics": {},
        "warnings": [],
        "notes": "No prediction has been generated in this session.",
    }


def render_gemini_assistant() -> None:
    """Render Gemini engineering assistant page."""
    st.markdown('<div class="main-title">Gemini Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="section-copy">
            AI-assisted interpretation of model results and composite engineering context.
        </p>
        """,
        unsafe_allow_html=True,
    )

    context = get_prediction_context()
    st.info(
        "ANN/MLP → actual tensile-strength prediction. Gemini → optional "
        "explanation and interpretation only."
    )
    st.markdown("## Model Result")
    metrics = context.get("validation_metrics", {})
    if get_app_mode(st.session_state) == APP_MODE_SIMPLE:
        metric_columns = st.columns(2)
        metric_columns[0].metric("Prediction Method", "Artificial Intelligence Prediction Model")
        metric_columns[1].metric(
            "Predicted Strength",
            format_metric_number(
                context.get("predicted_tensile_strength_mpa"),
                2,
                " MPa",
            ),
        )
    else:
        metric_columns = st.columns(5)
        metric_columns[0].metric("Model", context.get("model_name", "ANN/MLP"))
        metric_columns[1].metric(
            "Predicted Tensile Strength",
            format_metric_number(
                context.get("predicted_tensile_strength_mpa"),
                2,
                " MPa",
            ),
        )
        metric_columns[2].metric(
            "R²",
            format_metric_number(metrics.get("r2") if isinstance(metrics, dict) else None, 4),
        )
        metric_columns[3].metric(
            "MAE",
            format_metric_number(
                metrics.get("mae") if isinstance(metrics, dict) else None,
                2,
                " MPa",
            ),
        )
        metric_columns[4].metric(
            "RMSE",
            format_metric_number(
                metrics.get("rmse") if isinstance(metrics, dict) else None,
                2,
                " MPa",
            ),
        )
    if context.get("notes"):
        st.caption(str(context["notes"]))

    service = GeminiService()
    render_gemini_status(service)
    st.markdown("## Gemini Interpretation")

    prompt = st.text_area(
        "Ask Gemini",
        value=(
            "Explain this prediction in simple engineering terms. Discuss what "
            "the predicted tensile strength means, how the model metrics should "
            "be interpreted, what the model does and does not capture, and the "
            "limitations of using this prediction for an aerospace composite "
            "laminate. Do not invent material properties or engineering certification."
        ),
        height=150,
    )

    if st.button("Generate Engineering Guidance", type="primary"):
        if not service.is_configured:
            render_gemini_status(service)
            render_gemini_failure(
                "Assistant unavailable. Core CompositeAI features remain functional.",
                service.generate_text(""),
            )
            return

        with st.spinner("Generating Gemini response..."):
            result = service.engineering_guidance(
                user_prompt=prompt,
                prediction_context=context,
            )
            render_gemini_status(service, result)
            if not result.success:
                render_gemini_failure(
                    "Assistant unavailable. Core CompositeAI features remain functional.",
                    result,
                )
                return
        render_gemini_text_response(
            title="Gemini Engineering Guidance",
            text=result.text or "",
            caption=(
                "Generated by Gemini from displayed prediction context only. "
                "Underlying prediction and metrics come from validated CompositeAI artifacts."
            ),
        )


def render_placeholder_page(page_name: str) -> None:
    """Render placeholder page for future development."""
    st.markdown(f'<div class="main-title">{page_name}</div>', unsafe_allow_html=True)
    st.info(
        "This module is staged for demo navigation. Validated project pages remain "
        "available from the sidebar."
    )
    render_footer()


def render_report_generator() -> None:
    """Render final project report generator."""
    st.markdown('<div class="main-title">Report Generator</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="section-copy">
            Generate a complete technical report from the validated CompositeAI results.
        </p>
        """,
        unsafe_allow_html=True,
    )

    mode = get_app_mode(st.session_state)
    summary = executive_summary(mode)
    st.subheader("Executive Summary")
    st.info(summary["what_it_does"])
    if mode == APP_MODE_SIMPLE:
        st.caption("Simple Mode -> executive report output.")
    else:
        st.caption("Engineering Mode -> technical report output.")

    st.subheader("Project Details")
    left, right = st.columns(2)
    with left:
        student_1 = st.text_input("Student 1")
        student_2 = st.text_input("Student 2")
        student_3 = st.text_input("Student 3")
    with right:
        guide = st.text_input("Guide")
        department = st.text_input("Department")
        college = st.text_input("College / University")

    st.subheader("Artifact Check")
    artifact_rows = [
        {
            "Artifact": relative_path,
            "Status": "Available" if (PROJECT_ROOT / relative_path).exists() else "Unavailable",
        }
        for relative_path in ARTIFACT_PATHS.values()
    ]
    st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)
    if any(row["Status"] == "Unavailable" for row in artifact_rows):
        st.warning(
            "Missing artifacts will be labelled as data unavailable in the generated report."
        )

    st.subheader("Report Contents")
    if mode == APP_MODE_SIMPLE:
        selected_sections = [
            "Executive Summary",
            "Prediction Results",
            "Composite vs Aerospace Metals",
            "Engineering Visualizations",
            "Conclusions",
        ]
        st.dataframe(
            pd.DataFrame({"Section": selected_sections}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        selected_sections = []
        checkbox_columns = st.columns(3)
        for index, section in enumerate(SECTION_LABELS):
            with checkbox_columns[index % 3]:
                if st.checkbox(section, value=True, key=f"report_section_{section}"):
                    selected_sections.append(section)

    if st.button("Generate Final Report", type="primary"):
        if not selected_sections:
            st.error("Select at least one report section.")
            render_footer()
            return
        details = ProjectDetails(
            student_1=student_1,
            student_2=student_2,
            student_3=student_3,
            guide=guide,
            department=department,
            college=college,
        )
        try:
            file_name, pdf_bytes = build_report_download_payload(
                project_details=details,
                selected_sections=selected_sections,
            )
            output_path = save_final_report(pdf_bytes)
            html_file_name, html_text = build_report_html_payload()
            csv_file_name, csv_bytes = build_report_csv_payload()
            html_output = save_report_text_artifact(html_text, html_file_name)
            csv_output = save_report_text_artifact(csv_bytes, csv_file_name)
        except Exception as exc:
            LOGGER.exception("Report generation failed")
            st.error(f"Report generation failed: {exc}")
            render_footer()
            return

        st.success("Report generated successfully.")
        st.session_state["report_data"] = {
            "pdf_path": str(output_path),
            "html_path": str(html_output),
            "csv_path": str(csv_output),
            "selected_sections": selected_sections,
        }
        st.metric("PDF Size", f"{len(pdf_bytes) / 1024:.1f} KB")
        st.caption(f"Saved copy: {output_path}")
        st.caption(f"HTML copy: {html_output}")
        st.caption(f"CSV copy: {csv_output}")
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf",
        )
        st.download_button(
            label="Download HTML Report",
            data=html_text,
            file_name=html_file_name,
            mime="text/html",
        )
        st.download_button(
            label="Download CSV Report",
            data=csv_bytes,
            file_name=csv_file_name,
            mime="text/csv",
        )

    render_footer()


def render_about() -> None:
    """Render dual-audience About page."""
    mode = get_app_mode(st.session_state)
    summary = executive_summary(mode)
    st.markdown('<div class="main-title">About Project</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class="section-copy">
            {summary["headline"]}
        </p>
        """,
        unsafe_allow_html=True,
    )

    if mode == APP_MODE_SIMPLE:
        simple_sections = [
            ("Project Overview", summary["what_it_does"]),
            ("Why It Matters", summary["why_it_matters"]),
            ("Artificial Intelligence Role", "Artificial Intelligence Prediction Model estimates tensile strength from material and process inputs."),
            ("Engineering Role", "Composite Laminate Structural Analysis checks whether laminate behavior makes engineering sense under defined loading."),
            ("Project Workflow", "Dataset -> Prediction -> Structural Analysis -> Optimization -> Benchmark -> Report"),
        ]
        for title, copy in simple_sections:
            st.header(title)
            st.write(copy)
        st.subheader("What does this mean?")
        st.info(
            "CompositeAI is designed to help first-time users understand composite-strength prediction without reading raw equations or validation tables."
        )
        render_footer()
        return

    sections = [
        ("1. Project Overview", summary["what_it_does"]),
        ("2. Problem Statement", "Composite laminate design is difficult because data-driven prediction, laminate mechanics, material benchmarking, and reporting are usually split across separate tools."),
        ("3. Objectives", "Deliver fast tensile-strength prediction, physics verification, stacking optimization, benchmark comparison, and report generation in one traceable system."),
        ("4. Dataset Information", "Locked ML dataset contains 10,000 rows, 7 input features, and tensile strength target. Benchmark dataset contains detected material baselines for comparison."),
        ("5. Artificial Intelligence Pipeline", summary["ai_role"]),
        ("6. Physics Verification Pipeline", summary["physics_role"]),
        ("7. Stacking Optimization Workflow", "Validated CLT-based optimizer searches symmetric and balanced laminate candidates and maximizes failure load factor λcs."),
        ("8. Material Benchmarking", "Benchmark page compares predicted composite strength against a separate aerospace metals reference database. Reference values are advisory only and are not part of the ML training dataset."),
        ("9. Technologies Used", "Python, Streamlit, Scikit-Learn, Pandas, NumPy, Plotly, Joblib, ReportLab."),
        ("10. Model Performance", "ANN/MLP -> R² 0.9952, MAE 32.51 MPa, RMSE 43.38 MPa."),
        ("11. Limitations", "No experimental validation, no certification-grade verification, no sequence-aware ANN, no buckling optimization."),
        ("12. Future Scope", "Sequence-aware ML, broader material cards, buckling optimization, physics-informed learning, experimental validation, digital twin integration."),
        ("13. Abbreviations and Definitions", "ANN, MLP, CLT, ABD, MPa, R², MAE, RMSE, Q Matrix, Qbar Matrix, Lambda_cs are defined below."),
    ]
    for title, copy in sections:
        st.header(title)
        st.write(copy)

    st.subheader("Project Workflow")
    render_workflow_visual(mode)

    st.subheader("Abbreviations and Definitions")
    glossary_rows = [
        ("ANN", "Artificial Neural Network"),
        ("MLP", "Multi-Layer Perceptron"),
        ("CLT", "Classical Laminate Theory"),
        ("ABD", "Laminate structural stiffness matrix"),
        ("MPa", "Megapascal"),
        ("R²", "Prediction reliability score"),
        ("MAE", "Mean Absolute Error"),
        ("RMSE", "Root Mean Squared Error"),
        ("Q Matrix", "Material stiffness matrix"),
        ("Qbar Matrix", "Transformed ply stiffness matrix"),
        ("Lambda_cs", "Failure load factor"),
    ]
    st.dataframe(pd.DataFrame(glossary_rows, columns=["Term", "Meaning"]), use_container_width=True, hide_index=True)
    render_term_expander(["ANN/MLP", "CLT", "Tensile Strength", "Stacking Sequence", "Lambda_cs", "Q Matrix", "ABD Matrix", "Residual Error", "R² Score"])
    render_footer()


def find_dataset_files(source_dir: Path) -> list[Path]:
    """Find supported dataset files below a source directory."""
    if not source_dir.exists():
        return []
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def select_dataset_version() -> pd.DataFrame | None:
    """Select and load processed dataset version or current session data."""
    versions = list_processed_versions()
    current_data = get_current_dataset()
    choices = ["Current session dataset"] if current_data is not None else []
    choices.extend([path.name for path in versions])

    if not choices:
        return None

    selected = st.selectbox("Dataset version", choices)
    if selected == "Current session dataset":
        return get_current_dataset()

    selected_path = next(path for path in versions if path.name == selected)
    try:
        data = pd.read_csv(selected_path)
        st.session_state["processed_dataset"] = data
        return data
    except Exception as exc:
        LOGGER.exception("Failed to load processed dataset")
        st.error(f"Failed to load processed dataset: {exc}")
        return None


def get_current_dataset() -> pd.DataFrame | None:
    """Return processed dataset held in Streamlit session state."""
    data = st.session_state.get("processed_dataset")
    if isinstance(data, pd.DataFrame):
        return data
    return load_local_dataset_fallback()


def load_local_dataset_fallback(
    dataset_path: Path = LOCAL_DATASET_FALLBACK_PATH,
) -> pd.DataFrame | None:
    """Load bundled ML-ready dataset when no processed session dataset exists."""
    if not dataset_path.exists():
        return None
    try:
        data = pd.read_csv(dataset_path)
    except Exception as exc:
        LOGGER.exception("Failed to load local dataset fallback")
        st.warning(f"Local dataset load failed: {exc}")
        return None
    st.session_state["processed_dataset"] = data
    st.session_state["version_path"] = dataset_path
    return data


def issue_to_dict(issue: ValidationIssue) -> dict[str, Any]:
    """Convert validation issue dataclass to serializable dict."""
    return {
        "category": issue.category,
        "column": issue.column,
        "severity": issue.severity,
        "message": issue.message,
        "count": issue.count,
    }


def render_download_button(data: pd.DataFrame, file_name: str) -> None:
    """Render CSV download button for processed dataset."""
    st.download_button(
        "Download Processed Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )


def build_display_frame(rows: list[tuple[Any, Any]], columns: list[str]) -> pd.DataFrame:
    """Return string-safe dataframe for small Streamlit tables."""
    return pd.DataFrame(
        [(str(left), str(right)) for left, right in rows],
        columns=columns,
    )


def render_footer() -> None:
    """Render application footer."""
    st.markdown(
        '<div class="footer">CompositeAI | Dataset-ready aerospace laminate prototype</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run Streamlit app."""
    configure_page()
    initialize_session_state(st.session_state)
    selected_page = render_sidebar()
    renderers = {
        "Dashboard": render_dashboard,
        "About Project": render_about,
        "Workflow": render_workflow_page,
        "Dataset Explorer": render_dataset_explorer,
        "Dataset Profile": render_dataset_profile,
        "EDA": render_eda_page,
        "Feature Engineering": render_feature_engineering_page,
        "Preprocessing": render_preprocessing_page,
        "Model Performance": render_model_performance,
        "Strength Prediction": render_strength_prediction,
        "CLT Analysis": render_clt_analysis,
        "Stacking Optimizer": render_stacking_optimizer,
        "AI vs CLT Comparison": render_ai_vs_clt_comparison,
        "Composite vs Aerospace Metals": render_engineering_benchmark,
        "Strength Surface": render_strength_surface,
        "Optimization Landscape": render_optimization_landscape,
        "Ply Failure Map": render_ply_failure_map,
        "Material Benchmark": render_material_benchmark_3d,
        "Report Generator": render_report_generator,
        "Gemini Assistant": render_gemini_assistant,
    }
    validation = validate_navigation(renderers)
    if not validation["valid"]:
        st.error(
            "Navigation configuration invalid: "
            f"duplicates={validation['duplicates']}, "
            f"missing_renderers={validation['missing_renderers']}"
        )
        render_footer()
        return
    render_page(selected_page, renderers)


if __name__ == "__main__":
    main()
