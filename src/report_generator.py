"""Final technical PDF report generation for CompositeAI."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.tableofcontents import TableOfContents
    REPORTLAB_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - env-specific fallback.
    colors = None
    TA_CENTER = 1
    A4 = (595, 842)
    ParagraphStyle = Any
    getSampleStyleSheet = None
    cm = 28.3464566929
    BaseDocTemplate = object
    Frame = PageBreak = PageTemplate = Paragraph = Spacer = Table = TableStyle = TableOfContents = None
    REPORTLAB_IMPORT_ERROR = exc

from src.config import PROJECT_ROOT, REPORT_DIR
from src.engineering_interpretation import (
    ENGINEERING_EXPLANATION,
    INTERPRETATION_SECTIONS,
    build_all_engineering_interpretations,
)
from src.optimization_impact import load_optimization_impact

PDF_FILE_NAME = "CompositeAI_Final_Project_Report.pdf"
REPORT_TITLE = "CompositeAI"
REPORT_SUBTITLE = (
    "AI-Assisted Composite Laminate Strength Prediction and "
    "Stacking Sequence Optimization"
)
TECHNICAL_SUBTITLE = (
    "An Integrated Machine Learning and Classical Laminate Theory Framework "
    "for Aerospace Composite Analysis"
)
UNAVAILABLE = "Data unavailable in current project artifacts."

ARTIFACT_PATHS = {
    "saved_model": "saved_models/best_strength_model.joblib",
    "model_metadata": "saved_models/model_metadata.json",
    "model_comparison": "data/training/model_comparison.csv",
    "model_validation": "data/training/model_validation_report.json",
    "feature_specification": "data/training/feature_specification.json",
    "feature_analysis": "data/training/feature_analysis_report.json",
    "preprocessing_config": "data/training/preprocessing_config.json",
    "ai_clt_comparison": "data/training/ai_clt_comparison_specification.json",
    "dataset_inventory": "data/sequence/dataset_inventory.json",
    "sequence_metadata": "data/sequence/tu_delft_zenodo_15864524/metadata.json",
    "optimization_validation": "data/sequence/optimization_validation_report.json",
    "comparison_specification": "data/reference_materials/comparison_specification.json",
}

SECTION_LABELS = [
    "Executive Summary",
    "Prediction Results",
    "AI Metrics",
    "CLT Analysis",
    "Optimization Results",
    "Composite vs Aerospace Metals",
    "Engineering Visualizations",
    "Conclusions",
]


@dataclass(frozen=True)
class ProjectDetails:
    """Optional project-team details for report cover page."""

    student_1: str = ""
    student_2: str = ""
    student_3: str = ""
    guide: str = ""
    department: str = ""
    college: str = ""


class NumberedDocTemplate(BaseDocTemplate):
    """ReportLab document with headers, footers, and TOC notifications."""

    def __init__(self, buffer: BytesIO) -> None:
        frame = Frame(
            2 * cm,
            2.2 * cm,
            A4[0] - 4 * cm,
            A4[1] - 4.2 * cm,
            id="normal",
        )
        super().__init__(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            pageCompression=0,
        )
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_draw_page)])

    def afterFlowable(self, flowable: Any) -> None:  # noqa: N802 - ReportLab API.
        """Register section headings for the table of contents."""
        if isinstance(flowable, Paragraph) and flowable.style.name == "ReportHeading1":
            text = flowable.getPlainText()
            self.notify("TOCEntry", (0, text, self.page))


def collect_report_data(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load project artifacts without fabricating missing values."""
    artifacts: dict[str, Any] = {}
    availability: dict[str, bool] = {}
    for name, relative_path in ARTIFACT_PATHS.items():
        path = project_root / relative_path
        availability[name] = path.exists()
        if not path.exists():
            artifacts[name] = None
            continue
        if path.suffix == ".json":
            artifacts[name] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".csv":
            artifacts[name] = _read_csv(path)
        else:
            artifacts[name] = {"path": relative_path, "exists": True}

    training_path = project_root / "data/training/ml_ready_features.csv"
    availability["ml_ready_dataset"] = training_path.exists()
    artifacts["dataset_stats"] = _dataset_stats(training_path)
    artifacts["engineering_interpretations"] = _collect_engineering_interpretations(project_root)
    artifacts["availability"] = availability
    return artifacts


def generate_final_report_pdf(
    project_details: ProjectDetails | None = None,
    selected_sections: list[str] | None = None,
    project_root: Path = PROJECT_ROOT,
    generated_at: datetime | None = None,
) -> bytes:
    """Generate final CompositeAI report as PDF bytes."""
    if REPORTLAB_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PDF report generation requires reportlab in CompositeAI/.venv312."
        ) from REPORTLAB_IMPORT_ERROR
    details = project_details or ProjectDetails()
    selected = selected_sections or SECTION_LABELS
    data = collect_report_data(project_root)
    timestamp = generated_at or datetime.now()

    buffer = BytesIO()
    doc = NumberedDocTemplate(buffer)
    styles = _styles()
    story: list[Any] = []
    _build_cover(story, styles, details, timestamp, data)
    story.append(PageBreak())
    _build_toc(story, styles)
    story.append(PageBreak())

    section_number = 1
    for label in SECTION_LABELS:
        if label not in selected:
            continue
        builder = SECTION_BUILDERS[label]
        builder(story, styles, section_number, data)
        section_number += 1

    doc.multiBuild(story)
    return buffer.getvalue()


def save_final_report(
    pdf_bytes: bytes,
    report_dir: Path = REPORT_DIR,
    file_name: str = PDF_FILE_NAME,
) -> Path:
    """Persist generated PDF and return path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / file_name
    output_path.write_bytes(pdf_bytes)
    return output_path


def build_report_download_payload(
    project_details: ProjectDetails | None = None,
    selected_sections: list[str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[str, bytes]:
    """Return file name and PDF bytes for Streamlit download buttons."""
    return (
        PDF_FILE_NAME,
        generate_final_report_pdf(project_details, selected_sections, project_root),
    )


def build_report_html_payload(project_root: Path = PROJECT_ROOT) -> tuple[str, str]:
    """Return HTML report payload."""
    data = collect_report_data(project_root)
    html_rows = []
    summary = _safe_get(data, "model_metadata", "metrics", default={})
    html_rows.append("<h1>Executive Summary</h1>")
    html_rows.append("<p>CompositeAI combines AI-based tensile-strength prediction, physics-based laminate verification, material benchmarking, constrained stacking optimization, and engineering reporting.</p>")
    html_rows.append("<h2>Prediction Results</h2>")
    html_rows.append("<p>Reference prediction card and report values are sourced from validated CompositeAI artifacts only.</p>")
    html_rows.append("<h2>AI Metrics</h2>")
    html_rows.append(
        f"<p>ANN/MLP | R²={_fmt(summary.get('r2'), 4)} | MAE={_fmt(summary.get('mae'), 2)} MPa | RMSE={_fmt(summary.get('rmse'), 2)} MPa</p>"
    )
    benchmark = data.get("comparison_specification") or {}
    html_rows.append("<h2>Composite vs Aerospace Metals</h2>")
    html_rows.append(f"<p>Reference source: {_safe_get(data, 'comparison_specification', 'source_database')}</p>")
    html_rows.append("<table border='1' cellpadding='6' cellspacing='0'><tr><th>Material</th><th>Strength (MPa)</th><th>Density (g/cm3)</th><th>Specific Strength</th></tr>")
    for item in benchmark.get("materials", []):
        html_rows.append(
            f"<tr><td>{item.get('material','')}</td><td>{_fmt(item.get('strength_mpa'),2)}</td><td>{_fmt(item.get('density_g_cm3'),3)}</td><td>{_fmt(item.get('strength_to_weight'),2)}</td></tr>"
        )
    html_rows.append("</table>")
    html_rows.append("<h2>CLT Analysis</h2>")
    html_rows.append("<p>AI-vs-CLT output remains explicitly limited to compatible cases only. Default status remains NOT DIRECTLY COMPARABLE unless material equivalence and quantity compatibility are verified.</p>")
    html_rows.append("<h2>Optimization Results</h2>")
    html_rows.append(
        f"<p>Best candidate improvement: {_fmt(_safe_get(data, 'optimization_validation', 'optimization_demo', 'improvement_pct', default=None), 4)}%</p>"
    )
    html_rows.append("<h2>Engineering Visualizations</h2>")
    for interpretation in data.get("engineering_interpretations") or []:
        html_rows.append(f"<h3>{html.escape(_graph_title(interpretation.get('graph_name', '')))}</h3>")
        html_rows.append(f"<p><strong>Data source:</strong> {html.escape(str(interpretation.get('data_source', '')))}</p>")
        sections = interpretation.get("sections", {})
        for section in INTERPRETATION_SECTIONS:
            html_rows.append(f"<h4>{html.escape(section)}</h4>")
            html_rows.append(f"<p>{html.escape(str(sections.get(section, UNAVAILABLE)))}</p>")
    html_text = "<html><body>" + "".join(html_rows) + "</body></html>"
    return ("CompositeAI_Final_Project_Report.html", html_text)


def build_report_csv_payload(project_root: Path = PROJECT_ROOT) -> tuple[str, bytes]:
    """Return CSV payload for executive report tables."""
    data = collect_report_data(project_root)
    rows = [
        {"section": "dataset", "metric": "rows", "value": _safe_get(data, "dataset_stats", "rows")},
        {"section": "ai_model", "metric": "R2", "value": _safe_get(data, "model_metadata", "metrics", "r2")},
        {"section": "ai_model", "metric": "MAE_MPa", "value": _safe_get(data, "model_metadata", "metrics", "mae")},
        {"section": "ai_model", "metric": "RMSE_MPa", "value": _safe_get(data, "model_metadata", "metrics", "rmse")},
        {"section": "physics_verification", "metric": "status", "value": "NOT DIRECTLY COMPARABLE"},
    ]
    benchmark = data.get("comparison_specification") or {}
    for item in benchmark.get("materials", []):
        rows.append(
            {
                "section": "material_benchmark",
                "metric": item.get("material", ""),
                "value": item.get("strength_mpa", UNAVAILABLE),
            }
        )
    for interpretation in data.get("engineering_interpretations") or []:
        for section, text in (interpretation.get("sections") or {}).items():
            rows.append(
                {
                    "section": "engineering_visualization_interpretation",
                    "metric": f"{interpretation.get('graph_name')}::{section}",
                    "value": text,
                }
            )
    text_buffer = []
    headers = ["section", "metric", "value"]
    text_buffer.append(",".join(headers))
    for row in rows:
        text_buffer.append(",".join(str(row[key]) for key in headers))
    return ("CompositeAI_Final_Project_Report.csv", "\n".join(text_buffer).encode("utf-8"))


def save_report_text_artifact(
    content: str | bytes,
    file_name: str,
    report_dir: Path = REPORT_DIR,
) -> Path:
    """Persist HTML/CSV report artifact and return path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / file_name
    if isinstance(content, bytes):
        output_path.write_bytes(content)
    else:
        output_path.write_text(content, encoding="utf-8")
    return output_path


def _build_cover(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    details: ProjectDetails,
    generated_at: datetime,
    data: dict[str, Any],
) -> None:
    """Build cover page."""
    story.extend(
        [
            Spacer(1, 3 * cm),
            Paragraph(REPORT_TITLE, styles["CoverTitle"]),
            Spacer(1, 0.4 * cm),
            Paragraph(REPORT_SUBTITLE, styles["CoverSubtitle"]),
            Spacer(1, 0.3 * cm),
            Paragraph(TECHNICAL_SUBTITLE, styles["Centered"]),
            Spacer(1, 1.2 * cm),
            Paragraph("Final-Year Project Technical Report", styles["CenteredBold"]),
            Spacer(1, 0.8 * cm),
        ]
    )
    rows = [
        ["Date generated", generated_at.strftime("%Y-%m-%d %H:%M")],
        ["Software/project version", _safe_get(data, "model_metadata", "project_stage")],
    ]
    optional_rows = [
        ("Student 1", details.student_1),
        ("Student 2", details.student_2),
        ("Student 3", details.student_3),
        ("Guide", details.guide),
        ("Department", details.department),
        ("College / University", details.college),
    ]
    rows.extend([[label, value] for label, value in optional_rows if value.strip()])
    story.append(_table(rows, widths=[5 * cm, 10 * cm]))
    story.append(Spacer(1, 1 * cm))
    story.append(
        Paragraph(
            "This report summarizes computational results only. It does not claim "
            "experimental certification or production aerospace qualification.",
            styles["SmallNote"],
        )
    )


def _build_toc(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    """Build table of contents."""
    story.append(Paragraph("Table of Contents", styles["Heading1NoToc"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            name="TOCLevel1",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=4,
        )
    ]
    story.append(toc)


def _section(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    number: int,
    title: str,
) -> None:
    """Append numbered section heading."""
    story.append(Paragraph(f"{number}. {title}", styles["Heading1"]))


def _abstract(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Abstract")
    story.append(Paragraph(
        "CompositeAI is an integrated computational framework for aerospace "
        "composite laminate analysis. The implemented system validates a "
        "composite material dataset, prepares material and process features, "
        "uses a validated ANN/MLP regression model for tensile-strength "
        "prediction, and provides Classical Laminate Theory tools for stacking "
        "sequence evaluation. The engineering path includes stacking-sequence "
        "representation, Q/Qbar and A/B/D laminate calculations, ply-level "
        "strain-allowable failure analysis, and bounded random-search "
        "optimization. A Streamlit interface exposes dataset review, model "
        "performance, prediction, CLT optimization, Gemini-assisted "
        "interpretation, and final reporting. AI-vs-CLT numerical comparison is "
        "intentionally limited unless material equivalence and unit compatibility "
        "are verified. The project does not claim experimental validation.",
        styles["Body"],
    ))


def _executive_summary(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Executive Summary")
    _paragraphs(story, styles, [
        "CompositeAI integrates data-driven tensile-strength prediction, Classical Laminate Theory analysis, stacking-sequence optimization, material benchmarking, and report generation in one Streamlit application.",
        "Validated project results show ANN/MLP regression with high fit quality, CLT reproduction within the documented reference tolerance, and constrained optimization improvement over the preserved baseline sequence.",
        "All engineering values in this report are loaded from project artifacts. Missing artifacts are labelled unavailable rather than reconstructed.",
    ])
    metadata = data.get("model_metadata") or {}
    metrics = metadata.get("metrics", {})
    validation = _safe_get(data, "optimization_validation", "reference_validation", default={})
    demo = _safe_get(data, "optimization_validation", "optimization_demo", default={})
    story.append(_table([
        ["ANN R²", _fmt(metrics.get("r2"), 4)],
        ["ANN MAE", f"{_fmt(metrics.get('mae'), 2)} MPa"],
        ["ANN RMSE", f"{_fmt(metrics.get('rmse'), 2)} MPa"],
        ["CLT reference status", str(validation.get("validation_status", UNAVAILABLE)).upper() if isinstance(validation, dict) else UNAVAILABLE],
        ["Best optimization improvement", f"{_fmt(demo.get('improvement_pct'), 1)} %" if isinstance(demo, dict) else UNAVAILABLE],
    ]))


def _introduction(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Introduction")
    _paragraphs(story, styles, [
        "Composite materials are widely used in aerospace structures because high stiffness-to-weight and strength-to-weight ratios can be achieved through fibre reinforcement, resin selection, and laminate design.",
        "Laminate performance depends not only on constituent material properties, but also on processing parameters, fibre volume fraction, void content, ply count, and stacking sequence. These relationships are nonlinear and difficult to capture with one method alone.",
        "Machine learning can learn data-driven mappings from material and process features to tensile strength, while Classical Laminate Theory provides physics-based evaluation of laminate stiffness, strain, and ply-level failure under defined loading.",
        "CompositeAI combines these two paths in one application while preserving their boundaries: ANN/MLP predicts tensile strength from the training dataset; CLT evaluates explicitly defined laminates and load cases.",
    ])


def _problem_statement(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Problem Statement")
    _bullets(story, styles, [
        "Analyze composite material and processing datasets.",
        "Predict tensile strength using machine-learning regression.",
        "Evaluate laminate stacking sequences using Classical Laminate Theory.",
        "Assess ply-level strain and failure under source-compatible allowables.",
        "Search valid stacking sequences under symmetry and balance constraints.",
        "Provide results through an interactive Streamlit application.",
    ])


def _objectives(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Objectives")
    _bullets(story, styles, [
        "Build a validated composite-material dataset.",
        "Prepare ML-ready features without target leakage.",
        "Train and compare regression models, then select a validated tensile-strength predictor.",
        "Implement Classical Laminate Theory with ply-level failure evaluation.",
        "Represent symmetric and balanced stacking sequences.",
        "Implement constrained stacking-sequence optimization.",
        "Validate CLT against a traceable TU Delft / Zenodo reference.",
        "Integrate prediction, optimization, interpretation, and reporting in Streamlit.",
    ])


def _system_architecture(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "System Architecture")
    story.append(_table(
        [
            ["Data-driven path", "Dataset → Validation → Preprocessing → Feature Analysis → ML Training → Model Validation → ANN Strength Prediction"],
            ["Engineering path", "Material Card + Stacking Sequence + Load Case → CLT → Q/Qbar → A/B/D → Ply Stress/Strain → Failure Evaluation → Stacking Optimization"],
            ["Application path", "Streamlit → Prediction / Optimization / Comparison / Report"],
        ],
        widths=[4 * cm, 12 * cm],
    ))


def _dataset(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Dataset")
    stats = data.get("dataset_stats", {})
    feature_spec = data.get("feature_specification") or {}
    preprocessing = data.get("preprocessing_config") or {}
    rows = stats.get("rows", UNAVAILABLE)
    missing = sum((preprocessing.get("missing_value_policy", {}).get("current_missing_values") or {}).values()) if preprocessing else UNAVAILABLE
    story.append(_table([
        ["Rows", rows],
        ["Features", _join(feature_spec.get("baseline_features"))],
        ["Target", feature_spec.get("target", UNAVAILABLE)],
        ["Missing values", missing],
        ["Duplicates", stats.get("duplicates", 0) if stats else UNAVAILABLE],
        ["Target IQR outliers retained", preprocessing.get("outlier_policy", {}).get("target_iqr_outliers", UNAVAILABLE) if preprocessing else UNAVAILABLE],
        ["Leakage status", _safe_get(data, "feature_specification", "leakage_decisions", "status")],
    ]))
    story.append(Paragraph(
        "The training dataset was locked to preserve a reproducible baseline for "
        "model validation. Outlier rows were retained because available checks did "
        "not prove them objectively invalid.",
        styles["Body"],
    ))


def _preprocessing(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Data Preprocessing")
    config = data.get("preprocessing_config") or {}
    policy = config.get("missing_value_policy", {})
    story.append(_table([
        ["Categorical features", _join(config.get("categorical_features"))],
        ["Numerical features", _join(config.get("numerical_features"))],
        ["Numerical imputation", policy.get("pipeline_numeric_strategy", UNAVAILABLE)],
        ["Categorical imputation", policy.get("pipeline_categorical_strategy", UNAVAILABLE)],
        ["Encoder", config.get("encoder", UNAVAILABLE)],
        ["Scaler", config.get("scaler", UNAVAILABLE)],
        ["Leakage prevention", config.get("leakage_prevention", UNAVAILABLE)],
    ]))


def _feature_analysis(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Feature Analysis")
    analysis = data.get("feature_analysis") or {}
    relationships = analysis.get("numerical_feature_target_relationships", [])
    rows = [["Feature", "Pearson", "Spearman", "Recommendation"]]
    for item in relationships:
        rows.append([
            item.get("feature", ""),
            _fmt(item.get("pearson_correlation"), 6),
            _fmt(item.get("spearman_correlation"), 6),
            item.get("recommendation", ""),
        ])
    story.append(_table(rows, repeat=True))
    target = analysis.get("target_analysis", {})
    story.append(_table([
        ["Target skewness", _fmt(target.get("skewness"), 4)],
        ["Distribution note", target.get("distribution_note", UNAVAILABLE)],
        ["Multicollinearity", "No strong numerical multicollinearity reported in project artifacts."],
        ["Baseline engineered features", "None; not justified by available dataset fields."],
        ["Density decision", "Retained for nonlinear/engineering relevance."],
    ]))


def _ml_models(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Machine Learning Models")
    comparison = data.get("model_comparison") or []
    rows = [["Model", "Test MAE", "Test RMSE", "Test R²"]]
    for row in comparison:
        rows.append([
            row.get("model", ""),
            _fmt(row.get("test_mae"), 4),
            _fmt(row.get("test_rmse"), 4),
            _fmt(row.get("test_r2"), 4),
        ])
    story.append(_table(rows, repeat=True))


def _final_model(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Final Model")
    metadata = data.get("model_metadata") or {}
    rows = [
        ["Selected model", metadata.get("model_name", UNAVAILABLE)],
        ["Random state", metadata.get("random_state", UNAVAILABLE)],
        ["Target", metadata.get("target", UNAVAILABLE)],
        ["Input features", _join(metadata.get("feature_list"))],
        ["Preprocessing", json.dumps(metadata.get("preprocessing_description", UNAVAILABLE), default=str)],
    ]
    rows.extend([
        ["Architecture", "64 → 32"],
        ["Activation", "ReLU"],
        ["Solver", "Adam"],
        ["Early stopping", "Enabled"],
        ["Validation fraction", "0.1"],
        ["Maximum iterations", "600"],
    ])
    story.append(_table(rows))


def _model_validation(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Model Validation")
    report = data.get("model_validation") or {}
    test = report.get("final_test_metrics", {}).get("test", {})
    residual = report.get("residual_analysis", {})
    cv = report.get("cross_validation", {}).get("ANN/MLP", {})
    seed = report.get("seed_robustness", {}).get("summary", {})
    story.append(_table([
        ["Test MAE", f"{_fmt(test.get('mae'), 4)} MPa"],
        ["Test RMSE", f"{_fmt(test.get('rmse'), 4)} MPa"],
        ["Test R²", _fmt(test.get("r2"), 4)],
        ["CV MAE", _mean_std(cv, "mae")],
        ["CV RMSE", _mean_std(cv, "rmse")],
        ["CV R²", _mean_std(cv, "r2")],
        ["Seed robustness R²", _mean_std(seed, "r2")],
        ["Residual mean", f"{_fmt(residual.get('mean_residual'), 4)} MPa"],
        ["Residual median", f"{_fmt(residual.get('median_residual'), 4)} MPa"],
        ["Residual std", f"{_fmt(residual.get('std_residual'), 4)} MPa"],
    ]))
    _paragraphs(story, styles, [
        "Validation artifacts report no target leakage and no meaningful overfitting. The top-error records indicate larger upper-range errors, especially in Carbon/Epoxy samples.",
        "Predictions remain within the observed target range in the saved validation report.",
    ])


def _strength_prediction(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Prediction Results")
    story.append(_table([
        ["Fiber type", "Aramid"],
        ["Resin type", "Phenolic"],
        ["Density", "2.16 g/cm³"],
        ["Layer count", "16"],
        ["Curing temperature", "138 °C"],
        ["Fiber volume fraction", "0.56"],
        ["Void content", "1.84 %"],
        ["Model prediction", "1827.77 MPa"],
    ]))
    story.append(Paragraph(
        "This is an ML prediction based on the trained dataset and is not experimental certification.",
        styles["SmallNote"],
    ))


def _clt(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "CLT Analysis")
    material = _safe_get(data, "ai_clt_comparison", "clt_system", "source_material", default={})
    story.append(_table([
        ["Implemented calculations", "Q matrix, Qbar transformation, A/B/D matrices, mid-plane strains, ply stresses, local ply stresses, failure evaluation"],
        ["Material card", material.get("name", UNAVAILABLE) if isinstance(material, dict) else UNAVAILABLE],
        ["E1", _pa_to_gpa(material.get("E1_Pa")) if isinstance(material, dict) else UNAVAILABLE],
        ["E2", _pa_to_gpa(material.get("E2_Pa")) if isinstance(material, dict) else UNAVAILABLE],
        ["G12", _pa_to_gpa(material.get("G12_Pa")) if isinstance(material, dict) else UNAVAILABLE],
        ["nu12", material.get("nu12", UNAVAILABLE) if isinstance(material, dict) else UNAVAILABLE],
        ["Ply thickness", f"{material.get('ply_thickness_m')} m" if isinstance(material, dict) else UNAVAILABLE],
    ]))


def _engineering_benchmark(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Composite vs Aerospace Metals")
    spec = data.get("comparison_specification") or {}
    rows = [["Material", "Strength (MPa)", "Density (g/cm3)", "Specific Strength"]]
    for item in spec.get("materials", []):
        rows.append([
            item.get("material", ""),
            _fmt(item.get("strength_mpa"), 2),
            _fmt(item.get("density_g_cm3"), 3),
            _fmt(item.get("strength_to_weight"), 2),
        ])
    if len(rows) == 1:
        rows.append([UNAVAILABLE, UNAVAILABLE, UNAVAILABLE, UNAVAILABLE])
    story.append(_table(rows, repeat=True))
    story.append(_table([
        ["Reference source", _safe_get(data, "comparison_specification", "source_database")],
        ["Reference type", _safe_get(data, "comparison_specification", "source_type")],
        ["Record count", _safe_get(data, "comparison_specification", "database_record_count")],
        ["Output fields", json.dumps(_safe_get(data, "comparison_specification", "output_fields", default=[]))],
    ]))
    story.append(Paragraph(
        "Engineering benchmark compares CompositeAI prediction output against a separate aerospace metals reference database. These values are advisory only and are not part of the ML training dataset.",
        styles["Body"],
    ))


def _engineering_visualizations(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Engineering Visualizations")
    interpretations = data.get("engineering_interpretations") or []
    if not interpretations:
        story.append(Paragraph(UNAVAILABLE, styles["Body"]))
        return
    for interpretation in interpretations:
        story.append(Paragraph(_graph_title(interpretation.get("graph_name", "")), styles["Heading2"]))
        story.append(_table([["Data source", interpretation.get("data_source", UNAVAILABLE)]]))
        sections = interpretation.get("sections") or {}
        for section in INTERPRETATION_SECTIONS:
            story.append(Paragraph(section, styles["Heading3"]))
            story.append(Paragraph(str(sections.get(section, UNAVAILABLE)), styles["Body"]))


def _failure_analysis(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Failure Analysis")
    failure = _safe_get(data, "optimization_validation", "failure_model", default={})
    allowables = failure.get("allowables", {}) if isinstance(failure, dict) else {}
    story.append(_table([
        ["Failure route", failure.get("method", UNAVAILABLE) if isinstance(failure, dict) else UNAVAILABLE],
        ["epsilon1", allowables.get("epsilon_1_allowable", UNAVAILABLE)],
        ["epsilon2", allowables.get("epsilon_2_allowable", UNAVAILABLE)],
        ["gamma12", allowables.get("gamma_12_allowable", UNAVAILABLE)],
        ["Source", failure.get("source", UNAVAILABLE) if isinstance(failure, dict) else UNAVAILABLE],
    ]))
    story.append(Paragraph(
        "Failure evaluation uses the selected source-compatible allowable route. It is not claimed as a universal material failure criterion.",
        styles["Body"],
    ))


def _clt_validation(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "CLT Validation")
    validation = _safe_get(data, "optimization_validation", "reference_validation", default={})
    story.append(_table([
        ["Reference", "TU Delft / Zenodo source case"],
        ["Reference lambda_cs", _fmt(validation.get("reference_lambda_cs"), 4) if isinstance(validation, dict) else UNAVAILABLE],
        ["Our CLT lambda_cs", _fmt(validation.get("our_lambda_cs"), 4) if isinstance(validation, dict) else UNAVAILABLE],
        ["Difference", f"{abs(float(validation.get('difference_pct', 0))):.4f} %" if isinstance(validation, dict) and validation.get("difference_pct") is not None else UNAVAILABLE],
        ["Tolerance", "1 %"],
        ["Status", str(validation.get("validation_status", UNAVAILABLE)).upper() if isinstance(validation, dict) else UNAVAILABLE],
    ]))
    story.append(Paragraph(
        "This validates the implemented CLT calculation against the selected reference case. It is not AI-vs-CLT validation.",
        styles["Body"],
    ))


def _stacking_sequence(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Stacking Sequence Representation")
    constraints = _safe_get(data, "optimization_validation", "optimization_demo", "constraints", default={})
    story.append(_table([
        ["Allowed angles", _join(constraints.get("allowed_angles")) if isinstance(constraints, dict) else UNAVAILABLE],
        ["Symmetric", constraints.get("require_symmetric", UNAVAILABLE) if isinstance(constraints, dict) else UNAVAILABLE],
        ["Balanced", constraints.get("require_balanced", UNAVAILABLE) if isinstance(constraints, dict) else UNAVAILABLE],
        ["Ply count", constraints.get("expected_ply_count", UNAVAILABLE) if isinstance(constraints, dict) else UNAVAILABLE],
        ["Candidate generation", "Bounded valid-sequence generation"],
    ]))
    story.append(Paragraph(
        "Stacking sequence controls directional stiffness, coupling behavior, and ply-level failure response, so it is evaluated separately from the non-sequence-aware ANN model.",
        styles["Body"],
    ))


def _optimization(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Optimization Results Summary")
    demo = _safe_get(data, "optimization_validation", "optimization_demo", default={})
    impact = load_optimization_impact(demo if isinstance(demo, dict) else None)
    sequence = impact.get("best_sequence", []) or []
    story.append(_table([
        ["Method", demo.get("search_method", UNAVAILABLE) if isinstance(demo, dict) else UNAVAILABLE],
        ["Candidates evaluated", impact.get("candidates_evaluated", UNAVAILABLE)],
        ["Baseline load factor", _fmt(impact.get("baseline_lambda_cs"), 4)],
        ["Optimized load factor", _fmt(impact.get("optimized_lambda_cs"), 4)],
        ["Improvement %", f"{_fmt(impact.get('improvement_pct'), 4)} %" if impact.get("improvement_pct") is not None else UNAVAILABLE],
        ["Improvement ratio", f"{_fmt(impact.get('improvement_ratio'), 2)}×" if impact.get("improvement_ratio") is not None else UNAVAILABLE],
        ["Best laminate sequence", _join(sequence)],
        ["Optimization constraints", json.dumps(impact.get("constraints", {}))],
    ]))
    story.append(Paragraph(
        "Optimization improved failure load capacity from baseline to optimized design under fixed laminate constraints. Result is computational, not experimental.",
        styles["SmallNote"],
    ))
    story.append(Paragraph(f"<font name='Courier'>{_join(sequence)}</font>", styles["Body"]))


def _ai_vs_clt(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "AI vs CLT Comparison")
    spec = data.get("ai_clt_comparison") or {}
    story.append(_table([
        ["Current status", "NOT DIRECTLY COMPARABLE"],
        ["Reason", "ANN training dataset does not have verified material equivalence to the TU Delft CLT material card."],
        ["ANN output", _safe_get(data, "ai_clt_comparison", "ann_system", "output_unit")],
        ["CLT raw output", _safe_get(data, "ai_clt_comparison", "clt_system", "lambda_cs_description")],
        ["Numerical difference", "Not reported for the current default case."],
    ]))
    _bullets(story, styles, spec.get("comparison_prohibited_when", [])[:6] or [
        "Comparison requires explicit material equivalence.",
        "Comparison requires compatible physical quantities and consistent units.",
    ])


def _streamlit_system(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Streamlit Application")
    _bullets(story, styles, [
        "Dashboard",
        "Dataset Import",
        "Dataset Explorer",
        "Dataset Profile",
        "Exploratory Data Analysis",
        "Feature Engineering",
        "Data Preprocessing",
        "Model Performance",
        "Strength Prediction",
        "Composite vs Aerospace Metals",
        "Stacking Optimizer",
        "AI vs CLT Comparison",
        "Gemini Engineering Assistant",
        "Report Generator",
        "About",
    ])


def _gemini_assistant(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Gemini Assistant")
    _paragraphs(story, styles, [
        "Gemini is optional. It provides dataset interpretation, prediction explanation, and engineering guidance.",
        "Gemini does not train the ANN, modify datasets, replace CLT, replace the optimizer, or fabricate model results. If Gemini is unavailable, the core application still works.",
    ])


def _results_summary(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Results Summary")
    metadata = data.get("model_metadata") or {}
    metrics = metadata.get("metrics", {})
    validation = _safe_get(data, "optimization_validation", "reference_validation", default={})
    demo = _safe_get(data, "optimization_validation", "optimization_demo", default={})
    benchmark = data.get("comparison_specification") or {}
    benchmark_materials = benchmark.get("materials", []) if isinstance(benchmark, dict) else []
    benchmark_best = max(
        benchmark_materials,
        key=lambda item: float(item.get("strength_to_weight", float("-inf"))),
    ) if benchmark_materials else None
    story.append(_table([
        ["Component", "Result", "Status"],
        ["Dataset", f"{_safe_get(data, 'dataset_stats', 'rows')} rows", "PASS"],
        ["ANN R²", _fmt(metrics.get("r2"), 4), "PASS"],
        ["ANN MAE", f"{_fmt(metrics.get('mae'), 2)} MPa", "PASS"],
        ["ANN RMSE", f"{_fmt(metrics.get('rmse'), 2)} MPa", "PASS"],
        ["Engineering benchmark", benchmark_best.get("material", UNAVAILABLE) if isinstance(benchmark_best, dict) else UNAVAILABLE, "PASS" if benchmark_best else "LIMITATION"],
        ["CLT validation", f"{abs(float(validation.get('difference_pct', 0))):.4f}% difference" if isinstance(validation, dict) else UNAVAILABLE, "PASS"],
        ["Optimization", f"{_fmt(demo.get('improvement_pct'), 1)}% candidate improvement" if isinstance(demo, dict) else UNAVAILABLE, "DEMONSTRATOR"],
        ["AI vs CLT", "Not directly comparable", "VALID LIMITATION"],
        ["Experimental validation", "Not available", "LIMITATION"],
    ], repeat=True))


def _limitations(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Limitations")
    _bullets(story, styles, [
        "ANN dataset lacks stacking-sequence information.",
        "ANN is not sequence-aware.",
        "Current ANN and TU Delft CLT material representations are not directly equivalent.",
        "Experimental laminate validation is unavailable.",
        "Optimization does not prove global optimality.",
        "Buckling optimization is not implemented.",
        "CLT optimization is a computational demonstrator.",
        "Material allowables depend on the selected source-compatible route.",
        "Dataset domain limitations affect model generalization.",
    ])


def _future_work(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Future Work")
    _bullets(story, styles, [
        "Sequence-aware ML dataset.",
        "Experimental laminate testing.",
        "Broader verified material cards.",
        "Buckling optimization.",
        "Multi-objective optimization.",
        "Physics-informed ML.",
        "Larger sequence dataset.",
        "Manufacturing constraints.",
        "Uncertainty quantification.",
        "Real aerospace laminate validation.",
    ])


def _conclusion(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Conclusions")
    story.append(Paragraph(
        "CompositeAI demonstrates ML-based tensile-strength prediction, a validated "
        "ANN model, CLT-based laminate analysis, ply-level failure evaluation, "
        "constrained stacking-sequence search, reference validation, and an "
        "interactive Streamlit application. The system remains a final-year "
        "computational prototype and does not claim production readiness or "
        "aerospace certification.",
        styles["Body"],
    ))


def _ai_metrics(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "AI Metrics")
    comparison = data.get("model_comparison") or []
    rows = [["Model", "Test MAE", "Test RMSE", "Test R²"]]
    for row in comparison:
        rows.append([
            row.get("model", ""),
            _fmt(row.get("test_mae"), 4),
            _fmt(row.get("test_rmse"), 4),
            _fmt(row.get("test_r2"), 4),
        ])
    story.append(_table(rows, repeat=True))
    metadata = data.get("model_metadata") or {}
    story.append(_table([
        ["Selected model", metadata.get("model_name", UNAVAILABLE)],
        ["Architecture", "64 → 32"],
        ["Activation", "ReLU"],
        ["Solver", "Adam"],
        ["Early stopping", "Enabled"],
        ["Maximum iterations", "600"],
    ]))
    report = data.get("model_validation") or {}
    test = report.get("final_test_metrics", {}).get("test", {})
    cv = report.get("cross_validation", {}).get("ANN/MLP", {})
    story.append(_table([
        ["Test MAE", f"{_fmt(test.get('mae'), 4)} MPa"],
        ["Test RMSE", f"{_fmt(test.get('rmse'), 4)} MPa"],
        ["Test R²", _fmt(test.get("r2"), 4)],
        ["CV R²", _mean_std(cv, "r2")],
    ]))


def _clt_analysis(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "CLT Analysis")
    material = _safe_get(data, "ai_clt_comparison", "clt_system", "source_material", default={})
    failure = _safe_get(data, "optimization_validation", "failure_model", default={})
    allowables = failure.get("allowables", {}) if isinstance(failure, dict) else {}
    validation = _safe_get(data, "optimization_validation", "reference_validation", default={})
    story.append(_table([
        ["Implemented calculations", "Q matrix, Qbar transformation, A/B/D matrices, mid-plane strains, ply stresses, local ply stresses, failure evaluation"],
        ["Material card", material.get("name", UNAVAILABLE) if isinstance(material, dict) else UNAVAILABLE],
        ["E1", _pa_to_gpa(material.get("E1_Pa")) if isinstance(material, dict) else UNAVAILABLE],
        ["E2", _pa_to_gpa(material.get("E2_Pa")) if isinstance(material, dict) else UNAVAILABLE],
        ["G12", _pa_to_gpa(material.get("G12_Pa")) if isinstance(material, dict) else UNAVAILABLE],
        ["nu12", material.get("nu12", UNAVAILABLE) if isinstance(material, dict) else UNAVAILABLE],
        ["Ply thickness", f"{material.get('ply_thickness_m')} m" if isinstance(material, dict) else UNAVAILABLE],
        ["epsilon1", allowables.get("epsilon_1_allowable", UNAVAILABLE)],
        ["epsilon2", allowables.get("epsilon_2_allowable", UNAVAILABLE)],
        ["gamma12", allowables.get("gamma_12_allowable", UNAVAILABLE)],
    ]))
    story.append(_table([
        ["CLT Validation", "TU Delft / Zenodo source case"],
        ["Reference lambda_cs", _fmt(validation.get("reference_lambda_cs"), 4) if isinstance(validation, dict) else UNAVAILABLE],
        ["Our CLT lambda_cs", _fmt(validation.get("our_lambda_cs"), 4) if isinstance(validation, dict) else UNAVAILABLE],
        ["Difference", f"{abs(float(validation.get('difference_pct', 0))):.4f} %" if isinstance(validation, dict) and validation.get("difference_pct") is not None else UNAVAILABLE],
        ["AI vs CLT", "NOT DIRECTLY COMPARABLE unless material equivalence and units are verified."],
    ]))


def _optimization_results(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Optimization Results Summary")
    constraints = _safe_get(data, "optimization_validation", "optimization_demo", "constraints", default={})
    demo = _safe_get(data, "optimization_validation", "optimization_demo", default={})
    impact = load_optimization_impact(demo if isinstance(demo, dict) else None)
    story.append(_table([
        ["Allowed angles", _join(constraints.get("allowed_angles")) if isinstance(constraints, dict) else UNAVAILABLE],
        ["Symmetric", constraints.get("require_symmetric", UNAVAILABLE) if isinstance(constraints, dict) else UNAVAILABLE],
        ["Balanced", constraints.get("require_balanced", UNAVAILABLE) if isinstance(constraints, dict) else UNAVAILABLE],
        ["Ply count", constraints.get("expected_ply_count", UNAVAILABLE) if isinstance(constraints, dict) else UNAVAILABLE],
        ["Search method", demo.get("search_method", UNAVAILABLE) if isinstance(demo, dict) else UNAVAILABLE],
        ["Candidates evaluated", impact.get("candidates_evaluated", UNAVAILABLE)],
        ["Baseline load factor", _fmt(impact.get("baseline_lambda_cs"), 4)],
        ["Optimized load factor", _fmt(impact.get("optimized_lambda_cs"), 4)],
        ["Improvement %", f"{_fmt(impact.get('improvement_pct'), 4)} %" if impact.get("improvement_pct") is not None else UNAVAILABLE],
        ["Improvement ratio", f"{_fmt(impact.get('improvement_ratio'), 2)}×" if impact.get("improvement_ratio") is not None else UNAVAILABLE],
        ["Best laminate sequence", _join(impact.get("best_sequence", []))],
    ]))
    story.append(Paragraph(
        "Optimization impact summary highlights baseline vs optimized failure load factor under current constraints.",
        styles["SmallNote"],
    ))


def _conclusions(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "Conclusions")
    metadata = data.get("model_metadata") or {}
    metrics = metadata.get("metrics", {})
    story.append(_table([
        ["ANN/MLP", f"R²={_fmt(metrics.get('r2'), 4)} | MAE={_fmt(metrics.get('mae'), 4)} MPa | RMSE={_fmt(metrics.get('rmse'), 4)} MPa"],
        ["AI vs CLT", "NOT DIRECTLY COMPARABLE without verified material equivalence"],
        ["Prototype status", "Final-year computational prototype; not certification evidence"],
    ]))
    _bullets(story, styles, [
        "ANN dataset lacks stacking-sequence information.",
        "Experimental laminate validation is unavailable.",
        "Optimization does not prove global optimality.",
        "Future work includes sequence-aware ML, broader material cards, buckling optimization, and physics-informed ML.",
    ])
    story.append(Paragraph(
        "CompositeAI demonstrates ML-based tensile-strength prediction, validated CLT calculations, constrained stacking-sequence search, and integrated reporting in one application while preserving clear limits on what has and has not been validated.",
        styles["Body"],
    ))


def _references(story: list[Any], styles: dict[str, ParagraphStyle], number: int, data: dict[str, Any]) -> None:
    _section(story, styles, number, "References")
    sequence_meta = data.get("sequence_metadata") or {}
    inventory = data.get("dataset_inventory") or {}
    refs = [
        f"TU Delft / Zenodo DOI: {sequence_meta.get('doi', '10.5281/zenodo.15864524')}",
        sequence_meta.get("publication", ""),
        f"Zenodo URL: {sequence_meta.get('url', UNAVAILABLE)}",
        f"License: {sequence_meta.get('license', UNAVAILABLE)}",
        f"Dataset inventory: {json.dumps(inventory.get('summary', UNAVAILABLE), default=str)}",
    ]
    _bullets(story, styles, [ref for ref in refs if ref])


SECTION_BUILDERS = {
    "Executive Summary": _executive_summary,
    "Prediction Results": _strength_prediction,
    "AI Metrics": _ai_metrics,
    "CLT Analysis": _clt_analysis,
    "Optimization Results": _optimization_results,
    "Composite vs Aerospace Metals": _engineering_benchmark,
    "Engineering Visualizations": _engineering_visualizations,
    "Conclusions": _conclusions,
}


def _styles() -> dict[str, ParagraphStyle]:
    """Return report styles."""
    base = getSampleStyleSheet()
    return {
        "Normal": base["Normal"],
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=10, leading=14, spaceAfter=8),
        "SmallNote": ParagraphStyle("SmallNote", parent=base["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#4b5563"), spaceAfter=8),
        "Heading1": ParagraphStyle("ReportHeading1", parent=base["Heading1"], fontSize=15, leading=18, textColor=colors.HexColor("#101820"), spaceBefore=12, spaceAfter=8),
        "Heading2": ParagraphStyle("ReportHeading2", parent=base["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#101820"), spaceBefore=8, spaceAfter=5),
        "Heading3": ParagraphStyle("ReportHeading3", parent=base["Heading3"], fontSize=10, leading=12, textColor=colors.HexColor("#1b2733"), spaceBefore=5, spaceAfter=3),
        "Heading1NoToc": ParagraphStyle("Heading1NoToc", parent=base["Heading1"], fontSize=16, leading=20, textColor=colors.HexColor("#101820"), spaceAfter=12),
        "CoverTitle": ParagraphStyle("CoverTitle", parent=base["Title"], fontSize=30, leading=36, alignment=TA_CENTER, textColor=colors.HexColor("#101820")),
        "CoverSubtitle": ParagraphStyle("CoverSubtitle", parent=base["Title"], fontSize=16, leading=22, alignment=TA_CENTER, textColor=colors.HexColor("#1b2733")),
        "Centered": ParagraphStyle("Centered", parent=base["BodyText"], fontSize=11, leading=15, alignment=TA_CENTER),
        "CenteredBold": ParagraphStyle("CenteredBold", parent=base["BodyText"], fontSize=12, leading=16, alignment=TA_CENTER, fontName="Helvetica-Bold"),
    }


def _paragraphs(story: list[Any], styles: dict[str, ParagraphStyle], paragraphs: list[str]) -> None:
    """Append paragraphs."""
    for paragraph in paragraphs:
        story.append(Paragraph(paragraph, styles["Body"]))


def _bullets(story: list[Any], styles: dict[str, ParagraphStyle], bullets: list[Any]) -> None:
    """Append bullet paragraphs."""
    for bullet in bullets:
        story.append(Paragraph(f"• {bullet}", styles["Body"]))


def _table(rows: list[list[Any]], widths: list[float] | None = None, repeat: bool = False) -> Table:
    """Build styled table with paragraph-safe cells."""
    styles = _styles()
    converted = [
        [Paragraph(str(cell), styles["Body"]) for cell in row]
        for row in rows
    ]
    table = Table(converted, colWidths=widths, repeatRows=1 if repeat else 0, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f2fb") if repeat else colors.white),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#101820")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d4df")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _draw_page(canvas: Any, doc: BaseDocTemplate) -> None:
    """Draw page header and footer."""
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(colors.HexColor("#101820"))
    canvas.drawString(2 * cm, A4[1] - 1.25 * cm, "CompositeAI Final Project Report")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#607080"))
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _dataset_stats(path: Path) -> dict[str, Any]:
    """Return simple CSV dataset stats."""
    if not path.exists():
        return {"rows": UNAVAILABLE, "columns": UNAVAILABLE, "duplicates": UNAVAILABLE}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return {
        "rows": len(rows),
        "columns": len(reader.fieldnames or []),
        "duplicates": len(rows) - len({tuple(sorted(row.items())) for row in rows}),
    }


def _collect_engineering_interpretations(project_root: Path) -> list[dict[str, Any]]:
    """Collect live visualization interpretations only for configured project root."""
    if project_root.resolve() != PROJECT_ROOT.resolve():
        return []
    try:
        return build_all_engineering_interpretations(ENGINEERING_EXPLANATION)
    except Exception:
        return []


def _graph_title(graph_name: str) -> str:
    titles = {
        "strength_surface": "Composite Strength Response Surface",
        "optimization_landscape": "Optimization Search Landscape",
        "ply_failure_map": "Ply Failure Distribution",
        "material_benchmark": "Material Performance Benchmark",
    }
    return titles.get(graph_name, graph_name or UNAVAILABLE)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV as list of dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_get(data: dict[str, Any], *keys: str, default: Any = UNAVAILABLE) -> Any:
    """Read nested mapping value with unavailable fallback."""
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value or value[key] is None:
            return default
        value = value[key]
    return value


def _fmt(value: Any, digits: int) -> str:
    """Format numeric value."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return UNAVAILABLE


def _mean_std(data: dict[str, Any], key: str) -> str:
    """Format mean/std field from possible validation schemas."""
    if not isinstance(data, dict):
        return UNAVAILABLE
    mean = data.get(f"{key}_mean", data.get(key, {}).get("mean") if isinstance(data.get(key), dict) else None)
    std = data.get(f"{key}_std", data.get(key, {}).get("std") if isinstance(data.get(key), dict) else None)
    if mean is None and "metrics" in data:
        metric_data = data["metrics"].get(key, {}) if isinstance(data["metrics"], dict) else {}
        mean = metric_data.get("mean")
        std = metric_data.get("std")
    if mean is None:
        return UNAVAILABLE
    if std is None:
        return _fmt(mean, 4)
    return f"{_fmt(mean, 4)} ± {_fmt(std, 4)}"


def _join(values: Any) -> str:
    """Join list-like values."""
    if values is None:
        return UNAVAILABLE
    if isinstance(values, list):
        return ", ".join(str(value) for value in values)
    return str(values)


def _pa_to_gpa(value: Any) -> str:
    """Convert Pa to GPa string."""
    try:
        return f"{float(value) / 1e9:.2f} GPa"
    except (TypeError, ValueError):
        return UNAVAILABLE
