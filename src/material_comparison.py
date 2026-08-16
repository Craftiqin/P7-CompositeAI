"""Reference-database material comparison utilities for engineering benchmark."""

from __future__ import annotations

import html
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.material_benchmark import (
    DEFAULT_REFERENCE_DATABASE_PATH,
    calculate_specific_strength,
    generate_engineering_insights,
    load_reference_materials,
    rank_materials,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = (
    PROJECT_ROOT / "data" / "reference_materials" / "comparison_specification.json"
)
DEFAULT_HTML_REPORT_PATH = PROJECT_ROOT / "reports" / "material_benchmark_report.html"

DISCLAIMERS = [
    "Reference material properties are engineering benchmark values and are not part of the machine-learning training dataset.",
    "Benchmark comparisons are intended for preliminary engineering evaluation only.",
]


def compare_against_materials(
    predicted_strength_mpa: float,
    composite_density: float | None = None,
) -> list[dict[str, Any]]:
    """Return ranked reference-material comparison rows."""
    return rank_materials(predicted_strength_mpa, composite_density)


def summarize_material_comparison(
    predicted_strength_mpa: float,
    composite_density: float | None = None,
) -> dict[str, Any]:
    """Return summary ranking slices for the benchmark page."""
    comparisons = compare_against_materials(predicted_strength_mpa, composite_density)
    strongest = sorted(
        comparisons,
        key=lambda item: item["tensile_strength_mpa"],
        reverse=True,
    )
    lightest = sorted(
        comparisons,
        key=lambda item: item["density_g_cm3"] if item["density_g_cm3"] is not None else float("inf"),
    )
    best_specific = sorted(
        comparisons,
        key=lambda item: item["specific_strength"] if item["specific_strength"] is not None else float("-inf"),
        reverse=True,
    )
    closest_strength = min(
        comparisons,
        key=lambda item: abs(predicted_strength_mpa - item["tensile_strength_mpa"]),
    ) if comparisons else None

    return {
        "strongest_material": strongest[0] if strongest else None,
        "lightest_material": lightest[0] if lightest else None,
        "best_specific_strength": best_specific[0] if best_specific else None,
        "closest_strength_match": closest_strength,
        "composite_specific_strength": calculate_specific_strength(
            predicted_strength_mpa,
            composite_density,
        ),
        "strength_ranking": strongest,
        "density_ranking": lightest,
        "specific_strength_ranking": best_specific,
    }


def build_comparison_specification() -> dict[str, Any]:
    """Build JSON-safe specification for report integration."""
    materials = load_reference_materials()
    ranked = rank_materials(composite_strength_mpa=1.0)
    return {
        "source_type": "engineering_reference_database",
        "source_database": str(DEFAULT_REFERENCE_DATABASE_PATH),
        "database_record_count": len(materials),
        "output_fields": [
            "material",
            "category",
            "application",
            "tensile_strength_mpa",
            "density_g_cm3",
            "specific_strength",
            "difference_vs_composite_mpa",
            "strength_ratio",
            "density_ratio",
            "specific_strength_ratio",
        ],
        "materials": [
            {
                "material": item["material"],
                "category": item["category"],
                "application": item["application"],
                "strength_mpa": item["tensile_strength_mpa"],
                "density_g_cm3": item["density_g_cm3"],
                "strength_to_weight": item["specific_strength"],
            }
            for item in ranked
        ],
        "disclaimers": DISCLAIMERS,
    }


def build_material_benchmark_report(
    predicted_strength_mpa: float,
    composite_density: float | None = None,
) -> dict[str, Any]:
    """Return report-ready benchmark context."""
    comparison_rows = compare_against_materials(predicted_strength_mpa, composite_density)
    summary = summarize_material_comparison(predicted_strength_mpa, composite_density)
    return {
        "predicted_strength_mpa": float(predicted_strength_mpa),
        "composite_density_g_cm3": composite_density,
        "composite_specific_strength": summary["composite_specific_strength"],
        "source_database": str(DEFAULT_REFERENCE_DATABASE_PATH),
        "comparison_rows": comparison_rows,
        "summary": summary,
        "insights": generate_engineering_insights(
            predicted_strength_mpa,
            composite_density,
            comparison_rows,
        ),
        "disclaimers": DISCLAIMERS,
        "equations": {
            "strength_ratio": "strength_ratio = composite_strength / material_strength",
            "specific_strength": "specific_strength = tensile_strength / density",
            "density_ratio": "density_ratio = composite_density / material_density",
        },
    }


def export_comparison_csv(
    predicted_strength_mpa: float,
    composite_density: float | None = None,
) -> bytes:
    """Export comparison table as CSV."""
    frame = pd.DataFrame(compare_against_materials(predicted_strength_mpa, composite_density))
    return frame.to_csv(index=False).encode("utf-8")


def export_comparison_excel(
    predicted_strength_mpa: float,
    composite_density: float | None = None,
) -> bytes:
    """Export comparison table as Excel bytes."""
    frame = pd.DataFrame(compare_against_materials(predicted_strength_mpa, composite_density))
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="benchmark")
    return buffer.getvalue()


def generate_material_benchmark_html_report(
    predicted_strength_mpa: float,
    composite_density: float | None = None,
    output_path: Path = DEFAULT_HTML_REPORT_PATH,
) -> str:
    """Generate benchmark HTML report and persist to reports directory."""
    report = build_material_benchmark_report(predicted_strength_mpa, composite_density)
    rows_html = "".join(
        (
            "<tr>"
            f"<td>{html.escape(str(row['material']))}</td>"
            f"<td>{html.escape(str(row['category']))}</td>"
            f"<td>{row['tensile_strength_mpa']:.2f}</td>"
            f"<td>{_fmt_optional(row['density_g_cm3'])}</td>"
            f"<td>{_fmt_optional(row['specific_strength'])}</td>"
            f"<td>{row['difference_vs_composite_mpa']:.2f}</td>"
            f"<td>{row['strength_ratio']:.2f}</td>"
            "</tr>"
        )
        for row in report["comparison_rows"]
    )
    insights_html = "".join(f"<li>{html.escape(insight)}</li>" for insight in report["insights"])
    disclaimers_html = "".join(f"<li>{html.escape(text)}</li>" for text in report["disclaimers"])
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CompositeAI Composite vs Aerospace Metals Benchmark</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #101820; }}
    h1, h2 {{ color: #142a52; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #c8d4df; padding: 8px; text-align: left; }}
    th {{ background: #e8f2fb; }}
    .card {{ background: #f7fbff; border: 1px solid #c8d4df; padding: 16px; margin: 12px 0; }}
  </style>
</head>
<body>
  <h1>Composite vs Aerospace Metals Benchmark</h1>
  <p><strong>Reference source:</strong> {html.escape(report['source_database'])}</p>
  <div class="card">
    <h2>Composite Prediction</h2>
    <p><strong>Predicted Composite Strength:</strong> {report['predicted_strength_mpa']:.2f} MPa</p>
    <p><strong>Composite Density:</strong> {_fmt_optional(report['composite_density_g_cm3'])} g/cm³</p>
    <p><strong>Composite Specific Strength:</strong> {_fmt_optional(report['composite_specific_strength'])}</p>
  </div>
  <h2>Comparison Table</h2>
  <table>
    <thead>
      <tr>
        <th>Material</th>
        <th>Category</th>
        <th>Strength (MPa)</th>
        <th>Density (g/cm³)</th>
        <th>Specific Strength</th>
        <th>Difference vs Composite</th>
        <th>Strength Ratio</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <h2>Engineering Insights</h2>
  <ul>{insights_html}</ul>
  <h2>Disclaimers</h2>
  <ul>{disclaimers_html}</ul>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return html_text


def generate_material_benchmark_pdf_report(
    predicted_strength_mpa: float,
    composite_density: float | None = None,
) -> bytes:
    """Generate benchmark PDF bytes."""
    report = build_material_benchmark_report(predicted_strength_mpa, composite_density)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph("Composite vs Aerospace Metals Benchmark", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Reference source: {html.escape(report['source_database'])}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph(
            f"Predicted Composite Strength: {report['predicted_strength_mpa']:.2f} MPa",
            styles["Heading2"],
        ),
    ]
    if report["composite_density_g_cm3"] is not None:
        story.append(
            Paragraph(
                f"Composite Density: {report['composite_density_g_cm3']:.3f} g/cm³",
                styles["BodyText"],
            )
        )
    story.append(Spacer(1, 8))
    rows = [[
        "Material",
        "Category",
        "Strength (MPa)",
        "Density (g/cm³)",
        "Specific Strength",
        "Strength Ratio",
    ]]
    for row in report["comparison_rows"]:
        rows.append(
            [
                row["material"],
                row["category"],
                f"{row['tensile_strength_mpa']:.2f}",
                _fmt_optional(row["density_g_cm3"]),
                _fmt_optional(row["specific_strength"]),
                f"{row['strength_ratio']:.2f}",
            ]
        )
    table = Table(rows, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f2fb")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d4df")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#101820")),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Engineering Insights", styles["Heading2"]))
    for insight in report["insights"]:
        story.append(Paragraph(f"• {html.escape(insight)}", styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Disclaimers", styles["Heading2"]))
    for disclaimer in report["disclaimers"]:
        story.append(Paragraph(f"• {html.escape(disclaimer)}", styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()


def _fmt_optional(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.2f}"
