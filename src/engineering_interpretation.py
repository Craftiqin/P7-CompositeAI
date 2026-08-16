"""Data-derived engineering interpretation for 3D visualization outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from src.engineering_visualizations import (
    build_material_benchmark_3d,
    build_optimization_landscape,
    build_ply_failure_map,
    build_strength_response_surface,
)


SIMPLE_EXPLANATION = "simple"
ENGINEERING_EXPLANATION = "engineering"
INTERPRETATION_SECTIONS = [
    "Executive Summary",
    "What This Graph Shows",
    "Key Engineering Observations",
    "Design Implications",
    "Limitations",
]


def build_engineering_interpretation(
    graph_name: str,
    figure: go.Figure,
    metadata: dict[str, Any],
    mode: str = ENGINEERING_EXPLANATION,
) -> dict[str, Any]:
    """Return mode-aware interpretation derived from displayed chart data."""
    normalized_mode = SIMPLE_EXPLANATION if mode == SIMPLE_EXPLANATION else ENGINEERING_EXPLANATION
    if graph_name == "strength_surface":
        sections = _strength_surface_interpretation(figure, metadata, normalized_mode)
    elif graph_name == "optimization_landscape":
        sections = _optimization_interpretation(figure, metadata, normalized_mode)
    elif graph_name == "ply_failure_map":
        sections = _ply_failure_interpretation(figure, metadata, normalized_mode)
    elif graph_name == "material_benchmark":
        sections = _material_benchmark_interpretation(figure, metadata, normalized_mode)
    else:
        raise ValueError(f"Unsupported graph for interpretation: {graph_name}")
    return {
        "graph_name": graph_name,
        "mode": normalized_mode,
        "data_source": _data_source(graph_name, metadata),
        "sections": sections,
    }


def build_all_engineering_interpretations(mode: str = ENGINEERING_EXPLANATION) -> list[dict[str, Any]]:
    """Build report-ready interpretations for every engineering visualization."""
    builders = [
        ("strength_surface", build_strength_response_surface),
        ("optimization_landscape", build_optimization_landscape),
        ("ply_failure_map", build_ply_failure_map),
        ("material_benchmark", build_material_benchmark_3d),
    ]
    interpretations = []
    for graph_name, builder in builders:
        figure, metadata = builder()
        interpretations.append(build_engineering_interpretation(graph_name, figure, metadata, mode))
    return interpretations


def _strength_surface_interpretation(
    figure: go.Figure,
    metadata: dict[str, Any],
    mode: str,
) -> dict[str, str]:
    surface = figure.data[0]
    x_values = _as_float_array(surface.x)
    y_values = _as_float_array(surface.y)
    z_values = np.asarray(surface.z, dtype=float)
    max_row, max_col = np.unravel_index(int(np.argmax(z_values)), z_values.shape)
    min_row, min_col = np.unravel_index(int(np.argmin(z_values)), z_values.shape)
    max_strength = float(z_values[max_row, max_col])
    min_strength = float(z_values[min_row, min_col])
    max_fvf = float(x_values[max_col])
    max_temperature = float(y_values[max_row])
    min_fvf = float(x_values[min_col])
    min_temperature = float(y_values[min_row])
    fiber_effect = float(np.mean(np.ptp(z_values, axis=1)))
    temperature_effect = float(np.mean(np.ptp(z_values, axis=0)))
    dominant = "fiber volume fraction" if fiber_effect >= temperature_effect else "curing temperature"
    slope = np.abs(np.gradient(np.mean(z_values, axis=0), x_values))
    tail_start = max(1, int(len(slope) * 0.75))
    prior = float(np.mean(slope[:tail_start]))
    tail = float(np.mean(slope[tail_start:]))
    flattening = tail < prior

    if mode == SIMPLE_EXPLANATION:
        return {
            "Executive Summary": f"Predicted strength ranges from {min_strength:.1f} MPa to {max_strength:.1f} MPa across displayed manufacturing settings.",
            "What This Graph Shows": f"Chart varies fiber volume fraction from {x_values.min():.3f} to {x_values.max():.3f} and curing temperature from {y_values.min():.1f} °C to {y_values.max():.1f} °C, then uses trained model to predict strength.",
            "Key Engineering Observations": f"Highest displayed strength occurs near fiber volume fraction {max_fvf:.3f} and curing temperature {max_temperature:.1f} °C. Lowest occurs near {min_fvf:.3f} and {min_temperature:.1f} °C. Bigger visible influence comes from {dominant}.",
            "Design Implications": f"Moving toward region around fiber volume fraction {max_fvf:.3f} and curing temperature {max_temperature:.1f} °C gives strongest prediction within validated data range.",
            "Limitations": "Surface uses model inference inside training-data ranges. It does not prove experimental strength, manufacturability, or behavior outside displayed range.",
        }
    flattening_text = (
        f"Mean high-fiber-gradient {tail:.1f} MPa per fraction unit is below earlier gradient {prior:.1f}, indicating diminishing returns near upper fiber-volume range."
        if flattening
        else f"Mean high-fiber-gradient {tail:.1f} MPa per fraction unit does not fall below earlier gradient {prior:.1f}; clear diminishing returns are not supported by this grid."
    )
    return {
        "Executive Summary": f"ANN response surface predicts tensile-strength envelope of {min_strength:.1f}–{max_strength:.1f} MPa; maximum lies at V_f={max_fvf:.3f}, curing temperature={max_temperature:.1f} °C.",
        "What This Graph Shows": f"Plot sweeps V_f and curing temperature over ML-ready dataset bounds, holding other required ANN features at training-data medians/modes, then performs actual saved-model inference for {metadata['grid_points']} grid points.",
        "Key Engineering Observations": f"Average response range along V_f is {fiber_effect:.1f} MPa versus {temperature_effect:.1f} MPa along curing temperature, so stronger displayed sensitivity is {dominant}. {flattening_text}",
        "Design Implications": f"Process selection should prioritize validated region around V_f={max_fvf:.3f}, T_cure={max_temperature:.1f} °C when tensile strength is primary objective, while treating flattening behavior as model-local sensitivity not universal laminate law.",
        "Limitations": "ANN does not include stacking sequence mechanics. Surface depends on fixed median/mode values for non-plotted features and remains valid only inside observed dataset domain.",
    }


def _optimization_interpretation(
    figure: go.Figure,
    metadata: dict[str, Any],
    mode: str,
) -> dict[str, str]:
    candidate_trace = figure.data[0]
    load_factors = _as_float_array(candidate_trace.z)
    best_lambda = float(metadata["optimized_lambda_cs"])
    baseline_lambda = float(metadata["baseline_lambda_cs"])
    improvement = float(metadata["improvement_pct"])
    near_best_count = int(np.sum(load_factors >= 0.95 * best_lambda))
    median_lambda = float(np.median(load_factors))
    best_sequence = metadata.get("best_sequence", [])

    if mode == SIMPLE_EXPLANATION:
        return {
            "Executive Summary": f"Optimizer increased failure load factor from {baseline_lambda:.0f} to {best_lambda:.0f}, a {improvement:.1f}% improvement.",
            "What This Graph Shows": f"Chart shows {metadata['candidates_evaluated']} tested layer designs and their calculated failure load factors.",
            "Key Engineering Observations": f"{near_best_count} tested designs are within 95% of best result. Median tested design reaches λ_cs={median_lambda:.0f}.",
            "Design Implications": "Several strong candidates mean final choice can consider practical manufacturing constraints, not strength alone.",
            "Limitations": "Displayed candidates are evaluated search results, not proof of global best possible laminate.",
        }
    return {
        "Executive Summary": f"CLT optimizer raises λ_cs from {baseline_lambda:.0f} to {best_lambda:.0f}, producing +{improvement:.1f}% failure-load-factor gain under active constraints.",
        "What This Graph Shows": f"3D landscape plots {metadata['candidates_evaluated']} evaluated candidate sequences against ply count and strain-allowable λ_cs, with baseline and optimized designs highlighted.",
        "Key Engineering Observations": f"Best candidate sequence {_format_sequence(best_sequence)} exceeds candidate median λ_cs={median_lambda:.0f}; {near_best_count} candidates lie within 95% of optimum, indicating multiple high-performing laminate arrangements.",
        "Design Implications": "Search result supports stacking-sequence redesign as dominant lever for load-capacity improvement, while enabling secondary screening for balance, symmetry, ply drops, and manufacturing rules.",
        "Limitations": "Candidate list is ranked/evaluated output, not chronological convergence trace; convergence cannot be proven without iteration-order history. Optimization remains bounded by generated candidates and chosen failure criterion.",
    }


def _ply_failure_interpretation(
    figure: go.Figure,
    metadata: dict[str, Any],
    mode: str,
) -> dict[str, str]:
    trace = figure.data[0]
    ply_numbers = _as_float_array(trace.x)
    orientations = _as_float_array(trace.y)
    failure_indices = _as_float_array(trace.z)
    max_index_position = int(np.argmax(failure_indices))
    critical_ply = int(ply_numbers[max_index_position])
    critical_orientation = float(orientations[max_index_position])
    max_failure_index = float(failure_indices[max_index_position])
    threshold = float(np.percentile(failure_indices, 90))
    critical_mask = failure_indices >= threshold
    critical_orientations = orientations[critical_mask]
    dominant_orientation = _mode_orientation(critical_orientations)
    safety_margin = 1.0 - max_failure_index

    if mode == SIMPLE_EXPLANATION:
        return {
            "Executive Summary": f"Highest displayed failure index is {max_failure_index:.6f} at Ply {critical_ply}.",
            "What This Graph Shows": f"Chart checks all {metadata['ply_count']} plies and shows which layer is closest to failure under selected CLT load case.",
            "Key Engineering Observations": f"Most stressed orientation among top 10% failure-index plies is {dominant_orientation:.0f}°. Current maximum is far below 1.0.",
            "Design Implications": f"Ply {critical_ply} and {dominant_orientation:.0f}°-oriented plies should be checked first if load case or material allowables change.",
            "Limitations": "Failure index depends on current CLT material card, load case, and allowable method.",
        }
    return {
        "Executive Summary": f"Ply-level CLT map identifies Ply {critical_ply} at {critical_orientation:.0f}° as critical, with failure index {max_failure_index:.6f} at λ=1 and computed λ_cs={metadata['lambda_cs']:.2f}.",
        "What This Graph Shows": f"Distribution plots local ply orientation versus failure utilization for all {metadata['ply_count']} laminate plies from source-backed CLT evaluation.",
        "Key Engineering Observations": f"Top-decile failure-index plies concentrate most around {dominant_orientation:.0f}° orientation. Remaining margin to failure-index unity at λ=1 is {safety_margin:.6f}.",
        "Design Implications": f"Critical-ply review should focus on Ply {critical_ply}, adjacent plies, and {dominant_orientation:.0f}° orientation family before changing loads or allowables.",
        "Limitations": "Map represents strain-allowable criterion for current source case only. It does not include progressive damage, delamination, buckling, defects, or experimental scatter.",
    }


def _material_benchmark_interpretation(
    figure: go.Figure,
    metadata: dict[str, Any],
    mode: str,
) -> dict[str, str]:
    trace = figure.data[0]
    densities = _as_float_array(trace.x)
    strengths = _as_float_array(trace.z)
    materials = [str(value) for value in trace.text]
    customdata = np.asarray(trace.customdata, dtype=object)
    categories = [str(row[0]) for row in customdata]
    specific_strengths = np.asarray([float(row[1]) for row in customdata], dtype=float)
    strongest_index = int(np.argmax(strengths))
    lightest_index = int(np.argmin(densities))
    specific_index = int(np.argmax(specific_strengths))
    composite_indexes = [index for index, category in enumerate(categories) if "Composite" in category]
    best_composite_index = max(composite_indexes, key=lambda index: specific_strengths[index]) if composite_indexes else None
    aluminum_7075_index = materials.index("Aluminum 7075-T6") if "Aluminum 7075-T6" in materials else None
    ratio_text = _ratio_text(strengths, best_composite_index, aluminum_7075_index)
    missing = metadata.get("missing_requested_materials", [])

    if mode == SIMPLE_EXPLANATION:
        return {
            "Executive Summary": f"Strongest plotted material is {materials[strongest_index]} at {strengths[strongest_index]:.1f} MPa; best strength-per-weight material is {materials[specific_index]}.",
            "What This Graph Shows": f"Chart compares {metadata['row_count']} available materials by density, material category, and tensile strength.",
            "Key Engineering Observations": f"Lightest plotted material is {materials[lightest_index]} at {densities[lightest_index]:.3f} g/cm³. {ratio_text}",
            "Design Implications": "High specific strength favors lightweight aerospace design, but material choice still depends on temperature, corrosion, cost, damage tolerance, and manufacturing.",
            "Limitations": f"Only materials available in project data are shown. Missing requested materials: {_join_or_none(missing)}.",
        }
    return {
        "Executive Summary": f"Benchmark envelope shows {materials[strongest_index]} as maximum tensile-strength point ({strengths[strongest_index]:.1f} MPa) and {materials[specific_index]} as highest specific-strength point ({specific_strengths[specific_index]:.2f}).",
        "What This Graph Shows": f"3D material map plots density, category, and tensile strength from reference database rows plus ML-ready composite fiber-family aggregates; {metadata['row_count']} materials are displayed.",
        "Key Engineering Observations": f"Lowest density point is {materials[lightest_index]} ({densities[lightest_index]:.3f} g/cm³). {ratio_text} Specific-strength ranking separates lightweight composite efficiency from absolute tensile strength.",
        "Design Implications": "Aerospace screening should use specific strength for mass-critical structures and absolute strength for load-critical checks, then add environment, certification, joining, fatigue, and damage-tolerance constraints.",
        "Limitations": f"Composite points are training-dataset aggregates, not optimized laminate test coupons. Missing requested materials are not fabricated: {_join_or_none(missing)}.",
    }


def _data_source(graph_name: str, metadata: dict[str, Any]) -> str:
    if graph_name == "strength_surface":
        return str(metadata.get("dataset_path", ""))
    if graph_name == "optimization_landscape":
        return str(metadata.get("source", ""))
    if graph_name == "ply_failure_map":
        return str(metadata.get("source", ""))
    if graph_name == "material_benchmark":
        return f"{metadata.get('reference_database', '')}; {metadata.get('training_dataset', '')}"
    return ""


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _mode_orientation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    unique, counts = np.unique(values.astype(float), return_counts=True)
    return float(unique[int(np.argmax(counts))])


def _format_sequence(sequence: Any) -> str:
    if not isinstance(sequence, (list, tuple)) or not sequence:
        return "unavailable"
    return "[" + ", ".join(str(item) for item in sequence) + "]"


def _ratio_text(
    strengths: np.ndarray,
    composite_index: int | None,
    reference_index: int | None,
) -> str:
    if composite_index is None or reference_index is None or strengths[reference_index] == 0:
        return "Composite-to-Aluminum 7075-T6 ratio is unavailable from displayed rows."
    ratio = float(strengths[composite_index] / strengths[reference_index])
    return f"Best displayed composite strength is {ratio:.2f}× Aluminum 7075-T6."


def _join_or_none(values: Any) -> str:
    if not values:
        return "none"
    return ", ".join(str(value) for value in values)
