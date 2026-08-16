"""Real-data 3D engineering visualization builders for CompositeAI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.clt import calculate_strain_allowable_failure_load
from src.material_benchmark import calculate_specific_strength, load_reference_materials
from src.optimizer import default_demo_optimization, load_tu_delft_demo_case
from src.predict import REQUIRED_FEATURES, load_training_reference, predict_strength


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "training" / "ml_ready_features.csv"
REFERENCE_DATABASE_PATH = PROJECT_ROOT / "data" / "reference_materials" / "material_benchmark_database.json"


def build_strength_response_surface(
    current_context: dict[str, Any] | None = None,
    grid_size: int = 18,
) -> tuple[go.Figure, dict[str, Any]]:
    """Build ANN response surface from valid training ranges and model inference."""
    training_data = load_training_reference()
    base_input = _median_model_input(training_data)
    fvf_values = np.linspace(
        float(training_data["fiber_volume_fraction"].min()),
        float(training_data["fiber_volume_fraction"].max()),
        grid_size,
    )
    temperature_values = np.linspace(
        float(training_data["curing_temperature_c"].min()),
        float(training_data["curing_temperature_c"].max()),
        grid_size,
    )

    rows: list[dict[str, Any]] = []
    for curing_temperature in temperature_values:
        for fiber_volume_fraction in fvf_values:
            row = dict(base_input)
            row["fiber_volume_fraction"] = float(fiber_volume_fraction)
            row["curing_temperature_c"] = float(curing_temperature)
            rows.append(row)

    prediction_result = predict_strength(pd.DataFrame(rows))
    predictions = np.array(prediction_result["predictions"], dtype=float).reshape(
        len(temperature_values),
        len(fvf_values),
    )
    min_prediction = float(np.min(predictions))
    max_prediction = float(np.max(predictions))

    figure = go.Figure(
        data=[
            go.Surface(
                x=fvf_values,
                y=temperature_values,
                z=predictions,
                colorscale="Viridis",
                colorbar={"title": "MPa"},
                hovertemplate=(
                    "Fiber Volume Fraction=%{x:.3f}<br>"
                    "Curing Temperature=%{y:.1f} °C<br>"
                    "Predicted Strength=%{z:.2f} MPa<extra></extra>"
                ),
                name="ANN prediction surface",
            )
        ]
    )

    marker = _current_prediction_marker(current_context)
    if marker is not None:
        figure.add_trace(
            go.Scatter3d(
                x=[marker["fiber_volume_fraction"]],
                y=[marker["curing_temperature_c"]],
                z=[marker["predicted_tensile_strength_mpa"]],
                mode="markers",
                marker={"size": 7, "color": "#ff3b30", "symbol": "diamond"},
                name="Current user prediction",
                hovertemplate=(
                    "Current Prediction<br>"
                    "Fiber Volume Fraction=%{x:.3f}<br>"
                    "Curing Temperature=%{y:.1f} °C<br>"
                    "Predicted Strength=%{z:.2f} MPa<extra></extra>"
                ),
            )
        )

    _apply_3d_layout(
        figure,
        "Composite Strength Response Surface",
        "Fiber Volume Fraction",
        "Curing Temperature (°C)",
        "Predicted Tensile Strength (MPa)",
    )
    return figure, {
        "dataset_path": str(TRAINING_DATA_PATH),
        "grid_points": len(rows),
        "min_predicted_strength_mpa": min_prediction,
        "max_predicted_strength_mpa": max_prediction,
        "current_marker_available": marker is not None,
    }


def build_optimization_landscape(max_candidates: int = 250) -> tuple[go.Figure, dict[str, Any]]:
    """Build 3D optimizer landscape from evaluated candidate results."""
    result = default_demo_optimization(max_candidates=max_candidates)
    candidates = result["evaluated_candidates"]
    candidate_indexes = list(range(1, len(candidates) + 1))
    ply_counts = [len(row["sequence"]) for row in candidates]
    load_factors = [float(row["lambda_cs"]) for row in candidates]
    hover_text = [
        f"Sequence: {row['sequence']}<br>Failure Mode: {row['failure_mode']}"
        for row in candidates
    ]

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=candidate_indexes,
                y=ply_counts,
                z=load_factors,
                mode="markers",
                marker={
                    "size": 4,
                    "color": load_factors,
                    "colorscale": "Turbo",
                    "colorbar": {"title": "λ_cs"},
                },
                text=hover_text,
                name="Evaluated candidates",
                hovertemplate=(
                    "Candidate=%{x}<br>"
                    "Ply Count=%{y}<br>"
                    "λ_cs=%{z:.2f}<br>%{text}<extra></extra>"
                ),
            )
        ]
    )

    baseline = result["baseline"]
    best_index = load_factors.index(max(load_factors)) + 1
    _add_design_marker(
        figure,
        "Baseline",
        0,
        int(result["ply_count"]),
        float(baseline["lambda_cs"]),
        "#ff9500",
    )
    _add_design_marker(
        figure,
        "Optimized",
        best_index,
        int(result["ply_count"]),
        float(result["best_lambda_cs"]),
        "#00d084",
    )
    figure.update_layout(
        annotations=[
            {
                "text": f"Baseline: lambda_cs = {baseline['lambda_cs']:.0f}",
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.96,
                "showarrow": False,
            },
            {
                "text": f"Optimized: lambda_cs = {result['best_lambda_cs']:.0f}",
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.90,
                "showarrow": False,
            },
            {
                "text": f"Improvement: +{result['improvement_pct']:.1f}%",
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.84,
                "showarrow": False,
            },
        ]
    )
    _apply_3d_layout(
        figure,
        "Optimization Search Landscape",
        "Candidate Design Index",
        "Sequence Complexity / Ply Count",
        "Failure Load Factor (lambda_cs)",
    )
    return figure, {
        "source": "TU Delft source-backed optimizer",
        "candidates_evaluated": len(candidates),
        "baseline_lambda_cs": float(baseline["lambda_cs"]),
        "optimized_lambda_cs": float(result["best_lambda_cs"]),
        "improvement_pct": float(result["improvement_pct"]),
        "best_sequence": result["best_sequence"],
    }


def build_ply_failure_map() -> tuple[go.Figure, dict[str, Any]]:
    """Build ply failure map from CLT strain-allowable results."""
    case = load_tu_delft_demo_case()
    result = calculate_strain_allowable_failure_load(
        case["baseline_sequence"],
        case["material"],
        case["load_case"],
        case["allowables"],
    )
    ply_rows = result["ply_results"]
    ply_numbers = [row["ply_number"] for row in ply_rows]
    orientations = [row["angle_deg"] for row in ply_rows]
    failure_indices = [float(row["failure_index_at_lambda_1"]) for row in ply_rows]
    max_index = failure_indices.index(max(failure_indices))

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=ply_numbers,
                y=orientations,
                z=failure_indices,
                mode="markers+lines",
                marker={
                    "size": 5,
                    "color": failure_indices,
                    "colorscale": "Inferno",
                    "colorbar": {"title": "Failure Index"},
                },
                name="Ply failure index",
                hovertemplate=(
                    "Ply=%{x}<br>"
                    "Orientation=%{y}°<br>"
                    "Failure Index=%{z:.6f}<extra></extra>"
                ),
            ),
            go.Scatter3d(
                x=[ply_numbers[max_index]],
                y=[orientations[max_index]],
                z=[failure_indices[max_index]],
                mode="markers",
                marker={"size": 9, "color": "#ff3b30", "symbol": "diamond"},
                name="Critical ply",
                hovertemplate="Critical Ply=%{x}<br>Orientation=%{y}°<br>Failure Index=%{z:.6f}<extra></extra>",
            ),
        ]
    )
    _apply_3d_layout(
        figure,
        "Ply Failure Distribution",
        "Ply Number",
        "Ply Orientation",
        "Failure Index",
    )
    return figure, {
        "source": case["source_file"],
        "ply_count": len(ply_rows),
        "critical_ply": result["critical_ply"],
        "max_failure_index": max(failure_indices),
        "lambda_cs": float(result["lambda_cs"]),
    }


def build_material_benchmark_3d() -> tuple[go.Figure, dict[str, Any]]:
    """Build material 3D benchmark from reference DB and ML-ready composite data."""
    requested_materials = {
        "Aluminum 2024-T3",
        "Aluminum 7075-T6",
        "Titanium Ti-6Al-4V",
        "Stainless Steel 316L",
        "Carbon Fiber Composite",
        "Glass Fiber Composite",
        "Aramid Composite",
    }
    reference_rows = [
        {
            "material": row["material"],
            "category": row["category"],
            "density_g_cm3": float(row["density_g_cm3"]),
            "tensile_strength_mpa": float(row["tensile_strength_mpa"]),
            "specific_strength": calculate_specific_strength(
                row["tensile_strength_mpa"],
                row["density_g_cm3"],
            ),
            "source": "reference_database",
        }
        for row in load_reference_materials()
        if row["material"] in requested_materials
    ]
    composite_rows = _composite_material_rows()
    rows = reference_rows + composite_rows
    missing_requested = sorted(requested_materials - {row["material"] for row in rows})
    if not rows:
        raise ValueError("No material benchmark rows available from project data.")

    frame = pd.DataFrame(rows)
    categories = sorted(frame["category"].unique().tolist())
    category_codes = {category: index for index, category in enumerate(categories)}
    frame["category_code"] = frame["category"].map(category_codes)

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=frame["density_g_cm3"],
                y=frame["category_code"],
                z=frame["tensile_strength_mpa"],
                mode="markers",
                marker={
                    "size": 8,
                    "color": frame["specific_strength"],
                    "colorscale": "Viridis",
                    "colorbar": {"title": "Specific Strength"},
                },
                text=frame["material"],
                customdata=[
                    [row["category"], row["specific_strength"], row["source"]]
                    for row in rows
                ],
                hovertemplate=(
                    "Material=%{text}<br>"
                    "Density=%{x:.3f} g/cm³<br>"
                    "Category=%{customdata[0]}<br>"
                    "Tensile Strength=%{z:.2f} MPa<br>"
                    "Specific Strength=%{customdata[1]:.2f}<br>"
                    "Source=%{customdata[2]}<extra></extra>"
                ),
            )
        ]
    )
    _apply_3d_layout(
        figure,
        "Material Performance Benchmark",
        "Density (g/cm³)",
        "Material Category",
        "Tensile Strength (MPa)",
    )
    figure.update_layout(
        scene={
            **figure.layout.scene.to_plotly_json(),
            "yaxis": {
                "title": "Material Category",
                "tickmode": "array",
                "tickvals": list(category_codes.values()),
                "ticktext": list(category_codes.keys()),
            },
        }
    )
    return figure, {
        "reference_database": str(REFERENCE_DATABASE_PATH),
        "training_dataset": str(TRAINING_DATA_PATH),
        "materials_plotted": frame["material"].tolist(),
        "missing_requested_materials": missing_requested,
        "row_count": len(frame),
    }


def _median_model_input(training_data: pd.DataFrame) -> dict[str, Any]:
    """Build valid fixed model input from real training-data medians/modes."""
    values: dict[str, Any] = {}
    for feature in REQUIRED_FEATURES:
        series = training_data[feature]
        if pd.api.types.is_numeric_dtype(series):
            values[feature] = float(series.median())
        else:
            values[feature] = str(series.mode(dropna=True).iloc[0])
    return values


def _current_prediction_marker(current_context: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(current_context, dict):
        return None
    input_features = current_context.get("input_features")
    prediction = current_context.get("predicted_tensile_strength_mpa")
    if not isinstance(input_features, dict) or prediction is None:
        return None
    return {
        "fiber_volume_fraction": float(input_features["fiber_volume_fraction"]),
        "curing_temperature_c": float(input_features["curing_temperature_c"]),
        "predicted_tensile_strength_mpa": float(prediction),
    }


def _add_design_marker(
    figure: go.Figure,
    name: str,
    candidate_index: int,
    ply_count: int,
    lambda_cs: float,
    color: str,
) -> None:
    figure.add_trace(
        go.Scatter3d(
            x=[candidate_index],
            y=[ply_count],
            z=[lambda_cs],
            mode="markers",
            marker={"size": 9, "color": color, "symbol": "diamond"},
            name=name,
            hovertemplate=f"{name}<br>Candidate=%{{x}}<br>Ply Count=%{{y}}<br>λ_cs=%{{z:.2f}}<extra></extra>",
        )
    )


def _composite_material_rows() -> list[dict[str, Any]]:
    training_data = load_training_reference()
    rows: list[dict[str, Any]] = []
    name_map = {
        "Carbon": "Carbon Fiber Composite",
        "Glass": "Glass Fiber Composite",
        "Aramid": "Aramid Composite",
    }
    for fiber_type, material_name in name_map.items():
        subset = training_data[training_data["fiber_type"].astype(str) == fiber_type]
        if subset.empty:
            continue
        strength = float(subset["tensile_strength_mpa"].mean())
        density = float(subset["density_g_cm3"].mean())
        rows.append(
            {
                "material": material_name,
                "category": "Composite Dataset",
                "density_g_cm3": density,
                "tensile_strength_mpa": strength,
                "specific_strength": calculate_specific_strength(strength, density),
                "source": "ml_ready_training_dataset",
            }
        )
    return rows


def _apply_3d_layout(
    figure: go.Figure,
    title: str,
    x_title: str,
    y_title: str,
    z_title: str,
) -> None:
    figure.update_layout(
        title=title,
        template="plotly_dark",
        autosize=True,
        margin={"l": 0, "r": 0, "b": 0, "t": 48},
        scene={
            "xaxis_title": x_title,
            "yaxis_title": y_title,
            "zaxis_title": z_title,
        },
    )
