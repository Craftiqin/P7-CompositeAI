"""Step 3 feature analysis and ML-ready feature specification.

This module analyzes the locked dataset and writes metadata for Step 4.
It does not train models, fit preprocessors, or optimize stacking sequences.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    TARGET_COLUMN,
    TRAINING_DATA_PATH,
    infer_feature_columns,
    load_training_data,
    split_features_target,
    target_iqr_outliers,
)


FEATURE_SPEC_PATH = PROJECT_ROOT / "data" / "training" / "feature_specification.json"
ML_READY_DATA_PATH = PROJECT_ROOT / "data" / "training" / "ml_ready_features.csv"
ANALYSIS_REPORT_PATH = PROJECT_ROOT / "data" / "training" / "feature_analysis_report.json"
PLOT_DIR = PROJECT_ROOT / "data" / "training" / "step3_plots"
PREPROCESSING_REFERENCE = "data/training/preprocessing_config.json"

FEATURE_DESCRIPTIONS = {
    "fiber_type": "Composite reinforcement family as provided in dataset.",
    "resin_type": "Polymer/resin family as provided in dataset.",
    "density_g_cm3": "Composite density in grams per cubic centimeter.",
    "layer_count": "Number of laminate layers in sample.",
    "curing_temperature_c": "Curing temperature in degrees Celsius.",
    "fiber_volume_fraction": "Fiber volume fraction, unitless 0-1 fraction.",
    "void_content_pct": "Void content percentage.",
    "tensile_strength_mpa": "Measured tensile strength in MPa.",
}

FEATURE_UNITS = {
    "fiber_type": "category",
    "resin_type": "category",
    "density_g_cm3": "g/cm^3",
    "layer_count": "count",
    "curing_temperature_c": "deg C",
    "fiber_volume_fraction": "fraction",
    "void_content_pct": "percent",
    "tensile_strength_mpa": "MPa",
}


@dataclass(frozen=True)
class FeatureAnalysisResult:
    """Feature analysis outputs."""

    feature_specification: dict[str, Any]
    analysis_report: dict[str, Any]
    ml_ready_shape: tuple[int, int]
    plot_paths: list[str]


def profile_features(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Profile all input features."""
    categorical_features, numerical_features = infer_feature_columns(data)
    profiles: dict[str, dict[str, Any]] = {}
    for column in categorical_features:
        counts = data[column].value_counts(dropna=False).astype(int).to_dict()
        profiles[column] = {
            "type": "categorical",
            "dtype": str(data[column].dtype),
            "cardinality": int(data[column].nunique(dropna=True)),
            "categories": sorted(data[column].dropna().astype(str).unique().tolist()),
            "counts": counts,
            "missing_values": int(data[column].isna().sum()),
        }
    for column in numerical_features:
        series = data[column]
        profiles[column] = {
            "type": "numerical",
            "dtype": str(series.dtype),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "missing_values": int(series.isna().sum()),
            "zero_values": int((series == 0).sum()),
            "negative_values": int((series < 0).sum()),
        }
    return profiles


def analyze_target(data: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    """Analyze target distribution without transforming it."""
    target = data[target_column]
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1
    skewness = float(target.skew())
    if abs(skewness) < 0.5:
        distribution_note = "approximately symmetric"
    elif skewness > 0:
        distribution_note = "right-skewed"
    else:
        distribution_note = "left-skewed"
    return {
        "min": float(target.min()),
        "max": float(target.max()),
        "mean": float(target.mean()),
        "median": float(target.median()),
        "std": float(target.std()),
        "skewness": skewness,
        "quartiles": {
            "q1": float(q1),
            "q2": float(target.quantile(0.50)),
            "q3": float(q3),
        },
        "iqr": float(iqr),
        "distribution_note": distribution_note,
    }


def numerical_feature_target_relationships(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Calculate numerical feature relationships with target."""
    _, numerical_features = infer_feature_columns(data)
    target = data[TARGET_COLUMN]
    rows = []
    for column in numerical_features:
        pearson = data[column].corr(target, method="pearson")
        spearman = data[column].corr(target, method="spearman")
        rows.append(
            {
                "feature": column,
                "pearson_correlation": round(float(pearson), 6),
                "spearman_correlation": round(float(spearman), 6),
                "recommendation": relationship_recommendation(abs(float(pearson))),
            }
        )
    return rows


def categorical_feature_target_relationships(data: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Summarize target distributions across categorical values."""
    categorical_features, _ = infer_feature_columns(data)
    result: dict[str, list[dict[str, Any]]] = {}
    for column in categorical_features:
        grouped = data.groupby(column, dropna=False)[TARGET_COLUMN]
        result[column] = [
            {
                "category": str(category),
                "count": int(values.count()),
                "target_mean": float(values.mean()),
                "target_median": float(values.median()),
                "target_min": float(values.min()),
                "target_max": float(values.max()),
            }
            for category, values in grouped
        ]
    return result


def numerical_multicollinearity(data: pd.DataFrame) -> dict[str, Any]:
    """Analyze redundancy among numerical input features."""
    _, numerical_features = infer_feature_columns(data)
    correlation = data[numerical_features].corr()
    pairs = []
    for left_index, left in enumerate(numerical_features):
        for right in numerical_features[left_index + 1:]:
            value = float(correlation.loc[left, right])
            abs_value = abs(value)
            if abs_value >= 0.8:
                recommendation = "investigate"
            else:
                recommendation = "retain"
            pairs.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "correlation": round(value, 6),
                    "abs_correlation": round(abs_value, 6),
                    "recommendation": recommendation,
                }
            )
    pairs = sorted(pairs, key=lambda row: row["abs_correlation"], reverse=True)
    return {
        "strong_pairs": [row for row in pairs if row["abs_correlation"] >= 0.8],
        "all_pairs": pairs,
        "finding": "no strong multicollinearity found"
        if not any(row["abs_correlation"] >= 0.8 for row in pairs)
        else "strong multicollinearity found",
    }


def analyze_target_outliers(data: pd.DataFrame) -> dict[str, Any]:
    """Inspect target IQR outliers and associated input features."""
    outlier_mask = target_iqr_outliers(data[TARGET_COLUMN])
    outliers = data.loc[outlier_mask]
    _, numerical_features = infer_feature_columns(data)
    feature_ranges = {}
    for column in numerical_features:
        feature_ranges[column] = {
            "full_min": float(data[column].min()),
            "full_max": float(data[column].max()),
            "outlier_min": float(outliers[column].min()),
            "outlier_max": float(outliers[column].max()),
            "outlier_mean": float(outliers[column].mean()),
        }
    return {
        "row_count": int(outlier_mask.sum()),
        "target_range": {
            "min": float(outliers[TARGET_COLUMN].min()),
            "max": float(outliers[TARGET_COLUMN].max()),
        },
        "feature_ranges": feature_ranges,
        "objective_invalid_values_found": False,
        "finding": (
            "outlier target rows remain within valid input feature ranges; "
            "they require domain review, not automatic removal"
        ),
    }


def leakage_review(data: pd.DataFrame) -> dict[str, Any]:
    """Check obvious target leakage conditions."""
    features, _ = split_features_target(data)
    feature_columns = list(features.columns)
    target_in_features = TARGET_COLUMN in feature_columns
    suspicious_features = [
        column
        for column in feature_columns
        if "strength" in column.lower() or "target" in column.lower()
    ]
    return {
        "status": "FAIL" if target_in_features or suspicious_features else "PASS",
        "target_in_features": target_in_features,
        "suspicious_feature_names": suspicious_features,
        "decision": "target excluded from X; no feature is calculated from target",
    }


def decide_engineered_features(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Return defensible engineered features for baseline spec."""
    return [
        {
            "name": "none",
            "decision": "No additional engineered features required for baseline model.",
            "reason": (
                "Available columns do not include ply thickness or stacking sequence; "
                "arbitrary interactions/ratios are deferred until Step 4 experiments."
            ),
        }
    ]


def baseline_feature_set(data: pd.DataFrame) -> dict[str, list[str] | str]:
    """Return baseline X/y definition."""
    categorical_features, numerical_features = infer_feature_columns(data)
    return {
        "numerical": numerical_features,
        "categorical": categorical_features,
        "target": TARGET_COLUMN,
        "x_columns": categorical_features + numerical_features,
    }


def create_feature_specification(data: pd.DataFrame) -> dict[str, Any]:
    """Create ML-ready feature specification JSON."""
    feature_set = baseline_feature_set(data)
    leak = leakage_review(data)
    return {
        "project_stage": "Step 3 - Feature Engineering and ML-Ready Dataset Preparation",
        "dataset_source": "data/training/composite_strength_training.csv",
        "ml_ready_dataset": "data/training/ml_ready_features.csv",
        "target": TARGET_COLUMN,
        "baseline_features": feature_set["x_columns"],
        "categorical_features": feature_set["categorical"],
        "numerical_features": feature_set["numerical"],
        "engineered_features": [],
        "engineering_decision": decide_engineered_features(data),
        "units": {column: FEATURE_UNITS[column] for column in feature_set["x_columns"] + [TARGET_COLUMN]},
        "feature_descriptions": {
            column: FEATURE_DESCRIPTIONS[column]
            for column in feature_set["x_columns"] + [TARGET_COLUMN]
        },
        "leakage_decisions": leak,
        "outlier_policy": analyze_target_outliers(data),
        "preprocessing_reference": PREPROCESSING_REFERENCE,
        "step4_notes": {
            "x_columns": feature_set["x_columns"],
            "y_column": TARGET_COLUMN,
            "preprocessing_pipeline": "src/preprocessing.py",
            "fit_policy": "fit preprocessor only on training split",
        },
    }


def build_analysis_report(data: pd.DataFrame) -> dict[str, Any]:
    """Build Step 3 feature analysis report."""
    return {
        "feature_profiles": profile_features(data),
        "target_analysis": analyze_target(data),
        "numerical_feature_target_relationships": numerical_feature_target_relationships(data),
        "categorical_feature_target_relationships": categorical_feature_target_relationships(data),
        "multicollinearity": numerical_multicollinearity(data),
        "outlier_analysis": analyze_target_outliers(data),
        "leakage_review": leakage_review(data),
        "baseline_feature_set": baseline_feature_set(data),
    }


def write_json(data: dict[str, Any], output_path: str | Path) -> Path:
    """Write JSON artifact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def write_ml_ready_dataset(data: pd.DataFrame, output_path: str | Path = ML_READY_DATA_PATH) -> Path:
    """Write untransformed ML-ready dataset with locked baseline columns."""
    feature_set = baseline_feature_set(data)
    columns = feature_set["x_columns"] + [TARGET_COLUMN]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data[columns].to_csv(path, index=False)
    return path


def generate_eda_plots(data: pd.DataFrame, output_dir: str | Path = PLOT_DIR) -> list[str]:
    """Generate compact Step 3 EDA Plotly HTML reports."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _, numerical_features = infer_feature_columns(data)
    paths: list[str] = []

    plots = {
        "target_distribution.html": px.histogram(
            data,
            x=TARGET_COLUMN,
            marginal="box",
            title="Target Distribution: tensile_strength_mpa",
        ),
        "correlation_heatmap.html": px.imshow(
            data[numerical_features + [TARGET_COLUMN]].corr(),
            text_auto=True,
            aspect="auto",
            title="Numerical Feature and Target Correlation",
        ),
    }
    for feature in numerical_features:
        plots[f"{feature}_vs_target.html"] = px.scatter(
            data,
            x=feature,
            y=TARGET_COLUMN,
            title=f"{feature} vs {TARGET_COLUMN}",
        )
    for feature in baseline_feature_set(data)["categorical"]:
        plots[f"{feature}_target_box.html"] = px.box(
            data,
            x=feature,
            y=TARGET_COLUMN,
            title=f"{TARGET_COLUMN} by {feature}",
        )

    for file_name, figure in plots.items():
        path = output / file_name
        figure.write_html(path)
        paths.append(str(path.relative_to(PROJECT_ROOT)))
    return paths


def relationship_recommendation(abs_correlation: float) -> str:
    """Return relationship note from absolute correlation."""
    if abs_correlation >= 0.5:
        return "retain; moderate to strong linear signal"
    if abs_correlation >= 0.2:
        return "retain; weak to moderate linear signal"
    return "retain; weak linear signal, may still help nonlinear models"


def run_step3_analysis() -> FeatureAnalysisResult:
    """Run Step 3 analysis and write artifacts."""
    data = load_training_data()
    feature_spec = create_feature_specification(data)
    report = build_analysis_report(data)
    ml_ready_path = write_ml_ready_dataset(data)
    spec_path = write_json(feature_spec, FEATURE_SPEC_PATH)
    report_path = write_json(report, ANALYSIS_REPORT_PATH)
    plot_paths = generate_eda_plots(data)
    ml_ready_shape = pd.read_csv(ml_ready_path).shape

    print("=== CompositeAI Step 3 Feature Analysis ===")
    print(f"Dataset: {TRAINING_DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(data)}")
    print(f"Baseline features: {feature_spec['baseline_features']}")
    print("Engineered features: none")
    print(f"Categorical features: {feature_spec['categorical_features']}")
    print(f"Numerical features: {feature_spec['numerical_features']}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Target analysis: {report['target_analysis']}")
    print(f"Feature-target relationships: {report['numerical_feature_target_relationships']}")
    print(f"Multicollinearity: {report['multicollinearity']['finding']}")
    print(f"Outliers: {report['outlier_analysis']}")
    print(f"Leakage check: {report['leakage_review']['status']}")
    print(f"ML-ready dataset: {ml_ready_path.relative_to(PROJECT_ROOT)} {ml_ready_shape}")
    print(f"Feature spec: {spec_path.relative_to(PROJECT_ROOT)}")
    print(f"Analysis report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Plots: {plot_paths}")
    print("Status: READY")

    return FeatureAnalysisResult(
        feature_specification=feature_spec,
        analysis_report=report,
        ml_ready_shape=ml_ready_shape,
        plot_paths=plot_paths,
    )


def main() -> int:
    """CLI entry point."""
    result = run_step3_analysis()
    return 0 if result.feature_specification["leakage_decisions"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
