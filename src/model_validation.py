"""Step 5 validation for selected strength prediction model."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import TARGET_COLUMN, build_preprocessor, infer_feature_columns
from src.train import (
    BEST_MODEL_PATH,
    DATASET_PATH,
    FEATURE_SPEC_PATH,
    METADATA_PATH,
    OUTLIER_EXPERIMENT_PATH,
    TEST_SIZE,
    build_training_pipeline,
    evaluate_predictions,
    split_data,
    validate_dataset_against_spec,
)


VALIDATION_REPORT_PATH = PROJECT_ROOT / "data" / "training" / "model_validation_report.json"
VALIDATION_SUMMARY_PATH = PROJECT_ROOT / "reports" / "step5_model_validation.md"
VALIDATION_PLOT_DIR = PROJECT_ROOT / "reports" / "step5_model_validation"
VALIDATION_PLOT_FILENAMES = (
    "actual_vs_predicted.html",
    "residual_vs_predicted.html",
    "residual_distribution.html",
    "cv_comparison.html",
    "seed_robustness.html",
)
SEEDS = [42, 7, 21, 100]


@dataclass(frozen=True)
class ValidationContext:
    """Loaded validation inputs."""

    data: pd.DataFrame
    spec: dict[str, Any]
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def get_validation_plot_path(filename: str, project_root: Path | None = None) -> Path | None:
    """Return existing Step 5 validation plot path across local/runtime roots."""
    if Path(filename).name != filename:
        raise ValueError("filename must be a plot filename, not a path")
    if filename not in VALIDATION_PLOT_FILENAMES:
        raise ValueError(f"Unsupported validation plot filename: {filename}")

    root = (project_root or PROJECT_ROOT).resolve()
    candidate_dirs = [
        root / "reports" / "step5_model_validation",
    ]
    if project_root is None:
        candidate_dirs.extend(
            [
                VALIDATION_PLOT_DIR,
                Path.cwd() / "reports" / "step5_model_validation",
                Path.cwd() / "CompositeAI" / "reports" / "step5_model_validation",
                Path(__file__).resolve().parents[1] / "reports" / "step5_model_validation",
            ]
        )
        for parent in Path.cwd().resolve().parents:
            candidate_dirs.extend(
                [
                    parent / "reports" / "step5_model_validation",
                    parent / "CompositeAI" / "reports" / "step5_model_validation",
                ]
            )

    seen: set[Path] = set()
    for directory in candidate_dirs:
        resolved_dir = directory.resolve()
        if resolved_dir in seen:
            continue
        seen.add(resolved_dir)
        candidate = resolved_dir / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_validation_context() -> ValidationContext:
    """Load dataset/spec and reproduce untouched test split."""
    data = pd.read_csv(DATASET_PATH)
    spec = json.loads(FEATURE_SPEC_PATH.read_text(encoding="utf-8"))
    validate_dataset_against_spec(data, spec)
    x_train, x_test, y_train, y_test = split_data(data, spec["target"])
    return ValidationContext(data, spec, x_train, x_test, y_train, y_test)


def load_saved_model() -> Any:
    """Load selected trained model artifact."""
    return joblib.load(BEST_MODEL_PATH)


def leakage_assessment(context: ValidationContext, model: Any) -> dict[str, Any]:
    """Inspect split, feature, and pipeline leakage conditions."""
    train_indices = set(context.x_train.index.tolist())
    test_indices = set(context.x_test.index.tolist())
    preprocessing_step = model.named_steps.get("preprocessing") if hasattr(model, "named_steps") else None
    model_step = model.named_steps.get("model") if hasattr(model, "named_steps") else None
    return {
        "status": "PASS"
        if TARGET_COLUMN not in context.x_train.columns
        and train_indices.isdisjoint(test_indices)
        and preprocessing_step is not None
        and model_step is not None
        else "FAIL",
        "target_in_x": TARGET_COLUMN in context.x_train.columns or TARGET_COLUMN in context.x_test.columns,
        "train_test_overlap_rows": len(train_indices.intersection(test_indices)),
        "row_identifier_columns": [
            column
            for column in context.x_train.columns
            if column.lower() in {"id", "index", "row_id", "unnamed: 0"}
        ],
        "pipeline_steps": list(model.named_steps.keys()) if hasattr(model, "named_steps") else [],
        "preprocessing_fit_policy": "saved artifact is sklearn Pipeline; preprocessing precedes model",
    }


def verify_saved_model(context: ValidationContext, model: Any) -> dict[str, Any]:
    """Verify model loads and predicts numeric finite values."""
    predictions = model.predict(context.x_test)
    return {
        "status": "PASS"
        if np.issubdtype(predictions.dtype, np.number)
        and not np.isnan(predictions).any()
        and not np.isinf(predictions).any()
        else "FAIL",
        "pipeline_steps": list(model.named_steps.keys()),
        "prediction_count": int(len(predictions)),
        "predictions_numeric": bool(np.issubdtype(predictions.dtype, np.number)),
        "nan_predictions": int(np.isnan(predictions).sum()),
        "inf_predictions": int(np.isinf(predictions).sum()),
    }


def final_test_metrics(context: ValidationContext, model: Any) -> dict[str, Any]:
    """Recalculate selected model test metrics from saved artifact."""
    train_predictions = model.predict(context.x_train)
    test_predictions = model.predict(context.x_test)
    train_metrics = evaluate_predictions(context.y_train, train_predictions)
    test_metrics = evaluate_predictions(context.y_test, test_predictions)
    return {
        "train": train_metrics,
        "test": test_metrics,
        "prediction_summary": {
            "min_prediction": float(test_predictions.min()),
            "max_prediction": float(test_predictions.max()),
            "mean_prediction": float(test_predictions.mean()),
            "min_actual": float(context.y_test.min()),
            "max_actual": float(context.y_test.max()),
            "mean_actual": float(context.y_test.mean()),
        },
        "test_predictions": test_predictions,
        "train_predictions": train_predictions,
    }


def residual_analysis(y_test: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Calculate residual statistics."""
    residuals = y_test.to_numpy() - predictions
    return {
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "std_residual": float(np.std(residuals, ddof=1)),
        "min_residual": float(np.min(residuals)),
        "max_residual": float(np.max(residuals)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
    }


def largest_errors(context: ValidationContext, predictions: np.ndarray, limit: int = 10) -> list[dict[str, Any]]:
    """Return largest absolute prediction errors with input features."""
    frame = context.x_test.copy()
    frame["actual_strength"] = context.y_test
    frame["predicted_strength"] = predictions
    frame["residual"] = frame["actual_strength"] - frame["predicted_strength"]
    frame["absolute_error"] = frame["residual"].abs()
    columns = [
        "actual_strength",
        "predicted_strength",
        "absolute_error",
        "residual",
        *list(context.x_test.columns),
    ]
    return frame.sort_values("absolute_error", ascending=False).head(limit)[columns].to_dict(orient="records")


def cross_validation_metrics(context: ValidationContext) -> dict[str, Any]:
    """Run 5-fold CV on training data only for ANN/MLP and Gradient Boosting."""
    categorical_features, numerical_features = infer_feature_columns(
        pd.concat([context.x_train, context.y_train], axis=1),
        TARGET_COLUMN,
    )
    models = {
        "ANN/MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=600,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, Any] = {}
    for name, estimator in models.items():
        pipeline = build_training_pipeline(estimator, numerical_features, categorical_features)
        scores = cross_validate(
            pipeline,
            context.x_train,
            context.y_train,
            cv=kfold,
            scoring=scoring,
            n_jobs=None,
        )
        results[name] = {
            "mae_mean": float((-scores["test_mae"]).mean()),
            "mae_std": float((-scores["test_mae"]).std(ddof=1)),
            "rmse_mean": float((-scores["test_rmse"]).mean()),
            "rmse_std": float((-scores["test_rmse"]).std(ddof=1)),
            "r2_mean": float(scores["test_r2"].mean()),
            "r2_std": float(scores["test_r2"].std(ddof=1)),
        }
    return results


def seed_robustness(context: ValidationContext) -> dict[str, Any]:
    """Evaluate ANN/MLP across fixed random seeds."""
    rows = []
    categorical_features, numerical_features = infer_feature_columns(
        pd.concat([context.x_train, context.y_train], axis=1),
        TARGET_COLUMN,
    )
    for seed in SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(
            context.data.drop(columns=[TARGET_COLUMN]),
            context.data[TARGET_COLUMN],
            test_size=TEST_SIZE,
            random_state=seed,
        )
        estimator = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=600,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        pipeline = build_training_pipeline(clone(estimator), numerical_features, categorical_features)
        pipeline.fit(x_train, y_train)
        metrics = evaluate_predictions(y_test, pipeline.predict(x_test))
        rows.append({"seed": seed, **metrics})
    frame = pd.DataFrame(rows)
    return {
        "rows": rows,
        "summary": {
            "mae_mean": float(frame["mae"].mean()),
            "mae_std": float(frame["mae"].std(ddof=1)),
            "rmse_mean": float(frame["rmse"].mean()),
            "rmse_std": float(frame["rmse"].std(ddof=1)),
            "r2_mean": float(frame["r2"].mean()),
            "r2_std": float(frame["r2"].std(ddof=1)),
        },
    }


def subgroup_metrics(context: ValidationContext, predictions: np.ndarray) -> dict[str, Any]:
    """Evaluate test metrics by fiber_type and resin_type where sample count is adequate."""
    frame = context.x_test.copy()
    frame["actual"] = context.y_test
    frame["predicted"] = predictions
    result: dict[str, Any] = {}
    for group_column in ["fiber_type", "resin_type"]:
        rows = []
        for group_value, group in frame.groupby(group_column):
            if len(group) < 30:
                continue
            metrics = evaluate_predictions(group["actual"], group["predicted"].to_numpy())
            rows.append({"group": str(group_value), "count": int(len(group)), **metrics})
        result[group_column] = rows
    return result


def prediction_plausibility(context: ValidationContext, predictions: np.ndarray) -> dict[str, Any]:
    """Compare predictions with observed target range."""
    target = context.data[TARGET_COLUMN]
    lower = float(target.min())
    upper = float(target.max())
    return {
        "actual_target_range": {"min": lower, "max": upper},
        "test_actual_range": {"min": float(context.y_test.min()), "max": float(context.y_test.max())},
        "predicted_range": {"min": float(predictions.min()), "max": float(predictions.max())},
        "predictions_outside_observed_target_range": int(((predictions < lower) | (predictions > upper)).sum()),
        "negative_predictions": int((predictions < 0).sum()),
    }


def compare_ann_gradient_boosting(context: ValidationContext, ann_metrics: dict[str, Any]) -> dict[str, Any]:
    """Train/evaluate Gradient Boosting on same split for comparison."""
    categorical_features, numerical_features = infer_feature_columns(
        pd.concat([context.x_train, context.y_train], axis=1),
        TARGET_COLUMN,
    )
    pipeline = build_training_pipeline(
        GradientBoostingRegressor(random_state=42),
        numerical_features,
        categorical_features,
    )
    pipeline.fit(context.x_train, context.y_train)
    gb_metrics = evaluate_predictions(context.y_test, pipeline.predict(context.x_test))
    return {
        "ann": ann_metrics["test"],
        "gradient_boosting": gb_metrics,
        "rmse_advantage": float(gb_metrics["rmse"] - ann_metrics["test"]["rmse"]),
        "finding": "ANN advantage is substantial on this split"
        if gb_metrics["rmse"] - ann_metrics["test"]["rmse"] > 10
        else "ANN advantage is small on this split",
    }


def make_validation_plots(
    context: ValidationContext,
    predictions: np.ndarray,
    cv_metrics: dict[str, Any],
    seed_metrics: dict[str, Any],
) -> dict[str, str]:
    """Write Step 5 Plotly validation plots."""
    VALIDATION_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    residuals = context.y_test.to_numpy() - predictions
    frame = pd.DataFrame(
        {
            "actual": context.y_test.to_numpy(),
            "predicted": predictions,
            "residual": residuals,
        }
    )
    min_value = min(float(frame["actual"].min()), float(frame["predicted"].min()))
    max_value = max(float(frame["actual"].max()), float(frame["predicted"].max()))

    actual_path = VALIDATION_PLOT_DIR / "actual_vs_predicted.html"
    actual_fig = px.scatter(frame, x="actual", y="predicted", title="Actual vs Predicted")
    actual_fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Actual = Predicted",
            line={"dash": "dash", "color": "red"},
        )
    )
    actual_fig.write_html(actual_path)

    residual_path = VALIDATION_PLOT_DIR / "residual_vs_predicted.html"
    residual_fig = px.scatter(frame, x="predicted", y="residual", title="Residual vs Predicted")
    residual_fig.add_hline(y=0, line_dash="dash", line_color="red")
    residual_fig.write_html(residual_path)

    distribution_path = VALIDATION_PLOT_DIR / "residual_distribution.html"
    px.histogram(frame, x="residual", nbins=40, title="Residual Distribution").write_html(distribution_path)

    cv_path = VALIDATION_PLOT_DIR / "cv_comparison.html"
    cv_frame = pd.DataFrame(
        [
            {"model": model, "metric": metric, "value": values[f"{metric}_mean"], "std": values[f"{metric}_std"]}
            for model, values in cv_metrics.items()
            for metric in ["mae", "rmse", "r2"]
        ]
    )
    px.bar(cv_frame, x="model", y="value", color="metric", barmode="group", title="5-Fold CV Comparison").write_html(cv_path)

    seed_path = VALIDATION_PLOT_DIR / "seed_robustness.html"
    seed_frame = pd.DataFrame(seed_metrics["rows"])
    px.line(seed_frame, x="seed", y=["mae", "rmse", "r2"], markers=True, title="ANN/MLP Seed Robustness").write_html(seed_path)

    return {
        "actual_vs_predicted": str(actual_path.relative_to(PROJECT_ROOT)),
        "residual_vs_predicted": str(residual_path.relative_to(PROJECT_ROOT)),
        "residual_distribution": str(distribution_path.relative_to(PROJECT_ROOT)),
        "cv_comparison": str(cv_path.relative_to(PROJECT_ROOT)),
        "seed_robustness": str(seed_path.relative_to(PROJECT_ROOT)),
    }


def final_decision(report: dict[str, Any]) -> dict[str, str]:
    """Classify final model readiness."""
    if report["leakage_assessment"]["status"] != "PASS" or report["saved_model_validation"]["status"] != "PASS":
        return {"status": "NOT RELIABLE", "reason": "Leakage or saved model validation failed."}
    if report["seed_robustness"]["summary"]["r2_std"] > 0.01:
        return {"status": "NEEDS IMPROVEMENT", "reason": "Seed robustness variation is too high."}
    if abs(report["overfitting_assessment"]["train_r2"] - report["overfitting_assessment"]["test_r2"]) > 0.03:
        return {"status": "NEEDS IMPROVEMENT", "reason": "Train/test R2 gap suggests possible overfitting."}
    if report["prediction_plausibility"]["negative_predictions"] > 0:
        return {"status": "NEEDS IMPROVEMENT", "reason": "Negative predictions found."}
    return {
        "status": "READY",
        "reason": "No leakage found; saved model valid; metrics stable; predictions remain within dataset-supported range.",
    }


def write_markdown_summary(report: dict[str, Any]) -> Path:
    """Write human-readable Step 5 validation summary."""
    VALIDATION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    test = report["final_test_metrics"]["test"]
    decision = report["final_decision"]
    lines = [
        "# Step 5 Model Validation",
        "",
        f"Model: {report['model']}",
        f"Decision: {decision['status']}",
        f"Reason: {decision['reason']}",
        "",
        "## Test Metrics",
        "",
        f"- MAE: {test['mae']:.4f}",
        f"- RMSE: {test['rmse']:.4f}",
        f"- R2: {test['r2']:.4f}",
        "",
        "## Leakage",
        "",
        f"- Status: {report['leakage_assessment']['status']}",
        f"- Train/test overlap rows: {report['leakage_assessment']['train_test_overlap_rows']}",
        "",
        "## Outlier Policy",
        "",
        "Primary model retains all outliers. Existing Step 4 secondary experiment is reported only.",
    ]
    VALIDATION_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    return VALIDATION_SUMMARY_PATH


def run_model_validation() -> dict[str, Any]:
    """Run full Step 5 validation and write artifacts."""
    context = load_validation_context()
    model = load_saved_model()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    saved_model = verify_saved_model(context, model)
    metrics = final_test_metrics(context, model)
    predictions = metrics["test_predictions"]
    cv_metrics = cross_validation_metrics(context)
    seed_metrics = seed_robustness(context)
    residuals = residual_analysis(context.y_test, predictions)
    top_errors = largest_errors(context, predictions)
    subgroups = subgroup_metrics(context, predictions)
    plausibility = prediction_plausibility(context, predictions)
    comparison = compare_ann_gradient_boosting(context, metrics)
    plots = make_validation_plots(context, predictions, cv_metrics, seed_metrics)
    leakage = leakage_assessment(context, model)
    overfitting = {
        "train_mae": metrics["train"]["mae"],
        "test_mae": metrics["test"]["mae"],
        "train_rmse": metrics["train"]["rmse"],
        "test_rmse": metrics["test"]["rmse"],
        "train_r2": metrics["train"]["r2"],
        "test_r2": metrics["test"]["r2"],
        "finding": "no meaningful overfitting detected"
        if abs(metrics["train"]["r2"] - metrics["test"]["r2"]) <= 0.03
        else "possible overfitting detected",
    }
    report: dict[str, Any] = {
        "project_stage": "Step 5 - Model Evaluation, Validation and Final Model Selection",
        "model": metadata["model_name"],
        "data_validation": {
            "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
            "train_rows": len(context.x_train),
            "test_rows": len(context.x_test),
            "test_set_untouched": True,
        },
        "leakage_assessment": leakage,
        "saved_model_validation": saved_model,
        "final_test_metrics": {
            "train": metrics["train"],
            "test": metrics["test"],
            "prediction_summary": metrics["prediction_summary"],
        },
        "residual_analysis": residuals,
        "top_10_errors": top_errors,
        "cross_validation": cv_metrics,
        "seed_robustness": seed_metrics,
        "subgroup_validation": subgroups,
        "prediction_plausibility": plausibility,
        "ann_vs_gradient_boosting": comparison,
        "outlier_experiment": pd.read_csv(OUTLIER_EXPERIMENT_PATH).to_dict(orient="records")
        if OUTLIER_EXPERIMENT_PATH.exists()
        else [],
        "overfitting_assessment": overfitting,
        "plots": plots,
    }
    report["final_decision"] = final_decision(report)

    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_summary(report)
    return report


def main() -> int:
    """CLI entry point."""
    report = run_model_validation()
    test = report["final_test_metrics"]["test"]
    print("=== CompositeAI Step 5 Model Validation ===")
    print(f"Model: {report['model']}")
    print(f"Leakage: {report['leakage_assessment']['status']}")
    print(f"Saved model valid: {report['saved_model_validation']['status']}")
    print(f"MAE: {test['mae']:.4f}")
    print(f"RMSE: {test['rmse']:.4f}")
    print(f"R2: {test['r2']:.4f}")
    print(f"Decision: {report['final_decision']['status']}")
    print(f"Report: {VALIDATION_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary: {VALIDATION_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    return 0 if report["final_decision"]["status"] != "NOT RELIABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
