"""Step 4 model training and comparison utilities.

This module trains tensile-strength regression models with preprocessing inside
each sklearn Pipeline. Preprocessing is fit only on training rows.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import TARGET_COLUMN, build_preprocessor, infer_feature_columns


DATASET_PATH = PROJECT_ROOT / "data" / "training" / "ml_ready_features.csv"
FEATURE_SPEC_PATH = PROJECT_ROOT / "data" / "training" / "feature_specification.json"
COMPARISON_PATH = PROJECT_ROOT / "data" / "training" / "model_comparison.csv"
OUTLIER_EXPERIMENT_PATH = PROJECT_ROOT / "data" / "training" / "outlier_experiment.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_strength_model.joblib"
METADATA_PATH = PROJECT_ROOT / "saved_models" / "model_metadata.json"
PLOT_DIR = PROJECT_ROOT / "reports" / "step4_model_plots"
RANDOM_STATE = 42
TEST_SIZE = 0.20


@dataclass(frozen=True)
class ModelResult:
    """Train/test metrics for one trained model."""

    model: str
    train_mae: float
    test_mae: float
    train_rmse: float
    test_rmse: float
    train_r2: float
    test_r2: float


def load_feature_spec(path: str | Path = FEATURE_SPEC_PATH) -> dict[str, Any]:
    """Load Step 3 feature specification."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_ml_ready_data(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load ML-ready dataset artifact."""
    return pd.read_csv(path)


def validate_dataset_against_spec(data: pd.DataFrame, spec: dict[str, Any]) -> None:
    """Ensure dataset columns match Step 3 feature spec."""
    expected_columns = spec["baseline_features"] + [spec["target"]]
    if list(data.columns) != expected_columns:
        raise ValueError(
            "ML-ready dataset columns do not match feature specification. "
            f"Expected {expected_columns}, got {list(data.columns)}"
        )


def split_data(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create reproducible 80/20 train/test split."""
    features = data.drop(columns=[target_column])
    target = data[target_column]
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def build_model_registry(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Build baseline regression models."""
    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "ANN/MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=600,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


def build_training_pipeline(model: Any, numerical_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Combine preprocessing and estimator into one trainable pipeline."""
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, dict[str, np.ndarray]]]:
    """Train models and return metrics, pipelines, and predictions."""
    categorical_features, numerical_features = infer_feature_columns(
        pd.concat([x_train, y_train], axis=1),
        TARGET_COLUMN,
    )
    registry = models or build_model_registry()
    results: list[ModelResult] = []
    trained: dict[str, Pipeline] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}

    for name, model in registry.items():
        pipeline = build_training_pipeline(clone(model), numerical_features, categorical_features)
        pipeline.fit(x_train, y_train)
        train_pred = pipeline.predict(x_train)
        test_pred = pipeline.predict(x_test)
        train_metrics = evaluate_predictions(y_train, train_pred)
        test_metrics = evaluate_predictions(y_test, test_pred)
        results.append(
            ModelResult(
                model=name,
                train_mae=train_metrics["mae"],
                test_mae=test_metrics["mae"],
                train_rmse=train_metrics["rmse"],
                test_rmse=test_metrics["rmse"],
                train_r2=train_metrics["r2"],
                test_r2=test_metrics["r2"],
            )
        )
        trained[name] = pipeline
        predictions[name] = {"train": train_pred, "test": test_pred}

    comparison = pd.DataFrame([asdict(result) for result in results]).sort_values("test_rmse")
    return comparison, trained, predictions


def select_best_model(comparison: pd.DataFrame) -> str:
    """Select best model by test RMSE, then MAE, then R2."""
    ranked = comparison.sort_values(["test_rmse", "test_mae", "test_r2"], ascending=[True, True, False])
    return str(ranked.iloc[0]["model"])


def create_prediction_plots(
    y_test: pd.Series,
    predictions: np.ndarray,
    model_name: str,
    output_dir: str | Path = PLOT_DIR,
) -> dict[str, str]:
    """Create actual-vs-predicted, residual, and error distribution plots."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace("/", "_").replace(" ", "_")
    residuals = y_test.to_numpy() - predictions

    actual_pred_path = output / f"{safe_name}_actual_vs_predicted.html"
    min_value = min(float(y_test.min()), float(predictions.min()))
    max_value = max(float(y_test.max()), float(predictions.max()))
    plot_frame = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": predictions,
            "residual": residuals,
        }
    )
    actual_fig = px.scatter(
        plot_frame,
        x="actual",
        y="predicted",
        title=f"{model_name}: Actual vs Predicted",
        labels={"actual": "Actual tensile strength (MPa)", "predicted": "Predicted tensile strength (MPa)"},
    )
    actual_fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Actual = Predicted",
            line={"dash": "dash", "color": "red"},
        )
    )
    actual_fig.write_html(actual_pred_path)

    residual_path = output / f"{safe_name}_residuals.html"
    residual_fig = px.scatter(
        plot_frame,
        x="predicted",
        y="residual",
        title=f"{model_name}: Residual Plot",
        labels={"predicted": "Predicted tensile strength (MPa)", "residual": "Residual"},
    )
    residual_fig.add_hline(y=0, line_dash="dash", line_color="red")
    residual_fig.write_html(residual_path)

    error_path = output / f"{safe_name}_error_distribution.html"
    error_fig = px.histogram(
        plot_frame,
        x="residual",
        nbins=40,
        title=f"{model_name}: Prediction Error Distribution",
        labels={"residual": "Prediction error (MPa)"},
    )
    error_fig.write_html(error_path)

    return {
        "actual_vs_predicted": str(actual_pred_path.relative_to(PROJECT_ROOT)),
        "residuals": str(residual_path.relative_to(PROJECT_ROOT)),
        "error_distribution": str(error_path.relative_to(PROJECT_ROOT)),
    }


def run_outlier_experiment(
    best_model_name: str,
    trained_models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Run secondary experiment: remove target outliers from training rows only."""
    q1 = y_train.quantile(0.25)
    q3 = y_train.quantile(0.75)
    iqr = q3 - q1
    keep_mask = (y_train >= q1 - 1.5 * iqr) & (y_train <= q3 + 1.5 * iqr)
    model = clone(trained_models[best_model_name].named_steps["model"])
    categorical_features, numerical_features = infer_feature_columns(
        pd.concat([x_train, y_train], axis=1),
        TARGET_COLUMN,
    )
    pipeline = build_training_pipeline(model, numerical_features, categorical_features)
    pipeline.fit(x_train.loc[keep_mask], y_train.loc[keep_mask])
    test_pred = pipeline.predict(x_test)
    metrics = evaluate_predictions(y_test, test_pred)
    result = pd.DataFrame(
        [
            {
                "experiment": "all_data_primary",
                "model": best_model_name,
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "removed_train_outliers": 0,
                "test_mae": float(
                    mean_absolute_error(y_test, trained_models[best_model_name].predict(x_test))
                ),
                "test_rmse": float(
                    np.sqrt(mean_squared_error(y_test, trained_models[best_model_name].predict(x_test)))
                ),
                "test_r2": float(r2_score(y_test, trained_models[best_model_name].predict(x_test))),
            },
            {
                "experiment": "outlier_filtered_training_only",
                "model": best_model_name,
                "train_rows": int(keep_mask.sum()),
                "test_rows": int(len(y_test)),
                "removed_train_outliers": int((~keep_mask).sum()),
                "test_mae": metrics["mae"],
                "test_rmse": metrics["rmse"],
                "test_r2": metrics["r2"],
            },
        ]
    )
    return result


def save_artifacts(
    best_model_name: str,
    best_pipeline: Pipeline,
    comparison: pd.DataFrame,
    metadata: dict[str, Any],
    outlier_experiment: pd.DataFrame,
) -> None:
    """Save model, metadata, comparison, and outlier experiment artifacts."""
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    comparison.to_csv(COMPARISON_PATH, index=False)
    outlier_experiment.to_csv(OUTLIER_EXPERIMENT_PATH, index=False)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def package_versions() -> dict[str, str | None]:
    """Collect package/runtime versions."""
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "xgboost": None,
        "tensorflow": None,
    }
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except Exception:
        versions["xgboost"] = "not_available"
    try:
        import tensorflow as tf

        versions["tensorflow"] = tf.__version__
    except Exception:
        versions["tensorflow"] = "not_used"
    return versions


def run_training() -> dict[str, Any]:
    """Run Step 4 model training and artifact export."""
    data = load_ml_ready_data()
    spec = load_feature_spec()
    validate_dataset_against_spec(data, spec)
    x_train, x_test, y_train, y_test = split_data(data, spec["target"])
    comparison, trained_models, predictions = train_and_evaluate_models(x_train, x_test, y_train, y_test)
    best_model_name = select_best_model(comparison)
    best_pipeline = trained_models[best_model_name]
    plot_paths = create_prediction_plots(y_test, predictions[best_model_name]["test"], best_model_name)
    outlier_experiment = run_outlier_experiment(
        best_model_name,
        trained_models,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    best_row = comparison.loc[comparison["model"] == best_model_name].iloc[0].to_dict()
    metadata = {
        "project_stage": "Step 4 - Google Colab Model Training and Comparison",
        "training_environment": "local validation run; notebook is Google Colab-ready",
        "model_name": best_model_name,
        "training_dataset": "data/training/ml_ready_features.csv",
        "target": spec["target"],
        "feature_list": spec["baseline_features"],
        "train_test_split": {"test_size": TEST_SIZE, "train_rows": len(x_train), "test_rows": len(x_test)},
        "random_state": RANDOM_STATE,
        "preprocessing_description": {
            "numerical": "median imputation + StandardScaler",
            "categorical": "most-frequent imputation + OneHotEncoder(handle_unknown='ignore')",
            "fit_policy": "preprocessing fit only inside training Pipeline on training rows",
        },
        "training_date": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "mae": best_row["test_mae"],
            "rmse": best_row["test_rmse"],
            "r2": best_row["test_r2"],
            "train_mae": best_row["train_mae"],
            "train_rmse": best_row["train_rmse"],
            "train_r2": best_row["train_r2"],
        },
        "package_versions": package_versions(),
        "xgboost_status": "trained" if "XGBoost" in comparison["model"].tolist() else "not_available_locally",
        "plots": plot_paths,
        "artifacts": {
            "trained_model": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
            "comparison_csv": str(COMPARISON_PATH.relative_to(PROJECT_ROOT)),
            "outlier_experiment_csv": str(OUTLIER_EXPERIMENT_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    save_artifacts(best_model_name, best_pipeline, comparison, metadata, outlier_experiment)
    return {
        "comparison": comparison,
        "best_model": best_model_name,
        "metadata": metadata,
        "outlier_experiment": outlier_experiment,
    }


def main() -> int:
    """CLI entry point."""
    result = run_training()
    print("=== CompositeAI Step 4 Model Training ===")
    print(f"Dataset: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Best model: {result['best_model']}")
    print("\nModel comparison:")
    print(result["comparison"].to_string(index=False))
    print("\nOutlier experiment:")
    print(result["outlier_experiment"].to_string(index=False))
    print(f"\nSaved model: {BEST_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Metadata: {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Comparison CSV: {COMPARISON_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
