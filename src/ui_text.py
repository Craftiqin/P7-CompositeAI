"""Dual-audience UI copy and translation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_MODE_SIMPLE = "simple"
APP_MODE_ENGINEERING = "engineering"
DEFAULT_APP_MODE = APP_MODE_SIMPLE
TRAINING_DATASET_PATH = PROJECT_ROOT / "data" / "training" / "ml_ready_features.csv"
MODEL_METADATA_PATH = PROJECT_ROOT / "saved_models" / "model_metadata.json"

TERM_GLOSSARY: dict[str, dict[str, str]] = {
    "ANN/MLP": {
        "simple_label": "Artificial Intelligence Prediction Model",
        "tooltip": "Machine-learning model that predicts laminate strength from material and process inputs.",
        "explanation": "This is the trained prediction model used by CompositeAI for tensile-strength estimates.",
        "example": "If inputs change from Carbon/Epoxy to Glass/Polyester, this engine recalculates the predicted strength.",
    },
    "CLT": {
        "simple_label": "Composite Laminate Structural Analysis",
        "tooltip": "Mechanics model that checks laminate behavior using stiffness and loading equations.",
        "explanation": "Classical Laminate Theory calculates laminate stiffness, ply strains, and failure response.",
        "example": "CLT can show whether a stacking sequence is mechanically safe under a chosen load case.",
    },
    "Tensile Strength": {
        "simple_label": "Maximum Pulling Strength",
        "tooltip": "Maximum pulling stress a material can resist before failure.",
        "explanation": "Higher values usually indicate a stronger laminate under direct pulling loads.",
        "example": "A laminate rated at 1800 MPa can resist more pulling stress than one rated at 900 MPa.",
    },
    "Fiber Volume Fraction": {
        "simple_label": "Fiber Percentage Inside Material",
        "tooltip": "Share of laminate volume occupied by fibers.",
        "explanation": "This influences stiffness, strength, and weight.",
        "example": "A higher fiber percentage often improves strength, but manufacturing quality still matters.",
    },
    "Void Content": {
        "simple_label": "Manufacturing Defect Percentage",
        "tooltip": "Percentage of trapped voids or air pockets inside the material.",
        "explanation": "More voids usually reduce strength and quality.",
        "example": "A laminate with 5% void content is typically weaker than one with 1% void content.",
    },
    "Stacking Sequence": {
        "simple_label": "Layer Arrangement Pattern",
        "tooltip": "Order and orientation of laminate plies.",
        "explanation": "Ply arrangement controls directional stiffness and failure behavior.",
        "example": "A [0, 45, -45, 90] layup behaves differently from a [0, 0, 0, 0] layup.",
    },
    "Failure Index": {
        "simple_label": "Safety Utilization Score",
        "tooltip": "Indicator showing how close a ply is to its allowable limit.",
        "explanation": "Values closer to the limit indicate less remaining margin.",
        "example": "A higher failure index means the ply is using more of its allowable capacity.",
    },
    "Lambda_cs": {
        "simple_label": "Failure Load Capacity Factor",
        "tooltip": "Load multiplier before the selected CLT failure limit is reached.",
        "explanation": "Larger values mean the laminate can scale the defined load further before failure.",
        "example": "A failure load factor of 2 means the laminate can roughly double the current load case before the selected failure criterion triggers.",
    },
    "Q Matrix": {
        "simple_label": "Material Stiffness Matrix",
        "tooltip": "Lamina stiffness description in material directions.",
        "explanation": "This matrix is the starting point for ply stiffness calculations.",
        "example": "CLT uses the Q matrix before rotating it into laminate coordinates.",
    },
    "ABD Matrix": {
        "simple_label": "Laminate Structural Stiffness",
        "tooltip": "Combined extensional, coupling, and bending stiffness of the laminate.",
        "explanation": "This matrix describes how the whole laminate responds to load and curvature.",
        "example": "A larger A block means greater in-plane stiffness.",
    },
    "Residual Error": {
        "simple_label": "Prediction Difference",
        "tooltip": "Difference between predicted and observed value.",
        "explanation": "Residuals help show where the model underpredicts or overpredicts.",
        "example": "A +20 MPa residual means the model prediction is 20 MPa below the observed value.",
    },
    "R² Score": {
        "simple_label": "Model Accuracy Score",
        "tooltip": "Explained-variance score for model performance.",
        "explanation": "Values closer to 1 indicate stronger fit on validation data.",
        "example": "A score of 0.995 means the model explains nearly all variation in the validation target.",
    },
}


def get_app_mode(session_state: Any) -> str:
    """Return normalized app mode from Streamlit session state."""
    raw = str(session_state.get("app_mode", DEFAULT_APP_MODE)).strip().lower()
    return raw if raw in {APP_MODE_SIMPLE, APP_MODE_ENGINEERING} else DEFAULT_APP_MODE


def ui_term(term: str, mode: str) -> str:
    """Return mode-aware label for engineering term."""
    if mode != APP_MODE_SIMPLE:
        return term
    return TERM_GLOSSARY.get(term, {}).get("simple_label", term)


def term_help(term: str) -> str | None:
    """Return tooltip text for a glossary term."""
    return TERM_GLOSSARY.get(term, {}).get("tooltip")


def term_details(term: str) -> dict[str, str] | None:
    """Return full glossary entry if available."""
    entry = TERM_GLOSSARY.get(term)
    return dict(entry) if entry else None


def executive_summary(mode: str) -> dict[str, str]:
    """Return recruiter/judge-friendly summary copy."""
    if mode == APP_MODE_ENGINEERING:
        return {
            "headline": "Integrated ANN + CLT aerospace composite decision-support platform.",
            "what_it_does": "Predicts tensile strength from composite material/process inputs, checks laminate behavior with Classical Laminate Theory, benchmarks against discovered materials, and evaluates constrained stacking-sequence candidates.",
            "why_it_matters": "Combines data-driven speed with mechanics-based traceability for aerospace laminate studies.",
            "ai_role": "Validated ANN/MLP predicts tensile strength from locked dataset features.",
            "physics_role": "CLT computes laminate stiffness, ply strain/stress response, and source-compatible failure load factors.",
            "uniqueness": "One interface separates authoritative numerical engines from optional explanatory UX without fabricating comparisons.",
        }
    return {
        "headline": "AI plus engineering physics for smarter composite material decisions.",
        "what_it_does": "CompositeAI estimates how strong a composite laminate will be, explains what that result means, checks whether engineering physics supports the case, compares it with benchmark materials, and packages results into reports.",
        "why_it_matters": "It helps engineers, students, recruiters, and project judges understand technical value quickly without losing traceable numbers.",
        "ai_role": "Artificial intelligence learns patterns from composite data to predict strength fast.",
        "physics_role": "Engineering physics verifies laminate behavior under defined loads so decisions are not based on AI alone.",
        "uniqueness": "It combines prediction, verification, optimization, benchmark comparison, and reporting in one project-ready platform.",
    }


def prediction_distribution() -> dict[str, float]:
    """Return training-target distribution for dynamic prediction bands."""
    data = pd.read_csv(TRAINING_DATASET_PATH, usecols=["tensile_strength_mpa"])
    series = data["tensile_strength_mpa"].dropna()
    return {
        "min": float(series.min()),
        "q1": float(series.quantile(0.25)),
        "median": float(series.quantile(0.50)),
        "q3": float(series.quantile(0.75)),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def strength_status(predicted_strength_mpa: float) -> dict[str, str]:
    """Return dataset-relative status label and explanation."""
    dist = prediction_distribution()
    if predicted_strength_mpa >= dist["q3"]:
        return {
            "badge": "Excellent",
            "color": "green",
            "interpretation": "This prediction falls in the upper performance range of the locked training dataset.",
        }
    if predicted_strength_mpa >= dist["median"]:
        return {
            "badge": "Good",
            "color": "yellow",
            "interpretation": "This prediction is above the middle of the training distribution.",
        }
    if predicted_strength_mpa >= dist["q1"]:
        return {
            "badge": "Moderate",
            "color": "orange",
            "interpretation": "This prediction is below the median but still within the central dataset range.",
        }
    return {
        "badge": "Weak",
        "color": "red",
        "interpretation": "This prediction sits in the lower range of the training distribution.",
    }


def metadata_version() -> str:
    """Return project stage/version from model metadata if available."""
    if not MODEL_METADATA_PATH.exists():
        return "Unavailable"
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return str(metadata.get("project_stage", "Unavailable"))
