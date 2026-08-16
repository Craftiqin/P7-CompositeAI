"""Project configuration values."""

import os
from pathlib import Path

APP_TITLE = (
    "AI-Based Strength Prediction and Stacking Sequence Optimization of "
    "Aerospace Composite Laminates"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
KAGGLE_DATA_DIR = DATA_DIR / "kaggle"
NASA_DATA_DIR = DATA_DIR / "nasa"
PAPERS_DATA_DIR = DATA_DIR / "papers"
UPLOADED_DATA_DIR = DATA_DIR / "uploaded"
MERGED_DATA_DIR = DATA_DIR / "merged"
METADATA_DIR = DATA_DIR / "metadata"
MODEL_DIR = PROJECT_ROOT / "models"
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_models"
REPORT_DIR = PROJECT_ROOT / "reports"

DATASET_SOURCE_DIRS = {
    "Raw": RAW_DATA_DIR,
    "Kaggle": KAGGLE_DATA_DIR,
    "NASA": NASA_DATA_DIR,
    "Papers": PAPERS_DATA_DIR,
    "Uploaded": UPLOADED_DATA_DIR,
}

CONFIDENCE_THRESHOLD = 0.80
DEFAULT_GEMINI_PRIMARY_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_FALLBACK_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
)
GEMINI_PRIMARY_MODEL = os.getenv("GEMINI_PRIMARY_MODEL", DEFAULT_GEMINI_PRIMARY_MODEL)
GEMINI_FALLBACK_MODELS = tuple(
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        ",".join(DEFAULT_GEMINI_FALLBACK_MODELS),
    ).split(",")
    if model.strip()
)
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL_NAME = GEMINI_PRIMARY_MODEL
GEMINI_API_URL = f"{GEMINI_API_BASE_URL}/{GEMINI_MODEL_NAME}:generateContent"
