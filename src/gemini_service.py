"""Structured Gemini service for optional Streamlit assistants."""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.config import (
    CONFIDENCE_THRESHOLD,
    GEMINI_API_BASE_URL,
    GEMINI_FALLBACK_MODELS,
    GEMINI_PRIMARY_MODEL,
    PROJECT_ROOT,
)

LOGGER = logging.getLogger(__name__)


class IncompleteGeminiResponseError(ValueError):
    """Raised when Gemini returns truncated or otherwise incomplete text."""


@dataclass(frozen=True)
class GeminiResult:
    """Structured Gemini response safe for UI rendering."""

    success: bool
    text: str | None = None
    model_used: str | None = None
    attempts: int = 0
    status: str = "unavailable"
    fallback_used: bool = False
    error_type: str | None = None
    error_message: str | None = None
    technical_details: str | None = None
    attempted_models: tuple[str, ...] = ()


class GeminiService:
    """Small REST service with safe error mapping and prompt builders."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: int = 30,
        primary_model: str | None = None,
        fallback_models: list[str] | tuple[str, ...] | None = None,
        max_output_tokens: int = 1200,
    ) -> None:
        self.api_key = api_key if api_key is not None else self._load_api_key()
        self.timeout_seconds = timeout_seconds
        self.primary_model = primary_model or self._load_env_value(
            "GEMINI_PRIMARY_MODEL",
            GEMINI_PRIMARY_MODEL,
        )
        self.fallback_models = tuple(
            fallback_models
            if fallback_models is not None
            else self._load_fallback_models()
        )
        self.models = _dedupe_models((self.primary_model, *self.fallback_models))
        self.max_output_tokens = max_output_tokens

    @property
    def is_configured(self) -> bool:
        """Return True when API key exists."""
        return bool(self.api_key)

    def generate_text(self, prompt: str) -> GeminiResult:
        """Generate Gemini text, returning structured status instead of raising."""
        if not self.api_key:
            return GeminiResult(
                success=False,
                attempts=0,
                status="not_configured",
                error_type="missing_api_key",
                error_message="Gemini assistant is not configured.",
                technical_details=(
                    "Set GEMINI_API_KEY to enable AI-assisted explanations."
                ),
            )

        failures: list[GeminiResult] = []
        attempted_models: list[str] = []
        for model in self.models:
            attempted_models.append(model)
            request = self._build_request(prompt, model)
            result = self._attempt_request(request, model)
            attempts = len(attempted_models)
            if result.success:
                return GeminiResult(
                    success=True,
                    text=result.text,
                    model_used=model,
                    attempts=attempts,
                    status="success",
                    fallback_used=attempts > 1,
                    attempted_models=tuple(attempted_models),
                )
            failures.append(result)

        return _aggregate_failure(failures, attempted_models) or GeminiResult(
            success=False,
            attempts=len(attempted_models),
            status="unavailable",
            fallback_used=len(attempted_models) > 1,
            error_type="all_models_failed",
            error_message="Gemini assistant is currently unavailable.",
            technical_details="All configured Gemini models failed.",
            attempted_models=tuple(attempted_models),
        )

    def engineering_guidance(
        self,
        user_prompt: str,
        prediction_context: dict[str, Any],
    ) -> GeminiResult:
        """Generate advisory engineering text from displayed prediction context."""
        prompt = build_engineering_prompt(user_prompt, prediction_context)
        return self.generate_text(prompt)

    def dataset_interpretation(
        self,
        user_prompt: str,
        dataset_context: dict[str, Any],
    ) -> GeminiResult:
        """Generate advisory dataset interpretation from deterministic context."""
        prompt = build_dataset_prompt(user_prompt, dataset_context)
        return self.generate_text(prompt)

    def preprocessing_advice(
        self,
        title: str,
        eda_summary: dict[str, Any],
        preprocessing_context: dict[str, Any],
    ) -> GeminiResult:
        """Generate read-only EDA/preprocessing advice."""
        prompt = (
            "You are a read-only EDA and preprocessing advisor for aerospace "
            "composite laminate regression data. Provide dataset insights, "
            "outlier explanations, feature engineering suggestions, correlation "
            "explanations, potential data leakage risks, and preprocessing "
            "recommendations. Never modify data. Never fabricate values. Never "
            "claim that a model has been trained.\n\n"
            f"Advisor page: {title}\n\n"
            "EDA summary:\n"
            f"{json.dumps(_compact_mapping(eda_summary), indent=2, default=str)}\n\n"
            "Preprocessing context:\n"
            f"{json.dumps(_compact_mapping(preprocessing_context), indent=2, default=str)}"
        )
        return self.generate_text(prompt)

    def _build_request(
        self,
        prompt: str,
        model: str,
    ) -> urllib.request.Request:
        """Build Gemini REST request."""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.25,
                "topP": 0.85,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        return urllib.request.Request(
            f"{GEMINI_API_BASE_URL}/{model}:generateContent?key={self.api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _attempt_request(
        self,
        request: urllib.request.Request,
        model: str,
    ) -> GeminiResult:
        """Run one HTTP request and map every known failure class."""
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return GeminiResult(
                    success=True,
                    text=_extract_validated_text(payload),
                    model_used=model,
                    status="success",
                )
        except urllib.error.HTTPError as exc:
            return _http_error_result(exc, model)
        except IncompleteGeminiResponseError as exc:
            return GeminiResult(
                success=False,
                model_used=model,
                status="unavailable",
                error_type="incomplete_response",
                error_message="Gemini returned an incomplete response.",
                technical_details=str(exc),
            )
        except (TimeoutError, socket.timeout) as exc:
            return GeminiResult(
                success=False,
                model_used=model,
                status="unavailable",
                error_type="timeout",
                error_message="Gemini assistant timed out.",
                technical_details=str(exc) or "Request timeout.",
            )
        except urllib.error.URLError as exc:
            return GeminiResult(
                success=False,
                model_used=model,
                status="unavailable",
                error_type="network_error",
                error_message="Gemini assistant is currently unreachable.",
                technical_details=str(exc.reason),
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            return GeminiResult(
                success=False,
                model_used=model,
                status="unavailable",
                error_type="malformed_response",
                error_message="Gemini returned an unreadable response.",
                technical_details=str(exc),
            )
        except Exception as exc:  # pragma: no cover - final safety net.
            LOGGER.exception("Unexpected Gemini service failure")
            return GeminiResult(
                success=False,
                model_used=model,
                status="unavailable",
                error_type="request_error",
                error_message="Gemini assistant is currently unavailable.",
                technical_details=exc.__class__.__name__,
            )

    def _load_api_key(self) -> str | None:
        """Load Gemini API key from env, then local .env."""
        env_key = self._load_env_value("GEMINI_API_KEY")
        if env_key:
            return env_key
        return None

    def _load_fallback_models(self) -> tuple[str, ...]:
        """Load fallback model list from env/.env or defaults."""
        raw = self._load_env_value("GEMINI_FALLBACK_MODELS")
        if raw:
            return tuple(model.strip() for model in raw.split(",") if model.strip())
        return GEMINI_FALLBACK_MODELS

    def _load_env_value(self, name: str, default: str | None = None) -> str | None:
        """Load named value from environment, then local .env."""
        env_value = os.getenv(name)
        if env_value:
            return env_value
        env_path = PROJECT_ROOT / ".env"
        if not env_path.exists():
            return default

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
        return default


def build_engineering_prompt(
    user_prompt: str,
    prediction_context: dict[str, Any],
) -> str:
    """Build Gemini prompt for prediction-context interpretation."""
    confidence = _safe_float(prediction_context.get("confidence"), 1.0)
    threshold = _safe_float(
        prediction_context.get("confidence_threshold"),
        CONFIDENCE_THRESHOLD,
    )
    low_confidence_note = ""
    if confidence < threshold:
        low_confidence_note = (
            "Prediction confidence is below the configured threshold. Discuss "
            "uncertainty and design-review cautions without changing the ANN output."
        )

    return (
        "You are an engineering interpretation assistant for CompositeAI. The "
        "validated ANN/MLP model already produced the prediction. Gemini only "
        "explains supplied project context. Do not calculate a new prediction. "
        "Do not change, round incorrectly, truncate, or invent numerical values. "
        "Use supplied numerical values exactly when you mention them. Do not "
        "invent material properties, experimental results, or engineering "
        "certification. "
        f"{low_confidence_note}\n\n"
        "Displayed prediction context:\n"
        f"{format_prediction_context(prediction_context)}\n\n"
        "Explain:\n"
        "1. What the predicted tensile strength means.\n"
        "2. What the model metrics mean.\n"
        "3. What the model captures.\n"
        "4. What the model does not capture.\n"
        "5. Important engineering limitations.\n\n"
        "User request:\n"
        f"{user_prompt}"
    )


def build_dataset_prompt(
    user_prompt: str,
    dataset_context: dict[str, Any],
) -> str:
    """Build Gemini prompt for deterministic dataset-quality interpretation."""
    return (
        "You are a read-only dataset analysis assistant for aerospace composite "
        "laminate ML readiness. Do not modify, invent, or reinterpret numerical "
        "dataset values. Explain only the supplied facts. Never fabricate quality "
        "scores, missing values, duplicates, targets, materials, or validation "
        "findings.\n\n"
        "Deterministic CompositeAI dataset context:\n"
        f"{json.dumps(_compact_dataset_context(dataset_context), indent=2, default=str)}\n\n"
        "User request:\n"
        f"{user_prompt}"
    )


def format_prediction_context(prediction_context: dict[str, Any]) -> str:
    """Return readable prediction context for prompts and tests."""
    metrics = prediction_context.get("validation_metrics", {})
    lines = [
        f"Model: {prediction_context.get('model_name', 'ANN/MLP')}",
        f"Predicted tensile strength: {prediction_context.get('predicted_tensile_strength_mpa')} MPa",
        f"R2: {metrics.get('r2') if isinstance(metrics, dict) else None}",
        f"MAE (MPa): {metrics.get('mae') if isinstance(metrics, dict) else None}",
        f"RMSE (MPa): {metrics.get('rmse') if isinstance(metrics, dict) else None}",
        f"Warnings: {prediction_context.get('warnings', [])}",
        f"Notes: {prediction_context.get('notes', '')}",
    ]
    return "\n".join(lines)


def _extract_text(data: dict[str, Any]) -> str:
    """Extract text from Gemini REST response payload."""
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    if not text.strip():
        raise ValueError("Gemini returned empty response text.")
    return text


def _extract_validated_text(data: dict[str, Any]) -> str:
    """Extract Gemini text and reject incomplete or malformed output."""
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates.")

    finish_reason = str(candidates[0].get("finishReason", "")).upper()
    if finish_reason == "MAX_TOKENS":
        raise IncompleteGeminiResponseError("Gemini stopped at max tokens.")

    text = _extract_text(data)
    return _validate_response_text(text)


def _validate_response_text(text: str) -> str:
    """Reject partial, raw, or malformed Gemini text before UI render."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        raise ValueError("Gemini returned empty response text.")
    if _looks_like_raw_payload(normalized):
        raise ValueError("Gemini returned raw structured output.")
    if _has_unfinished_markdown(normalized):
        raise IncompleteGeminiResponseError("Gemini returned unfinished Markdown.")
    if _looks_truncated(normalized):
        raise IncompleteGeminiResponseError("Gemini returned suspiciously truncated text.")
    return normalized


def _looks_like_raw_payload(text: str) -> bool:
    """Detect raw JSON or Python-dict style payloads."""
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            return isinstance(parsed, (dict, list))
    return bool(
        re.fullmatch(r"\{['\"].*['\"]:.*\}", stripped, flags=re.DOTALL)
        or (
            re.fullmatch(r"\[?\{.*\}\]?", stripped, flags=re.DOTALL)
            and ("':" in stripped or '":' in stripped)
        )
    )


def _has_unfinished_markdown(text: str) -> bool:
    """Detect unbalanced Markdown markers that commonly indicate truncation."""
    if text.count("```") % 2 != 0:
        return True
    without_fences = text.replace("```", "")
    if without_fences.count("`") % 2 != 0:
        return True
    if text.count("**") % 2 != 0:
        return True
    return text.count("__") % 2 != 0


def _looks_truncated(text: str) -> bool:
    """Heuristic for clipped assistant responses."""
    lowered = text.lower().strip()
    if re.search(r"\*\*\d{1,3}$", text):
        return True
    if lowered in {"the model predicts", "here is", "the provided", "1.", "2.", "3.", "##"}:
        return True
    if lowered.startswith(("the model predicts", "the provided", "here is")) and len(text) < 80:
        return True
    return text.endswith(("**", "__", "`", "```", "(", "[", "{", "##"))


def _http_error_result(exc: urllib.error.HTTPError, model: str) -> GeminiResult:
    """Map HTTP status codes to UI-safe Gemini failures."""
    if exc.code == 404:
        return GeminiResult(
            success=False,
            model_used=model,
            status="unavailable",
            error_type="not_found",
            error_message="Gemini assistant is currently unavailable.",
            technical_details="HTTP 404. Check the configured Gemini API model/endpoint.",
        )
    if exc.code in {401, 403}:
        return GeminiResult(
            success=False,
            model_used=model,
            status="unavailable",
            error_type="auth_error",
            error_message="Gemini assistant is not authorized.",
            technical_details=f"HTTP {exc.code}. Check GEMINI_API_KEY permissions.",
        )
    if exc.code == 429:
        return GeminiResult(
            success=False,
            model_used=model,
            status="unavailable",
            error_type="rate_limited",
            error_message="Gemini assistant is rate limited.",
            technical_details="HTTP 429. Retry later or check quota.",
        )
    if exc.code >= 500:
        return GeminiResult(
            success=False,
            model_used=model,
            status="unavailable",
            error_type="server_error",
            error_message="Gemini assistant is currently unavailable.",
            technical_details=f"HTTP {exc.code}. Gemini service error.",
        )
    return GeminiResult(
        success=False,
        model_used=model,
        status="unavailable",
        error_type="request_error",
        error_message="Gemini assistant is currently unavailable.",
        technical_details=f"HTTP {exc.code}.",
    )


def _aggregate_failure(
    failures: list[GeminiResult],
    attempted_models: list[str],
) -> GeminiResult | None:
    """Combine failed model attempts into one UI-safe result."""
    if not failures:
        return None
    lines = ["Models attempted:"]
    lines.extend(f"- {model}" for model in attempted_models)
    lines.append("Failures:")
    for failure in failures:
        detail = failure.technical_details or failure.error_message or "Unavailable"
        lines.append(f"- {failure.model_used}: {failure.error_type} ({detail})")
    return GeminiResult(
        success=False,
        text=None,
        model_used=None,
        attempts=len(attempted_models),
        status="unavailable",
        fallback_used=len(attempted_models) > 1,
        error_type="all_models_failed",
        error_message="All configured Gemini models failed.",
        technical_details="\n".join(lines),
        attempted_models=tuple(attempted_models),
    )


def _dedupe_models(models: tuple[str | None, ...]) -> tuple[str, ...]:
    """Return configured models preserving order and removing duplicates."""
    ordered: list[str] = []
    seen: set[str] = set()
    for model in models:
        if not model:
            continue
        normalized = model.strip()
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return tuple(ordered)


def _compact_dataset_context(dataset_context: dict[str, Any]) -> dict[str, Any]:
    """Keep Gemini dataset context concise and deterministic."""
    dataset = dataset_context.get("dataset", {})
    validation = dataset_context.get("validation", {})
    profile = dataset_context.get("profile", {})
    return {
        "dataset": {
            "rows": dataset.get("rows"),
            "columns": dataset.get("columns"),
            "features": dataset.get("features"),
            "feature_count": dataset.get("feature_count"),
            "target": dataset.get("target"),
        },
        "validation": {
            "status": validation.get("status"),
            "quality_score": validation.get("quality_score"),
            "metrics": validation.get("metrics"),
            "issues": validation.get("issues"),
        },
        "feature_types": {
            "numeric_columns": profile.get("numeric_columns"),
            "categorical_columns": profile.get("categorical_columns"),
        },
    }


def _compact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    """Drop oversized nested fields from advisory context."""
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"data", "dataset", "dataframe", "records", "rows"}:
            continue
        if isinstance(item, dict):
            compact[key] = _compact_mapping(item)
        elif isinstance(item, (list, tuple)) and len(item) > 25:
            compact[key] = list(item[:25])
        else:
            compact[key] = item
    return compact


def _safe_float(value: Any, default: float) -> float:
    """Convert value to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
