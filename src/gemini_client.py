"""Compatibility wrapper for central Gemini service."""

from __future__ import annotations

from typing import Any

from src.gemini_service import GeminiService


class GeminiClientError(RuntimeError):
    """Raised when Gemini service cannot return usable text."""


class GeminiClient:
    """Backward-compatible client that delegates all requests to GeminiService."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: int = 30,
        max_retries: int = 0,
        retry_delay_seconds: float = 0.0,
    ) -> None:
        self.service = GeminiService(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    @property
    def is_configured(self) -> bool:
        """Return True when Gemini API key is available."""
        return self.service.is_configured

    def generate_engineering_guidance(
        self,
        user_prompt: str,
        prediction_context: dict[str, Any] | None = None,
    ) -> str:
        """Generate natural-language composite engineering guidance."""
        result = self.service.engineering_guidance(user_prompt, prediction_context or {})
        return self._unwrap(result)

    def analyze_dataset_quality(
        self,
        dataset_profile: dict[str, Any],
        validation_summary: dict[str, Any],
    ) -> str:
        """Summarize dataset quality without changing or inventing data."""
        result = self.service.dataset_interpretation(
            "Summarize dataset quality and risks.",
            {
                "dataset_profile": dataset_profile,
                "validation_summary": validation_summary,
            },
        )
        return self._unwrap(result)

    def advise_preprocessing(
        self,
        eda_summary: dict[str, Any],
        preprocessing_context: dict[str, Any],
    ) -> str:
        """Provide read-only EDA, outlier, feature, and preprocessing advice."""
        result = self.service.preprocessing_advice(
            "Preprocessing Advisor",
            eda_summary,
            preprocessing_context,
        )
        return self._unwrap(result)

    def generate_text(self, prompt: str) -> str:
        """Generate text through central Gemini service."""
        result = self.service.generate_text(prompt)
        return self._unwrap(result)

    def _unwrap(self, result: Any) -> str:
        """Return text or raise compatibility exception."""
        if result.success and result.text:
            return result.text
        detail = result.technical_details or result.error_message or "Gemini unavailable."
        raise GeminiClientError(detail)
