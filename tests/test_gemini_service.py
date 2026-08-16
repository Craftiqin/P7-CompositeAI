"""Tests for UI-safe Gemini service behavior."""

from __future__ import annotations

import json
import unittest
import urllib.error
from unittest.mock import patch

from src.gemini_service import (
    GeminiService,
    build_dataset_prompt,
    build_engineering_prompt,
    format_prediction_context,
)


class FakeResponse:
    """Minimal urlopen response context manager."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        """Return response payload bytes."""
        return self.payload


def gemini_payload(text: str, finish_reason: str = "STOP") -> dict[str, object]:
    """Build minimal Gemini REST payload for tests."""
    return {
        "candidates": [
            {
                "finishReason": finish_reason,
                "content": {
                    "parts": [{"text": text}],
                },
            }
        ]
    }


def http_error(code: int) -> urllib.error.HTTPError:
    """Create HTTPError with status code."""
    return urllib.error.HTTPError(
        url="https://example.test",
        code=code,
        msg="error",
        hdrs=None,
        fp=None,
    )


class GeminiServiceTest(unittest.TestCase):
    """Validate structured Gemini responses."""

    def make_service(self) -> GeminiService:
        """Return test service with deterministic model order."""
        return GeminiService(
            api_key="test-key",
            primary_model="primary-model",
            fallback_models=("fallback-one", "fallback-two"),
        )

    def test_missing_api_key_returns_not_configured(self) -> None:
        """Missing key returns clear status without request."""
        service = GeminiService(api_key="")
        result = service.generate_text("Explain prediction.")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "missing_api_key")
        self.assertIn("not configured", result.error_message or "")
        self.assertEqual(result.attempts, 0)

    @patch("urllib.request.urlopen")
    def test_success_extracts_text(self, mock_urlopen: object) -> None:
        """Successful Gemini payload returns response text."""
        payload = gemini_payload("Use ANN output as context only.")
        mock_urlopen.return_value = FakeResponse(json.dumps(payload).encode("utf-8"))
        result = self.make_service().generate_text("Prompt")
        self.assertTrue(result.success)
        self.assertEqual(result.text, "Use ANN output as context only.")
        self.assertEqual(result.model_used, "primary-model")
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.fallback_used)

    @patch("urllib.request.urlopen")
    def test_primary_max_tokens_falls_back_successfully(self, mock_urlopen: object) -> None:
        """MAX_TOKENS response moves to next configured model."""
        truncated = gemini_payload(
            "The model predicts a tensile strength of **16",
            finish_reason="MAX_TOKENS",
        )
        complete = gemini_payload("Complete fallback response with engineering limitations.")
        mock_urlopen.side_effect = [
            FakeResponse(json.dumps(truncated).encode("utf-8")),
            FakeResponse(json.dumps(complete).encode("utf-8")),
        ]
        result = self.make_service().generate_text("Prompt")
        self.assertTrue(result.success)
        self.assertEqual(result.text, "Complete fallback response with engineering limitations.")
        self.assertEqual(result.model_used, "fallback-one")
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.fallback_used)

    @patch("urllib.request.urlopen")
    def test_primary_404_falls_back_successfully(self, mock_urlopen: object) -> None:
        """Unavailable primary model moves to fallback model."""
        mock_urlopen.side_effect = [
            http_error(404),
            FakeResponse(json.dumps(gemini_payload("Fallback after 404.")).encode("utf-8")),
        ]
        result = self.make_service().generate_text("Prompt")
        self.assertTrue(result.success)
        self.assertEqual(result.model_used, "fallback-one")
        self.assertEqual(result.text, "Fallback after 404.")

    @patch("urllib.request.urlopen")
    def test_primary_429_falls_back_successfully(self, mock_urlopen: object) -> None:
        """Rate-limited primary moves to fallback model."""
        mock_urlopen.side_effect = [
            http_error(429),
            FakeResponse(json.dumps(gemini_payload("Fallback after quota.")).encode("utf-8")),
        ]
        result = self.make_service().generate_text("Prompt")
        self.assertTrue(result.success)
        self.assertEqual(result.model_used, "fallback-one")

    @patch("urllib.request.urlopen")
    def test_primary_timeout_falls_back_successfully(self, mock_urlopen: object) -> None:
        """Timeout on primary moves to fallback model."""
        mock_urlopen.side_effect = [
            TimeoutError("timed out"),
            FakeResponse(json.dumps(gemini_payload("Fallback after timeout.")).encode("utf-8")),
        ]
        result = self.make_service().generate_text("Prompt")
        self.assertTrue(result.success)
        self.assertEqual(result.model_used, "fallback-one")

    @patch("urllib.request.urlopen")
    def test_primary_empty_response_falls_back_successfully(self, mock_urlopen: object) -> None:
        """Empty primary response moves to fallback model."""
        mock_urlopen.side_effect = [
            FakeResponse(json.dumps(gemini_payload("")).encode("utf-8")),
            FakeResponse(json.dumps(gemini_payload("Fallback after empty response.")).encode("utf-8")),
        ]
        result = self.make_service().generate_text("Prompt")
        self.assertTrue(result.success)
        self.assertEqual(result.model_used, "fallback-one")

    @patch("urllib.request.urlopen")
    def test_all_models_fail_cleanly(self, mock_urlopen: object) -> None:
        """All model failures return one clean unavailable result."""
        mock_urlopen.side_effect = [http_error(404), http_error(429), TimeoutError("timed out")]
        result = self.make_service().generate_text("Prompt")
        self.assertFalse(result.success)
        self.assertIsNone(result.text)
        self.assertEqual(result.error_type, "all_models_failed")
        self.assertEqual(result.attempts, 3)
        self.assertTrue(result.fallback_used)
        self.assertIn("primary-model", result.technical_details or "")
        self.assertIn("fallback-two", result.technical_details or "")

    @patch("urllib.request.urlopen")
    def test_incomplete_response_never_reaches_ui(self, mock_urlopen: object) -> None:
        """Partial primary/fallback text is never returned as text."""
        payload = gemini_payload(
            "The model predicts a tensile strength of **16",
            finish_reason="MAX_TOKENS",
        )
        mock_urlopen.return_value = FakeResponse(json.dumps(payload).encode("utf-8"))
        result = self.make_service().generate_text("Prompt")
        self.assertFalse(result.success)
        self.assertIsNone(result.text)
        self.assertEqual(result.error_type, "all_models_failed")

    @patch("urllib.request.urlopen")
    def test_fragment_without_finish_reason_is_still_rejected(self, mock_urlopen: object) -> None:
        """Suspicious clipped fragments are not treated as valid Gemini text."""
        payload = gemini_payload("The model predicts")
        mock_urlopen.return_value = FakeResponse(json.dumps(payload).encode("utf-8"))
        result = self.make_service().generate_text("Prompt")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "all_models_failed")

    @patch("urllib.request.urlopen")
    def test_raw_json_text_is_rejected(self, mock_urlopen: object) -> None:
        """Structured payload echoed as text is rejected for UI safety."""
        payload = gemini_payload('{"prediction": 1827.77, "unit": "MPa"}')
        mock_urlopen.return_value = FakeResponse(json.dumps(payload).encode("utf-8"))
        result = self.make_service().generate_text("Prompt")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "all_models_failed")

    @patch("urllib.request.urlopen")
    def test_404_returns_clean_unavailable(self, mock_urlopen: object) -> None:
        """HTTP 404 is mapped to clean unavailable state."""
        mock_urlopen.side_effect = http_error(404)
        result = GeminiService(
            api_key="test-key",
            primary_model="primary-model",
            fallback_models=(),
        ).generate_text("Prompt")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "all_models_failed")
        self.assertEqual(
            result.error_message,
            "All configured Gemini models failed.",
        )
        self.assertIn("HTTP 404", result.technical_details or "")

    @patch("urllib.request.urlopen")
    def test_401_and_403_return_auth_error(self, mock_urlopen: object) -> None:
        """Auth status codes map to authorization failure."""
        for code in (401, 403):
            with self.subTest(code=code):
                mock_urlopen.side_effect = http_error(code)
                result = GeminiService(
                    api_key="test-key",
                    primary_model="primary-model",
                    fallback_models=(),
                ).generate_text("Prompt")
                self.assertFalse(result.success)
                self.assertEqual(result.error_type, "all_models_failed")
                self.assertIn("auth_error", result.technical_details or "")

    @patch("urllib.request.urlopen")
    def test_timeout_returns_timeout(self, mock_urlopen: object) -> None:
        """Timeout returns structured timeout result."""
        mock_urlopen.side_effect = TimeoutError("timed out")
        result = GeminiService(
            api_key="test-key",
            primary_model="primary-model",
            fallback_models=(),
        ).generate_text("Prompt")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "all_models_failed")

    @patch("urllib.request.urlopen")
    def test_network_error_returns_network_error(self, mock_urlopen: object) -> None:
        """URLError returns structured network failure."""
        mock_urlopen.side_effect = urllib.error.URLError("dns")
        result = GeminiService(
            api_key="test-key",
            primary_model="primary-model",
            fallback_models=(),
        ).generate_text("Prompt")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "all_models_failed")

    @patch("urllib.request.urlopen")
    def test_malformed_response_returns_error(self, mock_urlopen: object) -> None:
        """Invalid JSON or empty candidates is not treated as success."""
        mock_urlopen.return_value = FakeResponse(b"not-json")
        result = GeminiService(
            api_key="test-key",
            primary_model="primary-model",
            fallback_models=(),
        ).generate_text("Prompt")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "all_models_failed")
        self.assertIsNone(result.text)

    @patch("urllib.request.urlopen")
    def test_model_order_is_respected(self, mock_urlopen: object) -> None:
        """Requests use configured model order."""
        mock_urlopen.side_effect = [
            http_error(404),
            FakeResponse(json.dumps(gemini_payload("Fallback response.")).encode("utf-8")),
        ]
        result = self.make_service().generate_text("Prompt")
        urls = [call.args[0].full_url for call in mock_urlopen.call_args_list]
        self.assertTrue(urls[0].startswith("https://generativelanguage.googleapis.com/v1beta/models/primary-model"))
        self.assertTrue(urls[1].startswith("https://generativelanguage.googleapis.com/v1beta/models/fallback-one"))
        self.assertEqual(result.attempted_models, ("primary-model", "fallback-one"))

    @patch("urllib.request.urlopen")
    def test_same_model_is_not_retried_unnecessarily(self, mock_urlopen: object) -> None:
        """Duplicate fallback config is de-duplicated."""
        mock_urlopen.side_effect = [http_error(404), http_error(404)]
        service = GeminiService(
            api_key="test-key",
            primary_model="primary-model",
            fallback_models=("primary-model", "fallback-one", "fallback-one"),
        )
        result = service.generate_text("Prompt")
        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(result.attempted_models, ("primary-model", "fallback-one"))

    def test_prediction_context_formatting_has_no_raw_dict_dependency(self) -> None:
        """Prediction context format includes metrics as readable lines."""
        context = {
            "model_name": "ANN/MLP",
            "predicted_tensile_strength_mpa": 1827.766,
            "validation_metrics": {"r2": 0.9952, "mae": 32.51, "rmse": 43.38},
            "warnings": [],
            "notes": "Step 6 prediction only.",
        }
        formatted = format_prediction_context(context)
        self.assertIn("Model: ANN/MLP", formatted)
        self.assertIn("Predicted tensile strength", formatted)
        self.assertIn("R2: 0.9952", formatted)

    def test_engineering_prompt_keeps_ann_gemini_boundary(self) -> None:
        """Engineering prompt separates ANN prediction from Gemini guidance."""
        prompt = build_engineering_prompt(
            "Explain.",
            {
                "model_name": "ANN/MLP",
                "predicted_tensile_strength_mpa": 1827.7660899459502,
                "validation_metrics": {"r2": 0.9952, "mae": 32.51, "rmse": 43.38},
            },
        )
        self.assertIn("validated ANN/MLP model already produced the prediction", prompt)
        self.assertIn("Gemini only explains supplied project context", prompt)
        self.assertIn("Use supplied numerical values exactly", prompt)
        self.assertIn("Predicted tensile strength: 1827.7660899459502 MPa", prompt)
        self.assertIn("Do not invent material properties", prompt)

    def test_dataset_prompt_forbids_fabrication(self) -> None:
        """Dataset prompt includes deterministic context and no-invention rule."""
        context = {
            "dataset": {
                "rows": 10000,
                "columns": 8,
                "target": "tensile_strength_mpa",
            },
            "validation": {
                "quality_score": 75.78,
                "metrics": {"missing_values": 0, "duplicate_rows": 0},
            },
        }
        prompt = build_dataset_prompt("Summarize.", context)
        self.assertIn("10000", prompt)
        self.assertIn("75.78", prompt)
        self.assertIn("Do not modify, invent, or reinterpret", prompt)


if __name__ == "__main__":
    unittest.main()
