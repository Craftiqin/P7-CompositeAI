"""Optimization impact summary helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTIMIZATION_REPORT_PATH = (
    PROJECT_ROOT / "data" / "sequence" / "optimization_validation_report.json"
)


def calculate_improvement_pct(
    baseline_lambda_cs: float | None,
    optimized_lambda_cs: float | None,
) -> float | None:
    """Return percent improvement over baseline."""
    if not baseline_lambda_cs or not optimized_lambda_cs:
        return None
    return ((float(optimized_lambda_cs) - float(baseline_lambda_cs)) / float(baseline_lambda_cs)) * 100.0


def calculate_improvement_ratio(
    baseline_lambda_cs: float | None,
    optimized_lambda_cs: float | None,
) -> float | None:
    """Return optimized-to-baseline ratio."""
    if not baseline_lambda_cs or not optimized_lambda_cs:
        return None
    return float(optimized_lambda_cs) / float(baseline_lambda_cs)


def load_optimization_impact(
    optimization_result: dict[str, Any] | None = None,
    report_path: str | Path = DEFAULT_OPTIMIZATION_REPORT_PATH,
) -> dict[str, Any]:
    """Load latest optimization impact from session result or stored report."""
    if optimization_result:
        baseline = _safe_get(optimization_result, "baseline", "lambda_cs")
        if baseline is None:
            baseline = optimization_result.get("baseline_lambda_cs")
        optimized = optimization_result.get("best_lambda_cs")
        constraints = optimization_result.get("constraints", {})
        candidates = optimization_result.get("candidates_evaluated")
        best_sequence = optimization_result.get("best_sequence")
        improvement_pct = optimization_result.get("improvement_pct")
    else:
        data = json.loads(Path(report_path).read_text(encoding="utf-8"))
        demo = data.get("optimization_demo", {})
        baseline = demo.get("baseline_lambda_cs")
        optimized = demo.get("best_lambda_cs")
        constraints = demo.get("constraints", {})
        candidates = demo.get("candidates_evaluated")
        best_sequence = demo.get("best_sequence")
        improvement_pct = demo.get("improvement_pct")

    if improvement_pct is None:
        improvement_pct = calculate_improvement_pct(baseline, optimized)

    return {
        "baseline_lambda_cs": baseline,
        "optimized_lambda_cs": optimized,
        "improvement_pct": improvement_pct,
        "improvement_ratio": calculate_improvement_ratio(baseline, optimized),
        "constraints": constraints,
        "candidates_evaluated": candidates,
        "best_sequence": best_sequence,
    }


def _safe_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
