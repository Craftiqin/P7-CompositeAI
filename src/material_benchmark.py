"""Reference material benchmark utilities for aerospace metals comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_DATABASE_PATH = (
    PROJECT_ROOT / "data" / "reference_materials" / "material_benchmark_database.json"
)


def load_reference_materials(
    database_path: str | Path = DEFAULT_REFERENCE_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Load engineering reference materials database."""
    path = Path(database_path)
    materials = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(materials, list):
        raise ValueError("Reference materials database must contain a list of records")
    return [dict(item) for item in materials]


def calculate_strength_ratio(
    composite_strength_mpa: float,
    material_strength_mpa: float,
) -> float:
    """Return composite-to-material tensile strength ratio."""
    composite_strength = _validate_finite_positive(
        composite_strength_mpa,
        "composite_strength_mpa",
    )
    material_strength = _validate_finite_positive(
        material_strength_mpa,
        "material_strength_mpa",
    )
    return composite_strength / material_strength


def calculate_density_ratio(
    composite_density_g_cm3: float | None,
    material_density_g_cm3: float | None,
) -> float | None:
    """Return composite-to-material density ratio when both are available."""
    if composite_density_g_cm3 is None or material_density_g_cm3 is None:
        return None
    composite_density = _validate_finite_positive(
        composite_density_g_cm3,
        "composite_density_g_cm3",
    )
    material_density = _validate_finite_positive(
        material_density_g_cm3,
        "material_density_g_cm3",
    )
    return composite_density / material_density


def calculate_specific_strength(
    strength_mpa: float | None,
    density_g_cm3: float | None,
) -> float | None:
    """Return specific strength in MPa per g/cm³."""
    if strength_mpa is None or density_g_cm3 is None:
        return None
    strength = _validate_finite_positive(strength_mpa, "strength_mpa")
    density = _validate_finite_positive(density_g_cm3, "density_g_cm3")
    return strength / density


def rank_materials(
    composite_strength_mpa: float,
    composite_density_g_cm3: float | None = None,
    materials: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Calculate benchmark rows and ranking metrics."""
    composite_strength = _validate_finite_positive(
        composite_strength_mpa,
        "composite_strength_mpa",
    )
    composite_density = _validate_optional_positive(
        composite_density_g_cm3,
        "composite_density_g_cm3",
    )
    reference_materials = materials or load_reference_materials()
    composite_specific_strength = calculate_specific_strength(
        composite_strength,
        composite_density,
    )

    rows: list[dict[str, Any]] = []
    for item in reference_materials:
        material_strength = _validate_finite_positive(
            float(item["tensile_strength_mpa"]),
            "tensile_strength_mpa",
        )
        material_density = _validate_optional_positive(
            item.get("density_g_cm3"),
            "density_g_cm3",
        )
        material_specific_strength = calculate_specific_strength(
            material_strength,
            material_density,
        )
        row = {
            "material": str(item["material"]),
            "category": str(item["category"]),
            "application": str(item["application"]),
            "tensile_strength_mpa": material_strength,
            "density_g_cm3": material_density,
            "specific_strength": material_specific_strength,
            "difference_vs_composite_mpa": composite_strength - material_strength,
            "strength_ratio": calculate_strength_ratio(composite_strength, material_strength),
            "density_ratio": calculate_density_ratio(composite_density, material_density),
            "composite_specific_strength": composite_specific_strength,
            "specific_strength_ratio": (
                composite_specific_strength / material_specific_strength
                if composite_specific_strength is not None and material_specific_strength
                else None
            ),
        }
        rows.append(row)

    _apply_rank(rows, "tensile_strength_mpa", "strength_rank", descending=True)
    _apply_rank(rows, "density_g_cm3", "density_rank", descending=False)
    _apply_rank(rows, "specific_strength", "specific_strength_rank", descending=True)
    return sorted(
        rows,
        key=lambda item: (
            item["strength_rank"],
            item["density_rank"],
            item["material"],
        ),
    )


def generate_engineering_insights(
    composite_strength_mpa: float,
    composite_density_g_cm3: float | None = None,
    ranked_materials: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Generate factual engineering comparison statements."""
    rows = ranked_materials or rank_materials(
        composite_strength_mpa,
        composite_density_g_cm3,
    )
    if not rows:
        return ["Reference benchmark database is empty."]

    insights: list[str] = []
    strongest = max(rows, key=lambda item: item["tensile_strength_mpa"])
    weakest = min(rows, key=lambda item: item["tensile_strength_mpa"])
    best_specific = max(
        rows,
        key=lambda item: item["specific_strength"] or float("-inf"),
    )

    aluminum_7075 = next(
        (item for item in rows if item["material"] == "Aluminum 7075-T6"),
        None,
    )
    if aluminum_7075 is not None:
        insights.append(
            "This laminate is approximately "
            f"{aluminum_7075['strength_ratio']:.2f}× stronger than Aluminum 7075-T6."
        )

    insights.append(
        f"The highest reference tensile strength is {strongest['material']} at "
        f"{strongest['tensile_strength_mpa']:.0f} MPa."
    )
    insights.append(
        f"The lowest reference tensile strength is {weakest['material']} at "
        f"{weakest['tensile_strength_mpa']:.0f} MPa."
    )

    if composite_density_g_cm3 is not None and best_specific["specific_strength"] is not None:
        composite_specific_strength = calculate_specific_strength(
            composite_strength_mpa,
            composite_density_g_cm3,
        )
        if composite_specific_strength is not None:
            insights.append(
                "Composite specific strength is "
                f"{composite_specific_strength:.2f} MPa per g/cm³ versus "
                f"{best_specific['specific_strength']:.2f} for {best_specific['material']}."
            )
        lower_density_materials = [
            item for item in rows if item["density_ratio"] is not None and item["density_ratio"] < 1.0
        ]
        if lower_density_materials:
            lightest_comparison = min(lower_density_materials, key=lambda item: item["density_ratio"])
            insights.append(
                f"The composite density is lower than {lightest_comparison['material']} "
                f"by a factor of {1 / lightest_comparison['density_ratio']:.2f}×."
            )
    else:
        insights.append(
            f"Best reference specific strength is {best_specific['material']} at "
            f"{best_specific['specific_strength']:.2f} MPa per g/cm³."
        )

    return insights


def _apply_rank(
    rows: list[dict[str, Any]],
    field_name: str,
    rank_name: str,
    *,
    descending: bool,
) -> None:
    ranked = sorted(
        rows,
        key=lambda item: item[field_name] if item[field_name] is not None else (
            float("-inf") if descending else float("inf")
        ),
        reverse=descending,
    )
    for index, row in enumerate(ranked, start=1):
        row[rank_name] = index


def _validate_finite_positive(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be a finite positive number")
    return number


def _validate_optional_positive(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be a finite positive number when provided")
    return number
