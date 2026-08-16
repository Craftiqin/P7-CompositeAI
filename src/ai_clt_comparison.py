"""AI tensile-strength vs CLT load-capacity comparison backend.

This layer only compares ANN and CLT outputs when both are converted to the
same physical quantity with explicit units and verified material compatibility.
It does not retrain the ANN model and does not alter CLT equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.clt import (
    LaminaMaterial,
    LaminateLoadCase,
    StrainAllowables,
    calculate_strain_allowable_failure_load,
)


ANN_OUTPUT_NAME = "tensile_strength_mpa"
ANN_OUTPUT_UNIT = "MPa"
CLT_OUTPUT_NAME = "lambda_cs"
CLT_OUTPUT_UNIT = "dimensionless load factor"
LOAD_UNIT_N_PER_M = "N/m"
STRESS_MODE = "equivalent_laminate_tensile_stress_mpa"
LOAD_CAPACITY_MODE = "tensile_load_capacity_n_per_m"
SUPPORTED_COMPARISON_MODES = (STRESS_MODE, LOAD_CAPACITY_MODE)
MPA_TO_PA = 1_000_000.0
PA_TO_MPA = 1.0 / MPA_TO_PA


@dataclass(frozen=True)
class ComparisonCase:
    """Structured common case for AI-vs-CLT comparison.

    Material equivalence is caller-supplied evidence. The engine never assumes
    ANN category names such as "Carbon" equal a TU Delft CLT material card.
    """

    ann_input: dict[str, Any]
    stacking_sequence: list[int]
    material_card: LaminaMaterial | None
    load_case: LaminateLoadCase | None
    comparison_quantity: str = STRESS_MODE
    base_load_unit: str = LOAD_UNIT_N_PER_M
    material_equivalence_verified: bool = False
    material_equivalence_evidence: str | None = None
    allowables: StrainAllowables | None = None
    assumptions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MaterialCompatibility:
    """Material compatibility result."""

    status: str
    compatible: bool
    reason: str
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize compatibility result."""
        return {
            "status": self.status,
            "compatible": self.compatible,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ComparisonResult:
    """AI-vs-CLT comparison result with calculation trace."""

    comparable: bool
    reason: str
    ann_value: float | None
    ann_unit: str
    clt_value: float | None
    clt_unit: str
    common_quantity: str | None
    common_unit: str | None
    ann_common_value: float | None
    clt_common_value: float | None
    absolute_difference: float | None
    percentage_difference: float | None
    material_compatibility: dict[str, Any]
    assumptions: list[str]
    warnings: list[str]
    calculation_trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result for UI, reports, and tests."""
        return {
            "comparable": self.comparable,
            "reason": self.reason,
            "ann_value": self.ann_value,
            "ann_unit": self.ann_unit,
            "clt_value": self.clt_value,
            "clt_unit": self.clt_unit,
            "common_quantity": self.common_quantity,
            "common_unit": self.common_unit,
            "ann_common_value": self.ann_common_value,
            "clt_common_value": self.clt_common_value,
            "absolute_difference": self.absolute_difference,
            "percentage_difference": self.percentage_difference,
            "material_compatibility": self.material_compatibility,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "calculation_trace": self.calculation_trace,
        }


def compare_ai_and_clt(
    case: ComparisonCase,
    ann_result: dict[str, Any] | None = None,
) -> ComparisonResult:
    """Compare ANN and CLT only when same quantity/unit is defensible."""
    warnings: list[str] = []
    assumptions = list(case.assumptions)
    calculation_trace: list[dict[str, Any]] = []
    material_compatibility = assess_material_compatibility(case)

    ann_value = _extract_or_predict_ann_value(case, ann_result)
    pre_evaluation_rejection = _pre_evaluation_rejection_reason(case)
    clt_evaluation = None
    if pre_evaluation_rejection is None or not material_compatibility.compatible:
        clt_evaluation = _evaluate_clt_if_possible(case, calculation_trace, warnings)
    clt_value = _safe_float(clt_evaluation.get("lambda_cs")) if clt_evaluation else None

    if not material_compatibility.compatible:
        rejection_reason = material_compatibility.reason
    else:
        rejection_reason = pre_evaluation_rejection or _comparison_rejection_reason(
            case,
            material_compatibility,
            clt_evaluation,
        )
    if rejection_reason:
        return ComparisonResult(
            comparable=False,
            reason=rejection_reason,
            ann_value=ann_value,
            ann_unit=ANN_OUTPUT_UNIT,
            clt_value=clt_value,
            clt_unit=CLT_OUTPUT_UNIT,
            common_quantity=None,
            common_unit=None,
            ann_common_value=None,
            clt_common_value=None,
            absolute_difference=None,
            percentage_difference=None,
            material_compatibility=material_compatibility.to_dict(),
            assumptions=assumptions,
            warnings=warnings,
            calculation_trace=calculation_trace,
        )

    conversion = convert_to_common_quantity(case, ann_value, clt_value, calculation_trace)
    absolute_difference = abs(conversion["ann_common_value"] - conversion["clt_common_value"])
    percentage_difference = calculate_percentage_difference(
        conversion["ann_common_value"],
        conversion["clt_common_value"],
    )

    return ComparisonResult(
        comparable=True,
        reason="ANN and CLT converted to same physical quantity and unit.",
        ann_value=ann_value,
        ann_unit=ANN_OUTPUT_UNIT,
        clt_value=clt_value,
        clt_unit=CLT_OUTPUT_UNIT,
        common_quantity=case.comparison_quantity,
        common_unit=conversion["common_unit"],
        ann_common_value=conversion["ann_common_value"],
        clt_common_value=conversion["clt_common_value"],
        absolute_difference=absolute_difference,
        percentage_difference=percentage_difference,
        material_compatibility=material_compatibility.to_dict(),
        assumptions=assumptions
        + [
            "ANN tensile strength is treated as average laminate tensile stress for the verified same material case.",
            "CLT lambda_cs scales the supplied uniaxial Nx base load.",
            "Difference uses CLT common value as reference, not accuracy.",
        ],
        warnings=warnings,
        calculation_trace=calculation_trace,
    )


def assess_material_compatibility(case: ComparisonCase) -> MaterialCompatibility:
    """Return explicit material compatibility status."""
    if case.material_card is None:
        return MaterialCompatibility(
            status="NOT DIRECTLY COMPARABLE",
            compatible=False,
            reason="Missing thickness: CLT material card is missing.",
        )
    if not case.material_equivalence_verified:
        return MaterialCompatibility(
            status="NOT DIRECTLY COMPARABLE",
            compatible=False,
            reason=(
                "ANN dataset does not contain verified E1/E2/G12/nu12/ply_thickness "
                "for this material; equivalence to CLT material card is unproven."
            ),
            evidence=case.material_equivalence_evidence,
        )
    if not case.material_equivalence_evidence:
        return MaterialCompatibility(
            status="NOT DIRECTLY COMPARABLE",
            compatible=False,
            reason="Material equivalence flag is true but supporting evidence is missing.",
        )
    return MaterialCompatibility(
        status="COMPATIBLE",
        compatible=True,
        reason="Caller supplied verified material-equivalence evidence.",
        evidence=case.material_equivalence_evidence,
    )


def convert_to_common_quantity(
    case: ComparisonCase,
    ann_tensile_strength_mpa: float,
    clt_lambda_cs: float,
    calculation_trace: list[dict[str, Any]] | None = None,
) -> dict[str, float | str]:
    """Convert ANN and CLT outputs into configured common quantity."""
    trace = calculation_trace if calculation_trace is not None else []
    if case.material_card is None:
        raise ValueError("material_card is required for conversion")
    if case.load_case is None:
        raise ValueError("load_case is required for conversion")
    if case.comparison_quantity not in SUPPORTED_COMPARISON_MODES:
        raise ValueError(f"Unsupported comparison_quantity: {case.comparison_quantity}")

    total_thickness_m = len(case.stacking_sequence) * case.material_card.ply_thickness_m
    if total_thickness_m <= 0:
        raise ValueError("total laminate thickness must be greater than zero")

    base_nx_n_per_m = case.load_case.nx_n_per_m
    clt_failure_nx_n_per_m = clt_lambda_cs * base_nx_n_per_m
    ann_strength_pa = ann_tensile_strength_mpa * MPA_TO_PA
    ann_load_capacity_n_per_m = ann_strength_pa * total_thickness_m
    clt_equivalent_stress_mpa = clt_failure_nx_n_per_m / total_thickness_m * PA_TO_MPA

    trace.extend(
        [
            {
                "step": "Input ANN tensile strength",
                "equation": "ANN output",
                "value": ann_tensile_strength_mpa,
                "unit": "MPa",
            },
            {
                "step": "Input CLT load factor",
                "equation": "lambda_cs from calculate_strain_allowable_failure_load()",
                "value": clt_lambda_cs,
                "unit": "dimensionless",
            },
            {
                "step": "Laminate total thickness",
                "equation": "ply_count * ply_thickness_m",
                "value": total_thickness_m,
                "unit": "m",
            },
            {
                "step": "CLT failure Nx",
                "equation": "lambda_cs * base_Nx",
                "value": clt_failure_nx_n_per_m,
                "unit": "N/m",
            },
        ]
    )

    if case.comparison_quantity == STRESS_MODE:
        trace.append(
            {
                "step": "CLT equivalent laminate tensile stress",
                "equation": "failure_Nx / total_thickness",
                "value": clt_equivalent_stress_mpa,
                "unit": "MPa",
            }
        )
        return {
            "common_unit": "MPa",
            "ann_common_value": ann_tensile_strength_mpa,
            "clt_common_value": clt_equivalent_stress_mpa,
        }

    trace.append(
        {
            "step": "ANN tensile load capacity per unit width",
            "equation": "ANN_strength_Pa * total_thickness",
            "value": ann_load_capacity_n_per_m,
            "unit": "N/m",
        }
    )
    return {
        "common_unit": "N/m",
        "ann_common_value": ann_load_capacity_n_per_m,
        "clt_common_value": clt_failure_nx_n_per_m,
    }


def calculate_percentage_difference(ann_common_value: float, clt_common_value: float) -> float:
    """Return relative difference from CLT reference in percent."""
    if clt_common_value == 0:
        raise ValueError("CLT reference value must be non-zero")
    return abs(ann_common_value - clt_common_value) / abs(clt_common_value) * 100.0


def _extract_or_predict_ann_value(
    case: ComparisonCase,
    ann_result: dict[str, Any] | None,
) -> float:
    """Return ANN tensile-strength prediction in MPa."""
    active_result = ann_result
    if active_result is None:
        from src.predict import predict_strength

        active_result = predict_strength(case.ann_input)

    value = active_result.get("predicted_tensile_strength_mpa")
    if value is None:
        value = active_result.get(ANN_OUTPUT_NAME)
    ann_value = _safe_float(value)
    if ann_value is None:
        raise ValueError("ANN result does not contain finite tensile strength in MPa")
    return ann_value


def _evaluate_clt_if_possible(
    case: ComparisonCase,
    calculation_trace: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any] | None:
    """Evaluate CLT lambda_cs when required inputs exist."""
    if case.material_card is None:
        warnings.append("CLT result unavailable: missing material_card.")
        return None
    if case.load_case is None:
        warnings.append("CLT result unavailable: missing load_case.")
        return None
    try:
        result = calculate_strain_allowable_failure_load(
            case.stacking_sequence,
            case.material_card,
            case.load_case,
            case.allowables,
        )
    except ValueError as exc:
        warnings.append(f"CLT result unavailable: {exc}")
        return None
    calculation_trace.append(
        {
            "step": "CLT strain-allowable evaluation",
            "equation": "source-compatible lambda_cs route",
            "value": result["lambda_cs"],
            "unit": "dimensionless",
        }
    )
    return result


def _pre_evaluation_rejection_reason(case: ComparisonCase) -> str | None:
    """Return rejection reason that can be known before CLT evaluation."""
    if case.comparison_quantity not in SUPPORTED_COMPARISON_MODES:
        return f"Unsupported comparison quantity: {case.comparison_quantity}."
    if case.base_load_unit != LOAD_UNIT_N_PER_M:
        return "Invalid load units: CLT base Nx must be supplied in N/m."
    if case.load_case is None:
        return "Missing base load: CLT load_case is required."
    if case.material_card is None:
        return "Missing thickness: CLT material_card with ply_thickness_m is required."
    if case.material_card.ply_thickness_m <= 0:
        return "Missing thickness: ply_thickness_m must be greater than zero."
    if not case.stacking_sequence:
        return "Missing stacking sequence: CLT laminate sequence is required."
    if not _is_uniaxial_tensile_nx(case.load_case):
        return (
            "Invalid conversion: ANN tensile strength can only be compared with "
            "CLT uniaxial tensile Nx load case where Ny, Nxy, and moments are zero."
        )
    return None


def _comparison_rejection_reason(
    case: ComparisonCase,
    compatibility: MaterialCompatibility,
    clt_evaluation: dict[str, Any] | None,
) -> str | None:
    """Return reason when direct numerical comparison is invalid."""
    if not compatibility.compatible:
        return compatibility.reason
    if clt_evaluation is None:
        return "CLT result unavailable."
    clt_lambda = _safe_float(clt_evaluation.get("lambda_cs"))
    if clt_lambda is None or not math.isfinite(clt_lambda):
        return "Invalid conversion: CLT lambda_cs is missing or non-finite."
    return None


def _is_uniaxial_tensile_nx(load_case: LaminateLoadCase) -> bool:
    """Return True for supported tensile Nx-only comparison load."""
    return (
        load_case.nx_n_per_m > 0
        and load_case.ny_n_per_m == 0
        and load_case.nxy_n_per_m == 0
        and load_case.mx_n == 0
        and load_case.my_n == 0
        and load_case.mxy_n == 0
    )


def _safe_float(value: Any) -> float | None:
    """Return finite float or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
