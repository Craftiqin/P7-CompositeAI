"""CLT-based stacking-sequence optimization demonstrator.

This optimizer uses source-backed strain allowables and CLT stiffness. It does
not use the ANN/MLP model and does not claim experimental validation.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.clt import (
    LB_PER_IN_TO_N_PER_M,
    LaminaMaterial,
    LaminateLoadCase,
    StrainAllowables,
    calculate_strain_allowable_failure_load,
)
from src.sequence_data import extract_python_source_record, parse_material_card
from src.stacking_sequence import (
    DEFAULT_ALLOWED_ANGLES,
    LaminateSequence,
    SequenceConfig,
    estimate_search_space_size,
    generate_candidate_sequences,
    is_balanced,
    is_symmetric,
    validate_sequence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CASE = (
    PROJECT_ROOT
    / "data"
    / "sequence"
    / "tu_delft_zenodo_15864524"
    / "raw"
    / "D_Case1_pymoo_load.py"
)


@dataclass(frozen=True)
class OptimizationConfig:
    """Bounded optimization configuration."""

    sequence_config: SequenceConfig
    max_candidates: int = 250
    random_seed: int = 42
    top_n: int = 5


def load_tu_delft_demo_case(path: Path = DEFAULT_SOURCE_CASE) -> dict[str, Any]:
    """Load source-backed material, sequence, load, and reference values."""
    record = extract_python_source_record(path)
    material = parse_material_card(record)
    load_case = LaminateLoadCase(
        nx_n_per_m=-record.load_case["load_Nxx_lb_per_in"] * LB_PER_IN_TO_N_PER_M,
        ny_n_per_m=-record.load_case["load_Nyy_lb_per_in"] * LB_PER_IN_TO_N_PER_M,
    )
    return {
        "source_file": record.file_name,
        "material": material,
        "baseline_sequence": record.sequence,
        "load_case": load_case,
        "reference_outputs": record.reference_outputs,
        "allowables": StrainAllowables(),
    }


def validate_reference_case(path: Path = DEFAULT_SOURCE_CASE) -> dict[str, Any]:
    """Compare our source-compatible lambda_cs with preserved reference comment."""
    case = load_tu_delft_demo_case(path)
    result = calculate_strain_allowable_failure_load(
        case["baseline_sequence"],
        case["material"],
        case["load_case"],
        case["allowables"],
    )
    reference = case["reference_outputs"].get("reference_failure_load_lambda_cs")
    difference_pct = None
    status = "reference_missing"
    if reference:
        difference_pct = (result["lambda_cs"] - reference) / reference * 100.0
        status = "pass" if abs(difference_pct) <= 1.0 else "warning"
    return {
        "source_file": case["source_file"],
        "reference_lambda_cs": reference,
        "our_lambda_cs": result["lambda_cs"],
        "difference_pct": difference_pct,
        "validation_status": status,
        "assumptions": [
            "Uses source strain allowables epsilon_1=0.008, epsilon_2=0.029, gamma_12=0.015.",
            "Uses source Nxx/Nyy load convention in N/m without the 1.5 multiplier present in some scripts.",
            "Uses CLT A-matrix in-plane response; buckling is not optimized.",
        ],
    }


def optimize_stacking_sequence(
    material: LaminaMaterial,
    load_case: LaminateLoadCase,
    config: OptimizationConfig,
    baseline_sequence: list[int] | None = None,
    allowables: StrainAllowables | None = None,
) -> dict[str, Any]:
    """Find best sequence within configured bounded search space."""
    start = time.perf_counter()
    active_allowables = allowables or StrainAllowables()
    search_space = estimate_search_space_size(
        config.sequence_config.expected_ply_count,
        config.sequence_config.allowed_angles,
        config.sequence_config.require_symmetric,
    )
    candidates = _candidate_pool(config, search_space)
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        evaluation = calculate_strain_allowable_failure_load(
            candidate.sequence,
            material,
            load_case,
            active_allowables,
        )
        ranked.append(
            {
                "sequence": list(candidate.sequence),
                "lambda_cs": evaluation["lambda_cs"],
                "critical_ply": evaluation["critical_ply"],
                "failure_mode": evaluation["failure_mode"],
                "evaluation": evaluation,
            }
        )

    ranked.sort(key=lambda row: row["lambda_cs"], reverse=True)
    best = ranked[0]
    baseline = None
    if baseline_sequence is not None:
        baseline_result = calculate_strain_allowable_failure_load(
            baseline_sequence,
            material,
            load_case,
            active_allowables,
        )
        baseline = {
            "sequence": baseline_sequence,
            "lambda_cs": baseline_result["lambda_cs"],
            "critical_ply": baseline_result["critical_ply"],
            "failure_mode": baseline_result["failure_mode"],
        }

    improvement_pct = None
    if baseline and baseline["lambda_cs"]:
        improvement_pct = (best["lambda_cs"] - baseline["lambda_cs"]) / baseline["lambda_cs"] * 100.0

    return {
        "best_sequence": best["sequence"],
        "best_lambda_cs": best["lambda_cs"],
        "critical_ply": best["critical_ply"],
        "failure_mode": best["failure_mode"],
        "ply_count": config.sequence_config.expected_ply_count,
        "material": material.name,
        "load_case": load_case.vector().tolist(),
        "failure_criterion": "Source-compatible strain allowable",
        "constraints": {
            "allowed_angles": list(config.sequence_config.allowed_angles),
            "require_symmetric": config.sequence_config.require_symmetric,
            "require_balanced": config.sequence_config.require_balanced,
            "expected_ply_count": config.sequence_config.expected_ply_count,
        },
        "search_method": "enumeration" if search_space <= config.max_candidates else "bounded_random_search",
        "theoretical_search_space": search_space,
        "candidates_evaluated": len(candidates),
        "evaluated_candidates": ranked,
        "top_candidates": ranked[: config.top_n],
        "baseline": baseline,
        "improvement_pct": improvement_pct,
        "runtime_seconds": time.perf_counter() - start,
        "disclaimer": (
            "Best sequence found within configured search space and constraints; "
            "not experimentally validated."
        ),
    }


def _candidate_pool(config: OptimizationConfig, search_space: int) -> list[LaminateSequence]:
    """Return enumeration or random candidate pool without duplicates."""
    if search_space <= config.max_candidates:
        return generate_candidate_sequences(config.sequence_config, max_candidates=config.max_candidates)
    return _random_candidates(config)


def _random_candidates(config: OptimizationConfig) -> list[LaminateSequence]:
    """Generate deterministic bounded random valid candidates."""
    rng = random.Random(config.random_seed)
    sequence_config = config.sequence_config
    if sequence_config.expected_ply_count is None:
        raise ValueError("expected_ply_count is required")
    candidates: list[LaminateSequence] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    max_attempts = config.max_candidates * 200
    while len(candidates) < config.max_candidates and attempts < max_attempts:
        attempts += 1
        sequence = _random_sequence(sequence_config, rng)
        if sequence in seen:
            continue
        result = validate_sequence(
            sequence,
            allowed_angles=sequence_config.allowed_angles,
            require_symmetric=sequence_config.require_symmetric,
            require_balanced=sequence_config.require_balanced,
            expected_ply_count=sequence_config.expected_ply_count,
        )
        if result.valid:
            seen.add(sequence)
            candidates.append(LaminateSequence(sequence, sequence_config.allowed_angles))
    if not candidates:
        raise ValueError("No valid candidates generated under configured constraints.")
    return candidates


def _random_sequence(config: SequenceConfig, rng: random.Random) -> tuple[int, ...]:
    """Generate one random sequence under basic symmetry/balance pattern."""
    ply_count = config.expected_ply_count
    if ply_count is None:
        raise ValueError("expected_ply_count is required")
    if config.require_symmetric:
        half_count = (ply_count + 1) // 2
        half = [rng.choice(config.allowed_angles) for _ in range(half_count)]
        sequence = tuple(half + list(reversed(half[: ply_count // 2])))
    else:
        sequence = tuple(rng.choice(config.allowed_angles) for _ in range(ply_count))

    if config.require_balanced and 45 in config.allowed_angles and -45 in config.allowed_angles:
        sequence = _force_balanced(sequence, rng)
        if config.require_symmetric:
            half = sequence[: (ply_count + 1) // 2]
            sequence = tuple(half + tuple(reversed(half[: ply_count // 2])))
    return sequence


def _force_balanced(sequence: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    """Adjust +45/-45 counts without changing ply count."""
    values = list(sequence)
    count_pos = values.count(45)
    count_neg = values.count(-45)
    while count_pos != count_neg:
        if count_pos > count_neg:
            indexes = [idx for idx, value in enumerate(values) if value == 45]
            values[rng.choice(indexes)] = -45
            count_pos -= 1
            count_neg += 1
        else:
            indexes = [idx for idx, value in enumerate(values) if value == -45]
            values[rng.choice(indexes)] = 45
            count_pos += 1
            count_neg -= 1
    return tuple(values)


def optimizer_result_is_valid(result: dict[str, Any]) -> bool:
    """Return True when optimizer output satisfies configured constraints."""
    sequence = result["best_sequence"]
    constraints = result["constraints"]
    validation = validate_sequence(
        sequence,
        allowed_angles=constraints["allowed_angles"],
        require_symmetric=constraints["require_symmetric"],
        require_balanced=constraints["require_balanced"],
        expected_ply_count=constraints["expected_ply_count"],
    )
    return validation.valid and (
        not constraints["require_symmetric"] or is_symmetric(sequence)
    ) and (
        not constraints["require_balanced"] or is_balanced(sequence, constraints["allowed_angles"])
    )


def default_demo_optimization(max_candidates: int = 250) -> dict[str, Any]:
    """Run bounded TU Delft-backed demonstration optimization."""
    case = load_tu_delft_demo_case()
    config = OptimizationConfig(
        sequence_config=SequenceConfig(
            allowed_angles=DEFAULT_ALLOWED_ANGLES,
            require_symmetric=True,
            require_balanced=True,
            expected_ply_count=48,
        ),
        max_candidates=max_candidates,
        random_seed=42,
    )
    return optimize_stacking_sequence(
        material=case["material"],
        load_case=case["load_case"],
        config=config,
        baseline_sequence=case["baseline_sequence"],
        allowables=case["allowables"],
    )
