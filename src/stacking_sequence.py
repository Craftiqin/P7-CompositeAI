"""Stacking-sequence representation and validation utilities.

Step 7 defines sequence representation only. These functions do not evaluate
strength, train models, or optimize laminate designs.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable


DEFAULT_ALLOWED_ANGLES = (-45, 0, 45, 90)


@dataclass(frozen=True)
class SequenceConfig:
    """Configurable stacking-sequence constraints."""

    allowed_angles: tuple[int, ...] = DEFAULT_ALLOWED_ANGLES
    require_symmetric: bool = False
    require_balanced: bool = False
    expected_ply_count: int | None = None
    max_search_space: int = 100_000


@dataclass(frozen=True)
class LaminateSequence:
    """Ordered laminate ply-orientation representation in degrees."""

    sequence: tuple[int, ...]
    allowed_angles: tuple[int, ...] = DEFAULT_ALLOWED_ANGLES

    @property
    def ply_count(self) -> int:
        """Return number of plies."""
        return len(self.sequence)


@dataclass(frozen=True)
class SequenceValidationResult:
    """Validation result with clear reasons."""

    valid: bool
    reasons: list[str] = field(default_factory=list)


def normalize_angle(value: object) -> int:
    """Convert numeric angle to integer degrees."""
    if isinstance(value, bool):
        raise ValueError("Angle must be numeric, not boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Angle is not numeric: {value!r}") from exc
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError(f"Angle must be finite: {value!r}")
    if numeric % 1 != 0:
        raise ValueError(f"Angle must be whole degrees: {value!r}")
    return int(numeric)


def normalize_sequence(sequence: Iterable[object]) -> tuple[int, ...]:
    """Return ordered tuple of validated numeric angles."""
    return tuple(normalize_angle(angle) for angle in sequence)


def is_symmetric(sequence: Iterable[object]) -> bool:
    """Return True when sequence mirrors about mid-plane."""
    normalized = normalize_sequence(sequence)
    return normalized == tuple(reversed(normalized))


def is_balanced(sequence: Iterable[object], allowed_angles: Iterable[int] = DEFAULT_ALLOWED_ANGLES) -> bool:
    """Return True when +theta and -theta ply counts match.

    Exact Step 7 rule: for every positive angle whose negative angle is allowed,
    count(+theta) must equal count(-theta). Zero and 90-degree plies do not
    affect this balance check.
    """
    normalized = normalize_sequence(sequence)
    allowed = set(normalize_sequence(allowed_angles))
    for angle in sorted(angle for angle in allowed if angle > 0 and -angle in allowed):
        if normalized.count(angle) != normalized.count(-angle):
            return False
    return True


def validate_sequence(
    sequence: Iterable[object],
    allowed_angles: Iterable[int] = DEFAULT_ALLOWED_ANGLES,
    require_symmetric: bool = False,
    require_balanced: bool = False,
    expected_ply_count: int | None = None,
) -> SequenceValidationResult:
    """Validate sequence without modifying order or content."""
    reasons: list[str] = []

    try:
        normalized = normalize_sequence(sequence)
        allowed = set(normalize_sequence(allowed_angles))
    except ValueError as exc:
        return SequenceValidationResult(valid=False, reasons=[str(exc)])

    if not normalized:
        reasons.append("sequence must contain at least one ply")

    if expected_ply_count is not None:
        if expected_ply_count <= 0:
            reasons.append("expected_ply_count must be greater than zero")
        elif len(normalized) != expected_ply_count:
            reasons.append(
                f"sequence has {len(normalized)} plies; expected {expected_ply_count}"
            )

    invalid_angles = sorted(set(normalized) - allowed)
    if invalid_angles:
        reasons.append(f"angle(s) not allowed: {invalid_angles}")

    if require_symmetric and normalized and not is_symmetric(normalized):
        reasons.append("sequence is not symmetric")

    if require_balanced and normalized and not is_balanced(normalized, allowed):
        reasons.append("sequence is not balanced")

    return SequenceValidationResult(valid=not reasons, reasons=reasons)


def estimate_search_space_size(
    ply_count: int,
    allowed_angles: Iterable[int] = DEFAULT_ALLOWED_ANGLES,
    require_symmetric: bool = False,
) -> int:
    """Estimate raw candidate count before balance filtering."""
    if ply_count <= 0:
        raise ValueError("ply_count must be greater than zero")
    angle_count = len(tuple(normalize_sequence(allowed_angles)))
    variable_positions = math.ceil(ply_count / 2) if require_symmetric else ply_count
    return angle_count**variable_positions


def make_symmetric_sequence(half_sequence: tuple[int, ...], ply_count: int) -> tuple[int, ...]:
    """Mirror variable positions into full symmetric sequence."""
    if ply_count % 2 == 0:
        return half_sequence + tuple(reversed(half_sequence))
    return half_sequence + tuple(reversed(half_sequence[:-1]))


def generate_candidate_sequences(config: SequenceConfig, max_candidates: int | None = None) -> list[LaminateSequence]:
    """Generate valid candidate sequences under configured constraints.

    Generation is bounded. If raw search space exceeds config.max_search_space,
    caller must use Step 8 optimization/search instead of exhaustive generation.
    """
    if config.expected_ply_count is None:
        raise ValueError("expected_ply_count is required for candidate generation")

    search_space = estimate_search_space_size(
        config.expected_ply_count,
        config.allowed_angles,
        config.require_symmetric,
    )
    if search_space > config.max_search_space:
        raise ValueError(
            f"search space {search_space} exceeds limit {config.max_search_space}; "
            "use optimization/search strategy in Step 8"
        )

    limit = max_candidates or config.max_search_space
    positions = math.ceil(config.expected_ply_count / 2) if config.require_symmetric else config.expected_ply_count
    candidates: list[LaminateSequence] = []
    seen: set[tuple[int, ...]] = set()

    for combination in itertools.product(config.allowed_angles, repeat=positions):
        sequence = (
            make_symmetric_sequence(combination, config.expected_ply_count)
            if config.require_symmetric
            else combination
        )
        if sequence in seen:
            continue
        seen.add(sequence)
        result = validate_sequence(
            sequence,
            allowed_angles=config.allowed_angles,
            require_symmetric=config.require_symmetric,
            require_balanced=config.require_balanced,
            expected_ply_count=config.expected_ply_count,
        )
        if result.valid:
            candidates.append(
                LaminateSequence(
                    sequence=sequence,
                    allowed_angles=config.allowed_angles,
                )
            )
        if len(candidates) >= limit:
            break

    return candidates


def format_laminate_sequence(sequence: Iterable[object]) -> str:
    """Return simple ply stack text with mid-plane marker."""
    normalized = normalize_sequence(sequence)
    rows = [
        f"Ply {len(normalized) - offset:>2}   {angle:+d}°"
        for offset, angle in enumerate(reversed(normalized))
    ]
    midpoint = len(rows) // 2
    rows.insert(midpoint, "----------------\nMid-plane\n----------------")
    return "\n".join(rows)
