"""Classical Laminate Theory utilities for sequence-dependent evaluation.

This module is mechanics-only. It does not use the ANN/MLP model, fabricate
material properties, train models, or optimize stacking sequences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.stacking_sequence import normalize_sequence, validate_sequence


PA_PER_GPA = 1_000_000_000.0
LB_PER_IN_TO_N_PER_M = 4.448222 / (25.4 / 1000.0)


@dataclass(frozen=True)
class LaminaMaterial:
    """Orthotropic lamina material card using SI units.

    E1, E2, G12, Xt, Xc, Yt, Yc, and S use Pa. ply_thickness_m uses meters.
    Strength allowables are optional; failure evaluation is unavailable unless
    all five allowables are provided.
    """

    name: str
    e1_pa: float
    e2_pa: float
    g12_pa: float
    nu12: float
    ply_thickness_m: float
    xt_pa: float | None = None
    xc_pa: float | None = None
    yt_pa: float | None = None
    yc_pa: float | None = None
    s_pa: float | None = None

    def __post_init__(self) -> None:
        """Validate material values."""
        for field_name in ("e1_pa", "e2_pa", "g12_pa", "ply_thickness_m"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and > 0")
        if not math.isfinite(float(self.nu12)) or self.nu12 <= 0:
            raise ValueError("nu12 must be finite and > 0")
        nu21 = self.nu21
        if 1.0 - self.nu12 * nu21 <= 0:
            raise ValueError("invalid orthotropic constants: 1 - nu12*nu21 must be > 0")
        for field_name in ("xt_pa", "xc_pa", "yt_pa", "yc_pa", "s_pa"):
            value = getattr(self, field_name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{field_name} must be finite and > 0 when provided")

    @property
    def nu21(self) -> float:
        """Return reciprocal Poisson ratio."""
        return self.nu12 * self.e2_pa / self.e1_pa

    @property
    def has_strength_allowables(self) -> bool:
        """Return True when Maximum Stress criterion can be evaluated."""
        return all(
            value is not None
            for value in (self.xt_pa, self.xc_pa, self.yt_pa, self.yc_pa, self.s_pa)
        )


@dataclass(frozen=True)
class LaminateLoadCase:
    """Laminate loads using CLT units.

    Nx, Ny, Nxy are in N/m. Mx, My, Mxy are in N.
    """

    nx_n_per_m: float
    ny_n_per_m: float = 0.0
    nxy_n_per_m: float = 0.0
    mx_n: float = 0.0
    my_n: float = 0.0
    mxy_n: float = 0.0

    def __post_init__(self) -> None:
        """Validate finite load components."""
        for field_name in (
            "nx_n_per_m",
            "ny_n_per_m",
            "nxy_n_per_m",
            "mx_n",
            "my_n",
            "mxy_n",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")

    def vector(self) -> np.ndarray:
        """Return six-component load vector."""
        return np.array(
            [
                self.nx_n_per_m,
                self.ny_n_per_m,
                self.nxy_n_per_m,
                self.mx_n,
                self.my_n,
                self.mxy_n,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class StrainAllowables:
    """Source-backed strain allowables for Haftka-style failure route.

    Values are dimensionless engineering strains. Defaults are extracted from
    preserved TU Delft/Zenodo source files, which cite Haftka's 1993 paper.
    """

    epsilon_1_allowable: float = 0.008
    epsilon_2_allowable: float = 0.029
    gamma_12_allowable: float = 0.015
    source: str = "TU Delft/Zenodo source code citing Haftka 1993"

    def __post_init__(self) -> None:
        """Validate finite positive allowables."""
        for field_name in (
            "epsilon_1_allowable",
            "epsilon_2_allowable",
            "gamma_12_allowable",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and > 0")


def reduced_stiffness_matrix(material: LaminaMaterial) -> np.ndarray:
    """Return orthotropic reduced stiffness matrix Q under plane stress."""
    denominator = 1.0 - material.nu12 * material.nu21
    q11 = material.e1_pa / denominator
    q22 = material.e2_pa / denominator
    q12 = material.nu12 * material.e2_pa / denominator
    q66 = material.g12_pa
    return np.array(
        [
            [q11, q12, 0.0],
            [q12, q22, 0.0],
            [0.0, 0.0, q66],
        ],
        dtype=float,
    )


def transformed_stiffness_matrix(material: LaminaMaterial, angle_deg: float) -> np.ndarray:
    """Return transformed reduced stiffness matrix Qbar for ply angle."""
    angle = math.radians(float(angle_deg))
    m = math.cos(angle)
    n = math.sin(angle)
    q11, q12, q22, q66 = _q_terms(reduced_stiffness_matrix(material))

    m2 = m * m
    n2 = n * n
    m4 = m2 * m2
    n4 = n2 * n2
    mn = m * n

    qbar11 = q11 * m4 + 2.0 * (q12 + 2.0 * q66) * m2 * n2 + q22 * n4
    qbar22 = q11 * n4 + 2.0 * (q12 + 2.0 * q66) * m2 * n2 + q22 * m4
    qbar12 = (q11 + q22 - 4.0 * q66) * m2 * n2 + q12 * (m4 + n4)
    qbar16 = (q11 - q12 - 2.0 * q66) * m * m2 * n - (q22 - q12 - 2.0 * q66) * m * n * n2
    qbar26 = (q11 - q12 - 2.0 * q66) * m * n * n2 - (q22 - q12 - 2.0 * q66) * m * m2 * n
    qbar66 = (q11 + q22 - 2.0 * q12 - 2.0 * q66) * m2 * n2 + q66 * (m4 + n4)

    return np.array(
        [
            [qbar11, qbar12, qbar16],
            [qbar12, qbar22, qbar26],
            [qbar16, qbar26, qbar66],
        ],
        dtype=float,
    )


def _q_terms(q_matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Extract Q11, Q12, Q22, Q66."""
    return float(q_matrix[0, 0]), float(q_matrix[0, 1]), float(q_matrix[1, 1]), float(q_matrix[2, 2])


def laminate_z_coordinates(ply_count: int, ply_thickness_m: float) -> np.ndarray:
    """Return bottom-to-top z coordinates centered on laminate mid-plane."""
    if ply_count <= 0:
        raise ValueError("ply_count must be greater than zero")
    total_thickness = ply_count * ply_thickness_m
    return np.linspace(-total_thickness / 2.0, total_thickness / 2.0, ply_count + 1)


def abd_matrices(sequence: Iterable[object], material: LaminaMaterial) -> dict[str, np.ndarray]:
    """Calculate laminate A, B, D, and ABD matrices."""
    normalized = normalize_sequence(sequence)
    validation = validate_sequence(normalized)
    if not validation.valid:
        raise ValueError("; ".join(validation.reasons))

    z_values = laminate_z_coordinates(len(normalized), material.ply_thickness_m)
    a_matrix = np.zeros((3, 3), dtype=float)
    b_matrix = np.zeros((3, 3), dtype=float)
    d_matrix = np.zeros((3, 3), dtype=float)

    for ply_index, angle in enumerate(normalized):
        z_bottom = z_values[ply_index]
        z_top = z_values[ply_index + 1]
        qbar = transformed_stiffness_matrix(material, angle)
        a_matrix += qbar * (z_top - z_bottom)
        b_matrix += 0.5 * qbar * (z_top**2 - z_bottom**2)
        d_matrix += (1.0 / 3.0) * qbar * (z_top**3 - z_bottom**3)

    abd = np.block([[a_matrix, b_matrix], [b_matrix, d_matrix]])
    return {"A": a_matrix, "B": b_matrix, "D": d_matrix, "ABD": abd, "z": z_values}


def global_to_local_strain(global_strain: np.ndarray, angle_deg: float) -> np.ndarray:
    """Transform global engineering strain [ex, ey, gxy] to local [e1, e2, g12]."""
    angle = math.radians(float(angle_deg))
    m = math.cos(angle)
    n = math.sin(angle)
    ex, ey, gxy = global_strain
    return np.array(
        [
            m * m * ex + n * n * ey + m * n * gxy,
            n * n * ex + m * m * ey - m * n * gxy,
            -2.0 * m * n * ex + 2.0 * m * n * ey + (m * m - n * n) * gxy,
        ],
        dtype=float,
    )


def maximum_stress_failure_index(local_stress: np.ndarray, material: LaminaMaterial) -> float | None:
    """Return Maximum Stress failure index, or None if allowables missing."""
    if not material.has_strength_allowables:
        return None

    sigma_1, sigma_2, tau_12 = local_stress
    longitudinal = sigma_1 / material.xt_pa if sigma_1 >= 0 else abs(sigma_1) / material.xc_pa
    transverse = sigma_2 / material.yt_pa if sigma_2 >= 0 else abs(sigma_2) / material.yc_pa
    shear = abs(tau_12) / material.s_pa
    return float(max(longitudinal, transverse, shear))


def source_compatible_local_strains(exx: float, eyy: float, angle_deg: float) -> dict[str, float]:
    """Return local strains using preserved source's Haftka routine equations."""
    cosine = math.cos(math.radians(float(angle_deg)))
    sine = math.sin(math.radians(float(angle_deg)))
    return {
        "epsilon_1": cosine**2 * exx + sine**2 * eyy,
        "epsilon_2": sine**2 * exx + cosine**2 * eyy,
        "gamma_12": sine**2 * (eyy - exx),
    }


def strain_failure_index(local_strains: dict[str, float], allowables: StrainAllowables) -> tuple[float, str]:
    """Return strain failure index and governing mode."""
    ratios = {
        "epsilon_1": abs(local_strains["epsilon_1"]) / allowables.epsilon_1_allowable,
        "epsilon_2": abs(local_strains["epsilon_2"]) / allowables.epsilon_2_allowable,
        "gamma_12": abs(local_strains["gamma_12"]) / allowables.gamma_12_allowable,
    }
    mode = max(ratios, key=ratios.get)
    return float(ratios[mode]), mode


def calculate_strain_allowable_failure_load(
    sequence: Iterable[object],
    material: LaminaMaterial,
    load_case: LaminateLoadCase,
    allowables: StrainAllowables | None = None,
) -> dict[str, Any]:
    """Calculate source-compatible allowable load factor lambda_cs.

    The preserved TU Delft source uses in-plane A stiffness, Nxx/Nyy only, and
    Haftka-style strain allowables. Lambda scales the supplied load vector.
    """
    normalized = normalize_sequence(sequence)
    validation = validate_sequence(normalized)
    if not validation.valid:
        raise ValueError("; ".join(validation.reasons))
    if load_case.nxy_n_per_m or load_case.mx_n or load_case.my_n or load_case.mxy_n:
        raise ValueError("source-compatible lambda_cs supports Nx/Ny loads only")

    active_allowables = allowables or StrainAllowables()
    matrices = abd_matrices(normalized, material)
    a_matrix = matrices["A"]
    in_plane_stiffness = np.array(
        [[a_matrix[0, 0], a_matrix[0, 1]], [a_matrix[0, 1], a_matrix[1, 1]]],
        dtype=float,
    )
    base_strain = np.linalg.solve(
        in_plane_stiffness,
        np.array([load_case.nx_n_per_m, load_case.ny_n_per_m], dtype=float),
    )
    exx = float(base_strain[0])
    eyy = float(base_strain[1])

    ply_results: list[dict[str, Any]] = []
    governing: dict[str, Any] | None = None
    for ply_index, angle in enumerate(normalized, start=1):
        local = source_compatible_local_strains(exx, eyy, angle)
        failure_index, mode = strain_failure_index(local, active_allowables)
        if governing is None or failure_index > governing["failure_index_at_lambda_1"]:
            governing = {
                "critical_ply": ply_index,
                "failure_mode": mode,
                "failure_index_at_lambda_1": failure_index,
                "local_strains_at_lambda_1": local,
            }
        ply_results.append(
            {
                "ply_number": ply_index,
                "angle_deg": angle,
                "local_strains_at_lambda_1": local,
                "failure_index_at_lambda_1": failure_index,
                "failure_mode": mode,
            }
        )

    if governing is None or governing["failure_index_at_lambda_1"] <= 0:
        lambda_cs = math.inf
    else:
        lambda_cs = 1.0 / governing["failure_index_at_lambda_1"]

    return {
        "sequence": list(normalized),
        "ply_count": len(normalized),
        "material": material.name,
        "load_case": load_case.vector().tolist(),
        "failure_criterion": "Source-compatible strain allowable",
        "allowables": {
            "epsilon_1_allowable": active_allowables.epsilon_1_allowable,
            "epsilon_2_allowable": active_allowables.epsilon_2_allowable,
            "gamma_12_allowable": active_allowables.gamma_12_allowable,
            "source": active_allowables.source,
        },
        "lambda_cs": float(lambda_cs),
        "critical_ply": governing["critical_ply"] if governing else None,
        "failure_mode": governing["failure_mode"] if governing else None,
        "failure_index_at_lambda_1": governing["failure_index_at_lambda_1"] if governing else None,
        "midplane_strain_at_lambda_1": [exx, eyy, 0.0],
        "ply_results": ply_results,
    }


def evaluate_laminate(
    sequence: Iterable[object],
    material: LaminaMaterial,
    load_case: LaminateLoadCase,
) -> dict[str, Any]:
    """Evaluate laminate stiffness, strain, stress, and optional failure index."""
    normalized = normalize_sequence(sequence)
    validation = validate_sequence(normalized)
    if not validation.valid:
        raise ValueError("; ".join(validation.reasons))

    matrices = abd_matrices(normalized, material)
    response = np.linalg.solve(matrices["ABD"], load_case.vector())
    midplane_strain = response[:3]
    curvature = response[3:]
    q_matrix = reduced_stiffness_matrix(material)
    ply_results: list[dict[str, Any]] = []
    failure_indices: list[float] = []

    for ply_index, angle in enumerate(normalized):
        z_bottom = matrices["z"][ply_index]
        z_top = matrices["z"][ply_index + 1]
        z_mid = 0.5 * (z_bottom + z_top)
        global_strain = midplane_strain + z_mid * curvature
        local_strain = global_to_local_strain(global_strain, angle)
        qbar = transformed_stiffness_matrix(material, angle)
        global_stress = qbar @ global_strain
        local_stress = q_matrix @ local_strain
        failure_index = maximum_stress_failure_index(local_stress, material)
        if failure_index is not None:
            failure_indices.append(failure_index)
        ply_results.append(
            {
                "ply_number": ply_index + 1,
                "angle_deg": angle,
                "z_bottom_m": float(z_bottom),
                "z_top_m": float(z_top),
                "z_mid_m": float(z_mid),
                "global_strain": global_strain.tolist(),
                "local_strain": local_strain.tolist(),
                "global_stress_pa": global_stress.tolist(),
                "local_stress_pa": local_stress.tolist(),
                "failure_index": failure_index,
            }
        )

    max_failure_index = max(failure_indices) if failure_indices else None
    return {
        "sequence": list(normalized),
        "ply_count": len(normalized),
        "material": material.name,
        "load_case": load_case.vector().tolist(),
        "total_thickness_m": len(normalized) * material.ply_thickness_m,
        "A_matrix": matrices["A"].tolist(),
        "B_matrix": matrices["B"].tolist(),
        "D_matrix": matrices["D"].tolist(),
        "midplane_strain": midplane_strain.tolist(),
        "curvature": curvature.tolist(),
        "ply_results": ply_results,
        "failure_criterion": "Maximum Stress" if material.has_strength_allowables else None,
        "failure_evaluation_available": material.has_strength_allowables,
        "failure_index": max_failure_index,
        "failure_status": _failure_status(max_failure_index),
    }


def _failure_status(failure_index: float | None) -> str:
    """Return failure status from Maximum Stress failure index."""
    if failure_index is None:
        return "unavailable_missing_strength_allowables"
    return "fail" if failure_index >= 1.0 else "pass"


def optimization_readiness(material: LaminaMaterial | None = None) -> dict[str, Any]:
    """Return honest readiness status for sequence optimization."""
    if material is None:
        return {
            "ready": False,
            "missing_verified_inputs": [
                "LaminaMaterial card with E1, E2, G12, nu12, ply_thickness",
                "Maximum Stress allowables Xt, Xc, Yt, Yc, S",
                "verified load case magnitude",
            ],
            "reason": "No verified material card supplied.",
        }
    if not material.has_strength_allowables:
        return {
            "ready": False,
            "missing_verified_inputs": ["Xt", "Xc", "Yt", "Yc", "S"],
            "reason": "CLT stress calculation available; failure objective unavailable.",
        }
    return {"ready": True, "missing_verified_inputs": [], "reason": "Evaluator inputs supplied."}
