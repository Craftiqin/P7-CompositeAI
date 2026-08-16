"""Tests for Step 8 Classical Laminate Theory evaluator."""

from __future__ import annotations

import unittest

import numpy as np

from src.clt import (
    PA_PER_GPA,
    LaminaMaterial,
    LaminateLoadCase,
    StrainAllowables,
    abd_matrices,
    calculate_strain_allowable_failure_load,
    evaluate_laminate,
    global_to_local_strain,
    optimization_readiness,
    reduced_stiffness_matrix,
    strain_failure_index,
    transformed_stiffness_matrix,
)


def stiffness_only_material() -> LaminaMaterial:
    """Return deterministic material card for CLT math tests."""
    return LaminaMaterial(
        name="UnitTest Orthotropic Lamina",
        e1_pa=135.0 * PA_PER_GPA,
        e2_pa=10.0 * PA_PER_GPA,
        g12_pa=5.0 * PA_PER_GPA,
        nu12=0.30,
        ply_thickness_m=0.000125,
    )


def strength_material() -> LaminaMaterial:
    """Return deterministic material card with allowables for criterion tests."""
    return LaminaMaterial(
        name="UnitTest Strength Lamina",
        e1_pa=135.0 * PA_PER_GPA,
        e2_pa=10.0 * PA_PER_GPA,
        g12_pa=5.0 * PA_PER_GPA,
        nu12=0.30,
        ply_thickness_m=0.000125,
        xt_pa=1_500_000_000.0,
        xc_pa=1_200_000_000.0,
        yt_pa=40_000_000.0,
        yc_pa=150_000_000.0,
        s_pa=70_000_000.0,
    )


class Step8CltTest(unittest.TestCase):
    """Validate CLT mechanics without ANN coupling or optimization."""

    def test_q_matrix_symmetry(self) -> None:
        """Reduced stiffness matrix Q is 3x3 symmetric."""
        q_matrix = reduced_stiffness_matrix(stiffness_only_material())
        self.assertEqual(q_matrix.shape, (3, 3))
        self.assertTrue(np.allclose(q_matrix, q_matrix.T))

    def test_qbar_calculation(self) -> None:
        """0-degree Qbar equals Q; 90-degree Qbar swaps axial stiffness terms."""
        material = stiffness_only_material()
        q_matrix = reduced_stiffness_matrix(material)
        qbar_0 = transformed_stiffness_matrix(material, 0)
        qbar_90 = transformed_stiffness_matrix(material, 90)
        self.assertTrue(np.allclose(qbar_0, q_matrix, atol=1e-5))
        self.assertAlmostEqual(qbar_90[0, 0], q_matrix[1, 1], delta=1e-4)
        self.assertAlmostEqual(qbar_90[1, 1], q_matrix[0, 0], delta=1e-4)

    def test_abd_matrix_dimensions(self) -> None:
        """A/B/D matrices are 3x3 and ABD is 6x6."""
        matrices = abd_matrices([0, 45, -45, 90], stiffness_only_material())
        self.assertEqual(np.array(matrices["A"]).shape, (3, 3))
        self.assertEqual(np.array(matrices["B"]).shape, (3, 3))
        self.assertEqual(np.array(matrices["D"]).shape, (3, 3))
        self.assertEqual(np.array(matrices["ABD"]).shape, (6, 6))

    def test_symmetric_laminate_b_near_zero(self) -> None:
        """Symmetric laminate has negligible coupling matrix B."""
        matrices = abd_matrices([0, 45, -45, 90, 90, -45, 45, 0], stiffness_only_material())
        b_matrix = np.array(matrices["B"])
        self.assertTrue(np.allclose(b_matrix, np.zeros((3, 3)), atol=1e-8))

    def test_sequence_order_preserved(self) -> None:
        """Ply results preserve input order."""
        sequence = [0, 45, -45, 90]
        result = evaluate_laminate(sequence, stiffness_only_material(), LaminateLoadCase(nx_n_per_m=1000.0))
        self.assertEqual(result["sequence"], sequence)
        self.assertEqual([ply["angle_deg"] for ply in result["ply_results"]], sequence)

    def test_zero_degree_laminate(self) -> None:
        """0-degree laminate under Nx produces finite ply stresses."""
        result = evaluate_laminate([0, 0, 0, 0], stiffness_only_material(), LaminateLoadCase(nx_n_per_m=1000.0))
        stresses = np.array([ply["local_stress_pa"] for ply in result["ply_results"]])
        self.assertTrue(np.isfinite(stresses).all())
        self.assertGreater(abs(stresses[:, 0]).mean(), abs(stresses[:, 1]).mean())

    def test_ninety_degree_laminate(self) -> None:
        """90-degree laminate under Nx produces finite ply stresses."""
        result = evaluate_laminate([90, 90, 90, 90], stiffness_only_material(), LaminateLoadCase(nx_n_per_m=1000.0))
        stresses = np.array([ply["local_stress_pa"] for ply in result["ply_results"]])
        self.assertTrue(np.isfinite(stresses).all())

    def test_balanced_plus_minus_45_laminate(self) -> None:
        """Balanced +/-45 laminate evaluates ply-by-ply."""
        sequence = [45, -45, -45, 45]
        result = evaluate_laminate(sequence, stiffness_only_material(), LaminateLoadCase(nx_n_per_m=1000.0))
        self.assertEqual(len(result["ply_results"]), 4)
        self.assertTrue(np.isfinite(np.array(result["midplane_strain"])).all())

    def test_global_to_local_strain_orientation_effect(self) -> None:
        """Ply angle transformation changes local strain components."""
        global_strain = np.array([0.001, 0.0, 0.0])
        local_0 = global_to_local_strain(global_strain, 0)
        local_90 = global_to_local_strain(global_strain, 90)
        self.assertGreater(local_0[0], local_0[1])
        self.assertGreater(local_90[1], local_90[0])

    def test_invalid_material_rejected(self) -> None:
        """Invalid material properties fail early."""
        with self.assertRaisesRegex(ValueError, "e1_pa"):
            LaminaMaterial(
                name="Invalid",
                e1_pa=-1.0,
                e2_pa=10.0,
                g12_pa=5.0,
                nu12=0.3,
                ply_thickness_m=0.001,
            )

    def test_invalid_ply_angle_rejected(self) -> None:
        """Invalid ply angle is rejected by sequence validation."""
        with self.assertRaisesRegex(ValueError, "not allowed"):
            evaluate_laminate([0, 30], stiffness_only_material(), LaminateLoadCase(nx_n_per_m=1000.0))

    def test_invalid_load_rejected(self) -> None:
        """Non-finite load is rejected."""
        with self.assertRaisesRegex(ValueError, "nx_n_per_m"):
            LaminateLoadCase(nx_n_per_m=float("nan"))

    def test_missing_strength_allowables_safe(self) -> None:
        """Failure evaluation stays unavailable when allowables missing."""
        result = evaluate_laminate([0, 0], stiffness_only_material(), LaminateLoadCase(nx_n_per_m=1000.0))
        self.assertFalse(result["failure_evaluation_available"])
        self.assertIsNone(result["failure_index"])
        self.assertEqual(result["failure_status"], "unavailable_missing_strength_allowables")

    def test_maximum_stress_failure_available(self) -> None:
        """Maximum Stress failure index is calculated when all allowables exist."""
        result = evaluate_laminate([0, 0], strength_material(), LaminateLoadCase(nx_n_per_m=1000.0))
        self.assertTrue(result["failure_evaluation_available"])
        self.assertIsNotNone(result["failure_index"])
        self.assertGreaterEqual(result["failure_index"], 0.0)

    def test_optimization_not_ready_without_material_card(self) -> None:
        """Optimization readiness requires verified material inputs."""
        readiness = optimization_readiness()
        self.assertFalse(readiness["ready"])
        self.assertIn("No verified material card supplied", readiness["reason"])

    def test_optimization_not_ready_without_allowables(self) -> None:
        """Stress-only material is not enough for objective optimization."""
        readiness = optimization_readiness(stiffness_only_material())
        self.assertFalse(readiness["ready"])
        self.assertIn("Xt", readiness["missing_verified_inputs"])

    def test_invalid_strain_allowables_rejected(self) -> None:
        """Invalid source-compatible strain allowables fail early."""
        with self.assertRaisesRegex(ValueError, "epsilon_1_allowable"):
            StrainAllowables(epsilon_1_allowable=0.0)

    def test_strain_failure_modes(self) -> None:
        """Strain criterion detects longitudinal, transverse, and shear modes."""
        allowables = StrainAllowables()
        self.assertEqual(strain_failure_index({"epsilon_1": 0.009, "epsilon_2": 0.0, "gamma_12": 0.0}, allowables)[1], "epsilon_1")
        self.assertEqual(strain_failure_index({"epsilon_1": 0.0, "epsilon_2": 0.030, "gamma_12": 0.0}, allowables)[1], "epsilon_2")
        self.assertEqual(strain_failure_index({"epsilon_1": 0.0, "epsilon_2": 0.0, "gamma_12": 0.016}, allowables)[1], "gamma_12")

    def test_strain_failure_load_monotonicity(self) -> None:
        """Doubling base load halves allowable load factor."""
        material = stiffness_only_material()
        sequence = [0, 45, -45, 90, 90, -45, 45, 0]
        low = calculate_strain_allowable_failure_load(
            sequence,
            material,
            LaminateLoadCase(nx_n_per_m=-100.0),
        )
        high = calculate_strain_allowable_failure_load(
            sequence,
            material,
            LaminateLoadCase(nx_n_per_m=-200.0),
        )
        self.assertAlmostEqual(low["lambda_cs"] / 2.0, high["lambda_cs"], delta=1e-9)


if __name__ == "__main__":
    unittest.main()
