"""
Unit tests for Jacobian matrix calculation, eigenvalues, and linear stability classification.
"""

import numpy as np
import pytest
from fhn_core.model import FHNParameters, fhn_derivatives
from fhn_core.analysis import (
    compute_jacobian,
    compute_stability_metrics,
    classify_equilibrium,
    find_equilibria,
    StabilityClass,
)


def test_analytical_vs_finite_difference_jacobian():
    """
    Validates analytical Jacobian formula against 2nd-order central finite differences:
        ∂f_i/∂x_j ≈ (f_i(x + δ*e_j) - f_i(x - δ*e_j)) / (2*δ)
    """
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    v_test = -0.80485
    w_test = -0.13106

    J_analytical = compute_jacobian(v_test, params)

    delta = 1e-6
    state_center = np.array([v_test, w_test])

    # Perturbations
    f_v_plus = fhn_derivatives(0.0, state_center + np.array([delta, 0.0]), params)
    f_v_minus = fhn_derivatives(0.0, state_center - np.array([delta, 0.0]), params)

    f_w_plus = fhn_derivatives(0.0, state_center + np.array([0.0, delta]), params)
    f_w_minus = fhn_derivatives(0.0, state_center - np.array([0.0, delta]), params)

    J_numerical = np.zeros((2, 2))
    J_numerical[:, 0] = (f_v_plus - f_v_minus) / (2.0 * delta)
    J_numerical[:, 1] = (f_w_plus - f_w_minus) / (2.0 * delta)

    # Difference must be smaller than 1e-6
    max_err = np.max(np.abs(J_analytical - J_numerical))
    assert max_err < 1e-6, f"Max difference between analytical and numerical Jacobian: {max_err}"


def test_trace_determinant_eigenvalue_consistency():
    """
    Verifies that eigenvalues satisfy the characteristic polynomial:
        λ² - τ*λ + Δ = 0
    """
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    eqs = find_equilibria(params)
    assert len(eqs) > 0
    eq = eqs[0]

    trace = eq.trace
    det = eq.determinant
    disc = eq.discriminant

    # Trace equals sum of diagonal
    assert np.isclose(trace, eq.jacobian[0, 0] + eq.jacobian[1, 1])

    # Determinant equals J00*J11 - J01*J10
    det_calc = eq.jacobian[0, 0] * eq.jacobian[1, 1] - eq.jacobian[0, 1] * eq.jacobian[1, 0]
    assert np.isclose(det, det_calc)

    # Discriminant
    assert np.isclose(disc, trace**2 - 4.0 * det)

    # Characteristic polynomial evaluation at both eigenvalues
    for lam in eq.eigenvalues:
        char_val = lam**2 - trace * lam + det
        assert abs(char_val) < 1e-10, f"Characteristic polynomial residual {char_val} too high for {lam}"


def test_stability_classification_regimes():
    """
    Verifies stability classifier across distinct canonical eigenvalue signatures.
    """
    # 1. Stable focus: trace < 0, det > 0, disc < 0
    stab, is_st, _ = classify_equilibrium(trace=-0.5, determinant=1.0, discriminant=-3.75,
                                         eigenvalues=(-0.25 + 0.9682j, -0.25 - 0.9682j))
    assert stab == StabilityClass.STABLE_FOCUS
    assert is_st is True

    # 2. Stable node: trace < 0, det > 0, disc > 0
    stab, is_st, _ = classify_equilibrium(trace=-3.0, determinant=2.0, discriminant=1.0,
                                         eigenvalues=(-1.0 + 0j, -2.0 + 0j))
    assert stab == StabilityClass.STABLE_NODE
    assert is_st is True

    # 3. Unstable focus: trace > 0, det > 0, disc < 0
    stab, is_st, _ = classify_equilibrium(trace=0.4, determinant=1.0, discriminant=-3.84,
                                         eigenvalues=(0.2 + 0.9798j, 0.2 - 0.9798j))
    assert stab == StabilityClass.UNSTABLE_FOCUS
    assert is_st is False

    # 4. Saddle point: det < 0
    stab, is_st, _ = classify_equilibrium(trace=0.5, determinant=-1.5, discriminant=6.25,
                                         eigenvalues=(1.5 + 0j, -1.0 + 0j))
    assert stab == StabilityClass.SADDLE
    assert is_st is False

