"""
Unit tests for cubic equilibrium solving and residual verification.
"""

import numpy as np
import pytest
from fhn_core.model import FHNParameters
from fhn_core.analysis import find_equilibria, compute_nullclines


def test_standard_cardiac_equilibrium():
    """
    Validates equilibrium resolution for standard reference parameters:
        a = 0.7, b = 0.8, ε = 0.08, I = 0.5
    The cubic has a unique real root around v* ≈ -0.80485, w* ≈ -0.13106.
    """
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    equilibria = find_equilibria(params)

    assert len(equilibria) == 1, f"Expected 1 equilibrium, found {len(equilibria)}"
    eq = equilibria[0]

    # Verify expected coordinates
    assert np.isclose(eq.v_star, -0.80485, atol=1e-4)
    assert np.isclose(eq.w_star, -0.13106, atol=1e-4)

    # Residual verification: both residuals must be < 1e-10
    assert abs(eq.residual_v) < 1e-10, f"Residual v {eq.residual_v} exceeds tolerance"
    assert abs(eq.residual_w) < 1e-10, f"Residual w {eq.residual_w} exceeds tolerance"


def test_resting_state_equilibrium():
    """
    Validates polarized resting equilibrium with no bias current:
        a = 0.7, b = 0.8, ε = 0.08, I = 0.0
    For I = 0, the single stable resting equilibrium should satisfy:
        v* ≈ -1.1994, w* ≈ -0.6243
    """
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    equilibria = find_equilibria(params)

    assert len(equilibria) == 1
    eq = equilibria[0]

    assert abs(eq.residual_v) < 1e-10
    assert abs(eq.residual_w) < 1e-10
    assert eq.is_stable is True  # Resting state is asymptotically stable


def test_nullcline_intersection_coincidence():
    """
    Verifies that the analytical nullcline curves intersect exactly at the computed equilibria.
    """
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    equilibria = find_equilibria(params)
    assert len(equilibria) > 0

    for eq in equilibria:
        # Check w values on both nullclines at v = eq.v_star
        w_v = eq.v_star - (eq.v_star**3) / 3.0 + params.I
        w_w = (eq.v_star + params.a) / params.b

        assert np.isclose(w_v, eq.w_star, atol=1e-10)
        assert np.isclose(w_w, eq.w_star, atol=1e-10)
        assert np.isclose(w_v, w_w, atol=1e-10)


def test_multiple_equilibria_regime():
    """
    Tests parameters that produce three real equilibria (e.g. b > 1 with suitable a and I).
    Equation: b*(v*)³ + 3*(1-b)*v* + 3a - 3bI = 0
    When b = 1.5, 3*(1-b) = -1.5 < 0, allowing 3 real roots for small constant term.
    """
    params = FHNParameters(a=0.0, b=1.5, epsilon=0.05, I=0.0)
    equilibria = find_equilibria(params)

    assert len(equilibria) == 3, f"Expected 3 equilibria for bistable setup, got {len(equilibria)}"

    for eq in equilibria:
        assert abs(eq.residual_v) < 1e-10
        assert abs(eq.residual_w) < 1e-10

