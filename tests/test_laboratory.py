"""
Unit tests for Stage 6: Mathematical Analysis Laboratory.
Verifies Parts A through F:
- Part A: Nullclines and explicit intersection points.
- Part B: Cardano analytical roots and residual error verification (< 10⁻¹²).
- Part C: Analytical Jacobian, trace, determinant, discriminant, and Poincaré classification.
- Part D: Phase plane vector field and multi-trajectory integration.
- Part E: Bifurcation sweeps over I, ε, a, b and numerical bifurcation condition verification.
- Part F: State comparison laboratory and dynamical diagnosis synthesis.
"""

import pytest
import numpy as np
from fhn_core import (
    FHNParameters,
    compute_nullclines,
    find_equilibria,
    compute_jacobian,
    compute_stability_metrics,
    classify_equilibrium,
    compute_vector_field,
    rk4_solve,
    sweep_parameter_I,
    sweep_parameter_epsilon,
    sweep_parameter_a,
    sweep_parameter_b,
    analyze_bifurcation_conditions,
    compare_fhn_states,
    StabilityClass,
)


def test_part_a_nullclines_and_intersections():
    """Verify v-nullcline, w-nullcline, and that their intersections match equilibrium points."""
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    v_vals, w_v_null, w_w_null = compute_nullclines((-2.5, 2.5), params, num_points=200)

    # Check v-nullcline formula: w = v - v^3/3 + I
    expected_wv = v_vals - (v_vals**3) / 3.0 + params.I
    np.testing.assert_allclose(w_v_null, expected_wv, rtol=1e-10)

    # Check w-nullcline formula: w = (v + a) / b
    expected_ww = (v_vals + params.a) / params.b
    np.testing.assert_allclose(w_w_null, expected_ww, rtol=1e-10)

    # Check intersection matches find_equilibria
    eqs = find_equilibria(params)
    assert len(eqs) >= 1
    for eq in eqs:
        w_v = eq.v_star - (eq.v_star**3) / 3.0 + params.I
        w_w = (eq.v_star + params.a) / params.b
        assert abs(w_v - eq.w_star) < 1e-12
        assert abs(w_w - eq.w_star) < 1e-12
        assert abs(w_v - w_w) < 1e-12


def test_part_b_equilibrium_residuals_strict():
    """Verify Cardano equilibrium solver produces residual errors < 10^-12 across diverse parameter sets."""
    test_cases = [
        FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0),
        FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5),
        FHNParameters(a=0.5, b=0.6, epsilon=0.05, I=1.0),
        FHNParameters(a=-0.2, b=1.5, epsilon=0.1, I=0.0),  # Bistable / multi-root regime (b > 1)
    ]

    for p in test_cases:
        eqs = find_equilibria(p)
        assert len(eqs) >= 1
        for eq in eqs:
            # Check dv/dt residual: v - v^3/3 - w + I
            r_v = eq.v_star - (eq.v_star**3) / 3.0 - eq.w_star + p.I
            # Check dw/dt residual: ε(v + a - bw)
            r_w = p.epsilon * (eq.v_star + p.a - p.b * eq.w_star)

            assert abs(r_v) < 1e-12, f"v-residual {r_v} exceeds 1e-12 for params {p}"
            assert abs(r_w) < 1e-12, f"w-residual {r_w} exceeds 1e-12 for params {p}"


def test_part_c_analytical_jacobian_and_stability():
    """Verify analytical Jacobian J = [[1 - v^2, -1], [ε, -εb]], trace, det, and Poincaré classification."""
    p = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    eqs = find_equilibria(p)
    assert len(eqs) == 1
    eq = eqs[0]

    # Expected Jacobian
    expected_J = np.array([
        [1.0 - eq.v_star**2, -1.0],
        [p.epsilon, -p.epsilon * p.b],
    ])
    np.testing.assert_allclose(eq.jacobian, expected_J, rtol=1e-12)

    # Expected trace and determinant
    expected_trace = (1.0 - eq.v_star**2) - p.epsilon * p.b
    expected_det = (1.0 - eq.v_star**2) * (-p.epsilon * p.b) - (-1.0 * p.epsilon)
    expected_disc = expected_trace**2 - 4.0 * expected_det

    assert abs(eq.trace - expected_trace) < 1e-12
    assert abs(eq.determinant - expected_det) < 1e-12
    assert abs(eq.discriminant - expected_disc) < 1e-12

    # Verify eigenvalues match characteristic polynomial roots
    evals = eq.eigenvalues
    np.testing.assert_allclose(evals[0] + evals[1], eq.trace, rtol=1e-10)
    np.testing.assert_allclose(evals[0] * evals[1], eq.determinant, rtol=1e-10)

    # For I=0.5, trace is positive, det > 0, disc < 0 -> Unstable Focus
    assert eq.stability_class == StabilityClass.UNSTABLE_FOCUS
    assert not eq.is_stable


def test_part_d_phase_plane_trajectories_and_vector_field():
    """Verify vector field computation and that trajectories converge correctly."""
    p = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    eqs = find_equilibria(p)
    eq = eqs[0]

    # Test vector field grid
    V, W, dV, dW, dV_norm, dW_norm = compute_vector_field((-2.0, 2.0), (-1.0, 2.0), 15, p)
    assert V.shape == (15, 15)
    assert W.shape == (15, 15)
    assert dV.shape == (15, 15)
    assert dW.shape == (15, 15)
    assert dV_norm.shape == (15, 15)
    assert dW_norm.shape == (15, 15)

    # Test trajectory from distinct initial conditions converging to stable fixed point
    initial_conditions = [(-1.5, 0.5), (0.0, 1.0), (1.5, 0.0)]
    for init in initial_conditions:
        t, v, w = rk4_solve(init, (0.0, 80.0), 0.02, p)
        assert len(t) == 4001
        # Final point should be close to resting equilibrium (stable focus)
        assert abs(v[-1] - eq.v_star) < 0.05
        assert abs(w[-1] - eq.w_star) < 0.05


def test_part_e_sweeps_and_bifurcation_verification():
    """Verify sweeps over a, b and numerical bifurcation identification."""
    p_base = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)

    # 1. Sweep a
    sweep_a = sweep_parameter_a(a_min=0.0, a_max=1.4, num_points=60, base_params=p_base)
    assert sweep_a["parameter_name"] == "a"
    assert len(sweep_a["records"]) == 60
    assert len(sweep_a["hopf_candidates"]) >= 1

    # 2. Sweep b
    sweep_b = sweep_parameter_b(b_min=0.2, b_max=1.5, num_points=50, base_params=p_base)
    assert sweep_b["parameter_name"] == "b"
    assert sweep_b["fold_regime_possible"] is True

    # 3. Bifurcation analysis on I sweep
    sweep_I = sweep_parameter_I(I_min=0.0, I_max=1.0, num_points=100, base_params=p_base)
    bifs = analyze_bifurcation_conditions(sweep_I)
    assert len(bifs) >= 1
    hopf_bifs = [b for b in bifs if "Hopf" in b["type"]]
    assert len(hopf_bifs) >= 1
    # Check that critical value has near-zero trace
    h_crit = hopf_bifs[0]["critical_value"]
    assert 0.2 < h_crit < 0.4
    assert hopf_bifs[0]["determinant"] > 0
    assert abs(hopf_bifs[0]["transversality_rate"]) > 0


def test_part_f_compare_fhn_states():
    """Verify before-and-after comparison tool between resting and oscillatory states."""
    p_orig = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)  # Resting (stable focus)
    p_new = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)   # Oscillatory (unstable focus / limit cycle)

    res = compare_fhn_states(
        params_orig=p_orig,
        params_new=p_new,
        init_orig=(-1.0, 0.2),
        init_new=(-1.0, 0.2),
        t_end=80.0,
        dt=0.02,
    )

    assert res.param_deltas["ΔI"] == pytest.approx(0.5)
    assert res.param_deltas["Δε"] == pytest.approx(0.0)
    assert "Stable Focus" in res.stability_transition
    assert "Unstable Focus" in res.stability_transition
    assert "Hopf Excitation" in res.dynamical_diagnosis
    assert len(res.v_orig) == len(res.time_orig)
    assert len(res.v_new) == len(res.time_new)

    # v_new should exhibit large oscillations (amplitude > 2.0), whereas v_orig relaxes to constant
    v_orig_osc = np.max(res.v_orig[-500:]) - np.min(res.v_orig[-500:])
    v_new_osc = np.max(res.v_new) - np.min(res.v_new)
    assert v_orig_osc < 0.05
    assert v_new_osc > 2.0
