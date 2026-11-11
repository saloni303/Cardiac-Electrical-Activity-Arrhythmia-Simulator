"""
Unit tests for automated parameter sweeps, Hopf candidate detection, and bifurcation checks.
"""

import numpy as np
import pytest
from fhn_core.model import FHNParameters
from fhn_core.sweeps import (
    sweep_parameter_I,
    sweep_parameter_epsilon,
    find_hopf_candidates,
    check_fold_existence,
)


def test_sweep_parameter_I_execution():
    """
    Verifies that sweep over input current I produces monotonic grids and valid records.
    """
    base_params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    result = sweep_parameter_I(I_min=-0.2, I_max=1.8, num_points=25, base_params=base_params)

    assert len(result["I_values"]) == 25
    assert len(result["records"]) >= 25  # At least one equilibrium per I point

    # Check that record fields are fully populated
    for rec in result["records"]:
        assert "I" in rec
        assert "v_star" in rec
        assert "w_star" in rec
        assert "trace" in rec
        assert "determinant" in rec
        assert "max_re_lambda" in rec
        assert "stability_class" in rec
        assert np.isfinite(rec["v_star"])
        assert np.isfinite(rec["trace"])


def test_hopf_candidates_accuracy():
    """
    Verifies analytical Hopf candidate formula:
        (v*)² = 1 - ε*b
        I_H = (v*)³/3 - v* + (v* + a)/b
    """
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    candidates = find_hopf_candidates(params)

    assert len(candidates) == 2, f"Expected 2 Hopf candidates, found {len(candidates)}"

    expected_v_sq = 1.0 - 0.08 * 0.8  # 1 - 0.064 = 0.936
    expected_v_abs = np.sqrt(expected_v_sq)

    for cand in candidates:
        assert np.isclose(abs(cand["v_star"]), expected_v_abs, atol=1e-5)
        assert cand["determinant"] > 0  # Determinant must be positive at Hopf
        assert cand["is_transverse"] is True  # Crossing speed must be non-zero


def test_fold_nonexistence_for_cardiac_b():
    """
    Verifies that for b = 0.8, check_fold_existence confirms no saddle-node fold exists in I.
    """
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    fold_info = check_fold_existence(params)

    assert fold_info["has_real_fold"] is False
    assert fold_info["rhs_value"] < 0.0


def test_sweep_parameter_epsilon_execution():
    """
    Verifies that sweeping epsilon maintains invariant equilibrium coordinates (v*, w*)
    while varying trace and eigenvalues.
    """
    base_params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    result = sweep_parameter_epsilon(eps_min=0.01, eps_max=0.25, num_points=20, base_params=base_params)

    assert len(result["epsilon_values"]) == 20
    records = result["records"]
    assert len(records) == 20

    # Equilibrium v* and w* must be identical across all epsilon
    v_stars = [r["v_star"] for r in records]
    assert np.allclose(v_stars, v_stars[0])

    # Trace must strictly change with epsilon: dτ/dε = -b
    traces = [r["trace"] for r in records]
    assert not np.allclose(traces, traces[0])

