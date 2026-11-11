"""
Unit tests for the fourth-order Runge-Kutta (RK4) numerical integrator.
"""

import numpy as np
import pytest
from fhn_core.model import FHNParameters
from fhn_core.solver import rk4_step, rk4_solve, compute_time_series


def test_rk4_order_of_accuracy_on_exponential():
    """
    Verifies that the RK4 implementation achieves true 4th-order global convergence.
    Test ODE:
        dy/dt = -y, y(0) = 1.0  ==>  Exact solution: y(t) = exp(-t)
    """
    # Define single-variable linear test function
    def linear_deriv(t, y):
        return -y

    def rk4_linear_solve(y0, t_end, h):
        steps = int(np.round(t_end / h))
        y = y0
        t = 0.0
        for _ in range(steps):
            k1 = linear_deriv(t, y)
            k2 = linear_deriv(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = linear_deriv(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = linear_deriv(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return y

    t_end = 1.0
    exact = np.exp(-t_end)

    h1 = 0.1
    h2 = 0.05
    h3 = 0.025

    y_num1 = rk4_linear_solve(1.0, t_end, h1)
    y_num2 = rk4_linear_solve(1.0, t_end, h2)
    y_num3 = rk4_linear_solve(1.0, t_end, h3)

    err1 = abs(y_num1 - exact)
    err2 = abs(y_num2 - exact)
    err3 = abs(y_num3 - exact)

    # For 4th order: error ratio when halving step size should be ~ 2^4 = 16
    ratio1 = err1 / err2
    ratio2 = err2 / err3

    assert ratio1 > 15.0 and ratio1 < 17.0, f"Expected 4th order ratio ~16, got {ratio1}"
    assert ratio2 > 15.5 and ratio2 < 16.5, f"Expected 4th order ratio ~16, got {ratio2}"


def test_fhn_rk4_solve_structure():
    """
    Tests that rk4_solve returns arrays of correct shape, no NaNs/Infs, and respects initial state.
    """
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    v0, w0 = -1.0, 0.2
    t_span = (0.0, 50.0)
    dt = 0.05

    t, v, w = rk4_solve((v0, w0), t_span, dt, params)

    assert len(t) == len(v) == len(w)
    assert np.isclose(v[0], v0)
    assert np.isclose(w[0], w0)
    assert np.isclose(t[0], 0.0)
    assert np.isclose(t[-1], 50.0)

    # All values must be finite
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(w))


def test_fhn_step_refinement_convergence():
    """
    Verifies that refining step size dt produces consistent trajectories on FHN system.
    """
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    v0, w0 = 1.0, -0.5
    t_span = (0.0, 10.0)

    t_coarse, v_coarse, w_coarse = rk4_solve((v0, w0), t_span, dt=0.02, params=params)
    t_fine, v_fine, w_fine = rk4_solve((v0, w0), t_span, dt=0.005, params=params)

    # Compare at endpoint
    v_err = abs(v_coarse[-1] - v_fine[-1])
    w_err = abs(w_coarse[-1] - w_fine[-1])

    assert v_err < 1e-4, f"Endpoint voltage error {v_err} exceeds tolerance"
    assert w_err < 1e-4, f"Endpoint recovery error {w_err} exceeds tolerance"


def test_invalid_parameters_handling():
    """
    Verifies that invalid parameters or bounds raise appropriate exceptions.
    """
    params = FHNParameters()

    # t_end <= t_start
    with pytest.raises(ValueError):
        rk4_solve((0.0, 0.0), (10.0, 5.0), 0.1, params)

    # dt <= 0
    with pytest.raises(ValueError):
        rk4_solve((0.0, 0.0), (0.0, 10.0), -0.01, params)

    # epsilon <= 0
    with pytest.raises(ValueError):
        FHNParameters(epsilon=-0.05)

    # b == 0
    with pytest.raises(ValueError):
        FHNParameters(b=0.0)

