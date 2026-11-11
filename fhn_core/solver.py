"""
Fourth-order Runge-Kutta (RK4) ODE solver for the FitzHugh-Nagumo system.

Mathematical formulation:
    Given dx/dt = F(t, x) with x = [v, w]^T:
        k1 = F(t_n, x_n)
        k2 = F(t_n + h/2, x_n + (h/2)*k1)
        k3 = F(t_n + h/2, x_n + (h/2)*k2)
        k4 = F(t_n + h, x_n + h*k3)
        x_{n+1} = x_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
"""

from typing import Tuple, Callable, Optional, Dict, Any
import numpy as np
from .model import FHNParameters, fhn_derivatives


def rk4_step(
    t: float,
    state: np.ndarray,
    dt: float,
    params: FHNParameters,
    I_external: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    """
    Performs a single integration step using the classical 4th-order Runge-Kutta method.

    Parameters:
        t (float): Current time t_n.
        state (np.ndarray): State vector [v_n, w_n].
        dt (float): Time step size h.
        params (FHNParameters): Model parameters.
        I_external (callable, optional): Time-dependent stimulus function I(t).

    Returns:
        np.ndarray: Updated state vector x_{n+1} = [v_{n+1}, w_{n+1}].
    """
    k1 = fhn_derivatives(t, state, params, I_external)
    k2 = fhn_derivatives(t + 0.5 * dt, state + 0.5 * dt * k1, params, I_external)
    k3 = fhn_derivatives(t + 0.5 * dt, state + 0.5 * dt * k2, params, I_external)
    k4 = fhn_derivatives(t + dt, state + dt * k3, params, I_external)

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state


def rk4_solve(
    initial_state: Tuple[float, float],
    t_span: Tuple[float, float],
    dt: float,
    params: FHNParameters,
    I_external: Optional[Callable[[float], float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrates the FitzHugh-Nagumo ODE system over a time interval using the RK4 method.

    Parameters:
        initial_state (tuple): Initial values (v(0), w(0)).
        t_span (tuple): (t_start, t_end).
        dt (float): Integration step size h (must be positive).
        params (FHNParameters): Model parameters.
        I_external (callable, optional): Time-dependent input current function.

    Returns:
        t (np.ndarray): 1D array of discrete time points.
        v (np.ndarray): 1D array of membrane voltage values v(t).
        w (np.ndarray): 1D array of recovery variable values w(t).
    """
    t_start, t_end = t_span
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be strictly greater than t_start ({t_start}).")
    if dt <= 0:
        raise ValueError(f"Time step dt ({dt}) must be strictly positive.")

    # Calculate number of steps
    num_steps = int(np.ceil((t_end - t_start) / dt))
    t = np.linspace(t_start, t_end, num_steps + 1)
    actual_dt = t[1] - t[0] if num_steps > 0 else dt

    v = np.zeros(num_steps + 1, dtype=float)
    w = np.zeros(num_steps + 1, dtype=float)

    current_state = np.array(initial_state, dtype=float)
    v[0] = current_state[0]
    w[0] = current_state[1]

    for n in range(num_steps):
        current_state = rk4_step(t[n], current_state, actual_dt, params, I_external)
        v[n + 1] = current_state[0]
        w[n + 1] = current_state[1]

    return t, v, w


def compute_time_series(
    initial_state: Tuple[float, float],
    t_span: Tuple[float, float],
    dt: float,
    params: FHNParameters,
    I_external: Optional[Callable[[float], float]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper returning structured time series results along with diagnostics.
    """
    t, v, w = rk4_solve(initial_state, t_span, dt, params, I_external)

    return {
        "time": t,
        "v": v,
        "w": w,
        "v_max": float(np.max(v)),
        "v_min": float(np.min(v)),
        "w_max": float(np.max(w)),
        "w_min": float(np.min(w)),
        "dt": float(t[1] - t[0]) if len(t) > 1 else dt,
        "num_steps": len(t),
    }

