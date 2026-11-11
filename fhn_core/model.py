"""
Model definitions and vector field evaluations for the FitzHugh-Nagumo ODE system.

Mathematical formulation:
    dv/dt = f(v, w) = v - v³/3 - w + I
    dw/dt = g(v, w) = ε(v + a - bw)
"""

from dataclasses import dataclass
from typing import Tuple, Union, Callable, Optional
import numpy as np


@dataclass(frozen=True)
class FHNParameters:
    """
    Parameter container for the classical FitzHugh-Nagumo model.

    Attributes:
        a (float): Recovery nullcline intercept shaping parameter. Standard cardiac value: 0.7.
        b (float): Recovery nullcline slope shaping parameter (b != 0). Standard cardiac value: 0.8.
        epsilon (float): Timescale separation ratio (0 < ε << 1). Standard value: 0.08 or 0.01.
        I (float): Applied bias / stimulus current.
    """
    a: float = 0.7
    b: float = 0.8
    epsilon: float = 0.08
    I: float = 0.5

    def __post_init__(self):
        if self.epsilon <= 0.0:
            raise ValueError(f"Timescale parameter epsilon must be positive, got {self.epsilon}")
        if np.isclose(self.b, 0.0):
            raise ValueError("Parameter b cannot be zero as it appears in recovery nullcline slope.")


def fhn_derivatives(
    t: float,
    state: Union[np.ndarray, Tuple[float, float]],
    params: FHNParameters,
    I_external: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    """
    Computes instantaneous derivatives [dv/dt, dw/dt] for the FHN system.

    dv/dt = v - v³/3 - w + I
    dw/dt = ε(v + a - bw)

    Parameters:
        t (float): Current time.
        state (array-like): Current state vector [v, w].
        params (FHNParameters): Model parameter set.
        I_external (callable, optional): Time-varying current function I(t). If None, uses constant params.I.

    Returns:
        np.ndarray: Derivatives [dv/dt, dw/dt].
    """
    v = state[0]
    w = state[1]

    current_I = I_external(t) if I_external is not None else params.I

    dv_dt = v - (v**3) / 3.0 - w + current_I
    dw_dt = params.epsilon * (v + params.a - params.b * w)

    return np.array([dv_dt, dw_dt], dtype=float)


def compute_vector_field(
    v_bounds: Tuple[float, float],
    w_bounds: Tuple[float, float],
    grid_size: int,
    params: FHNParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a 2D grid in phase space and computes the local vector field.

    Parameters:
        v_bounds (tuple): (v_min, v_max).
        w_bounds (tuple): (w_min, w_max).
        grid_size (int): Number of grid intervals along each axis.
        params (FHNParameters): Model parameters.

    Returns:
        V (np.ndarray): 2D meshgrid of v coordinates.
        W (np.ndarray): 2D meshgrid of w coordinates.
        dV (np.ndarray): dv/dt at each grid point.
        dW (np.ndarray): dw/dt at each grid point.
        dV_norm (np.ndarray): Direction unit vector component along v.
        dW_norm (np.ndarray): Direction unit vector component along w.
    """
    v_vals = np.linspace(v_bounds[0], v_bounds[1], grid_size)
    w_vals = np.linspace(w_bounds[0], w_bounds[1], grid_size)
    V, W = np.meshgrid(v_vals, w_vals)

    dV = V - (V**3) / 3.0 - W + params.I
    dW = params.epsilon * (V + params.a - params.b * W)

    speed = np.sqrt(dV**2 + dW**2)
    # Avoid division by zero at stagnation / equilibrium points
    safe_speed = np.where(speed < 1e-12, 1.0, speed)
    dV_norm = np.where(speed < 1e-12, 0.0, dV / safe_speed)
    dW_norm = np.where(speed < 1e-12, 0.0, dW / safe_speed)

    return V, W, dV, dW, dV_norm, dW_norm

