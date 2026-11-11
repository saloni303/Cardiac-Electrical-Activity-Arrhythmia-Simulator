"""
Analytical and stability analysis module for the FitzHugh-Nagumo ODE system.

Mathematical formulation:
    dv/dt = f(v, w) = v - v³/3 - w + I = 0
    dw/dt = g(v, w) = ε(v + a - bw) = 0

Nullclines:
    v-nullcline: w = v - v³/3 + I
    w-nullcline: w = (v + a) / b  (b != 0)

Equilibrium cubic:
    b*(v*)³ + 3*(1 - b)*v* + 3*a - 3*b*I = 0
    w* = (v* + a) / b

Jacobian:
    J(v*, w*) = [[1 - (v*)², -1],
                 [ε,         -ε*b]]

Trace, Determinant, Discriminant, and Eigenvalues:
    τ = tr(J) = 1 - (v*)² - ε*b
    Δ = det(J) = ε*[1 - b + b*(v*)²]
    D = τ² - 4*Δ
    λ_{1,2} = (τ ± sqrt(D)) / 2
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .model import FHNParameters


class StabilityClass(str, Enum):
    """Classification of local linear stability for planar ODE equilibria."""
    STABLE_NODE = "Locally Asymptotically Stable Node"
    STABLE_FOCUS = "Locally Asymptotically Stable Focus (Spiral Sink)"
    UNSTABLE_NODE = "Unstable Node"
    UNSTABLE_FOCUS = "Unstable Focus (Spiral Source / Periodic Orbit Onset)"
    SADDLE = "Saddle Point (Unstable)"
    CENTER_OR_HOPF = "Non-Hyperbolic Center / Candidate Hopf Boundary"
    NON_HYPERBOLIC_FOLD = "Non-Hyperbolic Boundary (Zero Eigenvalue / Fold Candidate)"
    DEGENERATE_NODE = "Degenerate / Star Node"


@dataclass
class EquilibriumPoint:
    """
    Detailed mathematical profile of an equilibrium point (v*, w*).
    """
    v_star: float
    w_star: float
    residual_v: float
    residual_w: float
    jacobian: np.ndarray
    trace: float
    determinant: float
    discriminant: float
    eigenvalues: Tuple[complex, complex]
    stability_class: StabilityClass
    is_stable: bool
    summary_text: str


def compute_nullclines(
    v_bounds: Tuple[float, float],
    params: FHNParameters,
    num_points: int = 400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes coordinates for both the v-nullcline and w-nullcline.

    v-nullcline: w = v - v³/3 + I
    w-nullcline: w = (v + a) / b

    Parameters:
        v_bounds (tuple): Range (v_min, v_max).
        params (FHNParameters): Model parameters.
        num_points (int): Sampling density.

    Returns:
        v_grid (np.ndarray): 1D array of v values.
        w_v_nullcline (np.ndarray): w values corresponding to dv/dt = 0.
        w_w_nullcline (np.ndarray): w values corresponding to dw/dt = 0.
    """
    v_grid = np.linspace(v_bounds[0], v_bounds[1], num_points)
    w_v_nullcline = v_grid - (v_grid**3) / 3.0 + params.I
    w_w_nullcline = (v_grid + params.a) / params.b

    return v_grid, w_v_nullcline, w_w_nullcline


def compute_jacobian(v_star: float, params: FHNParameters) -> np.ndarray:
    """
    Evaluates the 2x2 Jacobian matrix at voltage state v*.

    J = [[1 - (v*)², -1],
         [ε,         -ε*b]]
    """
    return np.array([
        [1.0 - (v_star**2), -1.0],
        [params.epsilon, -params.epsilon * params.b]
    ], dtype=float)


def compute_stability_metrics(
    jacobian: np.ndarray,
) -> Tuple[float, float, float, Tuple[complex, complex]]:
    """
    Computes trace, determinant, discriminant, and eigenvalues for a 2x2 matrix.

    Returns:
        trace (float): tr(J)
        determinant (float): det(J)
        discriminant (float): tr(J)² - 4*det(J)
        eigenvalues (tuple of complex): (λ1, λ2)
    """
    trace = float(np.trace(jacobian))
    determinant = float(np.linalg.det(jacobian))
    discriminant = float(trace**2 - 4.0 * determinant)

    # Use characteristic equation roots for maximum numerical precision
    sqrt_disc = np.lib.scimath.sqrt(discriminant)
    lambda1 = complex((trace + sqrt_disc) / 2.0)
    lambda2 = complex((trace - sqrt_disc) / 2.0)

    return trace, determinant, discriminant, (lambda1, lambda2)


def classify_equilibrium(
    trace: float,
    determinant: float,
    discriminant: float,
    eigenvalues: Tuple[complex, complex],
    tolerance: float = 1e-9,
) -> Tuple[StabilityClass, bool, str]:
    """
    Classifies planar linear stability based on Poincaré trace-determinant criteria.

    Returns:
        (StabilityClass, is_stable, description_string)
    """
    lambda1, lambda2 = eigenvalues

    # Check for zero determinant (non-hyperbolic, at least one eigenvalue zero)
    if abs(determinant) < tolerance:
        return (
            StabilityClass.NON_HYPERBOLIC_FOLD,
            False,
            "Determinant vanishes (Δ ≈ 0). System has a zero eigenvalue; linearization is inconclusive without nonlinear terms."
        )

    # Saddle point: det < 0
    if determinant < -tolerance:
        return (
            StabilityClass.SADDLE,
            False,
            f"Saddle point (Δ = {determinant:.5f} < 0). One eigenvalue is positive real ({lambda1.real:.4f}) and one is negative real ({lambda2.real:.4f}). Trajectories approach along stable manifold and diverge along unstable manifold."
        )

    # Determinant is positive (Δ > 0)
    # Check trace near zero (Hopf boundary)
    if abs(trace) < tolerance:
        return (
            StabilityClass.CENTER_OR_HOPF,
            False,
            f"Non-hyperbolic center / candidate Hopf bifurcation (τ ≈ 0, Δ = {determinant:.5f} > 0). Eigenvalues are purely imaginary (±{abs(lambda1.imag):.4f}i). Linearization indicates neutral rotation."
        )

    if trace < -tolerance:
        # Asymptotically stable
        if discriminant > tolerance:
            return (
                StabilityClass.STABLE_NODE,
                True,
                f"Locally asymptotically stable node (τ < 0, Δ > 0, D > 0). Both eigenvalues are negative real ({lambda1.real:.4f}, {lambda2.real:.4f}). Perturbations decay non-oscillatory."
            )
        elif discriminant < -tolerance:
            return (
                StabilityClass.STABLE_FOCUS,
                True,
                f"Locally asymptotically stable focus / spiral sink (τ < 0, Δ > 0, D < 0). Complex conjugate eigenvalues with negative real part ({lambda1.real:.4f} ± {abs(lambda1.imag):.4f}i). Trajectories spiral inward toward resting potential."
            )
        else:
            return (
                StabilityClass.DEGENERATE_NODE,
                True,
                f"Stable degenerate node (repeated eigenvalue λ = {lambda1.real:.4f})."
            )
    else:
        # Unstable (trace > 0)
        if discriminant > tolerance:
            return (
                StabilityClass.UNSTABLE_NODE,
                False,
                f"Unstable node (τ > 0, Δ > 0, D > 0). Both eigenvalues are positive real ({lambda1.real:.4f}, {lambda2.real:.4f}). Perturbations grow monotonically."
            )
        elif discriminant < -tolerance:
            return (
                StabilityClass.UNSTABLE_FOCUS,
                False,
                f"Unstable focus / spiral source (τ > 0, Δ > 0, D < 0). Complex conjugate eigenvalues with positive real part ({lambda1.real:.4f} ± {abs(lambda1.imag):.4f}i). Trajectories spiral outward, typically organizing into an autorhythmic limit cycle."
            )
        else:
            return (
                StabilityClass.DEGENERATE_NODE,
                False,
                f"Unstable degenerate node (repeated eigenvalue λ = {lambda1.real:.4f})."
            )


def find_equilibria(
    params: FHNParameters,
    imag_tolerance: float = 1e-11,
    residual_tolerance: float = 1e-9,
) -> List[EquilibriumPoint]:
    """
    Finds all real equilibrium points of the FitzHugh-Nagumo system by solving
    the cubic polynomial in v*:

        b*(v*)³ + 3*(1 - b)*v* + (3*a - 3*b*I) = 0
        w* = (v* + a) / b

    Residual verification is performed for every root:
        r_v = v* - (v*)³/3 - w* + I
        r_w = ε*(v* + a - b*w*)
        Accept root iff max(|r_v|, |r_w|) < residual_tolerance.

    Returns:
        List[EquilibriumPoint]: Sorted list of real equilibria by v*.
    """
    # Polynomial coefficients for: b*v³ + 0*v² + 3*(1-b)*v + (3*a - 3*b*I) = 0
    coeffs = [
        params.b,
        0.0,
        3.0 * (1.0 - params.b),
        3.0 * params.a - 3.0 * params.b * params.I
    ]

    all_roots = np.roots(coeffs)

    equilibria: List[EquilibriumPoint] = []

    for root in all_roots:
        if abs(root.imag) > imag_tolerance:
            continue  # Discard non-real roots

        v_star = float(root.real)
        w_star = float((v_star + params.a) / params.b)

        # Residual verification
        res_v = float(v_star - (v_star**3) / 3.0 - w_star + params.I)
        res_w = float(params.epsilon * (v_star + params.a - params.b * w_star))

        if max(abs(res_v), abs(res_w)) > residual_tolerance:
            continue  # Fails residual verification gate

        J = compute_jacobian(v_star, params)
        trace, det, disc, lambdas = compute_stability_metrics(J)
        stab_class, is_stable, desc = classify_equilibrium(trace, det, disc, lambdas)

        eq = EquilibriumPoint(
            v_star=v_star,
            w_star=w_star,
            residual_v=res_v,
            residual_w=res_w,
            jacobian=J,
            trace=trace,
            determinant=det,
            discriminant=disc,
            eigenvalues=lambdas,
            stability_class=stab_class,
            is_stable=is_stable,
            summary_text=desc,
        )
        equilibria.append(eq)

    # Sort equilibria in ascending order of v*
    equilibria.sort(key=lambda ep: ep.v_star)
    return equilibria

