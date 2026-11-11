"""
Automated parameter sweeps and bifurcation analysis for the FitzHugh-Nagumo model.

Focus parameters:
    1. Applied bias current I (primary bifurcation parameter).
    2. Timescale separation ratio ε.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .model import FHNParameters
from .analysis import find_equilibria, compute_jacobian, compute_stability_metrics, classify_equilibrium


def find_hopf_candidates(params: FHNParameters) -> List[Dict[str, Any]]:
    """
    Computes analytical candidate Hopf bifurcation points by seeking where trace(J) = 0 with det(J) > 0.

    Condition:
        τ = 1 - (v*)² - ε*b = 0  ==>  (v*)² = 1 - ε*b
        I_H = (v*)³/3 - v* + (v* + a)/b

    Returns:
        List of candidate dictionaries containing v_star, w_star, I_H, det, frequency, transversality.
    """
    candidates = []
    val = 1.0 - params.epsilon * params.b

    if val > 0:
        v_h_plus = np.sqrt(val)
        v_h_minus = -np.sqrt(val)

        for v_h in [v_h_minus, v_h_plus]:
            w_h = (v_h + params.a) / params.b
            I_h = (v_h**3) / 3.0 - v_h + (v_h + params.a) / params.b

            # Verify determinant at this candidate
            det_h = params.epsilon * (1.0 - params.b + params.b * (v_h**2))
            freq = np.sqrt(det_h) if det_h > 0 else 0.0

            # Transversality check: dτ/dI = -2*v* * (dv*/dI)
            # From I(v*), dI/dv* = (v*)² - 1 + 1/b
            dI_dv = (v_h**2) - 1.0 + (1.0 / params.b)
            dv_dI = 1.0 / dI_dv if not np.isclose(dI_dv, 0.0) else np.nan
            dtau_dI = -2.0 * v_h * dv_dI if not np.isnan(dv_dI) else np.nan

            candidates.append({
                "v_star": float(v_h),
                "w_star": float(w_h),
                "I_hopf": float(I_h),
                "determinant": float(det_h),
                "frequency": float(freq),
                "transversality_dtau_dI": float(dtau_dI),
                "is_transverse": not np.isclose(dtau_dI, 0.0) if not np.isnan(dtau_dI) else False,
            })

    return candidates


def check_fold_existence(params: FHNParameters) -> Dict[str, Any]:
    """
    Evaluates whether a saddle-node fold in current I can occur for the current parameter b.

    Condition for fold in I:
        dI/dv* = (v*)² - 1 + 1/b = 0  ==>  (v*)² = 1 - 1/b
    """
    rhs = 1.0 - (1.0 / params.b)
    has_real_fold = rhs >= 0.0

    return {
        "rhs_value": float(rhs),
        "has_real_fold": bool(has_real_fold),
        "explanation": (
            f"For b = {params.b:.4f}, 1 - 1/b = {rhs:.4f} < 0. Since (v*)² must be non-negative for real roots, "
            "no real saddle-node fold exists as a function of I in this parameter regime."
            if not has_real_fold else
            f"Fold candidates exist at v* = ±sqrt({rhs:.4f})."
        )
    }


def sweep_parameter_I(
    I_min: float,
    I_max: float,
    num_points: int,
    base_params: FHNParameters,
) -> Dict[str, Any]:
    """
    Conducts a fine parameter sweep across input current I while keeping a, b, ε fixed.

    Returns:
        Dictionary containing arrays of I values, equilibrium branches, trace, determinant,
        eigenvalue real and imaginary parts, and stability labels.
    """
    I_values = np.linspace(I_min, I_max, num_points)

    records = []

    for I_val in I_values:
        p = FHNParameters(a=base_params.a, b=base_params.b, epsilon=base_params.epsilon, I=I_val)
        eqs = find_equilibria(p)

        for eq in eqs:
            records.append({
                "I": float(I_val),
                "v_star": float(eq.v_star),
                "w_star": float(eq.w_star),
                "trace": float(eq.trace),
                "determinant": float(eq.determinant),
                "discriminant": float(eq.discriminant),
                "lambda1_real": float(eq.eigenvalues[0].real),
                "lambda1_imag": float(eq.eigenvalues[0].imag),
                "lambda2_real": float(eq.eigenvalues[1].real),
                "lambda2_imag": float(eq.eigenvalues[1].imag),
                "max_re_lambda": float(max(eq.eigenvalues[0].real, eq.eigenvalues[1].real)),
                "stability_class": eq.stability_class.value,
                "is_stable": eq.is_stable,
            })

    hopf_candidates = find_hopf_candidates(base_params)
    fold_info = check_fold_existence(base_params)

    return {
        "I_values": I_values,
        "records": records,
        "hopf_candidates": hopf_candidates,
        "fold_info": fold_info,
        "base_params": base_params,
    }


def sweep_parameter_epsilon(
    eps_min: float,
    eps_max: float,
    num_points: int,
    base_params: FHNParameters,
) -> Dict[str, Any]:
    """
    Conducts a parameter sweep across timescale separation ratio ε while keeping a, b, I fixed.

    Demonstrates how ε controls damping, trace, determinant, and eigenvalue trajectories
    without shifting the equilibrium point coordinates (v*, w*).
    """
    eps_values = np.linspace(eps_min, eps_max, num_points)

    # Solve equilibrium coordinates once since they are independent of ε (for ε > 0)
    base_eqs = find_equilibria(base_params)

    records = []

    for eps_val in eps_values:
        p = FHNParameters(a=base_params.a, b=base_params.b, epsilon=eps_val, I=base_params.I)

        for base_eq in base_eqs:
            v_star = base_eq.v_star
            w_star = base_eq.w_star

            J = compute_jacobian(v_star, p)
            trace, det, disc, lambdas = compute_stability_metrics(J)
            stab_class, is_stable, desc = classify_equilibrium(trace, det, disc, lambdas)

            records.append({
                "epsilon": float(eps_val),
                "v_star": float(v_star),
                "w_star": float(w_star),
                "trace": float(trace),
                "determinant": float(det),
                "discriminant": float(disc),
                "lambda1_real": float(lambdas[0].real),
                "lambda1_imag": float(lambdas[0].imag),
                "lambda2_real": float(lambdas[1].real),
                "lambda2_imag": float(lambdas[1].imag),
                "max_re_lambda": float(max(lambdas[0].real, lambdas[1].real)),
                "stability_class": stab_class.value,
                "is_stable": is_stable,
            })

    return {
        "epsilon_values": eps_values,
        "records": records,
        "base_params": base_params,
    }

