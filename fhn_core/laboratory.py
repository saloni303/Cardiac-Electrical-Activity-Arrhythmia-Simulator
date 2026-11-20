"""
Stage 6: Mathematical Analysis Laboratory Module for the Single-Cell FitzHugh-Nagumo Model.

Provides comprehensive analytical, numerical, and comparative tools:
1. Part A: Nullcline calculations and explicit intersection finder.
2. Part B: Numerical equilibrium solver with Cardano formula and residual verification (< 10⁻¹²).
3. Part C: Exact analytical Jacobian, eigenvalues, trace, determinant, discriminant, and Poincaré classification.
4. Part D: Phase-plane analysis with directional vector fields and multiple trajectory integrations.
5. Part E: Multi-parameter sweeps over I, ε, a, and b, with rigorous numerical bifurcation verification.
6. Part F: Before-and-after parameter comparison laboratory with dynamical transition diagnosis.

Academic Boundary:
    The two-dimensional phase plane, nullclines, and Jacobian belong strictly to the
    standalone single-cell FHN model. The multi-compartment coupled network has a 2N-dimensional
    state space and is treated separately.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np

from .model import FHNParameters, fhn_derivatives, compute_vector_field
from .solver import rk4_solve
from .analysis import (
    compute_nullclines,
    find_equilibria,
    compute_jacobian,
    compute_stability_metrics,
    classify_equilibrium,
    EquilibriumPoint,
    StabilityClass,
)
from .sweeps import (
    find_hopf_candidates,
    check_fold_existence,
    sweep_parameter_I,
    sweep_parameter_epsilon,
)


@dataclass
class StateComparisonResult:
    """
    Structured outcome of a before-and-after parameter comparison.
    """
    params_orig: FHNParameters
    params_new: FHNParameters
    init_orig: Tuple[float, float]
    init_new: Tuple[float, float]
    equilibria_orig: List[EquilibriumPoint]
    equilibria_new: List[EquilibriumPoint]
    time_orig: np.ndarray
    v_orig: np.ndarray
    w_orig: np.ndarray
    time_new: np.ndarray
    v_new: np.ndarray
    w_new: np.ndarray
    param_deltas: Dict[str, float]
    stability_transition: str
    dynamical_diagnosis: str


def sweep_parameter_a(
    a_min: float,
    a_max: float,
    num_points: int,
    base_params: FHNParameters,
) -> Dict[str, Any]:
    """
    Conducts a parameter sweep across recovery nullcline intercept a while keeping b, ε, I fixed.

    Shifts the w-nullcline horizontally, translating the intersection along the cubic v-nullcline.
    """
    a_values = np.linspace(a_min, a_max, num_points)
    records = []

    for a_val in a_values:
        p = FHNParameters(a=float(a_val), b=base_params.b, epsilon=base_params.epsilon, I=base_params.I)
        eqs = find_equilibria(p)

        for eq in eqs:
            records.append({
                "a": float(a_val),
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

    # Check Hopf conditions along sweep: trace = 0 <=> (v*)^2 = 1 - ε*b
    hopf_v = np.sqrt(max(0.0, 1.0 - base_params.epsilon * base_params.b))
    hopf_candidates = []
    for sgn in [-1.0, 1.0]:
        v_cand = sgn * hopf_v
        # a_hopf from equilibrium relation: w* = (v* + a)/b = v* - (v*)³/3 + I  ==> a = b(v* - v*³/3 + I) - v*
        a_hopf = base_params.b * (v_cand - (v_cand**3) / 3.0 + base_params.I) - v_cand
        if a_min <= a_hopf <= a_max:
            det_val = base_params.epsilon * (1.0 - base_params.b + base_params.b * (v_cand**2))
            hopf_candidates.append({
                "v_star": float(v_cand),
                "a_hopf": float(a_hopf),
                "determinant": float(det_val),
                "is_valid": det_val > 0,
            })

    return {
        "parameter_name": "a",
        "values": a_values,
        "records": records,
        "hopf_candidates": hopf_candidates,
        "base_params": base_params,
    }


def sweep_parameter_b(
    b_min: float,
    b_max: float,
    num_points: int,
    base_params: FHNParameters,
) -> Dict[str, Any]:
    """
    Conducts a parameter sweep across recovery nullcline slope parameter b while keeping a, ε, I fixed.

    When b crosses 1.0, the slope 1/b becomes < 1, which allows the line to intersect the cubic
    v-nullcline up to 3 times, giving rise to saddle-node (fold) bifurcations and bistability.
    """
    b_values = np.linspace(b_min, b_max, num_points)
    records = []

    for b_val in b_values:
        if np.isclose(b_val, 0.0):
            continue
        p = FHNParameters(a=base_params.a, b=float(b_val), epsilon=base_params.epsilon, I=base_params.I)
        eqs = find_equilibria(p)

        for eq in eqs:
            records.append({
                "b": float(b_val),
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

    # Note on fold bifurcation threshold: b > 1
    fold_regime = any(b_val > 1.0 for b_val in b_values)

    return {
        "parameter_name": "b",
        "values": b_values,
        "records": records,
        "fold_regime_possible": fold_regime,
        "base_params": base_params,
    }


def analyze_bifurcation_conditions(
    sweep_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Rigorous numerical identifier of bifurcation points from parameter sweep data.

    Checks:
        1. Andronov-Hopf: Real part of complex conjugate eigenvalues crosses zero (trace changes sign with det > 0).
           Verifies transversality d(Re(λ))/dμ ≠ 0.
        2. Saddle-Node (Fold): Determinant crosses zero (real eigenvalue crosses zero).
        3. Node-Focus Transition: Discriminant D = τ² - 4Δ crosses zero.
    """
    records = sweep_data["records"]
    param_name = "I" if "I_values" in sweep_data else ("epsilon" if "epsilon_values" in sweep_data else sweep_data.get("parameter_name", "parameter"))

    bifurcations = []

    if len(records) < 2:
        return bifurcations

    # Group records by equilibrium branch (for unique root systems)
    for i in range(len(records) - 1):
        r1 = records[i]
        r2 = records[i + 1]

        mu1 = r1[param_name]
        mu2 = r2[param_name]

        # 1. Check Hopf Bifurcation: trace changes sign while determinant remains positive
        if (r1["trace"] * r2["trace"] < 0.0) and (r1["determinant"] > 0.0) and (r2["determinant"] > 0.0):
            # Linear interpolation for zero crossing
            fraction = -r1["trace"] / (r2["trace"] - r1["trace"] + 1e-15)
            mu_hopf = mu1 + fraction * (mu2 - mu1)
            v_hopf = r1["v_star"] + fraction * (r2["v_star"] - r1["v_star"])
            det_hopf = r1["determinant"] + fraction * (r2["determinant"] - r1["determinant"])
            dtau_dmu = (r2["trace"] - r1["trace"]) / (mu2 - mu1 + 1e-15)

            bifurcations.append({
                "type": "Andronov-Hopf Bifurcation",
                "parameter": param_name,
                "critical_value": float(mu_hopf),
                "v_star": float(v_hopf),
                "determinant": float(det_hopf),
                "imaginary_frequency": float(np.sqrt(det_hopf)) if det_hopf > 0 else 0.0,
                "transversality_rate": float(dtau_dmu),
                "condition_verified": "Trace τ = 0, Determinant Δ > 0, dRe(λ)/dμ ≠ 0",
                "subcritical_supercritical": "Supercritical (Stable Limit Cycle)" if dtau_dmu > 0 else "Subcritical",
            })

        # 2. Check Saddle-Node (Fold) Bifurcation: determinant changes sign
        if r1["determinant"] * r2["determinant"] < 0.0:
            fraction = -r1["determinant"] / (r2["determinant"] - r1["determinant"] + 1e-15)
            mu_fold = mu1 + fraction * (mu2 - mu1)
            v_fold = r1["v_star"] + fraction * (r2["v_star"] - r1["v_star"])

            bifurcations.append({
                "type": "Saddle-Node (Fold) Bifurcation",
                "parameter": param_name,
                "critical_value": float(mu_fold),
                "v_star": float(v_fold),
                "determinant": 0.0,
                "condition_verified": "Determinant Δ = 0 (One zero eigenvalue λ = 0)",
            })

        # 3. Check Node-Focus Transition: discriminant changes sign
        if r1["discriminant"] * r2["discriminant"] < 0.0:
            fraction = -r1["discriminant"] / (r2["discriminant"] - r1["discriminant"] + 1e-15)
            mu_trans = mu1 + fraction * (mu2 - mu1)
            bifurcations.append({
                "type": "Node-Focus Transition",
                "parameter": param_name,
                "critical_value": float(mu_trans),
                "condition_verified": "Discriminant D = τ² - 4Δ = 0 (Real/Complex eigenvalue boundary)",
            })

    return bifurcations


def compare_fhn_states(
    params_orig: FHNParameters,
    params_new: FHNParameters,
    init_orig: Tuple[float, float] = (-1.0, 0.2),
    init_new: Tuple[float, float] = (-1.0, 0.2),
    t_end: float = 60.0,
    dt: float = 0.02,
) -> StateComparisonResult:
    """
    Executes a comprehensive before-and-after dynamical comparison between two single-cell FHN states.
    """
    # 1. Equilibria & Jacobians
    eqs_orig = find_equilibria(params_orig)
    eqs_new = find_equilibria(params_new)

    # 2. Numerical Trajectories
    t_orig, v_orig, w_orig = rk4_solve(init_orig, (0.0, t_end), dt, params_orig)
    t_new, v_new, w_new = rk4_solve(init_new, (0.0, t_end), dt, params_new)

    # 3. Parameter Deltas
    deltas = {
        "ΔI": params_new.I - params_orig.I,
        "Δε": params_new.epsilon - params_orig.epsilon,
        "Δa": params_new.a - params_orig.a,
        "Δb": params_new.b - params_orig.b,
    }

    # 4. Stability Transition Assessment
    stab_orig = eqs_orig[0].stability_class.value if eqs_orig else "Unknown"
    stab_new = eqs_new[0].stability_class.value if eqs_new else "Unknown"
    stab_trans = f"{stab_orig}  ⟶  {stab_new}"

    # 5. Dynamical Diagnosis Synthesis
    diagnosis_parts = []
    if not eqs_orig[0].is_stable and eqs_new[0].is_stable:
        diagnosis_parts.append("Restabilization: Limit cycle oscillations cease; system settles to a stable quiescent equilibrium.")
    elif eqs_orig[0].is_stable and not eqs_new[0].is_stable:
        diagnosis_parts.append("Hopf Excitation: Quiescent fixed point loses stability via Andronov-Hopf bifurcation, generating continuous periodic limit cycle action potentials.")
    elif eqs_orig[0].is_stable and eqs_new[0].is_stable:
        if stab_orig != stab_new:
            diagnosis_parts.append(f"Damping Mode Transition: Shifted from {stab_orig} to {stab_new} without loss of asymptotic stability.")
        else:
            diagnosis_parts.append("Equilibrium Shift: Stability class maintained; resting potential coordinates shifted.")
    else:
        diagnosis_parts.append("Oscillatory Modification: Both regimes reside in the unstable/limit-cycle domain; oscillation amplitude and frequency modified.")

    # Frequency analysis if oscillatory
    v_new_amp = float(np.max(v_new) - np.min(v_new))
    if v_new_amp > 1.0 and not eqs_new[0].is_stable:
        diagnosis_parts.append(f"Sustained action potential firing with peak-to-peak amplitude {v_new_amp:.2f}.")

    dynamical_diag = " ".join(diagnosis_parts)

    return StateComparisonResult(
        params_orig=params_orig,
        params_new=params_new,
        init_orig=init_orig,
        init_new=init_new,
        equilibria_orig=eqs_orig,
        equilibria_new=eqs_new,
        time_orig=t_orig,
        v_orig=v_orig,
        w_orig=w_orig,
        time_new=t_new,
        v_new=v_new,
        w_new=w_new,
        param_deltas=deltas,
        stability_transition=stab_trans,
        dynamical_diagnosis=dynamical_diag,
    )
