"""
FitzHugh-Nagumo (FHN) Mathematical Core Package.

Stage 1: Standalone classical two-variable FitzHugh-Nagumo ODE system.
Stage 2: Coupled cardiac-cell network simulator.
Stage 3: Arrhythmia scenario engine.
"""

from .model import FHNParameters, fhn_derivatives, compute_vector_field
from .solver import rk4_step, rk4_solve, compute_time_series
from .analysis import (
    compute_nullclines,
    find_equilibria,
    compute_jacobian,
    compute_stability_metrics,
    classify_equilibrium,
    EquilibriumPoint,
    StabilityClass,
)
from .sweeps import sweep_parameter_I, sweep_parameter_epsilon
from .network import (
    RegionType,
    NetworkNode,
    NetworkEdge,
    CardiacNetwork,
    create_default_cardiac_network,
    solve_network,
    rk4_network_step,
)
from .activation import (
    detect_activations,
    compute_propagation_latencies,
    detect_conduction_failures,
)
from .scenarios import (
    ScenarioType,
    ScenarioConfig,
    ScenarioResult,
    run_scenario,
    compute_rhythm_trace,
    create_reentry_network,
    make_pacing_protocol,
)

__all__ = [
    # Stage 1: Single-cell
    "FHNParameters",
    "fhn_derivatives",
    "compute_vector_field",
    "rk4_step",
    "rk4_solve",
    "compute_time_series",
    "compute_nullclines",
    "find_equilibria",
    "compute_jacobian",
    "compute_stability_metrics",
    "classify_equilibrium",
    "EquilibriumPoint",
    "StabilityClass",
    "sweep_parameter_I",
    "sweep_parameter_epsilon",
    # Stage 2: Coupled Network
    "RegionType",
    "NetworkNode",
    "NetworkEdge",
    "CardiacNetwork",
    "create_default_cardiac_network",
    "solve_network",
    "rk4_network_step",
    "detect_activations",
    "compute_propagation_latencies",
    "detect_conduction_failures",
    # Stage 3: Arrhythmia Scenarios
    "ScenarioType",
    "ScenarioConfig",
    "ScenarioResult",
    "run_scenario",
    "compute_rhythm_trace",
    "create_reentry_network",
    "make_pacing_protocol",
]
