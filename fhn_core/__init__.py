"""
FitzHugh-Nagumo (FHN) Mathematical Core Package.

Stage 1: Standalone classical two-variable FitzHugh-Nagumo ODE system.
Stage 2: Coupled cardiac-cell network simulator.
Stage 3: Arrhythmia scenario engine.
Stage 4: Interactive anatomical cardiac visualization.
Stage 5: Rhythm visualization and activation analysis.
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
from .visualization import (
    map_state_to_color,
    extract_animation_frames,
    build_heart_visualization_html,
)
from .rhythm_analysis import (
    EventLogEntry,
    generate_numerical_event_log,
    compute_beat_to_beat_intervals,
    estimate_heart_rate_bpm,
    detect_missing_events,
    compute_true_ecg_lead_approximation,
    compare_with_normal_rhythm,
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
    # Stage 4: Interactive Anatomical Visualization
    "map_state_to_color",
    "extract_animation_frames",
    "build_heart_visualization_html",
    # Stage 5: Rhythm Visualization & Activation Analysis
    "EventLogEntry",
    "generate_numerical_event_log",
    "compute_beat_to_beat_intervals",
    "estimate_heart_rate_bpm",
    "detect_missing_events",
    "compute_true_ecg_lead_approximation",
    "compare_with_normal_rhythm",
]
