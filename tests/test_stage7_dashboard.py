"""
Unit tests for Stage 7: Integrated Cardiac Simulator Dashboard.
Verifies that all components required by the integrated dashboard operate correctly:
1. Unified multi-scale data consistency (network solver, heart visualizer, rhythm trace, and event timeline).
2. Dynamic parameter adjustments across global (I, ε, a, b, D) and regional compartments.
3. Session-state baseline experiment comparison logic (parameter deltas, latency shifts, rhythm differences).
4. Mathematical analysis consistency between single-cell Jacobian and network node parameters.
"""

import pytest
import numpy as np
from fhn_core import (
    FHNParameters,
    RegionType,
    NetworkNode,
    CardiacNetwork,
    create_default_cardiac_network,
    solve_network,
    detect_activations,
    compute_propagation_latencies,
    detect_conduction_failures,
    ScenarioType,
    ScenarioConfig,
    ScenarioResult,
    run_scenario,
    compute_rhythm_trace,
    extract_animation_frames,
    build_heart_visualization_html,
    generate_numerical_event_log,
    compute_beat_to_beat_intervals,
    estimate_heart_rate_bpm,
    find_equilibria,
    compute_jacobian,
    compute_stability_metrics,
    classify_equilibrium,
    compare_fhn_states,
)


def test_unified_simulation_engine_consistency():
    """Verify that the same numerical network run supplies visualizer, rhythm trace, and event timeline."""
    cfg = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=120.0)
    res = run_scenario(cfg)

    # 1. State arrays match time dimensions
    assert res.V.shape[0] == len(res.time)
    assert res.W.shape[0] == len(res.time)
    assert len(res.rhythm_trace) == len(res.time)
    assert res.V.shape[1] == res.network.num_nodes

    # 2. Visualizer consumes the same data without error
    html = build_heart_visualization_html(res)
    assert res.scenario_type.value in html
    assert "rhythmCanvas" in html

    # 3. Rhythm trace and event timeline refer to same simulation
    event_log = generate_numerical_event_log(res.activations, res.network)
    assert len(event_log) > 0
    # Earliest activation time should be SAN node
    assert "SAN" in event_log[0].region_name or "SA" in event_log[0].region_name
    assert event_log[0].simulation_time >= 0.0


def test_region_specific_parameter_customization():
    """Verify that custom regional excitability, recovery, and coupling parameters solve stably."""
    net = create_default_cardiac_network(av_conductance=0.25, global_coupling=0.65)

    # Modify SA node excitability
    net.nodes[0].params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.3)
    # Modify AV node recovery timescale (slower recovery)
    net.nodes[2].params = FHNParameters(a=0.7, b=0.8, epsilon=0.04, I=0.0)
    # Modify Ventricle coupling
    net.set_edge_conductance(4, 5, 0.45)  # Purkinje to RV

    sim = solve_network(net, t_span=(0.0, 50.0), dt=0.02)
    time = sim["time"]
    V = sim["V"]
    W = sim["W"]

    assert not np.isnan(V).any()
    assert not np.isnan(W).any()
    assert np.all(V > -3.0) and np.all(V < 3.0)

    activations = detect_activations(time, V, threshold=0.0)
    assert len(activations[0]) >= 1  # SA node fired


def test_experiment_comparison_baseline_delta_logic():
    """Verify that comparing a saved baseline with a modified parameter trial correctly extracts deltas."""
    # Baseline: Normal pacing (T_pace = 45.0, g_AV = 0.35)
    cfg_base = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, av_conductance=0.35, t_end=150.0)
    res_base = run_scenario(cfg_base)
    hr_base = estimate_heart_rate_bpm(res_base.mean_cycle_length)

    # Trial: Tachycardia pacing (T_pace = 35.0, g_AV = 0.35)
    cfg_trial = ScenarioConfig(scenario_type=ScenarioType.TACHYCARDIA, pacing_interval=35.0, av_conductance=0.35, t_end=150.0)
    res_trial = run_scenario(cfg_trial)
    hr_trial = estimate_heart_rate_bpm(res_trial.mean_cycle_length)

    # Comparison metrics
    delta_pacing = cfg_trial.pacing_interval - cfg_base.pacing_interval
    assert delta_pacing == pytest.approx(-10.0)

    # Heart rate should accelerate in trial
    assert hr_trial["bpm"] > hr_base["bpm"]
    rate_ratio = hr_trial["bpm"] / hr_base["bpm"]
    assert rate_ratio > 1.2


def test_mathematical_analysis_consistency_with_network_node():
    """Verify that single-cell Jacobian and stability metrics match the resting parameters of network nodes."""
    node_params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    eqs = find_equilibria(node_params)
    assert len(eqs) == 1
    eq = eqs[0]

    # Resting cardiac cells are asymptotically stable foci
    assert eq.is_stable
    assert eq.stability_class.name == "STABLE_FOCUS"
    assert eq.trace < 0
    assert eq.determinant > 0
    assert eq.discriminant < 0
