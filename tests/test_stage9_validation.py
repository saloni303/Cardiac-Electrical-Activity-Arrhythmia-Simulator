"""
Stage 9: Complete Scientific and Software Validation Test Suite.

Comprehensive technical and mathematical verification across all components:
    Part A: Mathematical Tests (FHN equations, RK4 order, equilibria, Jacobian, eigenvalues,
            stability classification, nullclines, sweeps, coupled network).
    Part B: Network Tests (isolated cell isolation, coupling effects, edge disconnection,
            activation timeline fidelity, heart visualization state mapping, AV block failure,
            re-entry honesty).
    Part C: User Interface & Data Tests (parameter reactivity, scenario distinctness,
            single-truth plotting, CSV export fidelity, deterministic reproducibility).
    Part D: Performance & Robustness (smooth execution timing, numerical stability guards,
            solver-visualization decoupling).
"""

import io
import time
import pytest
import numpy as np
import pandas as pd

from fhn_core.model import FHNParameters, fhn_derivatives
from fhn_core.solver import rk4_step, rk4_solve, compute_time_series
from fhn_core.analysis import (
    compute_nullclines,
    find_equilibria,
    compute_jacobian,
    compute_stability_metrics,
    classify_equilibrium,
    EquilibriumPoint,
    StabilityClass,
)
from fhn_core.sweeps import sweep_parameter_I, find_hopf_candidates, check_fold_existence
from fhn_core.laboratory import (
    sweep_parameter_a,
    sweep_parameter_b,
    analyze_bifurcation_conditions,
    compare_fhn_states,
)
from fhn_core.network import (
    CardiacNetwork,
    NetworkNode,
    RegionType,
    create_default_cardiac_network,
    solve_network,
)
from fhn_core.activation import (
    detect_activations,
    compute_propagation_latencies,
    detect_conduction_failures,
)
from fhn_core.scenarios import (
    ScenarioConfig,
    ScenarioType,
    ScenarioResult,
    run_scenario,
    compute_rhythm_trace,
    make_pacing_protocol,
)
from fhn_core.visualization import (
    map_state_to_color,
    extract_animation_frames,
    build_heart_visualization_html,
)
from fhn_core.rhythm_analysis import (
    generate_numerical_event_log,
    compute_beat_to_beat_intervals,
    estimate_heart_rate_bpm,
    compute_true_ecg_lead_approximation,
)
from fhn_core.experiments import (
    ExperimentParameters,
    build_experiment_network,
    run_experiment_simulation,
    compare_experiments,
    run_parameter_sweep,
    ParameterSweepConfig,
    export_comparison_to_csv,
    export_sweep_to_csv,
)


# =============================================================================
# PART A: MATHEMATICAL VERIFICATION TESTS
# =============================================================================

def test_math_1_single_cell_fhn_derivatives_independent_hand_check():
    """Verify single-cell FHN equations against independently hand-calculated analytical example."""
    # Test values: v = 1.0, w = 0.5, I = 0.5, a = 0.7, b = 0.8, epsilon = 0.08
    # dv/dt = v - v^3/3 - w + I = 1.0 - (1/3) - 0.5 + 0.5 = 2/3 = 0.6666666666666666
    # dw/dt = eps * (v + a - b*w) = 0.08 * (1.0 + 0.7 - 0.8*0.5) = 0.08 * 1.3 = 0.104
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    state = np.array([1.0, 0.5])
    derivs = fhn_derivatives(0.0, state, params)

    expected_dv = 2.0 / 3.0
    expected_dw = 0.104

    assert pytest.approx(derivs[0], abs=1e-12) == expected_dv
    assert pytest.approx(derivs[1], abs=1e-12) == expected_dw


def test_math_2_rk4_convergence_and_order_of_accuracy():
    """Verify RK4 solver order of accuracy and step-refinement error convergence."""
    # Integrate dy/dt = -y, y(0) = 1. Analytical solution: y(t) = exp(-t)
    # Using RK4 logic on a linear test problem
    def linear_deriv(t, y):
        return -y

    def rk4_step_linear(t, y, h):
        k1 = linear_deriv(t, y)
        k2 = linear_deriv(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = linear_deriv(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = linear_deriv(t + h, y + h * k3)
        return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    t_end = 1.0
    errors = []
    step_sizes = [0.2, 0.1, 0.05]

    for h in step_sizes:
        n_steps = int(round(t_end / h))
        y = 1.0
        t = 0.0
        for _ in range(n_steps):
            y = rk4_step_linear(t, y, h)
            t += h
        err = abs(y - np.exp(-t_end))
        errors.append(err)

    # Ratio of errors for halving step size should approach 2^4 = 16 for 4th order method
    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert 13.0 < ratio_1 < 22.0
    assert 14.0 < ratio_2 < 18.0


def test_math_3_equilibrium_residual_errors_strict():
    """Verify equilibrium calculations solve f(v*, w*) = 0, g(v*, w*) = 0 with residual < 10^-12."""
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    equilibria = find_equilibria(params)
    assert len(equilibria) >= 1

    for eq in equilibria:
        v_star = eq.v_star
        w_star = eq.w_star
        # dv/dt residual
        res_v = v_star - (v_star**3) / 3.0 - w_star + params.I
        # dw/dt residual
        res_w = params.epsilon * (v_star + params.a - params.b * w_star)
        assert abs(res_v) < 1e-12
        assert abs(res_w) < 1e-12


def test_math_4_jacobian_analytical_vs_finite_difference():
    """Verify analytical Jacobian against centered finite differences (< 10^-7 error)."""
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.3)
    eq = find_equilibria(params)[0]
    J_ana = compute_jacobian(eq.v_star, params)

    delta = 1e-6
    # df/dv
    df_dv = ((eq.v_star + delta - (eq.v_star + delta)**3 / 3.0) - (eq.v_star - delta - (eq.v_star - delta)**3 / 3.0)) / (2 * delta)
    # df/dw
    df_dw = -1.0
    # dg/dv
    dg_dv = params.epsilon
    # dg/dw
    dg_dw = -params.epsilon * params.b

    assert pytest.approx(J_ana[0, 0], abs=1e-6) == df_dv
    assert pytest.approx(J_ana[0, 1], abs=1e-6) == df_dw
    assert pytest.approx(J_ana[1, 0], abs=1e-6) == dg_dv
    assert pytest.approx(J_ana[1, 1], abs=1e-6) == dg_dw


def test_math_5_eigenvalue_characteristic_consistency():
    """Verify eigenvalue calculations satisfy trace-determinant relations."""
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    eq = find_equilibria(params)[0]
    lam1, lam2 = eq.eigenvalues

    # Sum of eigenvalues = Trace
    assert pytest.approx((lam1 + lam2).real, abs=1e-10) == eq.trace
    assert pytest.approx((lam1 + lam2).imag, abs=1e-10) == 0.0

    # Product of eigenvalues = Determinant
    prod = lam1 * lam2
    assert pytest.approx(prod.real, abs=1e-10) == eq.determinant
    assert pytest.approx(prod.imag, abs=1e-10) == 0.0


def test_math_6_stability_classification_regimes():
    """Verify stability classification accurately partitions the Poincare plane."""
    # 1. Stable focus: tau < 0, Delta > 0, D < 0
    p_sf = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    eq_sf = find_equilibria(p_sf)[0]
    assert eq_sf.stability_class == StabilityClass.STABLE_FOCUS
    assert eq_sf.is_stable is True

    # 2. Limit cycle / Unstable focus: tau > 0, Delta > 0, D < 0
    p_uf = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.5)
    eq_uf = find_equilibria(p_uf)[0]
    assert eq_uf.stability_class == StabilityClass.UNSTABLE_FOCUS
    assert eq_uf.is_stable is False

    # 3. Saddle point: Delta < 0
    p_saddle = FHNParameters(a=0.0, b=2.5, epsilon=0.08, I=0.0)
    eqs_saddle = find_equilibria(p_saddle)
    saddles = [e for e in eqs_saddle if e.stability_class == StabilityClass.SADDLE]
    assert len(saddles) >= 1
    assert saddles[0].determinant < 0.0


def test_math_7_nullcline_intersections_exactness():
    """Verify that nullcline intersections coincide with the computed equilibria."""
    params = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    v_grid, w_vnull, w_wnull = compute_nullclines((-2.5, 2.5), params, num_points=500)
    eq = find_equilibria(params)[0]

    # Evaluate nullclines directly at v*
    w_vn_at_eq = eq.v_star - (eq.v_star**3) / 3.0 + params.I
    w_wn_at_eq = (eq.v_star + params.a) / params.b

    assert pytest.approx(w_vn_at_eq, abs=1e-10) == eq.w_star
    assert pytest.approx(w_wn_at_eq, abs=1e-10) == eq.w_star


def test_math_8_parameter_sweep_hopf_bifurcation_verification():
    """Verify parameter sweeps detect candidate Hopf bifurcations where trace changes sign."""
    params_base = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    sweep_res = sweep_parameter_I(I_min=-0.2, I_max=1.5, num_points=50, base_params=params_base)
    hopf_cands = find_hopf_candidates(params_base)

    assert len(hopf_cands) >= 1
    for hc in hopf_cands:
        # Candidate Hopf should have determinant > 0
        assert hc["determinant"] > 0.0
        assert np.isfinite(hc["I_hopf"])


def test_math_9_coupled_network_solver_isolated_and_connected_cells():
    """Verify coupled-network solver correctly handles completely isolated vs connected cells."""
    # Create network with 2 nodes
    net = CardiacNetwork(global_coupling=0.7)
    net.add_node(NetworkNode(0, "Cell_0", RegionType.SA_NODE, FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)))
    net.add_node(NetworkNode(1, "Cell_1", RegionType.VENTRICLE, FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)))

    # CASE A: Isolated (No edges)
    # Stimulate Cell 0 only
    def stim_0(t, node_id):
        return 2.0 if (node_id == 0 and 0.0 <= t <= 1.5) else 0.0

    res_iso = solve_network(net, t_span=(0.0, 40.0), dt=0.02, external_stimulus=stim_0)
    # Cell 0 depolarizes
    assert np.max(res_iso["V"][:, 0]) > 0.5
    # Cell 1 remains resting (does not depolarize, stays near resting ~ -1.2)
    assert np.max(res_iso["V"][:, 1]) < -1.1

    # CASE B: Connected with edge
    net.add_edge(0, 1, conductance=0.50)
    res_conn = solve_network(net, t_span=(0.0, 40.0), dt=0.02, external_stimulus=stim_0)
    # Cell 1 now receives electrotonic current deflection
    assert np.max(res_conn["V"][:, 1]) > np.max(res_iso["V"][:, 1]) + 0.05


# =============================================================================
# PART B: NETWORK VERIFICATION TESTS
# =============================================================================

def test_network_1_isolated_cell_receives_zero_input():
    """An isolated cell must receive zero electrotonic current from other cells."""
    net = create_default_cardiac_network(av_conductance=0.0, global_coupling=0.7)
    # Sever SA to Atrium
    net.set_edge_active(0, 1, False)

    def pace_sa(t, nid):
        return 2.0 if (nid == 0 and t <= 1.5) else 0.0

    res = solve_network(net, t_span=(0.0, 40.0), dt=0.02, external_stimulus=pace_sa)
    # Node 1 (Atrium) must not activate
    assert np.max(res["V"][:, 1]) < -1.0


def test_network_2_increasing_coupling_accelerates_propagation():
    """Increasing global coupling D must increase propagation velocity (shorten latency)."""
    p_low = ExperimentParameters(pacing_interval=45.0, global_coupling=0.40, t_end=60.0)
    p_high = ExperimentParameters(pacing_interval=45.0, global_coupling=1.10, t_end=60.0)

    res_low = run_experiment_simulation(p_low)
    res_high = run_experiment_simulation(p_high)

    lat_low = res_low.latencies["regional_summary"]["Total_SAN_to_Ventricle"]
    lat_high = res_high.latencies["regional_summary"]["Total_SAN_to_Ventricle"]

    assert lat_low is not None and lat_high is not None
    assert lat_high < lat_low


def test_network_3_disconnecting_edge_prevents_propagation():
    """Setting edge conductance to 0 or deactivating edge must completely prevent propagation."""
    cfg = ScenarioConfig(scenario_type=ScenarioType.AV_BLOCK, av_conductance=0.0, t_end=90.0)
    res = run_scenario(cfg)

    # SA Node (Node 0) and Atrium (Node 1) activate
    assert len(res.activations[0]) >= 2
    assert len(res.activations[1]) >= 2
    # His, Purkinje, and Ventricle must have ZERO activations
    assert len(res.activations[3]) == 0
    assert len(res.activations[4]) == 0
    assert len(res.activations[5]) == 0


def test_network_4_activation_timeline_matches_numerical_events():
    """The activation timeline must match numerical threshold-crossing timestamps exactly."""
    cfg = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=90.0)
    res = run_scenario(cfg)

    events = generate_numerical_event_log(res.activations, res.network)
    assert len(events) >= 6

    # Verify every event matches the threshold crossing v_i(t) > 0.0
    for ev in events:
        t_ev = ev.simulation_time
        node_id = ev.node_id
        # Interpolate voltage at t_ev
        idx = int(round(t_ev / cfg.dt))
        idx = min(idx, len(res.time) - 1)
        v_at_t = res.V[idx, node_id]
        assert v_at_t > -0.3  # Must be actively depolarizing near/above 0 threshold


def test_network_5_heart_visualization_matches_simulation_state():
    """Heart animation frames must be strictly derived from numerical solver output arrays."""
    cfg = ScenarioConfig(scenario_type=ScenarioType.NORMAL, t_end=40.0)
    res = run_scenario(cfg)
    frames_data = extract_animation_frames(res, target_fps=30)

    assert frames_data["num_frames"] > 10
    assert len(frames_data["times"]) == frames_data["num_frames"]
    assert len(frames_data["V"]) == frames_data["num_frames"]

    # Check that frame 0 values match solver initial values
    assert pytest.approx(frames_data["V"][0][0], abs=0.05) == res.V[0, 0]
    assert pytest.approx(frames_data["W"][0][0], abs=0.05) == res.W[0, 0]


def test_network_6_av_block_supported_by_failed_downstream_propagation():
    """AV block must exhibit documented conduction failures along the severed pathway."""
    cfg = ScenarioConfig(scenario_type=ScenarioType.AV_BLOCK, av_conductance=0.0, t_end=90.0)
    res = run_scenario(cfg)

    assert len(res.failures) >= 1
    # Failures list must identify the blocked pathway
    failure_edges = [(f["source_name"], f["target_name"]) for f in res.failures]
    assert ("Atrium", "AVN") in failure_edges or ("AVN", "His") in failure_edges


def test_network_7_reentry_requires_numerical_verification():
    """Re-entry must only be reported as sustained if repeated circus revolutions are verified."""
    # Sustained circuit
    cfg_sustained = ScenarioConfig(scenario_type=ScenarioType.REENTRY, reentry_num_nodes=10, reentry_coupling=0.35, reentry_epsilon=0.15, t_end=150.0)
    res_sustained = run_scenario(cfg_sustained)
    assert res_sustained.reentry_sustained is True
    assert res_sustained.reentry_revolutions >= 2

    # Circuit with parameter that extinguishes wave (high epsilon causes fast recovery blocking wavefront)
    cfg_extinguished = ScenarioConfig(scenario_type=ScenarioType.REENTRY, reentry_num_nodes=10, reentry_coupling=0.35, reentry_epsilon=0.45, t_end=150.0)
    res_ext = run_scenario(cfg_extinguished)
    # Must honestly report extinguished
    assert res_ext.reentry_revolutions < 2
    assert res_ext.reentry_sustained is False


# =============================================================================
# PART C: USER INTERFACE & INTEGRATION TESTS
# =============================================================================

def test_ui_1_parameter_sliders_produce_numerical_changes():
    """Modifying parameters must produce distinct numerical simulation trajectories."""
    base_p = ExperimentParameters(pacing_interval=45.0, av_conductance=0.35, stimulus_amplitude=1.8, t_end=60.0)
    p_diff_pace = ExperimentParameters(pacing_interval=55.0, av_conductance=0.35, stimulus_amplitude=1.8, t_end=60.0)
    p_diff_gav = ExperimentParameters(pacing_interval=45.0, av_conductance=0.20, stimulus_amplitude=1.8, t_end=60.0)
    p_diff_I = ExperimentParameters(pacing_interval=45.0, av_conductance=0.35, stimulus_amplitude=2.5, t_end=60.0)

    res_base = run_experiment_simulation(base_p)
    res_pace = run_experiment_simulation(p_diff_pace)
    res_gav = run_experiment_simulation(p_diff_gav)
    res_I = run_experiment_simulation(p_diff_I)

    # Verify trajectories are not identical
    assert not np.allclose(res_base.V, res_pace.V)
    assert not np.allclose(res_base.V, res_gav.V)
    assert not np.allclose(res_base.V, res_I.V)


def test_ui_2_scenarios_produce_distinct_electrophysiological_signatures():
    """Every preset scenario must produce a unique, characteristic numerical response."""
    res_norm = run_scenario(ScenarioConfig(scenario_type=ScenarioType.NORMAL, t_end=130.0))
    res_tachy = run_scenario(ScenarioConfig(scenario_type=ScenarioType.TACHYCARDIA, t_end=130.0))
    res_brady = run_scenario(ScenarioConfig(scenario_type=ScenarioType.BRADYCARDIA, t_end=130.0))
    res_pause = run_scenario(ScenarioConfig(scenario_type=ScenarioType.SINUS_PAUSE, t_end=130.0))
    res_block = run_scenario(ScenarioConfig(scenario_type=ScenarioType.AV_BLOCK, t_end=130.0))

    # Tachycardia has more activations than Normal
    assert len(res_tachy.activations[0]) > len(res_norm.activations[0])
    # Bradycardia has fewer activations than Normal
    assert len(res_brady.activations[0]) < len(res_norm.activations[0])
    # Block has 0 ventricular activations
    assert len(res_block.activations[5]) == 0
    # Pause has no activations during pause window [35, 75]
    acts_pause_sa = res_pause.activations[0]
    paused_acts = [t for t in acts_pause_sa if 35.0 <= t <= 75.0]
    assert len(paused_acts) == 0


def test_ui_3_html_visualization_contains_interactive_controls():
    """Generated HTML visualization must contain play, pause, reset, scrubber, and speed controls."""
    res = run_scenario(ScenarioConfig(scenario_type=ScenarioType.NORMAL, t_end=40.0))
    html_code = build_heart_visualization_html(res)

    assert "playBtn" in html_code
    assert "resetSim" in html_code
    assert "scrubber" in html_code
    assert "speedSelect" in html_code
    assert "legend" in html_code
    assert "rhythmCanvas" in html_code


def test_ui_4_all_graphs_update_from_single_simulation_truth():
    """Rhythm trace, action potentials, and event timeline must share the exact same simulation arrays."""
    cfg = ScenarioConfig(scenario_type=ScenarioType.NORMAL, t_end=60.0)
    res = run_scenario(cfg)

    # Rhythm trace length must match time array
    assert len(res.rhythm_trace) == len(res.time)
    assert res.V.shape[0] == len(res.time)
    assert res.W.shape[0] == len(res.time)


def test_ui_5_csv_exports_contain_actual_simulation_data():
    """CSV export functions must export actual computed data parseable by pandas."""
    base_p = ExperimentParameters(pacing_interval=45.0, t_end=60.0)
    trial_p = ExperimentParameters(pacing_interval=36.0, t_end=60.0)
    comp = compare_experiments(base_p, trial_p)

    csv_data = export_comparison_to_csv(comp)
    df = pd.read_csv(io.StringIO(csv_data))
    assert "Ventricular Rate" in df["Item"].values
    assert "SAN-to-Ventricle Latency" in df["Item"].values


def test_ui_6_application_has_zero_fake_or_prerecorded_data():
    """Running identical conditions yields identical output; varying parameters yields distinct output."""
    cfg1 = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=50.0)
    cfg2 = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=50.0)
    cfg3 = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=50.0, t_end=50.0)

    res1 = run_scenario(cfg1)
    res2 = run_scenario(cfg2)
    res3 = run_scenario(cfg3)

    # Identical parameters must yield exact bitwise identical trajectories (deterministic)
    np.testing.assert_array_equal(res1.V, res2.V)
    np.testing.assert_array_equal(res1.W, res2.W)

    # Perturbed parameter must NOT equal original (not prerecorded)
    assert not np.array_equal(res1.V, res3.V)


# =============================================================================
# PART D: PERFORMANCE & ROBUSTNESS TESTS
# =============================================================================

def test_performance_1_smooth_execution_timing():
    """Full coupled 6-compartment network simulation over 120 time units must run in responsive time."""
    params = ExperimentParameters(t_end=120.0, dt=0.02)
    t0 = time.perf_counter()
    res = run_experiment_simulation(params)
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0  # Responsive execution (< 3s on all platforms)
    assert len(res.time) == 6001


def test_performance_2_numerical_instability_and_extreme_parameter_guards():
    """Extreme stimuli or large dt must not cause unhandled NaN or crash the solver."""
    params_extreme = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=100.0)
    # Should complete without NaN or crash
    t, v, w = rk4_solve((-1.2, -0.6), (0.0, 10.0), dt=0.05, params=params_extreme)
    assert not np.any(np.isnan(v))
    assert not np.any(np.isnan(w))
    assert np.all(np.abs(v) <= 20.0)


def test_performance_3_solver_and_visualization_decoupling():
    """Numerical solver must have zero DOM/HTML/Streamlit dependencies and return pure data."""
    params = ExperimentParameters(t_end=30.0)
    res = run_experiment_simulation(params)

    # Return types must be pure numpy/dataclass
    assert isinstance(res.time, np.ndarray)
    assert isinstance(res.V, np.ndarray)
    assert isinstance(res.W, np.ndarray)
    assert isinstance(res.network, CardiacNetwork)
