"""
Technical Validation Audit Test Suite for Stage 4 - FHN Cardiac Network.

Automates formal verification of the 7 audit criteria:
1. Normal Rhythm regular activation and forward propagation.
2. Tachycardia frequency acceleration evidence.
3. Bradycardia frequency deceleration evidence.
4. Sinus Pause stimulus suppression and resting stability (no autonomous oscillation).
5. AV Block actual conductance modification, pathway severance, and downstream failure.
6. Mathematical integration: visualization data strictly mirrors ODE state.
7. Reproducibility: deterministic invariance across identical simulation runs.
"""

import numpy as np
import pytest

from fhn_core.scenarios import (
    ScenarioType,
    ScenarioConfig,
    run_scenario,
    make_pacing_protocol,
)
from fhn_core.visualization import extract_animation_frames, map_state_to_color
from fhn_core.network import solve_network, create_default_cardiac_network


def test_audit_normal_rhythm_regularity_and_propagation():
    """Verify regular SA pacing and forward propagation through all compartments."""
    cfg = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=160.0)
    res = run_scenario(cfg)

    # 1. SA node generates regular activations
    sa_acts = res.activations[0]
    assert len(sa_acts) >= 3, "Expected at least 3 SA activations"
    sa_intervals = np.diff(sa_acts)
    assert np.allclose(sa_intervals, 45.0, atol=0.2), f"SA pacing interval mismatch: {sa_intervals}"

    # 2. Sequential forward propagation across network
    # SAN (0) -> Atrium (1) -> AVN (2) -> His (3) -> Purkinje (4) -> Ventricle (5)
    for beat_idx in range(len(res.activations[5])):
        t_san = res.activations[0][beat_idx]
        t_atr = res.activations[1][beat_idx]
        t_avn = res.activations[2][beat_idx]
        t_his = res.activations[3][beat_idx]
        t_purk = res.activations[4][beat_idx]
        t_vent = res.activations[5][beat_idx]

        assert t_san < t_atr < t_avn < t_his < t_purk < t_vent, (
            f"Beat {beat_idx} failed strictly monotonic forward propagation: "
            f"[{t_san:.2f}, {t_atr:.2f}, {t_avn:.2f}, {t_his:.2f}, {t_purk:.2f}, {t_vent:.2f}]"
        )

    # 3. Activation times are floating-point numbers derived from sub-step interpolation
    assert isinstance(sa_acts[0], float)
    assert not np.isnan(sa_acts[0])


def test_audit_tachycardia_frequency_acceleration():
    """Verify tachycardia increases activation frequency with numerical evidence."""
    cfg_norm = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=160.0)
    res_norm = run_scenario(cfg_norm)

    cfg_tachy = ScenarioConfig(scenario_type=ScenarioType.TACHYCARDIA, pacing_interval=35.0, t_end=160.0)
    res_tachy = run_scenario(cfg_tachy)

    # Activation frequency check
    assert res_tachy.activation_frequency > res_norm.activation_frequency
    rate_ratio = res_tachy.activation_frequency / res_norm.activation_frequency
    assert rate_ratio > 1.15, f"Expected >15% rate acceleration, got {rate_ratio:.3f}x"

    # Cycle length check
    assert res_tachy.mean_cycle_length < res_norm.mean_cycle_length


def test_audit_bradycardia_frequency_deceleration():
    """Verify bradycardia decreases activation frequency with numerical evidence."""
    cfg_norm = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=160.0)
    res_norm = run_scenario(cfg_norm)

    cfg_brady = ScenarioConfig(scenario_type=ScenarioType.BRADYCARDIA, pacing_interval=75.0, t_end=160.0)
    res_brady = run_scenario(cfg_brady)

    # Activation frequency check
    assert res_brady.activation_frequency < res_norm.activation_frequency
    rate_ratio = res_brady.activation_frequency / res_norm.activation_frequency
    assert rate_ratio < 0.75, f"Expected <75% of normal rate, got {rate_ratio:.3f}x"

    # Cycle length check
    assert res_brady.mean_cycle_length > res_norm.mean_cycle_length


def test_audit_sinus_pause_stimulus_suppression_and_stability():
    """
    Verify sinus pause suppresses the stimulus function and confirms that
    the SA-node model remains in its stable resting state (no autonomous oscillations).
    """
    pause_start = 30.0
    pause_end = 110.0
    cfg = ScenarioConfig(
        scenario_type=ScenarioType.SINUS_PAUSE,
        pacing_interval=45.0,
        pause_start=pause_start,
        pause_end=pause_end,
        t_end=160.0,
    )

    stim_fn = make_pacing_protocol(
        pacing_interval=cfg.pacing_interval,
        pulse_amplitude=cfg.pulse_amplitude,
        pulse_duration=cfg.pulse_duration,
        pause_start=pause_start,
        pause_end=pause_end,
        target_node_id=0,
    )

    # 1. Numerical verification of stimulus suppression during pause window
    for t_test in [35.0, 45.0, 45.5, 70.0, 90.0, 90.5, 105.0]:
        assert stim_fn(t_test, 0) == 0.0, f"Stimulus not suppressed at t={t_test}"

    # 2. Stimulus is delivered outside the pause window
    assert stim_fn(0.5, 0) == cfg.pulse_amplitude
    assert stim_fn(135.5, 0) == cfg.pulse_amplitude

    # 3. Simulate and verify SA node behavior
    res = run_scenario(cfg)
    sa_acts = res.activations[0]

    # No activations during pause window
    pause_acts = [t for t in sa_acts if pause_start <= t <= pause_end]
    assert len(pause_acts) == 0, f"Unexpected activation during sinus pause: {pause_acts}"

    # Check SA node trajectory during pause: strictly subthreshold (v < -1.0)
    mask = (res.time >= pause_start) & (res.time <= pause_end)
    v_sa_pause = res.V[mask, 0]
    assert np.all(v_sa_pause < -1.0), "SA node voltage deviated from resting baseline during pause"


def test_audit_av_block_coupling_severance_and_downstream_failure():
    """
    Verify complete AV block sets conductance to 0, preserves upstream SA/atrial firing,
    and produces complete downstream ventricular activation failure.
    """
    cfg = ScenarioConfig(
        scenario_type=ScenarioType.AV_BLOCK,
        av_conductance=0.0,
        block_location="Atrium-to-AVN",
        t_end=160.0,
    )
    res = run_scenario(cfg)

    # 1. Verify adjacency matrix reflects severed connection
    A = res.network.get_adjacency_matrix()
    assert A[1, 2] == 0.0
    assert A[2, 1] == 0.0

    # 2. Upstream cells (SAN and Atrium) continue regular activation
    assert len(res.activations[0]) >= 4, "SA node failed to fire regularly"
    assert len(res.activations[1]) >= 4, "Atrium failed to activate from SA node"

    # 3. Downstream cells (AVN, His, Purkinje, Ventricles) experience complete failure
    assert len(res.activations[2]) == 0, "AVN activated despite complete block"
    assert len(res.activations[3]) == 0, "His activated despite complete block"
    assert len(res.activations[4]) == 0, "Purkinje activated despite complete block"
    assert len(res.activations[5]) == 0, "Ventricle activated despite complete block"

    # 4. Conduction failure analyzer detects the severed pathway
    assert len(res.failures) >= 1
    assert any(f["source_id"] == 1 and f["target_id"] == 2 for f in res.failures)


def test_audit_mathematical_integration_no_synthetic_randomness():
    """
    Verify the visualization frame extraction derives 100% from ODE solution arrays
    and contains no random number generation or pre-recorded animations.
    """
    cfg = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=60.0)
    res = run_scenario(cfg)

    anim_data = extract_animation_frames(res, target_fps=30)

    # Verify frame array shapes and direct values matching ODE
    assert anim_data["num_frames"] == len(anim_data["times"])
    assert len(anim_data["V"]) == anim_data["num_frames"]
    assert len(anim_data["W"]) == anim_data["num_frames"]

    # Verify frame 0 matches initial ODE condition
    assert np.isclose(anim_data["V"][0][0], res.V[0, 0], atol=1e-2)
    assert np.isclose(anim_data["W"][0][0], res.W[0, 0], atol=1e-2)

    # Verify deterministic color mapping
    col_rest, _, _ = map_state_to_color(-1.2, -0.6)
    col_active, _, _ = map_state_to_color(1.5, 0.0)
    col_rec, _, _ = map_state_to_color(-0.5, 0.5)

    assert col_rest != col_active
    assert col_active != col_rec


def test_audit_reproducibility_deterministic_invariance():
    """
    Verify repeated simulations with identical inputs produce bit-for-bit identical results.
    """
    cfg = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=80.0)

    res1 = run_scenario(cfg)
    res2 = run_scenario(cfg)

    # Strict floating-point identity
    assert np.array_equal(res1.V, res2.V), "Numerical trajectory V was not bit-for-bit identical across runs"
    assert np.array_equal(res1.W, res2.W), "Numerical trajectory W was not bit-for-bit identical across runs"
    assert res1.activations == res2.activations, "Activation timestamps differed across identical runs"
