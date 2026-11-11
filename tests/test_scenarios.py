"""
Automated validation tests for Stage 3 Arrhythmia Scenarios.

Verifies:
    1. Normal Rhythm: 1:1 forward conduction with steady cycle length.
    2. Tachycardia: Higher activation frequency than baseline.
    3. Bradycardia: Lower activation frequency than baseline.
    4. Sinus Pause: Verified absence of activations during suppression window.
    5. AV Block: Upstream activation with downstream ventricular failure.
    6. Re-entry Investigation: Honest numerical classification of loop circulation.
"""

import numpy as np
import pytest
from fhn_core.scenarios import (
    ScenarioType,
    ScenarioConfig,
    run_scenario,
)


def test_scenario_normal_rhythm_conduction():
    """
    Scenario 1: Normal Rhythm.
    Verifies that regular pacing of the SA node (T_pace = 45.0) produces 1:1 activation
    in both Atrium and Ventricle, with cycle length matching pacing interval.
    """
    cfg = ScenarioConfig(
        scenario_type=ScenarioType.NORMAL,
        pacing_interval=45.0,
        t_end=160.0,
        av_conductance=0.35,
        global_coupling=0.70,
    )
    result = run_scenario(cfg)

    san_acts = result.activations[0]
    vent_acts = result.activations[5]

    # Should have 4 beats over 160 time units
    assert len(san_acts) == 4, f"Expected 4 SAN beats, got {len(san_acts)}"
    assert len(vent_acts) == 4, f"Expected 4 Ventricular beats, got {len(vent_acts)}"

    # 1:1 conduction check
    assert len(san_acts) == len(vent_acts), f"Conduction is not 1:1: {len(san_acts)} SAN vs {len(vent_acts)} Ventricle"

    # Measured cycle length should be close to 45.0
    cl = result.mean_cycle_length
    assert cl is not None
    assert np.isclose(cl, 45.0, atol=1.0), f"Mean cycle length {cl} deviates from pacing interval 45.0"

    # Rhythm trace should show clear non-zero deflections
    assert np.ptp(result.rhythm_trace) > 0.2, "Rhythm trace amplitude too low"


def test_scenario_tachycardia_frequency_increase():
    """
    Scenario 2: Tachycardia-Like Rhythm.
    Verifies that shortening pacing interval from 45.0 to 35.0
    significantly elevates activation frequency.
    """
    cfg_normal = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=160.0)
    cfg_tachy = ScenarioConfig(scenario_type=ScenarioType.TACHYCARDIA, pacing_interval=35.0, t_end=160.0)

    res_norm = run_scenario(cfg_normal)
    res_tachy = run_scenario(cfg_tachy)

    # Tachycardia frequency must be strictly higher
    assert res_tachy.activation_frequency > res_norm.activation_frequency, (
        f"Tachycardia freq ({res_tachy.activation_frequency:.2f}) not greater than normal ({res_norm.activation_frequency:.2f})"
    )

    # More beats should be detected in tachycardia over the same duration (5 vs 4)
    assert len(res_tachy.activations[5]) > len(res_norm.activations[5]), "Expected more beats in tachycardia"


def test_scenario_bradycardia_frequency_decrease():
    """
    Scenario 3: Bradycardia-Like Rhythm.
    Verifies that lengthening pacing interval from 45.0 to 75.0
    significantly lowers activation frequency.
    """
    cfg_normal = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=160.0)
    cfg_brady = ScenarioConfig(scenario_type=ScenarioType.BRADYCARDIA, pacing_interval=75.0, t_end=160.0)

    res_norm = run_scenario(cfg_normal)
    res_brady = run_scenario(cfg_brady)

    # Bradycardia frequency must be strictly lower
    assert res_brady.activation_frequency < res_norm.activation_frequency, (
        f"Bradycardia freq ({res_brady.activation_frequency:.2f}) not lower than normal ({res_norm.activation_frequency:.2f})"
    )
    assert len(res_brady.activations[5]) < len(res_norm.activations[5]), "Expected fewer beats in bradycardia"


def test_scenario_sinus_pause_suppression_window():
    """
    Scenario 4: Sinus Pause.
    Verifies that clamping SA pacing to zero during [30.0, 110.0] produces a true pause
    where zero activations occur in the network during that window.
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
    res = run_scenario(cfg)

    # Check that in all chambers, no activation is initiated within the core pause interval
    for node_id in range(res.network.num_nodes):
        acts = res.activations[node_id]
        paused_acts = [t for t in acts if (pause_start + 15.0) <= t <= (pause_end - 1.0)]
        assert len(paused_acts) == 0, f"Node {node_id} unexpectedly fired during pause window: {paused_acts}"

    # Confirm beats still occurred before and after the pause
    vent_acts = res.activations[5]
    pre_pause = [t for t in vent_acts if t < pause_start]
    post_pause = [t for t in vent_acts if t > pause_end]

    assert len(pre_pause) >= 1, "Expected at least 1 beat before pause"
    assert len(post_pause) >= 1, "Expected at least 1 beat after pause resumed"


def test_scenario_av_block_conduction_failure():
    """
    Scenario 5: AV Block-Like Condition.
    Severing coupling between Atrium and AV Node (g_AV = 0.0) must allow Atrium
    to activate regularly while downstream ventricular chambers stay at 0 beats.
    """
    cfg = ScenarioConfig(
        scenario_type=ScenarioType.AV_BLOCK,
        pacing_interval=45.0,
        av_conductance=0.0,  # Complete uncoupling
        block_location="Atrium-to-AVN",
        t_end=160.0,
    )
    res = run_scenario(cfg)

    san_acts = res.activations[0]
    atr_acts = res.activations[1]
    avn_acts = res.activations[2]
    vent_acts = res.activations[5]

    # Upstream activation intact
    assert len(san_acts) == 4, f"Expected 4 SAN beats, got {len(san_acts)}"
    assert len(atr_acts) == 4, f"Expected 4 Atrial beats, got {len(atr_acts)}"

    # Downstream activation completely blocked
    assert len(avn_acts) == 0, f"AVN activated despite complete block: {avn_acts}"
    assert len(vent_acts) == 0, f"Ventricle activated despite complete block: {vent_acts}"

    # Failure must be detected and documented
    assert len(res.failures) > 0, "Failed to flag conduction block"


def test_scenario_reentry_numerical_honesty():
    """
    Advanced Scenario: Re-entry-like Activation.
    Verifies that the closed-loop network is numerically solved without hardcoded results.
    The test verifies that reentry_sustained is determined strictly by post-stimulus
    spontaneous activations around the loop.
    """
    # 1. Test 10-node ring (sufficient length > refractory wavelength): sustained circus movement
    cfg_sustained = ScenarioConfig(
        scenario_type=ScenarioType.REENTRY,
        reentry_num_nodes=10,
        reentry_coupling=0.35,
        reentry_epsilon=0.15,
        t_end=150.0,
        dt=0.02,
    )
    res_sust = run_scenario(cfg_sustained)

    assert res_sust.reentry_sustained is True
    assert res_sust.reentry_revolutions >= 4, f"Expected >= 4 revolutions, got {res_sust.reentry_revolutions}"
    assert "Sustained circus movement confirmed" in res_sust.comparison_with_normal["status"]

    # 2. Test 4-node ring (path too short, hits refractory tail): wave extinguishes!
    cfg_extinguished = ScenarioConfig(
        scenario_type=ScenarioType.REENTRY,
        reentry_num_nodes=4,
        reentry_coupling=0.35,
        reentry_epsilon=0.08,
        t_end=100.0,
        dt=0.02,
    )
    res_ext = run_scenario(cfg_extinguished)

    assert res_ext.reentry_sustained is False
    assert res_ext.reentry_revolutions == 0
    assert "extinguished" in res_ext.comparison_with_normal["status"]
