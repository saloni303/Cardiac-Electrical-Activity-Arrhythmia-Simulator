"""
Unit tests for Stage 5: Rhythm Visualization & Activation Analysis.

Verifies:
1. Numerical event log generation and chronologically sorted entries.
2. Beat-to-beat interval series calculations and variability metrics.
3. Heart rate (BPM) estimation across normal, tachycardic, bradycardic, and asystolic regimes.
4. Detection of missing expected impulses in Sinus Pause.
5. Detection of missing downstream activations in AV Conduction Block.
6. Einthoven Lead II dipole projection and source-to-electrode documentation.
7. Quantitative comparative analysis against Normal Sinus Rhythm.
"""

import numpy as np
import pytest

from fhn_core.scenarios import (
    ScenarioType,
    ScenarioConfig,
    run_scenario,
)
from fhn_core.rhythm_analysis import (
    generate_numerical_event_log,
    compute_beat_to_beat_intervals,
    estimate_heart_rate_bpm,
    detect_missing_events,
    compute_true_ecg_lead_approximation,
    compare_with_normal_rhythm,
)


def test_generate_numerical_event_log_structure():
    """Verify numerical event log contains proper keys, timestamps, and chronological order."""
    cfg = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=100.0)
    res = run_scenario(cfg)

    log_entries = generate_numerical_event_log(res.activations, res.network, threshold=0.0)
    assert len(log_entries) > 0

    # Verify chronological ordering
    times = [entry.simulation_time for entry in log_entries]
    assert times == sorted(times)

    # Verify dictionary export format
    sample_dict = log_entries[0].to_dict()
    assert "Time (u)" in sample_dict
    assert "Region" in sample_dict
    assert "Threshold Crossing (v)" in sample_dict
    assert "Interval (Δt u)" in sample_dict
    assert "Conduction Status" in sample_dict

    # First beat of each region should have interval = None
    first_entries = [e for e in log_entries if e.beat_index == 0]
    for e in first_entries:
        assert e.previous_activation_time is None
        assert e.interval_since_previous is None


def test_compute_beat_to_beat_intervals_normal():
    """Verify beat-to-beat interval calculations for normal regular rhythm."""
    cfg = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=150.0)
    res = run_scenario(cfg)

    metrics = compute_beat_to_beat_intervals(res.activations, monitored_node_id=5)
    assert metrics["count"] >= 3
    assert len(metrics["intervals"]) >= 2
    assert np.isclose(metrics["mean_interval"], 45.0, atol=0.5)
    assert metrics["std_interval"] < 0.5
    assert metrics["is_regular"] is True


def test_estimate_heart_rate_bpm_scenarios():
    """Verify BPM estimation across normal, tachycardic, bradycardic, and blocked conditions."""
    # Normal: ~45 u -> ~66.7 BPM
    hr_norm = estimate_heart_rate_bpm(45.0, time_unit_seconds=0.020)
    assert hr_norm["is_meaningful"] is True
    assert 65.0 <= hr_norm["bpm"] <= 68.0
    assert "Normal-Range" in hr_norm["classification"]

    # Tachycardia: ~35 u -> ~85.7 BPM
    hr_tachy = estimate_heart_rate_bpm(35.0, time_unit_seconds=0.020)
    assert hr_tachy["is_meaningful"] is True
    assert hr_tachy["bpm"] > hr_norm["bpm"]

    # Bradycardia: ~75 u -> ~40.0 BPM
    hr_brady = estimate_heart_rate_bpm(75.0, time_unit_seconds=0.020)
    assert hr_brady["is_meaningful"] is True
    assert hr_brady["bpm"] < hr_norm["bpm"]
    assert "Bradycardia-Like" in hr_brady["classification"]

    # Complete Block / Asystole: None
    hr_none = estimate_heart_rate_bpm(None)
    assert hr_none["is_meaningful"] is False
    assert hr_none["bpm"] is None
    assert "Asystole" in hr_none["classification"]


def test_detect_missing_events_sinus_pause():
    """Verify detection of missing expected SA impulses during the sinus pause window."""
    cfg = ScenarioConfig(
        scenario_type=ScenarioType.SINUS_PAUSE,
        pacing_interval=45.0,
        pause_start=30.0,
        pause_end=110.0,
        t_end=160.0,
    )
    res = run_scenario(cfg)

    missing_info = detect_missing_events(res)
    missing_sa = missing_info["missing_sa_impulses"]

    # In interval [30, 110], expected pulses at t=45 and t=90 should be missing
    assert len(missing_sa) >= 2
    exp_times = [m["expected_time"] for m in missing_sa]
    assert 45.0 in exp_times
    assert 90.0 in exp_times


def test_detect_missing_events_av_block():
    """Verify detection of missing downstream ventricular activations in complete AV block."""
    cfg = ScenarioConfig(
        scenario_type=ScenarioType.AV_BLOCK,
        av_conductance=0.0,
        block_location="Atrium-to-AVN",
        t_end=160.0,
    )
    res = run_scenario(cfg)

    missing_info = detect_missing_events(res)
    assert missing_info["num_atrial_beats"] >= 3
    assert missing_info["num_ventricular_beats"] == 0
    assert len(missing_info["missing_downstream_events"]) == missing_info["num_atrial_beats"]
    assert "Conduction failure" in missing_info["missing_downstream_events"][0]["reason"]


def test_compute_true_ecg_lead_approximation():
    """Verify Einthoven Lead II projection calculation and physical documentation string."""
    cfg = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=45.0, t_end=60.0)
    res = run_scenario(cfg)

    lead_trace, doc_text = compute_true_ecg_lead_approximation(
        res.time, res.V, res.network, lead_name="Lead_II"
    )

    assert len(lead_trace) == len(res.time)
    assert np.all(np.isfinite(lead_trace))
    assert "Einthoven Lead_II Dipole Approximation" in doc_text
    assert "homogeneous, isotropic, quasi-static" in doc_text


def test_compare_with_normal_rhythm():
    """Verify quantitative delta comparisons against baseline normal rhythm."""
    cfg_tachy = ScenarioConfig(scenario_type=ScenarioType.TACHYCARDIA, pacing_interval=35.0, t_end=100.0)
    res_tachy = run_scenario(cfg_tachy)

    comp = compare_with_normal_rhythm(res_tachy)
    assert comp["scenario_cycle_length"] < comp["baseline_cycle_length"]
    assert comp["rate_ratio"] > 1.15
    assert comp["conduction_success"] is True
