"""
Unit and Integration Tests for Stage 8: Parameter Experiment Laboratory.

Verifies:
    1. Controlled experiment creation, network building, and region parameter overrides.
    2. Side-by-side comparison between baseline and trial states.
    3. Conduction block detection (AV junction weakening).
    4. Conduction acceleration without arrhythmia (coupling D increase).
    5. Frequency acceleration (tachycardia) and deceleration (bradycardia).
    6. Single-cell equilibrium shift and stability tracking.
    7. Automated parameter sweep execution and metric extraction.
    8. CSV serialization and scientific interpretation validity.
"""

import pytest
import numpy as np
import pandas as pd

from fhn_core.experiments import (
    ExperimentParameters,
    ExperimentComparisonResult,
    ParameterSweepPoint,
    ParameterSweepConfig,
    build_experiment_network,
    run_experiment_simulation,
    compare_experiments,
    run_parameter_sweep,
    export_comparison_to_csv,
    export_sweep_to_csv,
    compute_conduction_latency_san_to_ventricle,
)
from fhn_core.scenarios import ScenarioType
from fhn_core.network import RegionType


def test_build_experiment_network_and_region_overrides():
    """Verify that custom parameters and regional overrides are correctly applied."""
    params = ExperimentParameters(
        cell_a=0.65,
        cell_b=0.75,
        cell_epsilon=0.09,
        cell_I_bias=0.05,
        av_conductance=0.22,
        global_coupling=0.85,
        region_overrides={
            "AVN": {"epsilon": 0.035, "a": 0.72},
            "Ventricle": {"epsilon": 0.055},
        }
    )

    net = build_experiment_network(params)
    assert net.num_nodes == 6
    assert net.global_coupling == 0.85

    # Check AV conductance edge (1 -> 2)
    av_edge = [e for e in net.edges if e.source_id == 1 and e.target_id == 2][0]
    assert pytest.approx(av_edge.conductance, abs=1e-5) == 0.22

    # Check Atrium node (Node 1) - base parameters
    atrium_node = net.nodes[1]
    assert atrium_node.params.a == 0.65
    assert atrium_node.params.epsilon == 0.09

    # Check AVN node (Node 2) - overridden parameters
    avn_node = net.nodes[2]
    assert avn_node.params.epsilon == 0.035
    assert avn_node.params.a == 0.72
    assert avn_node.params.b == 0.75

    # Check Ventricle node (Node 5) - overridden epsilon
    vent_node = net.nodes[5]
    assert vent_node.params.epsilon == 0.055
    assert vent_node.params.a == 0.65


def test_experiment_baseline_simulation_run():
    """Verify that a baseline simulation executes reliably with 1:1 forward conduction."""
    params = ExperimentParameters(
        scenario_type=ScenarioType.NORMAL,
        pacing_interval=45.0,
        av_conductance=0.35,
        global_coupling=0.70,
        t_end=90.0,
    )

    res = run_experiment_simulation(params)
    assert res.time[-1] == pytest.approx(90.0, abs=0.1)

    # SA node and Ventricle should both activate
    assert len(res.activations[0]) >= 2
    assert len(res.activations[5]) >= 2
    assert len(res.failures) == 0

    latency = compute_conduction_latency_san_to_ventricle(res.activations, params.pacing_interval)
    assert latency is not None
    assert 10.0 < latency < 30.0


def test_compare_experiments_conduction_block():
    """Verify that reducing AV conductance causes complete block and is diagnosed as arrhythmia."""
    base_p = ExperimentParameters(
        pacing_interval=45.0,
        av_conductance=0.35,
        t_end=90.0,
    )
    trial_p = ExperimentParameters(
        pacing_interval=45.0,
        av_conductance=0.00,  # Severed conduction
        t_end=90.0,
    )

    comp = compare_experiments(base_p, trial_p)

    assert comp.conduction_drops_baseline == 0
    assert comp.conduction_drops_trial > 0
    assert comp.bpm_baseline is not None and comp.bpm_baseline > 40.0
    assert comp.bpm_trial is None  # Ventricle blocked
    assert comp.is_arrhythmia_induced is True
    assert "Conduction Block" in comp.scientific_interpretation


def test_compare_experiments_coupling_preserves_rhythm():
    """
    Verify scientific honesty: increasing coupling D increases conduction velocity
    without inducing an arrhythmia (1:1 rhythm preserved).
    """
    base_p = ExperimentParameters(
        pacing_interval=45.0,
        global_coupling=0.70,
        t_end=90.0,
    )
    trial_p = ExperimentParameters(
        pacing_interval=45.0,
        global_coupling=0.95,  # Faster conduction velocity
        t_end=90.0,
    )

    comp = compare_experiments(base_p, trial_p)

    assert comp.conduction_drops_baseline == 0
    assert comp.conduction_drops_trial == 0
    # Both should have valid similar heart rate
    assert comp.bpm_baseline is not None
    assert comp.bpm_trial is not None
    assert abs(comp.bpm_trial - comp.bpm_baseline) < 2.0

    # Latency in trial should be shorter due to higher coupling D
    assert comp.san_to_vent_latency_baseline is not None
    assert comp.san_to_vent_latency_trial is not None
    assert comp.san_to_vent_latency_trial < comp.san_to_vent_latency_baseline

    # Must NOT flag as arrhythmia!
    assert comp.is_arrhythmia_induced is False
    assert "Conduction Velocity Shift" in comp.scientific_interpretation or "Physiological" in comp.scientific_interpretation


def test_compare_experiments_rate_modulation():
    """Verify rate acceleration in tachycardia vs deceleration in bradycardia."""
    base_p = ExperimentParameters(pacing_interval=45.0, t_end=120.0)
    tachy_p = ExperimentParameters(pacing_interval=36.0, t_end=120.0)
    brady_p = ExperimentParameters(pacing_interval=75.0, t_end=120.0)

    comp_tachy = compare_experiments(base_p, tachy_p)
    assert comp_tachy.bpm_trial > comp_tachy.bpm_baseline
    assert "Frequency Acceleration" in comp_tachy.scientific_interpretation

    comp_brady = compare_experiments(base_p, brady_p)
    assert comp_brady.bpm_trial < comp_brady.bpm_baseline
    assert "Frequency Deceleration" in comp_brady.scientific_interpretation


def test_parameter_sweep_av_conductance():
    """Verify automated sweep over AV conductance captures the block-to-conduction threshold."""
    sweep_cfg = ParameterSweepConfig(
        parameter_name="av_conductance",
        range_min=0.00,
        range_max=0.35,
        num_steps=5,
        base_params=ExperimentParameters(pacing_interval=45.0, t_end=60.0),
    )

    points, summary = run_parameter_sweep(sweep_cfg)
    assert len(points) == 5

    # First point (g_AV = 0.0) should have conduction failures
    assert points[0].param_value == 0.0
    assert points[0].conduction_success is False
    assert points[0].conduction_drops > 0

    # Last point (g_AV = 0.35) should conduct cleanly
    assert points[-1].param_value == 0.35
    assert points[-1].conduction_success is True
    assert points[-1].conduction_drops == 0

    assert "Parameter Sweep Analysis" in summary


def test_csv_export_format():
    """Verify CSV export string generation for both comparison and parameter sweeps."""
    base_p = ExperimentParameters(pacing_interval=45.0, av_conductance=0.35, t_end=60.0)
    trial_p = ExperimentParameters(pacing_interval=45.0, av_conductance=0.05, t_end=60.0)
    comp = compare_experiments(base_p, trial_p)

    csv_text = export_comparison_to_csv(comp)
    assert "Category,Item,State A (Baseline),State B (Trial),Delta (B - A)" in csv_text
    assert "AV Nodal Conductance (g_AV)" in csv_text
    assert "Ventricular Rate" in csv_text

    # Verify parseable by pandas
    import io
    df = pd.read_csv(io.StringIO(csv_text))
    assert len(df) > 5

    # Test sweep CSV export
    pt1 = ParameterSweepPoint(0.1, 72.0, 45.0, True, 0, 15.2, -1.2, "Stable Focus", -0.05, 0.8, False)
    pt2 = ParameterSweepPoint(0.2, None, None, False, 2, None, -1.2, "Stable Focus", -0.05, 0.8, False)
    sweep_csv = export_sweep_to_csv([pt1, pt2], "D")
    df_sweep = pd.read_csv(io.StringIO(sweep_csv))
    assert len(df_sweep) == 2
    assert "Param_D" in df_sweep.columns
    assert "Ventricular_BPM" in df_sweep.columns
