"""
Unit tests for Stage 4: Interactive Cardiac Visualization Module.

Verifies:
1. State-to-color mapping across resting, depolarized, and refractory regimes.
2. Animation frame extraction from continuous ODE solutions.
3. Standalone HTML5/SVG/Canvas visualizer generation.
4. Numerical fidelity: visualization reflects actual simulated ODE states.
"""

import json
import numpy as np
import pytest

from fhn_core.visualization import (
    map_state_to_color,
    extract_animation_frames,
    build_heart_visualization_html,
)
from fhn_core.scenarios import (
    ScenarioType,
    ScenarioConfig,
    run_scenario,
    create_reentry_circuit,
)
from fhn_core.network import (
    create_default_cardiac_network,
    solve_network,
)


def test_map_state_to_color_resting():
    """Verify resting polarized state (v ≈ -1.2, w ≈ -0.6) maps to slate dark color."""
    color, state_name, intensity = map_state_to_color(v=-1.2, w=-0.6)
    assert color == "rgb(30, 41, 59)" or color == "#1e293b"
    assert "Resting" in state_name
    assert intensity == 0.0


def test_map_state_to_color_depolarized():
    """Verify upstroke/depolarized state (v > 0.0) maps to active golden color."""
    color, state_name, intensity = map_state_to_color(v=1.5, w=0.0)
    assert "rgb(" in color
    assert "Activating" in state_name or "Depolarized" in state_name
    assert 0.0 < intensity <= 1.0


def test_map_state_to_color_refractory():
    """Verify repolarizing refractory state (w > -0.35, v < 0.0) maps to violet color."""
    color, state_name, intensity = map_state_to_color(v=-0.5, w=0.8)
    assert "rgb(" in color
    assert "Refractory" in state_name or "Recovery" in state_name
    assert 0.0 < intensity <= 1.0


def test_extract_animation_frames_normal():
    """Verify animation frames extracted from Normal Sinus Rhythm scenario."""
    cfg = ScenarioConfig(
        scenario_type=ScenarioType.NORMAL,
        pacing_interval=45.0,
        t_end=90.0,
        dt=0.02,
    )
    res = run_scenario(cfg)
    anim_data = extract_animation_frames(res, target_fps=20)

    assert anim_data["num_frames"] > 0
    assert len(anim_data["times"]) == anim_data["num_frames"]
    assert len(anim_data["V"]) == anim_data["num_frames"]
    assert len(anim_data["W"]) == anim_data["num_frames"]
    assert len(anim_data["rhythm"]) == anim_data["num_frames"]
    assert anim_data["is_reentry"] is False
    assert len(anim_data["nodes"]) == 6

    # Verify no NaN or Inf
    for frame_v in anim_data["V"]:
        for val in frame_v:
            assert np.isfinite(val)


def test_extract_animation_frames_reentry():
    """Verify animation frames extracted from Re-entry scenario."""
    cfg = ScenarioConfig(
        scenario_type=ScenarioType.REENTRY,
        pacing_interval=45.0,
        t_end=60.0,
        dt=0.02,
        reentry_num_nodes=10,
        reentry_coupling=0.35,
        reentry_epsilon=0.15,
    )
    res = run_scenario(cfg)
    anim_data = extract_animation_frames(res, target_fps=20)

    assert anim_data["is_reentry"] is True
    assert len(anim_data["nodes"]) == 10
    assert anim_data["num_frames"] > 0


def test_build_heart_visualization_html():
    """Verify standalone HTML generation contains all required interactive components."""
    cfg = ScenarioConfig(
        scenario_type=ScenarioType.NORMAL,
        pacing_interval=45.0,
        t_end=50.0,
        dt=0.02,
    )
    res = run_scenario(cfg)
    html_output = build_heart_visualization_html(res)

    assert isinstance(html_output, str)
    assert "<!DOCTYPE html>" in html_output
    assert "<svg" in html_output
    assert "<canvas" in html_output
    assert "chamber_RA" in html_output
    assert "chamber_LA" in html_output
    assert "chamber_RV" in html_output
    assert "chamber_LV" in html_output
    assert "nodeMarker_0" in html_output  # SA Node
    assert "nodeMarker_2" in html_output  # AV Node
    assert "playBtn" in html_output
    assert "timeScrubber" in html_output
    assert "speedSelect" in html_output
    assert "rhythmCanvas" in html_output
    assert "Simplified Educational Model" in html_output


def test_visualization_numerical_fidelity():
    """Verify that visualization frame colors directly reflect numerical excitation."""
    cfg = ScenarioConfig(
        scenario_type=ScenarioType.NORMAL,
        pacing_interval=45.0,
        t_end=60.0,
        dt=0.02,
    )
    res = run_scenario(cfg)
    anim_data = extract_animation_frames(res, target_fps=30)

    # Locate peak excitation of SA node
    v_sa_series = [frame[0] for frame in anim_data["V"]]
    peak_sa_idx = int(np.argmax(v_sa_series))
    peak_v = v_sa_series[peak_sa_idx]
    peak_w = anim_data["W"][peak_sa_idx][0]

    assert peak_v > 0.0, "Expected SA node to achieve depolarized positive voltage"
    col, state, intensity = map_state_to_color(peak_v, peak_w)
    assert "Activating" in state or "Depolarized" in state
    assert intensity > 0.5
