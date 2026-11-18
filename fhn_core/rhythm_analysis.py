"""
Stage 5: Rhythm Visualization & Activation Analysis Module.

Provides comprehensive algorithmic analysis of coupled FitzHugh-Nagumo cardiac network outputs:
1. Activation-derived electrical rhythm traces (dipole projection) and far-field lead approximations.
2. High-precision beat-to-beat interval calculations (RR-like tachogram sequence).
3. Dimensionless-to-physical heart rate estimation in beats per minute (BPM).
4. Multi-compartment activation-event timelines with explicit detection of:
   - Missing expected SA impulses (Sinus Pause)
   - Missing downstream activations (AV Conduction Block)
5. Comprehensive numerical event logging (time, region, threshold crossing, interval).
6. Quantitative comparative analysis against Normal Sinus Rhythm.

Academic & Biophysical Disclaimers:
    - Rhythm traces are algorithmically derived from the actual numerical ODE states.
    - They are NOT clinically validated surface 12-lead ECGs and must never be used for diagnosis.
    - Heart rate (BPM) mapping uses an explicit dimensionless timescale calibration factor
      (default: 1 model unit ≈ 20 ms).
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .network import CardiacNetwork, RegionType
from .scenarios import ScenarioResult, ScenarioType, ScenarioConfig, run_scenario


@dataclass
class EventLogEntry:
    """
    Individual entry in the numerical activation event log.
    """
    simulation_time: float
    node_id: int
    region_name: str
    threshold_crossing: float
    previous_activation_time: Optional[float]
    interval_since_previous: Optional[float]
    beat_index: int
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Time (u)": round(self.simulation_time, 2),
            "Region": self.region_name,
            "Node ID": self.node_id,
            "Threshold Crossing (v)": round(self.threshold_crossing, 3),
            "Previous Activation (u)": round(self.previous_activation_time, 2) if self.previous_activation_time is not None else "-",
            "Interval (Δt u)": round(self.interval_since_previous, 2) if self.interval_since_previous is not None else "First Beat",
            "Beat #": self.beat_index + 1,
            "Conduction Status": self.status,
        }


def generate_numerical_event_log(
    activations: Dict[int, List[float]],
    network: CardiacNetwork,
    threshold: float = 0.0,
) -> List[EventLogEntry]:
    """
    Generates a chronological, multi-channel numerical activation event log.

    Every upward threshold crossing across all network compartments is recorded with:
    - Sub-step interpolated timestamp
    - Previous activation time for that region
    - Interval since preceding activation (local cycle length)
    - Conduction status flag
    """
    entries: List[EventLogEntry] = []

    for node_id, t_acts in activations.items():
        if node_id >= network.num_nodes:
            continue
        region_name = network.nodes[node_id].name

        prev_t = None
        for beat_idx, t_act in enumerate(t_acts):
            interval = (t_act - prev_t) if prev_t is not None else None

            # Determine qualitative status
            if prev_t is None:
                status = "Initial Beat / Baseline"
            elif interval is not None:
                if interval < 38.0:
                    status = "Accelerated / Tachycardic"
                elif interval > 60.0:
                    status = "Delayed / Bradycardic"
                else:
                    status = "Regular Forward Cascade"
            else:
                status = "Active"

            entry = EventLogEntry(
                simulation_time=float(t_act),
                node_id=node_id,
                region_name=region_name,
                threshold_crossing=float(threshold),
                previous_activation_time=prev_t,
                interval_since_previous=interval,
                beat_index=beat_idx,
                status=status,
            )
            entries.append(entry)
            prev_t = t_act

    # Sort chronologically by simulation time
    entries.sort(key=lambda e: e.simulation_time)
    return entries


def compute_beat_to_beat_intervals(
    activations: Dict[int, List[float]],
    monitored_node_id: int = 5,
) -> Dict[str, Any]:
    """
    Computes beat-to-beat (RR-like) interval series and variability metrics for a target region.
    Default monitored node is Node 5 (Ventricular Myocardium).
    """
    t_acts = activations.get(monitored_node_id, [])

    if len(t_acts) < 2:
        return {
            "intervals": [],
            "beat_indices": [],
            "mean_interval": None,
            "std_interval": None,
            "min_interval": None,
            "max_interval": None,
            "count": len(t_acts),
            "is_regular": False,
        }

    intervals = [float(t_acts[k + 1] - t_acts[k]) for k in range(len(t_acts) - 1)]
    beat_indices = list(range(1, len(t_acts)))

    mean_int = float(np.mean(intervals))
    std_int = float(np.std(intervals))
    min_int = float(np.min(intervals))
    max_int = float(np.max(intervals))

    # Regularity threshold (standard deviation < 5% of mean)
    is_reg = bool(std_int < 0.05 * mean_int) if mean_int > 0 else False

    return {
        "intervals": intervals,
        "beat_indices": beat_indices,
        "mean_interval": mean_int,
        "std_interval": std_int,
        "min_interval": min_int,
        "max_interval": max_int,
        "count": len(t_acts),
        "is_regular": is_reg,
    }


def estimate_heart_rate_bpm(
    mean_cycle_length: Optional[float],
    time_unit_seconds: float = 0.020,
) -> Dict[str, Any]:
    """
    Converts dimensionless FHN cycle length into an estimated physiological heart rate (BPM).

    Timescale Calibration:
        In standard cardiac electrophysiology:
            Action potential duration (APD) ≈ 250 - 300 ms.
            In FHN (a=0.7, b=0.8, ε=0.08), APD ≈ 11.3 model units.
            Calibrated scaling factor: 1 model unit ≈ 20 ms = 0.020 seconds.
            Normal resting cycle length (45.0 units) = 45 * 0.020 s = 0.900 s ≈ 66.7 BPM.

    Formula:
        HR (BPM) = 60 / (Cycle Length [units] * time_unit_seconds)
    """
    if mean_cycle_length is None or mean_cycle_length <= 0.0:
        return {
            "bpm": None,
            "cycle_length_units": None,
            "cycle_length_ms": None,
            "time_unit_seconds": time_unit_seconds,
            "is_meaningful": False,
            "classification": "Asystole / No Sustained Ventricular Rhythm",
            "explanation": "Insufficient or absent ventricular activation sequence to compute heart rate.",
        }

    cl_seconds = mean_cycle_length * time_unit_seconds
    cl_ms = cl_seconds * 1000.0
    bpm = 60.0 / cl_seconds

    # Clinical classification analogues
    if bpm > 100.0:
        classification = "Tachycardia-Like Rate (> 100 BPM analogue)"
    elif bpm < 60.0:
        classification = "Bradycardia-Like Rate (< 60 BPM analogue)"
    else:
        classification = "Normal-Range Rate (60 - 100 BPM analogue)"

    return {
        "bpm": round(bpm, 1),
        "cycle_length_units": round(mean_cycle_length, 2),
        "cycle_length_ms": round(cl_ms, 1),
        "time_unit_seconds": time_unit_seconds,
        "is_meaningful": True,
        "classification": classification,
        "explanation": f"Calculated from mean ventricular cycle length {mean_cycle_length:.2f} units using 1 unit = {time_unit_seconds*1000:.0f} ms.",
    }


def detect_missing_events(
    scenario_result: ScenarioResult,
) -> Dict[str, Any]:
    """
    Identifies expected vs missed impulses and conduction dropouts across scenarios:
    1. Sinus Pause: detects expected SA node impulses that were suppressed during the pause window.
    2. AV Block: detects atrial depolarizations that failed to propagate to downstream ventricles.
    """
    config = scenario_result.config
    activations = scenario_result.activations
    net = scenario_result.network

    missing_sa_impulses = []
    missing_downstream_events = []

    # 1. Sinus Pause Analysis: Expected SA node pacing events
    if config.scenario_type == ScenarioType.SINUS_PAUSE:
        pause_start = config.pause_start or 30.0
        pause_end = config.pause_end or 110.0
        pacing_interval = config.pacing_interval

        # Determine expected pulse times across simulation horizon
        num_expected_pulses = int(config.t_end // pacing_interval) + 1
        expected_times = [k * pacing_interval for k in range(num_expected_pulses) if k * pacing_interval < config.t_end]

        actual_sa_acts = activations.get(0, [])

        for t_exp in expected_times:
            # If expected time falls inside pause window, check if it occurred
            if pause_start <= t_exp <= pause_end:
                # Was an actual SA activation observed near t_exp?
                matched = any(abs(t_act - t_exp) < 5.0 for t_act in actual_sa_acts)
                if not matched:
                    missing_sa_impulses.append({
                        "expected_time": round(t_exp, 2),
                        "window": f"[{pause_start:.1f}, {pause_end:.1f}]",
                        "status": "Suppressed (Sinus Arrest / Pause)",
                    })

    # 2. AV Conduction Block Analysis: Atrial vs Downstream Ventricle Dropouts
    atrial_acts = activations.get(1, [])
    ventricle_acts = activations.get(5, [])

    if config.scenario_type == ScenarioType.AV_BLOCK or (len(atrial_acts) > len(ventricle_acts)):
        for beat_idx, t_atr in enumerate(atrial_acts):
            # Check if there is a corresponding ventricular beat within a reasonable conduction window
            # (conduction latency normal is ~13-16 units; window = 25 units)
            matching_vent = [tv for tv in ventricle_acts if 0.0 < (tv - t_atr) < 28.0]
            if not matching_vent:
                missing_downstream_events.append({
                    "atrial_beat_index": beat_idx + 1,
                    "atrial_time": round(t_atr, 2),
                    "blocked_downstream_target": "AVN -> His -> Purkinje -> Ventricle",
                    "reason": f"Conduction failure at junction (conductance = {config.av_conductance:.3f})",
                })

    return {
        "missing_sa_impulses": missing_sa_impulses,
        "missing_downstream_events": missing_downstream_events,
        "num_atrial_beats": len(atrial_acts),
        "num_ventricular_beats": len(ventricle_acts),
        "conduction_ratio": f"{len(atrial_acts)}:{len(ventricle_acts)}" if len(atrial_acts) > 0 else "0:0",
    }


def compute_true_ecg_lead_approximation(
    time: np.ndarray,
    V_matrix: np.ndarray,
    network: CardiacNetwork,
    lead_name: str = "Lead_II",
) -> Tuple[np.ndarray, str]:
    """
    Computes an idealized far-field Einthoven limb lead projection from net dipole current moments.

    Source-to-Electrode Physical Formulation:
        Let each edge (i, j) with gap-junction conductance g_ij carry an instantaneous intercellular current:
            I_ij(t) = g_ij * (v_j(t) - v_i(t))
        The cardiac dipole vector P(t) is the sum of all edge current dipole vectors:
            P(t) = Σ_{(i,j)} I_ij(t) * (r_j - r_i)
        Standard Einthoven Lead II projects P(t) along the 60° vector from right arm to left leg:
            V_Lead_II(t) = P(t) · [cos(60°), sin(60°)] = P_x(t) * 0.5 + P_y(t) * (√3 / 2)

    Assumptions & Limitations:
        1. Far-field dipole approximation: assumes the heart's spatial extent is small
           compared to distance to torso surface electrodes.
        2. Homogeneous, isotropic volume conductor: neglects inhomogeneous conductivities of
           lungs (low conductivity), intracardiac blood pools (high conductivity), ribs, and skin.
        3. Quasi-static assumption: neglects capacitive displacement currents (valid for frequencies < 1 kHz).
        4. Not a clinical 12-lead diagnostic ECG: educational approximation only.
    """
    num_steps = len(time)
    lead_trace = np.zeros(num_steps, dtype=float)

    # Lead projection vectors in normalized coordinates
    if lead_name == "Lead_I":
        # Horizontal axis from RA to LA (180° to 0°)
        unit_vec = np.array([1.0, 0.0])
    elif lead_name == "Lead_III":
        # Axis from LA to LL (120°)
        unit_vec = np.array([-0.5, np.sqrt(3) / 2.0])
    else:
        # Default Lead II: Axis from RA to LL (60°)
        unit_vec = np.array([0.5, np.sqrt(3) / 2.0])

    for edge in network.edges:
        if not edge.is_active or edge.conductance <= 0.0:
            continue
        u, v = edge.source_id, edge.target_id
        node_u, node_v = network.nodes[u], network.nodes[v]

        dx = node_v.pos_x - node_u.pos_x
        dy = node_v.pos_y - node_u.pos_y

        v_diff = V_matrix[:, v] - V_matrix[:, u]
        edge_current = edge.conductance * v_diff

        # Dipole moment projection along lead vector
        proj = dx * unit_vec[0] + dy * unit_vec[1]
        lead_trace += edge_current * proj

    # Light smoothing for visual continuity
    kernel_size = 5
    if len(lead_trace) > kernel_size:
        kernel = np.ones(kernel_size) / kernel_size
        lead_trace = np.convolve(lead_trace, kernel, mode="same")

    doc_text = (
        f"Einthoven {lead_name} Dipole Approximation: Formulated by summing intercellular current dipoles "
        f"P(t) = Σ g_ij (v_j - v_i) Δr_ij and projecting onto unit vector {unit_vec.round(3)}. "
        "Assumes a homogeneous, isotropic, quasi-static torso volume conductor. Non-diagnostic."
    )

    return lead_trace, doc_text


def compare_with_normal_rhythm(
    scenario_result: ScenarioResult,
    baseline_result: Optional[ScenarioResult] = None,
) -> Dict[str, Any]:
    """
    Generates a rigorous quantitative comparison of the given scenario against Normal Sinus Rhythm.
    """
    if baseline_result is None:
        # Run standard baseline normal configuration
        norm_cfg = ScenarioConfig(
            scenario_type=ScenarioType.NORMAL,
            pacing_interval=45.0,
            t_end=scenario_result.config.t_end,
            dt=scenario_result.config.dt,
        )
        baseline_result = run_scenario(norm_cfg)

    # Ventricular cycle lengths
    cl_scen = scenario_result.mean_cycle_length
    cl_norm = baseline_result.mean_cycle_length

    # Activation rates (BPM)
    hr_scen = estimate_heart_rate_bpm(cl_scen)
    hr_norm = estimate_heart_rate_bpm(cl_norm)

    # Conduction integrity
    acts_scen_vent = scenario_result.activations.get(5, [])
    acts_norm_vent = baseline_result.activations.get(5, [])

    # Latencies
    lat_scen = scenario_result.latencies.get("regional_summary", {}).get("Total_SAN_to_Ventricle")
    lat_norm = baseline_result.latencies.get("regional_summary", {}).get("Total_SAN_to_Ventricle")

    rate_ratio = (scenario_result.activation_frequency / baseline_result.activation_frequency) if baseline_result.activation_frequency > 0 else 0.0

    return {
        "scenario_name": scenario_result.scenario_type.value,
        "baseline_name": baseline_result.scenario_type.value,
        "scenario_cycle_length": cl_scen,
        "baseline_cycle_length": cl_norm,
        "cycle_length_delta": (cl_scen - cl_norm) if (cl_scen and cl_norm) else None,
        "scenario_bpm": hr_scen["bpm"],
        "baseline_bpm": hr_norm["bpm"],
        "rate_ratio": round(rate_ratio, 2),
        "scenario_vent_beats": len(acts_scen_vent),
        "baseline_vent_beats": len(acts_norm_vent),
        "scenario_total_latency": lat_scen,
        "baseline_total_latency": lat_norm,
        "conduction_success": bool(len(acts_scen_vent) > 0 and len(scenario_result.failures) == 0),
        "summary": scenario_result.comparison_with_normal.get("status", ""),
    }
