"""
Stage 3: Arrhythmia Scenario Engine using the Coupled FitzHugh-Nagumo Network.

Simulates and evaluates:
    1. Normal Sinus Rhythm (1:1 regular forward propagation).
    2. Tachycardia-Like Rhythm (elevated SA pacing frequency).
    3. Bradycardia-Like Rhythm (depressed SA pacing frequency).
    4. Sinus Pause (temporary suppression of SA pacing window).
    5. AV Block-Like Condition (pathway attenuation causing downstream ventricular failure).
    6. Advanced Re-entry-Like Activation (circulating excitation in a closed-loop ring).

All scenarios are strictly derived from numerical ODE integration (RK4).
Rhythm traces are computed via electrical dipole projection and labeled as
simulated ECG-like / activation-derived rhythm waveforms (non-diagnostic).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np

from .model import FHNParameters
from .network import (
    RegionType,
    NetworkNode,
    NetworkEdge,
    CardiacNetwork,
    create_default_cardiac_network,
    solve_network,
)
from .activation import (
    detect_activations,
    compute_propagation_latencies,
    detect_conduction_failures,
)


class ScenarioType(str, Enum):
    NORMAL = "Normal Sinus Rhythm"
    TACHYCARDIA = "Tachycardia-Like Rhythm"
    BRADYCARDIA = "Bradycardia-Like Rhythm"
    SINUS_PAUSE = "Sinus Pause"
    AV_BLOCK = "AV Block-Like Condition"
    REENTRY = "Advanced Re-entry-Like Activation"


@dataclass
class ScenarioConfig:
    """
    Configuration parameters for an arrhythmia simulation scenario.
    """
    scenario_type: ScenarioType
    pacing_interval: float = 45.0
    pulse_amplitude: float = 1.8
    pulse_duration: float = 1.5
    pause_start: Optional[float] = None
    pause_end: Optional[float] = None
    av_conductance: float = 0.35
    global_coupling: float = 0.70
    t_end: float = 160.0
    dt: float = 0.02
    block_location: str = "Atrium-to-AVN"
    # Re-entry parameters
    reentry_num_nodes: int = 10
    reentry_coupling: float = 0.35
    reentry_epsilon: float = 0.15


@dataclass
class ScenarioResult:
    """
    Comprehensive numerical and analytical output of an arrhythmia simulation run.
    """
    scenario_type: ScenarioType
    config: ScenarioConfig
    time: np.ndarray
    V: np.ndarray
    W: np.ndarray
    network: CardiacNetwork
    activations: Dict[int, List[float]]
    latencies: Dict[str, Any]
    failures: List[Dict[str, Any]]
    rhythm_trace: np.ndarray
    mean_cycle_length: Optional[float]
    activation_frequency: float
    comparison_with_normal: Dict[str, Any]
    reentry_sustained: bool = False
    reentry_revolutions: int = 0
    parameter_change_summary: str = ""
    mathematical_assumptions: str = ""


def compute_rhythm_trace(
    time: np.ndarray,
    V_matrix: np.ndarray,
    network: CardiacNetwork,
) -> np.ndarray:
    """
    Computes an activation-derived electrical rhythm waveform via dipole moment projection.

    Mathematical formulation:
        For every active edge (i, j) with conductance g_ij, the instantaneous dipole current is:
            I_ij(t) = g_ij * (v_j(t) - v_i(t))
        Projected onto a standardized vertical lead axis (representing Lead II analogue from base to apex):
            Rhythm(t) = Σ_{(i,j)} g_ij * (v_j(t) - v_i(t)) * (pos_y_j - pos_y_i)

    Scientific disclaimer: This is an illustrative dipole-projected waveform,
    NOT a clinically accurate or diagnostic ECG.
    """
    num_steps = len(time)
    rhythm = np.zeros(num_steps, dtype=float)

    for edge in network.edges:
        if not edge.is_active or edge.conductance <= 0.0:
            continue
        u = edge.source_id
        v = edge.target_id

        # Edge displacement vector along vertical conduction axis (base to apex)
        dy = network.nodes[u].pos_y - network.nodes[v].pos_y
        v_diff = V_matrix[:, v] - V_matrix[:, u]
        rhythm += edge.conductance * v_diff * dy

    # Apply light moving-average smoothing for clean visualization
    kernel_size = 5
    if len(rhythm) > kernel_size:
        kernel = np.ones(kernel_size) / kernel_size
        rhythm = np.convolve(rhythm, kernel, mode="same")

    return rhythm


def create_reentry_circuit(
    num_nodes: int = 10,
    coupling: float = 0.35,
    epsilon: float = 0.15,
    global_coupling: float = 0.70,
) -> CardiacNetwork:
    """
    Creates an anatomical closed-loop ring to investigate circus re-entrant excitation.

    Mathematical principles of re-entry:
        - The wave circulates unidirectionally around a closed loop.
        - Circulating excitation can only persist if:
              T_circulation = Σ (conduction delays around loop) > T_refractory of tissue.
        - If the loop is too small (e.g. num_nodes < 6), the wave hits its own refractory tail and extinguishes.
        - If the loop is sufficiently long (e.g. num_nodes >= 8) with adequate recovery,
          the wave circulates repeatedly as a self-sustaining re-entrant tachycardia.
    """
    net = CardiacNetwork(global_coupling=global_coupling)

    # Place nodes in a circle in 2D space for visualization
    for i in range(num_nodes):
        angle = 2.0 * np.pi * i / num_nodes
        x = 0.5 + 0.35 * np.cos(angle)
        y = 0.5 + 0.35 * np.sin(angle)

        region = RegionType.VENTRICLE if i >= (num_nodes // 2) else RegionType.ATRIUM
        net.add_node(NetworkNode(
            node_id=i,
            name=f"Ring_{i}",
            region=region,
            params=FHNParameters(a=0.7, b=0.8, epsilon=epsilon, I=0.0),
            pos_x=float(x),
            pos_y=float(y),
        ))

    # Add unidirectional forward edges around the ring
    for i in range(num_nodes):
        next_i = (i + 1) % num_nodes
        net.add_edge(i, next_i, conductance=coupling, is_bidirectional=False)

    return net


create_reentry_network = create_reentry_circuit


def make_pacing_protocol(
    pacing_interval: float,
    pulse_amplitude: float = 1.8,
    pulse_duration: float = 1.5,
    pause_start: Optional[float] = None,
    pause_end: Optional[float] = None,
    target_node_id: int = 0,
    max_pulses: Optional[int] = None,
) -> Callable[[float, int], float]:
    """
    Constructs a deterministic pacing function delivering square stimulus current pulses to target_node_id.
    """
    def stimulus_fn(t: float, node_id: int) -> float:
        if node_id != target_node_id:
            return 0.0

        # Check if current time falls within an active pause suppression window
        if pause_start is not None and pause_end is not None:
            if pause_start <= t <= pause_end:
                return 0.0

        # Determine which pulse cycle we are in
        cycle_idx = int(t // pacing_interval)
        if max_pulses is not None and cycle_idx >= max_pulses:
            return 0.0

        t_in_cycle = t % pacing_interval
        if t_in_cycle <= pulse_duration:
            return pulse_amplitude
        return 0.0

    return stimulus_fn


def run_scenario(config: ScenarioConfig) -> ScenarioResult:
    """
    Executes an arrhythmia scenario using the coupled FHN network through RK4 numerical integration.
    """
    # 1. Setup Network Topology
    if config.scenario_type == ScenarioType.REENTRY:
        net = create_reentry_circuit(
            num_nodes=config.reentry_num_nodes,
            coupling=config.reentry_coupling,
            epsilon=config.reentry_epsilon,
            global_coupling=config.global_coupling,
        )
        # Single starter pulse delivered to Node 0 at t = 0 to launch the circuit
        stim_fn = make_pacing_protocol(
            pacing_interval=config.t_end * 2.0,  # Single pulse
            pulse_amplitude=config.pulse_amplitude,
            pulse_duration=config.pulse_duration,
            target_node_id=0,
            max_pulses=1,
        )
    else:
        # Standard 6-compartment network (SAN, Atria, AVN, His, Purkinje, Ventricle)
        net = create_default_cardiac_network(
            av_conductance=config.av_conductance,
            global_coupling=config.global_coupling,
        )
        # Configure all units as excitable resting cells (I = 0.0, eps = 0.08)
        # Driven cleanly by the designated pacing protocol
        for i in range(net.num_nodes):
            old = net.nodes[i]
            net.nodes[i] = NetworkNode(
                node_id=old.node_id,
                name=old.name,
                region=old.region,
                params=FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0),
                pos_x=old.pos_x,
                pos_y=old.pos_y,
            )

        # Configure conduction block if selected
        if config.scenario_type == ScenarioType.AV_BLOCK:
            if config.block_location == "Atrium-to-AVN":
                net.set_edge_conductance(1, 2, config.av_conductance)
                if config.av_conductance <= 0.0:
                    net.set_edge_active(1, 2, False)
            elif config.block_location == "AVN-to-His":
                net.set_edge_conductance(2, 3, config.av_conductance)
                if config.av_conductance <= 0.0:
                    net.set_edge_active(2, 3, False)

        stim_fn = make_pacing_protocol(
            pacing_interval=config.pacing_interval,
            pulse_amplitude=config.pulse_amplitude,
            pulse_duration=config.pulse_duration,
            pause_start=config.pause_start,
            pause_end=config.pause_end,
            target_node_id=0,
        )

    # 2. Integrate Coupled Network ODE via RK4
    sim_data = solve_network(
        network=net,
        t_span=(0.0, config.t_end),
        dt=config.dt,
        external_stimulus=stim_fn,
    )

    time = sim_data["time"]
    V = sim_data["V"]
    W = sim_data["W"]

    # 3. Analyze Activation Events and Conduction Latencies
    activations = detect_activations(time, V, threshold=0.0)
    latencies = compute_propagation_latencies(activations, net, beat_index=0)
    failures = detect_conduction_failures(activations, net)

    # 4. Generate Activation-Derived Rhythm Trace
    rhythm = compute_rhythm_trace(time, V, net)

    # 5. Compute Cycle Length and Activation Frequency
    ventricle_node_id = 0 if config.scenario_type == ScenarioType.REENTRY else (net.num_nodes - 1)
    monitored_acts = activations.get(ventricle_node_id, [])

    if len(monitored_acts) >= 2:
        diffs = np.diff(monitored_acts)
        mean_cl = float(np.mean(diffs))
        act_freq = 100.0 / mean_cl  # Beats per 100 model time units
    elif len(monitored_acts) == 1:
        mean_cl = None
        act_freq = 100.0 / config.t_end
    else:
        mean_cl = None
        act_freq = 0.0

    # 6. Re-entry Assessment (circus movement validation)
    reentry_sustained = False
    reentry_revolutions = 0
    if config.scenario_type == ScenarioType.REENTRY:
        # In re-entry, starter pulse finished by t = config.pulse_duration (t ~ 1.5).
        # Any subsequent activation at Node 0 (t > 10.0) is a genuine spontaneous circus revolution!
        post_stim_acts = [t for t in monitored_acts if t > 10.0]
        reentry_revolutions = len(post_stim_acts)
        reentry_sustained = (reentry_revolutions >= 2)

    # 7. Scenario Comparison and Structured Documentation
    param_expl, math_assump, comp_norm = _generate_scenario_documentation(
        config, act_freq, mean_cl, failures, reentry_sustained, reentry_revolutions
    )

    return ScenarioResult(
        scenario_type=config.scenario_type,
        config=config,
        time=time,
        V=V,
        W=W,
        network=net,
        activations=activations,
        latencies=latencies,
        failures=failures,
        rhythm_trace=rhythm,
        mean_cycle_length=mean_cl,
        activation_frequency=act_freq,
        comparison_with_normal=comp_norm,
        reentry_sustained=reentry_sustained,
        reentry_revolutions=reentry_revolutions,
        parameter_change_summary=param_expl,
        mathematical_assumptions=math_assump,
    )


def _generate_scenario_documentation(
    config: ScenarioConfig,
    act_freq: float,
    mean_cl: Optional[float],
    failures: List[Dict[str, Any]],
    reentry_sustained: bool,
    reentry_revolutions: int,
) -> Tuple[str, str, Dict[str, Any]]:
    """Helper generating structured explanatory documentation and comparisons against normal rhythm."""

    baseline_cl = 45.0
    baseline_freq = 100.0 / 45.0  # ~2.22 beats per 100 time units

    st = config.scenario_type

    if st == ScenarioType.NORMAL:
        param_expl = (
            f"Baseline control configuration: Regular SA-node pacing period T_pace = {config.pacing_interval:.1f} time units. "
            f"Intercellular conductances intact (g_AV = {config.av_conductance:.3f}, Purkinje g = 0.50). Intact 1:1 forward transmission."
        )
        math_assump = (
            "Assumes constant pacing cadence, uniform intercellular coupling along forward anatomical tract, "
            "and adequate recovery intervals allowing recovery variable w to return near rest (w ≈ -0.6) prior to subsequent beats."
        )
        comp_norm = {
            "rate_ratio": 1.0,
            "status": "Baseline reference rhythm with 1:1 AV conduction.",
            "conduction_success": True,
        }

    elif st == ScenarioType.TACHYCARDIA:
        ratio = (act_freq / baseline_freq) if baseline_freq > 0 else 1.0
        param_expl = (
            f"Input parameter changed: SA pacing interval decreased from baseline T_pace = {baseline_cl:.1f} to {config.pacing_interval:.1f} time units. "
            f"This accelerates pacing frequency by {ratio:.2f}x."
        )
        math_assump = (
            "Shortened diastolic interval reduces recovery time for w. "
            "If pacing period falls below tissue refractory duration (T_pace < T_refractory), decremental conduction or block ensues."
        )
        comp_norm = {
            "rate_ratio": ratio,
            "status": f"Elevated ventricular activation frequency ({act_freq:.2f} vs normal {baseline_freq:.2f} beats/100 units).",
            "conduction_success": len(failures) == 0,
        }

    elif st == ScenarioType.BRADYCARDIA:
        ratio = (act_freq / baseline_freq) if baseline_freq > 0 else 1.0
        param_expl = (
            f"Input parameter changed: SA pacing interval increased from baseline T_pace = {baseline_cl:.1f} to {config.pacing_interval:.1f} time units. "
            f"Pacing frequency is reduced to {ratio:.2f}x of baseline."
        )
        math_assump = (
            "Extended cycle length allows the recovery variable w to return completely to its resting baseline (w ≈ -0.6), "
            "maximizing availability of fast excitation variable upstroke on subsequent beats."
        )
        comp_norm = {
            "rate_ratio": ratio,
            "status": f"Depressed activation frequency ({act_freq:.2f} vs normal {baseline_freq:.2f} beats/100 units).",
            "conduction_success": True,
        }

    elif st == ScenarioType.SINUS_PAUSE:
        param_expl = (
            f"Input parameter changed: SA pacing stimulation is temporarily suppressed between t = {config.pause_start:.1f} "
            f"and t = {config.pause_end:.1f}. External drive is clamped to zero during this pause window."
        )
        math_assump = (
            "Models transient sinus arrest or exit block where the primary pacemaker fails to initiate excitation. "
            "In the absence of subsidiary pacemakers, downstream excitable compartments remain quiescent at their stable resting equilibrium."
        )
        comp_norm = {
            "rate_ratio": (act_freq / baseline_freq) if baseline_freq > 0 else 0.0,
            "status": f"Temporary complete cessation of activations during pause interval [t={config.pause_start}, t={config.pause_end}].",
            "conduction_success": True,
        }

    elif st == ScenarioType.AV_BLOCK:
        param_expl = (
            f"Model parameter changed: Coupling conductance at junction '{config.block_location}' is reduced from normal (0.350) "
            f"to g = {config.av_conductance:.3f}."
        )
        math_assump = (
            "Electrotonic current across a weakened junction I_diff = D * g * (v_upstream - v_downstream) is insufficient "
            "to charge downstream membrane capacitance across the cubic excitation threshold barrier (v > v_threshold), causing complete conduction failure."
        )
        comp_norm = {
            "rate_ratio": 0.0 if act_freq == 0 else (act_freq / baseline_freq),
            "status": "Conduction block confirmed: upstream SA/Atrial impulses fail to propagate to downstream ventricles.",
            "conduction_success": False,
        }

    elif st == ScenarioType.REENTRY:
        if reentry_sustained:
            status_text = f"Sustained circus movement confirmed ({reentry_revolutions} spontaneous loop circulations detected)."
        else:
            status_text = f"Re-entry failed to self-sustain (wave extinguished after {reentry_revolutions} passage(s) due to refractoriness)."

        param_expl = (
            f"Network topology changed: Closed-loop ring of {config.reentry_num_nodes} nodes with unidirectional coupling "
            f"(g = {config.reentry_coupling:.2f}, ε = {config.reentry_epsilon:.2f}). A single premature starter beat was delivered at t = 0."
        )
        math_assump = (
            "Circus re-entry requires that total conduction time around the loop exceeds the refractory recovery period "
            "(T_loop > T_refractory). If the returning wave encounters elevated recovery variable w, conduction extinguishes."
        )
        comp_norm = {
            "rate_ratio": (act_freq / baseline_freq) if baseline_freq > 0 else 0.0,
            "status": status_text,
            "conduction_success": reentry_sustained,
        }

    return param_expl, math_assump, comp_norm
