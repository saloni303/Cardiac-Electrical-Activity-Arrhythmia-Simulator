"""
Stage 8: Parameter Experiment Laboratory Module.

Enables controlled, rigorous electrophysiological experimentation on the coupled
FitzHugh-Nagumo cardiac network and single-cell dynamical system.

Workflow:
    1. Configure and run a baseline scenario.
    2. Save baseline parameters, dynamical states, and activation sequences.
    3. Modify a controlled parameter (I, ε, a, b, D, T_pace, g_AV, or region-specific parameters).
    4. Run trial simulation.
    5. Side-by-side comparative analysis of:
        - Heart activation sequences and activation timestamps.
        - Conduction latencies and block / dropout failures.
        - Transmembrane voltages v_i(t) and recovery variables w_i(t).
        - Activation-derived rhythm waveforms Φ(t).
        - Single-cell equilibrium coordinates, analytical Jacobians, and stability.
    6. Automated parameter sweep engine across selectable ranges.
    7. CSV data export and scientifically honest dynamical interpretations.
"""

from dataclasses import dataclass, field
import io
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from .model import FHNParameters
from .network import (
    CardiacNetwork,
    NetworkNode,
    RegionType,
    create_default_cardiac_network,
    solve_network,
)
from .analysis import (
    find_equilibria,
    compute_jacobian,
    compute_stability_metrics,
    classify_equilibrium,
    EquilibriumPoint,
    StabilityClass,
)
from .activation import (
    detect_activations,
    compute_propagation_latencies,
    detect_conduction_failures,
)
from .scenarios import (
    ScenarioConfig,
    ScenarioType,
    ScenarioResult,
    run_scenario,
    compute_rhythm_trace,
    make_pacing_protocol,
)
from .rhythm_analysis import estimate_heart_rate_bpm


@dataclass
class ExperimentParameters:
    """
    Complete configuration container for controlled cardiac electrophysiology experiments.
    """
    scenario_type: ScenarioType = ScenarioType.NORMAL
    pacing_interval: float = 45.0
    stimulus_amplitude: float = 1.8
    pulse_duration: float = 1.5
    pause_start: Optional[float] = None
    pause_end: Optional[float] = None
    av_conductance: float = 0.35
    global_coupling: float = 0.70
    t_end: float = 120.0
    dt: float = 0.02
    cell_a: float = 0.70
    cell_b: float = 0.80
    cell_epsilon: float = 0.08
    cell_I_bias: float = 0.0
    # Region-specific overrides: {"SAN": {"epsilon": 0.08}, "AVN": {"epsilon": 0.04}, ...}
    region_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def copy(self) -> "ExperimentParameters":
        """Creates a deep copy of the parameter set."""
        import copy
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Converts parameters to a flat dictionary for display and export."""
        d = {
            "scenario_type": self.scenario_type.value,
            "pacing_interval": self.pacing_interval,
            "stimulus_amplitude": self.stimulus_amplitude,
            "av_conductance": self.av_conductance,
            "global_coupling": self.global_coupling,
            "cell_a": self.cell_a,
            "cell_b": self.cell_b,
            "cell_epsilon": self.cell_epsilon,
            "cell_I_bias": self.cell_I_bias,
            "t_end": self.t_end,
            "dt": self.dt,
        }
        if self.pause_start is not None:
            d["pause_start"] = self.pause_start
        if self.pause_end is not None:
            d["pause_end"] = self.pause_end
        for reg, overrides in self.region_overrides.items():
            for p_name, p_val in overrides.items():
                d[f"{reg}_{p_name}"] = p_val
        return d


def build_experiment_network(params: ExperimentParameters) -> CardiacNetwork:
    """
    Constructs a 6-compartment cardiac network with explicit per-compartment parameters,
    junction conductances, and global diffusive coupling.
    """
    net = CardiacNetwork(global_coupling=params.global_coupling)

    # Standard compartments
    compartments = [
        (0, "SAN", RegionType.SA_NODE, 0.75, 0.90),
        (1, "Atrium", RegionType.ATRIUM, 0.50, 0.75),
        (2, "AVN", RegionType.AV_NODE, 0.50, 0.55),
        (3, "His", RegionType.HIS_BUNDLE, 0.50, 0.40),
        (4, "Purkinje", RegionType.PURKINJE, 0.35, 0.22),
        (5, "Ventricle", RegionType.VENTRICLE, 0.50, 0.08),
    ]

    for node_id, name, region, px, py in compartments:
        # Base cellular parameters
        a_val = params.cell_a
        b_val = params.cell_b
        eps_val = params.cell_epsilon
        I_val = params.cell_I_bias

        # Apply regional overrides if defined
        if name in params.region_overrides:
            ov = params.region_overrides[name]
            a_val = ov.get("a", a_val)
            b_val = ov.get("b", b_val)
            eps_val = ov.get("epsilon", eps_val)
            I_val = ov.get("I", I_val)

        net.add_node(NetworkNode(
            node_id=node_id,
            name=name,
            region=region,
            params=FHNParameters(a=a_val, b=b_val, epsilon=eps_val, I=I_val),
            pos_x=px,
            pos_y=py,
        ))

    # Add conduction pathways
    net.add_edge(0, 1, conductance=0.35)                     # SA -> Atrium
    net.add_edge(1, 2, conductance=params.av_conductance)     # Atrium -> AV Node (Configurable)
    net.add_edge(2, 3, conductance=0.35)                     # AV Node -> His
    net.add_edge(3, 4, conductance=0.45)                     # His -> Purkinje
    net.add_edge(4, 5, conductance=0.50)                     # Purkinje -> Ventricle

    return net


def run_experiment_simulation(params: ExperimentParameters) -> ScenarioResult:
    """
    Simulates the cardiac conduction network for a given parameter set.
    """
    net = build_experiment_network(params)

    cfg = ScenarioConfig(
        scenario_type=params.scenario_type,
        pacing_interval=params.pacing_interval,
        pulse_amplitude=params.stimulus_amplitude,
        pulse_duration=params.pulse_duration,
        pause_start=params.pause_start,
        pause_end=params.pause_end,
        av_conductance=params.av_conductance,
        global_coupling=params.global_coupling,
        t_end=params.t_end,
        dt=params.dt,
        custom_network=net,
    )

    return run_scenario(cfg)


def compute_conduction_latency_san_to_ventricle(
    activations: Dict[int, List[float]],
    pacing_interval: float,
) -> Optional[float]:
    """
    Calculates the mean transit time from SA-node activation (node 0)
    to ventricular activation (node 5).
    """
    san_acts = activations.get(0, [])
    vent_acts = activations.get(5, [])

    if not san_acts or not vent_acts:
        return None

    latencies = []
    for t_san in san_acts:
        # Look for earliest ventricular activation after t_san within one pacing cycle + buffer
        matching = [t_v for t_v in vent_acts if t_san < t_v < t_san + pacing_interval * 1.5]
        if matching:
            latencies.append(matching[0] - t_san)

    if not latencies:
        return None

    return float(np.mean(latencies))


@dataclass
class ExperimentComparisonResult:
    """
    Comprehensive comparative analysis of Baseline (Condition A) vs Trial (Condition B).
    """
    baseline_params: ExperimentParameters
    trial_params: ExperimentParameters
    baseline_result: ScenarioResult
    trial_result: ScenarioResult
    changed_parameters: List[Dict[str, Any]]
    single_cell_baseline_eq: Optional[EquilibriumPoint]
    single_cell_trial_eq: Optional[EquilibriumPoint]
    bpm_baseline: Optional[float]
    bpm_trial: Optional[float]
    mean_cl_baseline: Optional[float]
    mean_cl_trial: Optional[float]
    san_to_vent_latency_baseline: Optional[float]
    san_to_vent_latency_trial: Optional[float]
    conduction_drops_baseline: int
    conduction_drops_trial: int
    conduction_status_baseline: str
    conduction_status_trial: str
    is_arrhythmia_induced: bool
    scientific_interpretation: str


def identify_changed_parameters(
    base: ExperimentParameters,
    trial: ExperimentParameters,
) -> List[Dict[str, Any]]:
    """
    Compares baseline and trial parameter dataclasses and identifies all differences.
    """
    diffs = []
    keys = [
        ("scenario_type", "Arrhythmia Scenario", False),
        ("pacing_interval", "SA Pacing Interval (T_pace)", True),
        ("stimulus_amplitude", "Stimulus Current Amplitude (I)", True),
        ("av_conductance", "AV Nodal Conductance (g_AV)", True),
        ("global_coupling", "Global Electrical Coupling (D)", True),
        ("cell_a", "Global Parameter a", True),
        ("cell_b", "Global Parameter b", True),
        ("cell_epsilon", "Global Recovery Timescale (ε)", True),
        ("cell_I_bias", "Cellular Bias Current (I_bias)", True),
        ("pause_start", "Sinus Pause Start", True),
        ("pause_end", "Sinus Pause End", True),
    ]

    for field_name, label, is_num in keys:
        v_base = getattr(base, field_name)
        v_trial = getattr(trial, field_name)
        if v_base != v_trial:
            diff_val = (v_trial - v_base) if (is_num and v_base is not None and v_trial is not None) else "N/A"
            diffs.append({
                "field": field_name,
                "label": label,
                "baseline": v_base,
                "trial": v_trial,
                "difference": diff_val,
            })

    # Check region overrides
    all_regions = set(list(base.region_overrides.keys()) + list(trial.region_overrides.keys()))
    for reg in sorted(all_regions):
        ov_base = base.region_overrides.get(reg, {})
        ov_trial = trial.region_overrides.get(reg, {})
        for p_name in ["a", "b", "epsilon", "I"]:
            val_b = ov_base.get(p_name, getattr(base, f"cell_{p_name}" if p_name != "I" else "cell_I_bias"))
            val_t = ov_trial.get(p_name, getattr(trial, f"cell_{p_name}" if p_name != "I" else "cell_I_bias"))
            if abs(val_b - val_t) > 1e-6:
                diffs.append({
                    "field": f"{reg}_{p_name}",
                    "label": f"Region {reg}: {p_name}",
                    "baseline": val_b,
                    "trial": val_t,
                    "difference": val_t - val_b,
                })

    return diffs


def generate_scientific_interpretation(
    diffs: List[Dict[str, Any]],
    bpm_base: Optional[float],
    bpm_trial: Optional[float],
    cl_base: Optional[float],
    cl_trial: Optional[float],
    lat_base: Optional[float],
    lat_trial: Optional[float],
    drops_base: int,
    drops_trial: int,
    eq_base: Optional[EquilibriumPoint],
    eq_trial: Optional[EquilibriumPoint],
) -> Tuple[bool, str]:
    """
    Synthesizes a short, rigorous scientific interpretation based on numerical data.
    Explicitly states when rhythm remains unchanged without falsely implying arrhythmia.
    """
    is_arrhythmia = False
    notes = []

    # 1. Conduction Block Check
    if drops_trial > drops_base and (bpm_trial is None or bpm_trial < 1e-3):
        is_arrhythmia = True
        notes.append(
            f"**Severe Conduction Block Detected:** Trial simulation exhibited {drops_trial} dropped activation "
            f"events (vs {drops_base} in baseline). Conduction failed downstream; ventricles were completely uncoupled. "
            f"This occurs because the electrotonic coupling current fell below the excitation threshold barrier."
        )
    elif drops_trial > drops_base:
        is_arrhythmia = True
        notes.append(
            f"**Partial Conduction Dropout:** Trial exhibited {drops_trial} dropped beats (intermittent conduction failure), "
            f"analogous to high-grade second-degree AV block."
        )

    # 2. Rate / Frequency Dynamics
    if bpm_base is not None and bpm_trial is not None:
        bpm_diff = bpm_trial - bpm_base
        if bpm_diff > 10.0:
            notes.append(
                f"**Frequency Acceleration (Tachycardia-Like):** Ventricular rate accelerated by {bpm_diff:+.1f} BPM "
                f"({bpm_base:.1f} → {bpm_trial:.1f} BPM). Decreased pacing interval or increased excitability "
                f"shortened the cycle length."
            )
        elif bpm_diff < -10.0:
            notes.append(
                f"**Frequency Deceleration (Bradycardia-Like):** Ventricular rate decelerated by {bpm_diff:+.1f} BPM "
                f"({bpm_base:.1f} → {bpm_trial:.1f} BPM). Prolonged quiescent recovery periods were maintained between activations."
            )
        elif abs(bpm_diff) <= 2.0 and drops_trial == 0:
            notes.append(
                f"**Physiological Homeostasis Maintained:** Ventricular rate remained virtually constant "
                f"({bpm_base:.1f} vs {bpm_trial:.1f} BPM). Conduction remained 1:1 synchronous."
            )

    # 3. Conduction Latency (Velocity)
    if lat_base is not None and lat_trial is not None and not is_arrhythmia:
        lat_diff = lat_trial - lat_base
        if abs(lat_diff) > 0.5:
            pct = (lat_diff / lat_base) * 100.0
            direction = "prolonged (slowed velocity)" if lat_diff > 0 else "shortened (accelerated velocity)"
            notes.append(
                f"**Conduction Velocity Shift:** Atrioventricular transit latency was {direction} by {abs(lat_diff):.2f} time units "
                f"({pct:+.1f}%). 1:1 synchrony was preserved without conduction block."
            )

    # 4. Single-cell Equilibrium & Stability
    if eq_base and eq_trial:
        dv = eq_trial.v_star - eq_base.v_star
        if abs(dv) > 0.05 or eq_base.stability_class != eq_trial.stability_class:
            notes.append(
                f"**Cellular Dynamical Shift:** Single-cell equilibrium shifted from v* = {eq_base.v_star:.3f} "
                f"({eq_base.stability_class.value}) to v* = {eq_trial.v_star:.3f} ({eq_trial.stability_class.value}). "
                f"Jacobian trace τ = {eq_trial.trace:.4f}, determinant Δ = {eq_trial.determinant:.4f}."
            )

    if not notes:
        notes.append(
            "**No Significant Electrophysiological Disruption:** The modified parameters remained within the "
            "tissue's normal physiological operating basin. Action potential generation, propagation latencies, "
            "and ventricular activation rate were preserved. No arrhythmia induced."
        )

    return is_arrhythmia, " ".join(notes)


def compare_experiments(
    baseline_params: ExperimentParameters,
    trial_params: ExperimentParameters,
    baseline_res: Optional[ScenarioResult] = None,
    trial_res: Optional[ScenarioResult] = None,
) -> ExperimentComparisonResult:
    """
    Executes and compares two controlled simulation runs (Condition A vs Condition B).
    """
    if baseline_res is None:
        baseline_res = run_experiment_simulation(baseline_params)
    if trial_res is None:
        trial_res = run_experiment_simulation(trial_params)

    # Metrics
    hr_base = estimate_heart_rate_bpm(baseline_res.mean_cycle_length)
    hr_trial = estimate_heart_rate_bpm(trial_res.mean_cycle_length)
    bpm_base = hr_base.get("bpm")
    bpm_trial = hr_trial.get("bpm")

    cl_base = baseline_res.mean_cycle_length
    cl_trial = trial_res.mean_cycle_length

    lat_base = compute_conduction_latency_san_to_ventricle(baseline_res.activations, baseline_params.pacing_interval)
    lat_trial = compute_conduction_latency_san_to_ventricle(trial_res.activations, trial_params.pacing_interval)

    drops_base = len(baseline_res.failures)
    drops_trial = len(trial_res.failures)

    # Standalone single cell equilibrium
    p_cell_base = FHNParameters(a=baseline_params.cell_a, b=baseline_params.cell_b, epsilon=baseline_params.cell_epsilon, I=baseline_params.cell_I_bias)
    p_cell_trial = FHNParameters(a=trial_params.cell_a, b=trial_params.cell_b, epsilon=trial_params.cell_epsilon, I=trial_params.cell_I_bias)

    eqs_base = find_equilibria(p_cell_base)
    eqs_trial = find_equilibria(p_cell_trial)
    eq_base = eqs_base[0] if eqs_base else None
    eq_trial = eqs_trial[0] if eqs_trial else None

    # Status strings
    status_base = "1:1 Synchronous" if drops_base == 0 else f"Block ({drops_base} drops)"
    status_trial = "1:1 Synchronous" if drops_trial == 0 else f"Block ({drops_trial} drops)"

    diffs = identify_changed_parameters(baseline_params, trial_params)
    is_arrhythmia, interp = generate_scientific_interpretation(
        diffs=diffs,
        bpm_base=bpm_base,
        bpm_trial=bpm_trial,
        cl_base=cl_base,
        cl_trial=cl_trial,
        lat_base=lat_base,
        lat_trial=lat_trial,
        drops_base=drops_base,
        drops_trial=drops_trial,
        eq_base=eq_base,
        eq_trial=eq_trial,
    )

    return ExperimentComparisonResult(
        baseline_params=baseline_params,
        trial_params=trial_params,
        baseline_result=baseline_res,
        trial_result=trial_res,
        changed_parameters=diffs,
        single_cell_baseline_eq=eq_base,
        single_cell_trial_eq=eq_trial,
        bpm_baseline=bpm_base,
        bpm_trial=bpm_trial,
        mean_cl_baseline=cl_base,
        mean_cl_trial=cl_trial,
        san_to_vent_latency_baseline=lat_base,
        san_to_vent_latency_trial=lat_trial,
        conduction_drops_baseline=drops_base,
        conduction_drops_trial=drops_trial,
        conduction_status_baseline=status_base,
        conduction_status_trial=status_trial,
        is_arrhythmia_induced=is_arrhythmia,
        scientific_interpretation=interp,
    )


# =============================================================================
# AUTOMATED PARAMETER SWEEP ENGINE
# =============================================================================

@dataclass
class ParameterSweepPoint:
    """Represents simulation metrics for a single point in a parameter sweep."""
    param_value: float
    bpm: Optional[float]
    mean_cycle_length: Optional[float]
    conduction_success: bool
    conduction_drops: int
    san_to_vent_latency: Optional[float]
    equilibrium_v_star: float
    stability_class: str
    trace: float
    determinant: float
    is_oscillatory: bool


@dataclass
class ParameterSweepConfig:
    """Configuration for an automatic parameter sweep."""
    parameter_name: str  # "I", "epsilon", "a", "b", "D", "pacing_interval", "av_conductance", "region_avn_eps"
    range_min: float
    range_max: float
    num_steps: int = 15
    base_params: ExperimentParameters = field(default_factory=ExperimentParameters)


def run_parameter_sweep(
    sweep_cfg: ParameterSweepConfig,
) -> Tuple[List[ParameterSweepPoint], str]:
    """
    Executes an automated parameter sweep over a designated parameter range,
    recording outcomes and diagnosing dynamical transitions.
    """
    values = np.linspace(sweep_cfg.range_min, sweep_cfg.range_max, sweep_cfg.num_steps)
    points: List[ParameterSweepPoint] = []

    for val in values:
        p = sweep_cfg.base_params.copy()
        # Set target parameter
        if sweep_cfg.parameter_name == "I":
            p.stimulus_amplitude = float(val)
        elif sweep_cfg.parameter_name == "epsilon":
            p.cell_epsilon = float(val)
        elif sweep_cfg.parameter_name == "a":
            p.cell_a = float(val)
        elif sweep_cfg.parameter_name == "b":
            p.cell_b = float(val)
        elif sweep_cfg.parameter_name == "D":
            p.global_coupling = float(val)
        elif sweep_cfg.parameter_name == "pacing_interval":
            p.pacing_interval = float(val)
        elif sweep_cfg.parameter_name == "av_conductance":
            p.av_conductance = float(val)
        elif sweep_cfg.parameter_name == "region_avn_eps":
            p.region_overrides["AVN"] = {"epsilon": float(val)}
        elif sweep_cfg.parameter_name == "region_vent_eps":
            p.region_overrides["Ventricle"] = {"epsilon": float(val)}

        # Run simulation with moderate horizon for quick responsive sweeps
        p.t_end = min(sweep_cfg.base_params.t_end, 90.0)
        res = run_experiment_simulation(p)

        hr = estimate_heart_rate_bpm(res.mean_cycle_length)
        bpm = hr.get("bpm")
        drops = len(res.failures)
        success = (drops == 0 and bpm is not None and bpm > 0)
        lat = compute_conduction_latency_san_to_ventricle(res.activations, p.pacing_interval)

        # Standalone cell equilibrium
        p_cell = FHNParameters(a=p.cell_a, b=p.cell_b, epsilon=p.cell_epsilon, I=p.cell_I_bias)
        eqs = find_equilibria(p_cell)
        eq = eqs[0] if eqs else None

        points.append(ParameterSweepPoint(
            param_value=float(val),
            bpm=bpm,
            mean_cycle_length=res.mean_cycle_length,
            conduction_success=success,
            conduction_drops=drops,
            san_to_vent_latency=lat,
            equilibrium_v_star=eq.v_star if eq else 0.0,
            stability_class=eq.stability_class.value if eq else "Unknown",
            trace=eq.trace if eq else 0.0,
            determinant=eq.determinant if eq else 0.0,
            is_oscillatory=(eq.trace > 0 and eq.determinant > 0) if eq else False,
        ))

    # Generate sweep summary interpretation
    p_name = sweep_cfg.parameter_name
    valid_bpms = [pt.bpm for pt in points if pt.bpm is not None]
    valid_lats = [pt.san_to_vent_latency for pt in points if pt.san_to_vent_latency is not None]
    block_points = [pt.param_value for pt in points if not pt.conduction_success]

    interp_parts = [
        f"**Parameter Sweep Analysis ({p_name} ∈ [{sweep_cfg.range_min:.3f}, {sweep_cfg.range_max:.3f}]):**"
    ]

    if block_points:
        interp_parts.append(
            f"Conduction block observed for {len(block_points)} of {len(points)} tested steps "
            f"(e.g., at {p_name} ≈ {block_points[0]:.3f})."
        )
    else:
        interp_parts.append(
            "Conduction remained 1:1 robust across the entire scanned parameter range."
        )

    if valid_bpms:
        interp_parts.append(
            f"Observed ventricular rates ranged from {min(valid_bpms):.1f} to {max(valid_bpms):.1f} BPM."
        )

    if valid_lats:
        interp_parts.append(
            f"Atrioventricular transit latencies ranged from {min(valid_lats):.2f} to {max(valid_lats):.2f} time units."
        )

    return points, " ".join(interp_parts)


# =============================================================================
# CSV EXPORT HELPERS
# =============================================================================

def export_comparison_to_csv(comp: ExperimentComparisonResult) -> str:
    """
    Serializes baseline vs trial comparative metrics into a clean CSV format.
    """
    rows = []
    # 1. Changed parameters
    for diff in comp.changed_parameters:
        rows.append({
            "Category": "Parameter Changed",
            "Item": diff["label"],
            "State A (Baseline)": str(diff["baseline"]),
            "State B (Trial)": str(diff["trial"]),
            "Delta (B - A)": str(diff["difference"]),
        })

    # 2. Conduction & Rate metrics
    bpm_a = f"{comp.bpm_baseline:.1f} BPM" if comp.bpm_baseline is not None else "0.0 BPM (Blocked)"
    bpm_b = f"{comp.bpm_trial:.1f} BPM" if comp.bpm_trial is not None else "0.0 BPM (Blocked)"
    diff_bpm = f"{comp.bpm_trial - comp.bpm_baseline:+.1f} BPM" if (comp.bpm_baseline and comp.bpm_trial) else "N/A"

    cl_a = f"{comp.mean_cl_baseline:.2f} u" if comp.mean_cl_baseline is not None else "N/A"
    cl_b = f"{comp.mean_cl_trial:.2f} u" if comp.mean_cl_trial is not None else "N/A"
    diff_cl = f"{comp.mean_cl_trial - comp.mean_cl_baseline:+.2f} u" if (comp.mean_cl_baseline and comp.mean_cl_trial) else "N/A"

    lat_a = f"{comp.san_to_vent_latency_baseline:.2f} u" if comp.san_to_vent_latency_baseline is not None else "N/A"
    lat_b = f"{comp.san_to_vent_latency_trial:.2f} u" if comp.san_to_vent_latency_trial is not None else "N/A"
    diff_lat = f"{comp.san_to_vent_latency_trial - comp.san_to_vent_latency_baseline:+.2f} u" if (comp.san_to_vent_latency_baseline and comp.san_to_vent_latency_trial) else "N/A"

    rows.extend([
        {"Category": "Conduction Metric", "Item": "Ventricular Rate", "State A (Baseline)": bpm_a, "State B (Trial)": bpm_b, "Delta (B - A)": diff_bpm},
        {"Category": "Conduction Metric", "Item": "Mean Cycle Length", "State A (Baseline)": cl_a, "State B (Trial)": cl_b, "Delta (B - A)": diff_cl},
        {"Category": "Conduction Metric", "Item": "SAN-to-Ventricle Latency", "State A (Baseline)": lat_a, "State B (Trial)": lat_b, "Delta (B - A)": diff_lat},
        {"Category": "Conduction Metric", "Item": "Conduction Failures / Drops", "State A (Baseline)": str(comp.conduction_drops_baseline), "State B (Trial)": str(comp.conduction_drops_trial), "Delta (B - A)": f"{comp.conduction_drops_trial - comp.conduction_drops_baseline:+d}"},
        {"Category": "Conduction Metric", "Item": "Conduction Status", "State A (Baseline)": comp.conduction_status_baseline, "State B (Trial)": comp.conduction_status_trial, "Delta (B - A)": "Arrhythmia Induced" if comp.is_arrhythmia_induced else "Preserved"},
    ])

    # 3. Cellular Stability
    if comp.single_cell_baseline_eq and comp.single_cell_trial_eq:
        eq_a = comp.single_cell_baseline_eq
        eq_b = comp.single_cell_trial_eq
        rows.extend([
            {"Category": "Cellular Dynamics", "Item": "Equilibrium Voltage v*", "State A (Baseline)": f"{eq_a.v_star:.4f}", "State B (Trial)": f"{eq_b.v_star:.4f}", "Delta (B - A)": f"{eq_b.v_star - eq_a.v_star:+.4f}"},
            {"Category": "Cellular Dynamics", "Item": "Equilibrium Recovery w*", "State A (Baseline)": f"{eq_a.w_star:.4f}", "State B (Trial)": f"{eq_b.w_star:.4f}", "Delta (B - A)": f"{eq_b.w_star - eq_a.w_star:+.4f}"},
            {"Category": "Cellular Dynamics", "Item": "Jacobian Trace τ", "State A (Baseline)": f"{eq_a.trace:.4f}", "State B (Trial)": f"{eq_b.trace:.4f}", "Delta (B - A)": f"{eq_b.trace - eq_a.trace:+.4f}"},
            {"Category": "Cellular Dynamics", "Item": "Jacobian Determinant Δ", "State A (Baseline)": f"{eq_a.determinant:.4f}", "State B (Trial)": f"{eq_b.determinant:.4f}", "Delta (B - A)": f"{eq_b.determinant - eq_a.determinant:+.4f}"},
            {"Category": "Cellular Dynamics", "Item": "Stability Regime", "State A (Baseline)": eq_a.stability_class.value, "State B (Trial)": eq_b.stability_class.value, "Delta (B - A)": "Transitioned" if eq_a.stability_class != eq_b.stability_class else "Invariant"},
        ])

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def export_sweep_to_csv(points: List[ParameterSweepPoint], parameter_name: str) -> str:
    """
    Serializes multi-step parameter sweep results into CSV format.
    """
    data = []
    for pt in points:
        data.append({
            f"Param_{parameter_name}": pt.param_value,
            "Ventricular_BPM": pt.bpm if pt.bpm is not None else 0.0,
            "Mean_Cycle_Length": pt.mean_cycle_length if pt.mean_cycle_length is not None else 0.0,
            "Conduction_Success": pt.conduction_success,
            "Conduction_Drops": pt.conduction_drops,
            "SAN_to_Vent_Latency": pt.san_to_vent_latency if pt.san_to_vent_latency is not None else 0.0,
            "Equilibrium_v_star": pt.equilibrium_v_star,
            "Stability_Class": pt.stability_class,
            "Trace": pt.trace,
            "Determinant": pt.determinant,
            "Is_Oscillatory": pt.is_oscillatory,
        })
    df = pd.DataFrame(data)
    return df.to_csv(index=False)
