"""
Activation Detection, Conduction Latency, and Conduction Failure Analysis.

Provides algorithmic detection of:
    - Cellular depolarization activation times (t_act).
    - Segment-to-segment conduction latencies.
    - Conduction blocks and failed propagation events.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from .network import CardiacNetwork, RegionType


def detect_activations(
    time: np.ndarray,
    V_matrix: np.ndarray,
    threshold: float = 0.0,
    min_refractory_period: float = 8.0,
) -> Dict[int, List[float]]:
    """
    Detects action potential depolarization activation times for every node in the network.

    An activation event is logged when the voltage crosses upward through `threshold`
    (dV/dt > 0), separated from any preceding activation by at least `min_refractory_period`.

    Parameters:
        time (np.ndarray): 1D array of time steps.
        V_matrix (np.ndarray): 2D array of shape (num_steps, num_nodes).
        threshold (float): Depolarization threshold crossing voltage (default: 0.0).
        min_refractory_period (float): Minimum duration before a subsequent beat can be logged.

    Returns:
        Dict[int, List[float]]: Map from node_id to list of activation timestamps [t_1, t_2, ...].
    """
    num_steps, num_nodes = V_matrix.shape
    activations: Dict[int, List[float]] = {i: [] for i in range(num_nodes)}

    dt = time[1] - time[0] if len(time) > 1 else 0.02

    for node_id in range(num_nodes):
        v_trace = V_matrix[:, node_id]
        last_act_time = -1e9

        for step in range(1, num_steps):
            t_curr = time[step]
            # Upward zero-crossing detection
            if v_trace[step - 1] <= threshold < v_trace[step]:
                if (t_curr - last_act_time) >= min_refractory_period:
                    # Linear interpolation for sub-step accuracy:
                    v_prev = v_trace[step - 1]
                    v_next = v_trace[step]
                    fraction = (threshold - v_prev) / (v_next - v_prev + 1e-15)
                    exact_t = time[step - 1] + fraction * dt

                    activations[node_id].append(float(exact_t))
                    last_act_time = exact_t

    return activations


def compute_propagation_latencies(
    activations: Dict[int, List[float]],
    network: CardiacNetwork,
    beat_index: int = 0,
) -> Dict[str, Any]:
    """
    Calculates edge-by-edge conduction latencies and overall propagation metrics for a specified beat.

    Returns:
        Dict containing:
            - 'edge_latencies': Map from (source, target) to latency Δt.
            - 'regional_summary': Key anatomical transitions (SA->Atria, Atria->AVN, etc.).
            - 'total_conduction_time': Time from SAN activation to Ventricular activation.
            - 'beat_available': True if beat was detected across monitored nodes.
    """
    edge_latencies: Dict[Tuple[int, int], Optional[float]] = {}
    n = network.num_nodes

    # Check each edge
    for edge in network.edges:
        src = edge.source_id
        tgt = edge.target_id

        src_acts = activations.get(src, [])
        tgt_acts = activations.get(tgt, [])

        if len(src_acts) > beat_index and len(tgt_acts) > beat_index:
            t_src = src_acts[beat_index]
            # Find the first target activation that occurs after source activation
            subsequent_tgt = [t for t in tgt_acts if t >= t_src]
            if subsequent_tgt:
                latency = subsequent_tgt[0] - t_src
                edge_latencies[(src, tgt)] = float(latency)
            else:
                edge_latencies[(src, tgt)] = None
        else:
            edge_latencies[(src, tgt)] = None

    # Regional summary for standard 6-region chain (0: SAN, 1: Atria, 2: AVN, 3: His, 4: Purkinje, 5: Ventricle)
    regional = {}
    if n >= 6:
        sa_act = activations[0][beat_index] if len(activations[0]) > beat_index else None
        atr_act = activations[1][beat_index] if len(activations[1]) > beat_index else None
        avn_act = activations[2][beat_index] if len(activations[2]) > beat_index else None
        his_act = activations[3][beat_index] if len(activations[3]) > beat_index else None
        purk_act = activations[4][beat_index] if len(activations[4]) > beat_index else None
        vent_act = activations[5][beat_index] if len(activations[5]) > beat_index else None

        regional["SA_to_Atrium"] = (atr_act - sa_act) if (atr_act and sa_act and atr_act >= sa_act) else None
        regional["Atrium_to_AVN"] = (avn_act - atr_act) if (avn_act and atr_act and avn_act >= atr_act) else None
        regional["AVN_to_His"] = (his_act - avn_act) if (his_act and avn_act and his_act >= avn_act) else None
        regional["His_to_Purkinje"] = (purk_act - his_act) if (purk_act and his_act and purk_act >= his_act) else None
        regional["Purkinje_to_Ventricle"] = (vent_act - purk_act) if (vent_act and purk_act and vent_act >= purk_act) else None
        regional["Total_SAN_to_Ventricle"] = (vent_act - sa_act) if (vent_act and sa_act and vent_act >= sa_act) else None

    return {
        "edge_latencies": edge_latencies,
        "regional_summary": regional,
        "beat_index": beat_index,
    }


def detect_conduction_failures(
    activations: Dict[int, List[float]],
    network: CardiacNetwork,
    max_expected_latency: float = 35.0,
) -> List[Dict[str, Any]]:
    """
    Identifies failed propagation events along all active pathways in the network.

    A failure is detected when an upstream node fires, but the downstream node
    fails to fire within `max_expected_latency` following the upstream activation.

    Returns:
        List of failure report dictionaries detailing location, timestamp, and severity.
    """
    failures = []

    for edge in network.edges:
        if not edge.is_active or edge.conductance <= 0.0:
            failures.append({
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "source_name": network.nodes[edge.source_id].name,
                "target_name": network.nodes[edge.target_id].name,
                "failure_type": "Pathway Severed / Anatomical Block",
                "reason": f"Edge conductance is zero or deactivated (conductance = {edge.conductance:.3f}).",
            })
            continue

        src_acts = activations.get(edge.source_id, [])
        tgt_acts = activations.get(edge.target_id, [])

        for act_idx, t_src in enumerate(src_acts):
            # Find downstream activations within reasonable propagation window [t_src, t_src + max_expected_latency]
            conducted = [t for t in tgt_acts if 0.0 <= (t - t_src) <= max_expected_latency]
            if not conducted:
                failures.append({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "source_name": network.nodes[edge.source_id].name,
                    "target_name": network.nodes[edge.target_id].name,
                    "source_time": t_src,
                    "beat_number": act_idx + 1,
                    "failure_type": "Functional Conduction Block",
                    "reason": (
                        f"{network.nodes[edge.source_id].name} depolarized at t = {t_src:.2f}, "
                        f"but downstream {network.nodes[edge.target_id].name} failed to activate within {max_expected_latency} units."
                    ),
                })

    return failures

