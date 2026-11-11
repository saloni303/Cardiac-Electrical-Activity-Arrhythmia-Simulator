"""
Unit tests for the coupled cardiac-cell network, emergent propagation, and conduction blocks.
"""

import numpy as np
import pytest
from fhn_core.model import FHNParameters
from fhn_core.network import (
    RegionType,
    NetworkNode,
    NetworkEdge,
    CardiacNetwork,
    create_default_cardiac_network,
    solve_network,
)
from fhn_core.activation import (
    detect_activations,
    compute_propagation_latencies,
    detect_conduction_failures,
)


def test_isolated_cells_uncoupled_dynamics():
    """
    Test 1: Isolated cells (uncoupled, D = 0).
    Verifies that SA node exhibits autorhythmic oscillation (limit cycle) while
    atrial and ventricular units remain quiescent at their stable resting equilibrium.
    """
    net = create_default_cardiac_network()
    net.global_coupling = 0.0  # Zero electrical coupling

    sim = solve_network(net, t_span=(0.0, 60.0), dt=0.02)
    time = sim["time"]
    V = sim["V"]

    # SA Node (node 0) is oscillatory (I = 0.50) -> must oscillate (max - min > 2.0)
    v_sa = V[:, 0]
    sa_excursion = np.max(v_sa) - np.min(v_sa)
    assert sa_excursion > 2.0, f"SA Node failed to oscillate when isolated (excursion {sa_excursion})"

    # Atrium (node 1) and Ventricle (node 5) are excitable resting units (I = 0.0)
    # Without coupling, they must remain quiescent near resting state (v ≈ -1.2)
    v_atrium = V[:, 1]
    v_ventricle = V[:, 5]

    assert (np.max(v_atrium) - np.min(v_atrium)) < 0.1, "Atrium erroneously oscillated without coupling"
    assert (np.max(v_ventricle) - np.min(v_ventricle)) < 0.1, "Ventricle erroneously oscillated without coupling"


def test_connected_cells_diffusive_equilibration():
    """
    Test 2: Connected cells with passive electrotonic flow.
    Two identical resting cells coupled with D > 0, one initialized at v = 1.0, other at v = -1.0.
    The diffusive coupling current D * (v2 - v1) must pull them towards each other.
    """
    net = CardiacNetwork(global_coupling=1.0)
    p = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)

    net.add_node(NetworkNode(0, "Cell_A", RegionType.ATRIUM, p))
    net.add_node(NetworkNode(1, "Cell_B", RegionType.ATRIUM, p))
    net.add_edge(0, 1, conductance=0.5)

    # Initial state: Cell A excited, Cell B polarized
    initial_v = np.array([1.0, -1.0])
    initial_w = np.array([0.0, 0.0])

    sim = solve_network(net, t_span=(0.0, 1.0), dt=0.01, initial_v=initial_v, initial_w=initial_w)
    V = sim["V"]

    # Initial difference is 2.0
    initial_diff = abs(V[0, 0] - V[0, 1])
    # After 1 unit of passive diffusion, the voltage difference must decrease
    final_diff = abs(V[-1, 0] - V[-1, 1])

    assert final_diff < initial_diff, f"Diffusive coupling failed to reduce voltage gradient: {final_diff} >= {initial_diff}"


def test_sequential_excitation_propagation():
    """
    Test 3: Stimulation propagation across the network.
    Verifies that pacemaking impulse from the SA node propagates sequentially
    through the conduction system:
        SAN -> Atrium -> AV Node -> His -> Purkinje -> Ventricle
    and that all propagation latencies are strictly positive (emergent wave delay).
    """
    net = create_default_cardiac_network(av_conductance=0.15, global_coupling=0.7)

    # Simulate for 50 time units (enough for initial wave propagation)
    sim = solve_network(net, t_span=(0.0, 50.0), dt=0.02)
    time = sim["time"]
    V = sim["V"]

    activations = detect_activations(time, V, threshold=0.0)

    # All 6 compartments must activate at least once
    for i in range(6):
        assert len(activations[i]) > 0, f"Node {i} ({net.nodes[i].name}) failed to activate"

    # Verify strictly sequential first-beat activation times:
    t_san = activations[0][0]
    t_atr = activations[1][0]
    t_avn = activations[2][0]
    t_his = activations[3][0]
    t_purk = activations[4][0]
    t_vent = activations[5][0]

    assert t_san < t_atr, f"Atrium fired before SAN: {t_atr} <= {t_san}"
    assert t_atr < t_avn, f"AV node fired before Atrium: {t_avn} <= {t_atr}"
    assert t_avn < t_his, f"His bundle fired before AV node: {t_his} <= {t_avn}"
    assert t_his < t_purk, f"Purkinje fired before His bundle: {t_purk} <= {t_his}"
    assert t_purk < t_vent, f"Ventricle fired before Purkinje: {t_vent} <= {t_purk}"

    latencies = compute_propagation_latencies(activations, net, beat_index=0)
    total_time = latencies["regional_summary"]["Total_SAN_to_Ventricle"]
    assert total_time is not None and total_time > 0.0, "Total conduction time must be positive"


def test_conduction_block_detection_on_severed_pathway():
    """
    Test 4: Conduction block and failed propagation.
    Sever the connection between Atrium and AV Node (set conductance = 0).
    Verifies that:
        1. SAN and Atrium still activate.
        2. AV Node, His, Purkinje, and Ventricle fail to activate.
        3. detect_conduction_failures flags the block at the AV junction.
    """
    net = create_default_cardiac_network(av_conductance=0.0, global_coupling=0.7)
    # Explicitly sever edge
    net.set_edge_active(1, 2, is_active=False)

    sim = solve_network(net, t_span=(0.0, 50.0), dt=0.02)
    time = sim["time"]
    V = sim["V"]

    activations = detect_activations(time, V, threshold=0.0)

    # SAN and Atrium must activate
    assert len(activations[0]) > 0, "SAN failed to activate"
    assert len(activations[1]) > 0, "Atrium failed to activate"

    # Downstream chambers must NOT activate
    assert len(activations[2]) == 0, "AV node activated despite severed pathway"
    assert len(activations[3]) == 0, "His bundle activated despite severed pathway"
    assert len(activations[4]) == 0, "Purkinje activated despite severed pathway"
    assert len(activations[5]) == 0, "Ventricle activated despite severed pathway"

    # Conduction failure analysis must detect the block
    failures = detect_conduction_failures(activations, net)
    assert len(failures) > 0, "Failed to detect conduction failure"

    # Check that block is at Atrium -> AVN junction
    blocked_targets = [f["target_name"] for f in failures]
    assert "AVN" in blocked_targets, f"Expected AVN to be blocked, got: {blocked_targets}"


def test_refractory_suppression_of_premature_stimulus():
    """
    Test 5: Recovery variable w mediates refractory behavior.
    If a cell is given an external current pulse while w is high (recovery phase),
    it fails to generate a full action potential.
    """
    net = CardiacNetwork(global_coupling=0.0)
    p = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0)
    net.add_node(NetworkNode(0, "Test_Cell", RegionType.VENTRICLE, p))

    # Case A: Stimulate from resting polarized state (v = -1.2, w = -0.6)
    def stimulus_resting(t, node_id):
        return 1.5 if (2.0 <= t <= 3.0) else 0.0

    sim_rest = solve_network(net, (0.0, 15.0), 0.02,
                             initial_v=np.array([-1.2]), initial_w=np.array([-0.6]),
                             external_stimulus=stimulus_resting)
    v_rest_max = np.max(sim_rest["V"][:, 0])
    assert v_rest_max > 1.0, f"Cell failed to activate from resting state (peak {v_rest_max})"

    # Case B: Stimulate with the same pulse but when recovery variable w is high (refractory: w = 1.2)
    def stimulus_refractory(t, node_id):
        return 1.5 if (2.0 <= t <= 3.0) else 0.0

    sim_refrac = solve_network(net, (0.0, 15.0), 0.02,
                               initial_v=np.array([-1.2]), initial_w=np.array([1.2]),
                               external_stimulus=stimulus_refractory)
    v_refrac_max = np.max(sim_refrac["V"][:, 0])

    # In refractory state, voltage cannot reach full upstroke
    assert v_refrac_max < 0.0, f"Refractory cell unexpectedly generated full AP: peak {v_refrac_max}"

