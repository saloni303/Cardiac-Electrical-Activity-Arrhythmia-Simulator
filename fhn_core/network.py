"""
Coupled Cardiac-Cell Network Module using the FitzHugh-Nagumo ODE System.

Model formulation:
    dv_i/dt = v_i - (v_i)³/3 - w_i + I_i(t) + D * Σ_j A_ij * (v_j - v_i)
    dw_i/dt = ε_i * (v_i + a_i - b_i * w_i)

Where:
    - i, j index discrete cells or functional compartments.
    - A_ij is the directed or undirected network coupling weight (conductance).
    - D is the global electrical coupling coefficient.
    - Each cell maintains independent states (v_i, w_i) and regional parameters (a_i, b_i, ε_i, I_i).
    - I_i(t) is external stimulus or intrinsic pacemaking current.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
from .model import FHNParameters


class RegionType(str, Enum):
    """Functional cardiac regions in the electrical conduction network."""
    SA_NODE = "Sinoatrial (SA) Node"
    ATRIUM = "Atrial Myocardium"
    AV_NODE = "Atrioventricular (AV) Node"
    HIS_BUNDLE = "Bundle of His"
    PURKINJE = "Purkinje Network"
    VENTRICLE = "Ventricular Myocardium"


@dataclass
class NetworkNode:
    """
    Represents an excitable unit or functional compartment in the cardiac network.

    Attributes:
        node_id (int): Unique numeric index (0 to N-1).
        name (str): Descriptive label (e.g. 'SAN_P', 'RA_Myo').
        region (RegionType): Functional anatomical classification.
        params (FHNParameters): Local FHN parameters (a, b, ε, I_bias).
        pos_x (float): Normalized 2D X coordinate for graph visualization.
        pos_y (float): Normalized 2D Y coordinate for graph visualization.
    """
    node_id: int
    name: str
    region: RegionType
    params: FHNParameters
    pos_x: float = 0.0
    pos_y: float = 0.0


@dataclass
class NetworkEdge:
    """
    Represents an electrical connection (gap-junction pathway) between two nodes.

    Attributes:
        source_id (int): Index of source node.
        target_id (int): Index of target node.
        conductance (float): Intercellular coupling conductance g_ij >= 0.
        is_bidirectional (bool): True if coupling is bidirectional (A_ij = A_ji).
        is_active (bool): True if conduction pathway is intact; False if severed.
    """
    source_id: int
    target_id: int
    conductance: float = 1.0
    is_bidirectional: bool = True
    is_active: bool = True


class CardiacNetwork:
    """
    Explicit graph-based cardiac conduction network with coupled FHN units.
    """

    def __init__(self, global_coupling: float = 1.0):
        self.global_coupling = float(global_coupling)
        self.nodes: Dict[int, NetworkNode] = {}
        self.edges: List[NetworkEdge] = []
        self._cached_adjacency: Optional[np.ndarray] = None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def add_node(self, node: NetworkNode) -> None:
        self.nodes[node.node_id] = node
        self._cached_adjacency = None

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        conductance: float = 1.0,
        is_bidirectional: bool = True,
        is_active: bool = True,
    ) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Both nodes ({source_id}, {target_id}) must exist before adding an edge.")

        edge = NetworkEdge(
            source_id=source_id,
            target_id=target_id,
            conductance=conductance,
            is_bidirectional=is_bidirectional,
            is_active=is_active,
        )
        self.edges.append(edge)
        self._cached_adjacency = None

    def set_edge_active(self, source_id: int, target_id: int, is_active: bool) -> None:
        """Toggles active state of edge(s) connecting source and target."""
        for edge in self.edges:
            if (edge.source_id == source_id and edge.target_id == target_id) or (
                edge.is_bidirectional and edge.source_id == target_id and edge.target_id == source_id
            ):
                edge.is_active = is_active
        self._cached_adjacency = None

    def set_edge_conductance(self, source_id: int, target_id: int, conductance: float) -> None:
        """Modulates coupling conductance between two nodes."""
        for edge in self.edges:
            if (edge.source_id == source_id and edge.target_id == target_id) or (
                edge.is_bidirectional and edge.source_id == target_id and edge.target_id == source_id
            ):
                edge.conductance = max(0.0, float(conductance))
        self._cached_adjacency = None

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Builds and caches the N x N coupling matrix A_ij representing inter-node conductances.
        """
        if self._cached_adjacency is not None:
            return self._cached_adjacency

        n = self.num_nodes
        A = np.zeros((n, n), dtype=float)

        for edge in self.edges:
            if not edge.is_active or edge.conductance <= 0.0:
                continue
            u, v = edge.source_id, edge.target_id
            A[u, v] = edge.conductance
            if edge.is_bidirectional:
                A[v, u] = edge.conductance

        self._cached_adjacency = A
        return A

    def get_parameter_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts regional parameters as ordered 1D numpy arrays: (a, b, ε, I_bias).
        """
        n = self.num_nodes
        a_arr = np.zeros(n, dtype=float)
        b_arr = np.zeros(n, dtype=float)
        eps_arr = np.zeros(n, dtype=float)
        I_arr = np.zeros(n, dtype=float)

        for i in range(n):
            node = self.nodes[i]
            a_arr[i] = node.params.a
            b_arr[i] = node.params.b
            eps_arr[i] = node.params.epsilon
            I_arr[i] = node.params.I

        return a_arr, b_arr, eps_arr, I_arr

    def compute_derivatives(
        self,
        t: float,
        V: np.ndarray,
        W: np.ndarray,
        a_arr: np.ndarray,
        b_arr: np.ndarray,
        eps_arr: np.ndarray,
        I_base: np.ndarray,
        A_mat: np.ndarray,
        external_stimulus: Optional[Callable[[float, int], float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes instantaneous derivatives [dv/dt, dw/dt] for all N coupled cells simultaneously.

        Vectorized coupling term:
            I_diff_i = D * Σ_j A_ij * (v_j - v_i)
        """
        n = len(V)

        # External stimulation vector
        I_stim = np.zeros(n, dtype=float)
        if external_stimulus is not None:
            for i in range(n):
                I_stim[i] = external_stimulus(t, i)

        # Discrete gap-junction diffusive currents:
        # For each cell i: sum_j A_ij * (v_j - v_i) = (A @ V)_i - (sum_j A_ij) * v_i
        row_sums = np.sum(A_mat, axis=1)
        diffusive_current = (A_mat @ V) - (row_sums * V)

        # dv/dt = v - v³/3 - w + I_base + I_stim + D * diffusive_current
        dV_dt = V - (V**3) / 3.0 - W + I_base + I_stim + self.global_coupling * diffusive_current

        # dw/dt = ε * (v + a - b * w)
        dW_dt = eps_arr * (V + a_arr - b_arr * W)

        return dV_dt, dW_dt


def rk4_network_step(
    t: float,
    V: np.ndarray,
    W: np.ndarray,
    dt: float,
    network: CardiacNetwork,
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    eps_arr: np.ndarray,
    I_base: np.ndarray,
    A_mat: np.ndarray,
    external_stimulus: Optional[Callable[[float, int], float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes a single RK4 integration step across all 2N network state variables.
    """
    # k1
    kV1, kW1 = network.compute_derivatives(t, V, W, a_arr, b_arr, eps_arr, I_base, A_mat, external_stimulus)

    # k2
    t_half = t + 0.5 * dt
    V_k2 = V + 0.5 * dt * kV1
    W_k2 = W + 0.5 * dt * kW1
    kV2, kW2 = network.compute_derivatives(t_half, V_k2, W_k2, a_arr, b_arr, eps_arr, I_base, A_mat, external_stimulus)

    # k3
    V_k3 = V + 0.5 * dt * kV2
    W_k3 = W + 0.5 * dt * kW2
    kV3, kW3 = network.compute_derivatives(t_half, V_k3, W_k3, a_arr, b_arr, eps_arr, I_base, A_mat, external_stimulus)

    # k4
    t_full = t + dt
    V_k4 = V + dt * kV3
    W_k4 = W + dt * kW3
    kV4, kW4 = network.compute_derivatives(t_full, V_k4, W_k4, a_arr, b_arr, eps_arr, I_base, A_mat, external_stimulus)

    # State update
    V_next = V + (dt / 6.0) * (kV1 + 2.0 * kV2 + 2.0 * kV3 + kV4)
    W_next = W + (dt / 6.0) * (kW1 + 2.0 * kW2 + 2.0 * kW3 + kW4)

    return V_next, W_next


def solve_network(
    network: CardiacNetwork,
    t_span: Tuple[float, float],
    dt: float,
    initial_v: Optional[np.ndarray] = None,
    initial_w: Optional[np.ndarray] = None,
    external_stimulus: Optional[Callable[[float, int], float]] = None,
) -> Dict[str, Any]:
    """
    Simulates the coupled cardiac network over t_span = (t_start, t_end) with step size dt.

    Returns:
        Dict containing:
            'time': np.ndarray of shape (num_steps + 1,)
            'V': np.ndarray of shape (num_steps + 1, num_nodes)
            'W': np.ndarray of shape (num_steps + 1, num_nodes)
            'network': CardiacNetwork reference
    """
    t_start, t_end = t_span
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be greater than t_start ({t_start}).")
    if dt <= 0:
        raise ValueError(f"Step size dt ({dt}) must be positive.")

    n = network.num_nodes
    num_steps = int(np.ceil((t_end - t_start) / dt))
    t_array = np.linspace(t_start, t_end, num_steps + 1)
    actual_dt = t_array[1] - t_array[0] if num_steps > 0 else dt

    # Pre-extract cached network arrays for high-performance vectorization
    a_arr, b_arr, eps_arr, I_base = network.get_parameter_arrays()
    A_mat = network.get_adjacency_matrix()

    # Pre-allocate output buffers
    V_traj = np.zeros((num_steps + 1, n), dtype=float)
    W_traj = np.zeros((num_steps + 1, n), dtype=float)

    # Initialize state
    if initial_v is not None:
        V_traj[0, :] = initial_v
    else:
        # Default resting state
        V_traj[0, :] = -1.2

    if initial_w is not None:
        W_traj[0, :] = initial_w
    else:
        # Default recovery state
        W_traj[0, :] = -0.6

    # Time integration loop
    curr_V = V_traj[0, :].copy()
    curr_W = W_traj[0, :].copy()

    for step in range(num_steps):
        t_curr = t_array[step]
        curr_V, curr_W = rk4_network_step(
            t_curr, curr_V, curr_W, actual_dt, network,
            a_arr, b_arr, eps_arr, I_base, A_mat, external_stimulus
        )
        V_traj[step + 1, :] = curr_V
        W_traj[step + 1, :] = curr_W

    return {
        "time": t_array,
        "V": V_traj,
        "W": W_traj,
        "network": network,
        "dt": actual_dt,
        "num_nodes": n,
    }


def create_default_cardiac_network(
    av_conductance: float = 0.08,
    global_coupling: float = 0.5,
) -> CardiacNetwork:
    """
    Factory function creating the initial calibrated 6-region cardiac conduction network.

    Network Topology & Anatomical Roles:
        Node 0: SA Node (Autorhythmic / Pacemaker, I = 0.50, oscillatory)
        Node 1: Atrial Myocardium (Excitable, I = 0.0, resting)
        Node 2: AV Node (Slower conduction, low coupling, higher recovery ε=0.04, I = 0.0)
        Node 3: Bundle of His (Fast septal highway, I = 0.0)
        Node 4: Purkinje Network (High-conductance terminal arborization, I = 0.0)
        Node 5: Ventricular Myocardium (Working ventricle, excitable, I = 0.0)

    Conductance Pathways:
        SA -> Atria:        g = 0.35 (Fast propagation from pacemaker into atria)
        Atria -> AV Node:   g = av_conductance (Configurable bottleneck; low g produces AV delay)
        AV Node -> His:     g = 0.35 (Relays impulse to specialized fast conduction system)
        His -> Purkinje:    g = 0.45 (Rapid septal transmission)
        Purkinje -> Ventr:  g = 0.50 (High-velocity spread into ventricular myocardium)
    """
    net = CardiacNetwork(global_coupling=global_coupling)

    # 1. SA Node: Autorhythmic pacemaker (I = 0.50 drives limit-cycle firing)
    net.add_node(NetworkNode(
        node_id=0,
        name="SAN",
        region=RegionType.SA_NODE,
        params=FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.50),
        pos_x=0.75, pos_y=0.90,
    ))

    # 2. Atrial Region: Excitable quiescent myocardium (I = 0.0, resting at v ≈ -1.2)
    net.add_node(NetworkNode(
        node_id=1,
        name="Atrium",
        region=RegionType.ATRIUM,
        params=FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0),
        pos_x=0.50, pos_y=0.75,
    ))

    # 3. AV Node: Critical delay unit. Lower ε=0.04 extends refractory period; lower conductance.
    net.add_node(NetworkNode(
        node_id=2,
        name="AVN",
        region=RegionType.AV_NODE,
        params=FHNParameters(a=0.7, b=0.8, epsilon=0.04, I=0.0),
        pos_x=0.50, pos_y=0.55,
    ))

    # 4. His Bundle: Fast conduction pathway towards ventricles
    net.add_node(NetworkNode(
        node_id=3,
        name="His",
        region=RegionType.HIS_BUNDLE,
        params=FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0),
        pos_x=0.50, pos_y=0.40,
    ))

    # 5. Purkinje Region: Rapid terminal distributor
    net.add_node(NetworkNode(
        node_id=4,
        name="Purkinje",
        region=RegionType.PURKINJE,
        params=FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=0.0),
        pos_x=0.35, pos_y=0.22,
    ))

    # 6. Ventricular Myocardium: Final working chamber activation
    net.add_node(NetworkNode(
        node_id=5,
        name="Ventricle",
        region=RegionType.VENTRICLE,
        params=FHNParameters(a=0.7, b=0.8, epsilon=0.06, I=0.0),
        pos_x=0.50, pos_y=0.08,
    ))

    # Add edges with specific conductances
    net.add_edge(0, 1, conductance=0.35)             # SA -> Atrium
    net.add_edge(1, 2, conductance=av_conductance)   # Atrium -> AV Node (Adjustable Bottleneck)
    net.add_edge(2, 3, conductance=0.35)             # AV Node -> His
    net.add_edge(3, 4, conductance=0.45)             # His -> Purkinje
    net.add_edge(4, 5, conductance=0.50)             # Purkinje -> Ventricle

    return net

