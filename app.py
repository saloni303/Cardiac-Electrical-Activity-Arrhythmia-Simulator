"""
Interactive Streamlit Application:
    Stage 1: Standalone FitzHugh-Nagumo Mathematical Engine
    Stage 2: Coupled Cardiac-Cell Conduction Network
    Stage 3: Arrhythmia Scenario Engine (Normal, Tachycardia, Bradycardia, Sinus Pause, AV Block, Re-entry)

Academic Project:
    ODE-Based Simulation of Cardiac Electrical Activity and Arrhythmia Dynamics Using the FitzHugh-Nagumo Model
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Core imports
from fhn_core.model import FHNParameters, compute_vector_field, fhn_derivatives
from fhn_core.solver import rk4_solve, compute_time_series
from fhn_core.analysis import (
    compute_nullclines,
    find_equilibria,
    compute_jacobian,
    compute_stability_metrics,
    classify_equilibrium,
    StabilityClass,
)
from fhn_core.sweeps import (
    sweep_parameter_I,
    sweep_parameter_epsilon,
    find_hopf_candidates,
    check_fold_existence,
)
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
from fhn_core.scenarios import (
    ScenarioType,
    ScenarioConfig,
    ScenarioResult,
    run_scenario,
    compute_rhythm_trace,
    create_reentry_circuit,
)

# Page configuration
st.set_page_config(
    page_title="Cardiac Arrhythmia & FHN Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for scientific styling
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 12px;
    }
    .stAlert {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR: APP MODE NAVIGATION & SCIENTIFIC DISCLAIMER
# ---------------------------------------------------------
st.sidebar.title("⚡ Cardiac FHN Simulator")
app_mode = st.sidebar.radio(
    "Select Simulation Module:",
    [
        "🏥 Arrhythmia Scenario Engine (Stage 3)",
        "🌐 Coupled Conduction Network (Stage 2)",
        "⚡ Standalone Single-Cell Engine (Stage 1)",
    ],
    index=0,
)

st.sidebar.divider()

st.sidebar.info("""
**Academic & Scientific Disclaimers:**
- Single-cell: 2-variable FitzHugh-Nagumo ODE.
- Coupled network: Multi-compartment discrete conduction graph.
- Rhythm traces are simulated via electrical dipole projection.
- **Not** a full human heart simulation; does **not** produce clinical ECGs or diagnose real patients.
- Demonstration parameters are pedagogical analogues.
""")


# =============================================================================
# WORKSPACE 3: ARRHYTHMIA SCENARIO ENGINE (STAGE 3)
# =============================================================================
if app_mode == "🏥 Arrhythmia Scenario Engine (Stage 3)":
    st.title("🏥 Cardiac Arrhythmia Scenario Engine (Stage 3)")
    st.caption("Numerically Driven Simulation of Cardiac Rhythms, Conduction Blocks, and Circus Re-entry")

    st.sidebar.subheader("Arrhythmia Scenarios")
    selected_scenario_name = st.sidebar.selectbox(
        "Select Rhythm Scenario:",
        [
            "Scenario 1: Normal Sinus Rhythm",
            "Scenario 2: Tachycardia-Like Rhythm",
            "Scenario 3: Bradycardia-Like Rhythm",
            "Scenario 4: Sinus Pause",
            "Scenario 5: AV Block-Like Condition",
            "Advanced Scenario: Re-entry-Like Activation",
        ],
        index=0,
    )

    # Map name to ScenarioType
    if "Normal" in selected_scenario_name:
        sc_type = ScenarioType.NORMAL
        default_pacing = 45.0
        default_t_end = 160.0
    elif "Tachycardia" in selected_scenario_name:
        sc_type = ScenarioType.TACHYCARDIA
        default_pacing = 35.0
        default_t_end = 160.0
    elif "Bradycardia" in selected_scenario_name:
        sc_type = ScenarioType.BRADYCARDIA
        default_pacing = 75.0
        default_t_end = 160.0
    elif "Sinus Pause" in selected_scenario_name:
        sc_type = ScenarioType.SINUS_PAUSE
        default_pacing = 45.0
        default_t_end = 160.0
    elif "AV Block" in selected_scenario_name:
        sc_type = ScenarioType.AV_BLOCK
        default_pacing = 45.0
        default_t_end = 160.0
    else:
        sc_type = ScenarioType.REENTRY
        default_pacing = 45.0
        default_t_end = 150.0

    st.sidebar.subheader("Interactive Scenario Controls")

    # Dynamic controls based on scenario
    if sc_type == ScenarioType.REENTRY:
        reentry_nodes = st.sidebar.slider(
            "Loop Size (Number of Nodes)",
            min_value=4, max_value=12, value=10, step=1,
            help="Small loops (e.g. 4 nodes) extinguish because conduction time < refractoriness. Larger loops (8-12 nodes) sustain circus movement."
        )
        reentry_g = st.sidebar.slider("Ring Coupling (g)", min_value=0.15, max_value=0.55, value=0.35, step=0.05)
        reentry_eps = st.sidebar.slider("Recovery Rate (ε)", min_value=0.05, max_value=0.25, value=0.15, step=0.02)
        pacing_val = 100.0
        pause_start = None
        pause_end = None
        g_av_val = 0.35
        block_loc = "Atrium-to-AVN"
    else:
        reentry_nodes = 10
        reentry_g = 0.35
        reentry_eps = 0.15
        pacing_val = st.sidebar.slider(
            "SA Pacing Interval (T_pace)",
            min_value=25.0, max_value=100.0, value=float(default_pacing), step=2.0,
            help="Time period between consecutive pacing pulses delivered to the SA node."
        )

        if sc_type == ScenarioType.SINUS_PAUSE:
            pause_start = st.sidebar.slider("Pause Start Time", min_value=10.0, max_value=80.0, value=30.0, step=5.0)
            pause_end = st.sidebar.slider("Pause End Time", min_value=pause_start + 10.0, max_value=140.0, value=110.0, step=5.0)
        else:
            pause_start = None
            pause_end = None

        if sc_type == ScenarioType.AV_BLOCK:
            block_loc = st.sidebar.selectbox("Block Location:", ["Atrium-to-AVN", "AVN-to-His"])
            g_av_val = st.sidebar.slider("Junction Conductance (g)", min_value=0.0, max_value=0.35, value=0.0, step=0.02,
                                         help="Set to 0.0 for complete block; intermediate values illustrate delayed/decremental transmission.")
        else:
            block_loc = "Atrium-to-AVN"
            g_av_val = 0.35

    t_end_val = st.sidebar.slider("Simulation Horizon (T_end)", min_value=80.0, max_value=240.0, value=float(default_t_end), step=10.0)

    # Build scenario config
    cfg = ScenarioConfig(
        scenario_type=sc_type,
        pacing_interval=pacing_val,
        pause_start=pause_start,
        pause_end=pause_end,
        av_conductance=g_av_val,
        global_coupling=0.70,
        t_end=t_end_val,
        dt=0.02,
        block_location=block_loc,
        reentry_num_nodes=reentry_nodes,
        reentry_coupling=reentry_g,
        reentry_epsilon=reentry_eps,
    )

    # Execute simulation
    res = run_scenario(cfg)

    # Summary metric bar
    m_cols = st.columns(4)
    with m_cols[0]:
        st.metric("Pacing Interval (T_pace)", f"{cfg.pacing_interval:.1f} units" if sc_type != ScenarioType.REENTRY else "Single Starter")
    with m_cols[1]:
        st.metric("Ventricular Rate", f"{res.activation_frequency:.2f} / 100u")
    with m_cols[2]:
        rate_ratio = res.comparison_with_normal.get("rate_ratio", 1.0)
        st.metric("Relative to Normal", f"{rate_ratio:.2f}x" if rate_ratio > 0 else "0.0x (Blocked)")
    with m_cols[3]:
        if sc_type == ScenarioType.REENTRY:
            status_label = "Sustained Circus" if res.reentry_sustained else "Extinguished"
            st.metric("Re-entry Status", status_label)
        else:
            status_label = "1:1 Conduction" if res.comparison_with_normal.get("conduction_success") else "Conduction Block"
            st.metric("Conduction Status", status_label)

    # Main Tabs for Stage 3
    sc_tab_traces, sc_tab_timing, sc_tab_comp, sc_tab_math = st.tabs([
        "📈 Action Potentials & Simulated Rhythm Waveform",
        "⏱️ Activation Timestamps & Conduction Latencies",
        "⚖️ Comparative Analysis vs Normal Rhythm",
        "📐 Mathematical Formulation & Verification",
    ])

    # -----------------------------------------------------
    # TAB 1: ACTION POTENTIALS & SIMULATED RHYTHM WAVEFORM
    # -----------------------------------------------------
    with sc_tab_traces:
        st.subheader("1. Synchronized Membrane Voltages v_i(t)")
        fig_v = go.Figure()
        colors = ["#2563eb", "#0284c7", "#d97706", "#7c3aed", "#ec4899", "#dc2626", "#059669", "#4f46e5", "#b45309", "#0e7490", "#9333ea", "#15803d"]

        num_nodes_to_plot = min(res.network.num_nodes, 10)
        for i in range(num_nodes_to_plot):
            node_name = res.network.nodes[i].name
            fig_v.add_trace(go.Scatter(
                x=res.time,
                y=res.V[:, i],
                mode="lines",
                name=f"{node_name}",
                line=dict(color=colors[i % len(colors)], width=1.8),
            ))

        fig_v.update_layout(
            height=380,
            margin=dict(l=40, r=20, t=20, b=20),
            xaxis_title="Time t (dimensionless model units)",
            yaxis_title="Voltage-Like State v_i",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_v, use_container_width=True)

        st.subheader("2. Simulated ECG-Like Rhythm Waveform")
        st.caption("Dipole moment projection onto the vertical heart axis: Φ(t) = Σ g_ij (v_j - v_i) · dy_ij. Demonstrates rhythm cadence, pauses, and deflections.")

        fig_rhythm = go.Figure()
        fig_rhythm.add_trace(go.Scatter(
            x=res.time,
            y=res.rhythm_trace,
            mode="lines",
            name="Simulated Rhythm",
            line=dict(color="#7c3aed", width=2.2),
        ))
        fig_rhythm.update_layout(
            height=260,
            margin=dict(l=40, r=20, t=20, b=20),
            xaxis_title="Time t (dimensionless model units)",
            yaxis_title="Projected Dipole Potential",
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig_rhythm, use_container_width=True)

    # -----------------------------------------------------
    # TAB 2: ACTIVATION TIMESTAMPS & LATENCIES
    # -----------------------------------------------------
    with sc_tab_timing:
        st.subheader("Activation Timestamps and Regional Delays")

        col_t1, col_t2 = st.columns([1.2, 1])

        with col_t1:
            st.markdown("#### Detected Depolarization Timestamps (t_act)")
            rows_act = []
            for i in range(res.network.num_nodes):
                acts = res.activations.get(i, [])
                acts_str = ", ".join(f"{t:.2f}" for t in acts[:5]) + ("..." if len(acts) > 5 else "") if acts else "No activation"
                rows_act.append({
                    "Node ID": i,
                    "Name": res.network.nodes[i].name,
                    "Region": res.network.nodes[i].region.value,
                    "Beats": len(acts),
                    "Timestamps (t)": acts_str,
                })
            st.dataframe(pd.DataFrame(rows_act), use_container_width=True)

        with col_t2:
            st.markdown("#### Regional Conduction Latencies (Beat #1)")
            reg_sum = res.latencies.get("regional_summary", {})
            rows_lat = []
            for seg, d_t in reg_sum.items():
                rows_lat.append({
                    "Segment Pathway": seg.replace("_", " "),
                    "Delay (Δt)": f"{d_t:.3f} units" if d_t is not None else "BLOCKED",
                })
            st.dataframe(pd.DataFrame(rows_lat), use_container_width=True)

            if res.failures:
                st.error(f"⚠ **Conduction Failure Detected:** {len(res.failures)} block(s) logged along transmission pathway.")
                for f in res.failures:
                    st.caption(f"- **{f['failure_type']}:** {f['reason']}")
            else:
                st.success("✓ **Conduction Intact:** All monitored pathways conducted successfully.")

    # -----------------------------------------------------
    # TAB 3: COMPARATIVE ANALYSIS VS NORMAL RHYTHM
    # -----------------------------------------------------
    with sc_tab_comp:
        st.subheader("Comparative Analysis: Scenario vs. Normal Sinus Rhythm")

        comp_data = [
            {"Metric": "Scenario Type", "Normal Rhythm": "Normal Sinus Rhythm", f"{res.scenario_type.value}": res.scenario_type.value},
            {"Metric": "SA Pacing Interval", "Normal Rhythm": "45.0 time units", f"{res.scenario_type.value}": f"{cfg.pacing_interval:.1f} time units" if sc_type != ScenarioType.REENTRY else "Single starter beat"},
            {"Metric": "Ventricular Activation Rate", "Normal Rhythm": "2.23 beats / 100 units", f"{res.scenario_type.value}": f"{res.activation_frequency:.2f} beats / 100 units"},
            {"Metric": "Conduction Success", "Normal Rhythm": "100% (1:1 Conduction)", f"{res.scenario_type.value}": "1:1 Forward" if res.comparison_with_normal.get('conduction_success') else "Conduction Failure / Block"},
            {"Metric": "Rate Ratio", "Normal Rhythm": "1.00x", f"{res.scenario_type.value}": f"{res.comparison_with_normal.get('rate_ratio', 1.0):.2f}x"},
        ]
        st.table(pd.DataFrame(comp_data).set_index("Metric"))

        st.markdown("### Mechanistic Electrophysiological Description")
        st.info(f"""
        **Parameter Alteration:**  
        {res.parameter_change_summary}

        **Functional Result:**  
        {res.comparison_with_normal.get('status')}
        """)

        if sc_type == ScenarioType.REENTRY:
            st.markdown("### Re-entry Circus Movement Evaluation")
            if res.reentry_sustained:
                st.success(f"""
                ✓ **Sustained Re-entry Confirmed:**  
                The wave completed **{res.reentry_revolutions} spontaneous revolutions** around the closed ring after the starter stimulus ceased.
                Because the conduction time around the loop ($T_{{\\text{{loop}}}} \\approx {res.mean_cycle_length:.1f}$) exceeds the refractory recovery duration,
                the tissue ahead of the wavefront remains fully excitable, supporting self-sustaining circus movement.
                """)
            else:
                st.warning(f"""
                ⚠ **Re-entry Extinguished:**  
                The wave died out after {res.reentry_revolutions} revolution(s).
                The loop conduction time is shorter than the refractory recovery period of the tissue ($T_{{\\text{{loop}}}} < T_{{\\text{{refractory}}}}$).
                The advancing wavefront ran directly into the refractory tail of preceding excitation, terminating the circus wave.
                *Try increasing the loop size (number of nodes) or decreasing coupling to lengthen loop traversal time.*
                """)

    # -----------------------------------------------------
    # TAB 4: MATHEMATICAL ASSUMPTIONS & TESTS
    # -----------------------------------------------------
    with sc_tab_math:
        st.subheader("Mathematical Assumptions, Limitations & Verification")

        st.markdown(f"""
        **Mathematical Assumptions:**  
        {res.mathematical_assumptions}

        **Governing Coupled Network Equations:**
        $$\\frac{{dv_i}}{{dt}} = v_i - \\frac{{v_i^3}}{{3}} - w_i + I_i(t) + D \\sum_{{j \\in \\mathcal{{N}}(i)}} A_{{ij}} (v_j - v_i)$$
        $$\\frac{{dw_i}}{{dt}} = \\varepsilon_i(v_i + a_i - b_i w_i)$$

        **Refractoriness and Conduction Wavelength:**
        - Refractory state is represented by the slow recovery variable $w_i(t)$.
        - Conduction wavelength $\\lambda = \\theta \\times \\text{{ERP}}$, where $\\theta$ is conduction velocity and $\\text{{ERP}}$ is effective refractory period.
        - Conduction block occurs when incoming activation arrives while $w_i$ is elevated above resting potential ($w_i > -0.4$).
        """)

        st.divider()
        st.markdown("### Automated Test Suite Execution")
        if st.button("▶ Run All 26 Unit Tests (Stages 1, 2 & 3)"):
            with st.spinner("Executing pytest suite across tests/..."):
                import subprocess
                p_res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"], capture_output=True, text=True)
                if p_res.returncode == 0:
                    st.success("All 26 Unit Tests Passed Successfully!")
                    st.code(p_res.stdout)
                else:
                    st.error("Some tests failed:")
                    st.code(p_res.stdout + "\n" + p_res.stderr)


# =============================================================================
# WORKSPACE 2: COUPLED CARDIAC NETWORK (STAGE 2)
# =============================================================================
elif app_mode == "🌐 Coupled Conduction Network (Stage 2)":
    st.title("🌐 Coupled Cardiac-Cell Conduction Network (Stage 2)")
    st.caption("Emergent Conduction Delays, Gap-Junction Coupling, Refractoriness, and Block Detection")

    st.sidebar.subheader("Network Presets")
    net_preset = st.sidebar.selectbox(
        "Load Network Conduction Preset:",
        [
            "Normal Conduction Sequence",
            "Prolonged AV Conduction Delay (1st Degree Analogue)",
            "AV Conduction Block (Complete Uncoupling)",
            "High Coupling / Accelerated Spread",
        ],
        index=0,
    )

    if net_preset == "Normal Conduction Sequence":
        default_g_av, default_D, default_eps_av = 0.35, 0.70, 0.08
    elif net_preset == "Prolonged AV Conduction Delay (1st Degree Analogue)":
        default_g_av, default_D, default_eps_av = 0.08, 0.70, 0.04
    elif net_preset == "AV Conduction Block (Complete Uncoupling)":
        default_g_av, default_D, default_eps_av = 0.00, 0.70, 0.08
    elif net_preset == "High Coupling / Accelerated Spread":
        default_g_av, default_D, default_eps_av = 0.50, 1.00, 0.08

    st.sidebar.subheader("Conduction & Coupling Controls")
    g_av_val = st.sidebar.slider("AV Conductance (g_AV)", 0.0, 0.50, float(default_g_av), 0.01)
    D_val = st.sidebar.slider("Global Coupling (D)", 0.1, 1.5, float(default_D), 0.05)
    eps_av_val = st.sidebar.slider("AV Timescale (ε_AV)", 0.01, 0.15, float(default_eps_av), 0.005)
    t_end_net = st.sidebar.slider("Duration (T_end)", 40.0, 180.0, 100.0, 10.0)

    net = create_default_cardiac_network(av_conductance=g_av_val, global_coupling=D_val)
    avn_node = net.nodes[2]
    net.nodes[2] = NetworkNode(
        node_id=avn_node.node_id, name=avn_node.name, region=avn_node.region,
        params=FHNParameters(a=avn_node.params.a, b=avn_node.params.b, epsilon=eps_av_val, I=avn_node.params.I),
        pos_x=avn_node.pos_x, pos_y=avn_node.pos_y,
    )

    sim_res = solve_network(net, t_span=(0.0, t_end_net), dt=0.02)
    t_net = sim_res["time"]
    V_net = sim_res["V"]
    W_net = sim_res["W"]

    activations = detect_activations(t_net, V_net, threshold=0.0)
    latencies = compute_propagation_latencies(activations, net, beat_index=0)
    failures = detect_conduction_failures(activations, net)

    net_cols = st.columns(4)
    with net_cols[0]:
        st.metric("AV Conductance (g_AV)", f"{g_av_val:.3f}")
    with net_cols[1]:
        st.metric("Global Coupling (D)", f"{D_val:.2f}")
    with net_cols[2]:
        tot_time = latencies["regional_summary"].get("Total_SAN_to_Ventricle")
        st.metric("Total Latency", f"{tot_time:.2f} units" if tot_time is not None else "BLOCKED")
    with net_cols[3]:
        st.metric("Conduction Status", "✓ Normal Wave" if not failures else f"⚠ Block ({len(failures)} path)")

    col_graph, col_traces = st.columns([1, 1.2])
    with col_graph:
        st.subheader("Network Topology")
        fig_g = go.Figure()
        for edge in net.edges:
            u, v = net.nodes[edge.source_id], net.nodes[edge.target_id]
            is_act = edge.is_active and edge.conductance > 0.0
            fig_g.add_trace(go.Scatter(
                x=[u.pos_x, v.pos_x], y=[u.pos_y, v.pos_y], mode="lines",
                line=dict(color="#3b82f6" if is_act else "#ef4444", width=max(1.5, edge.conductance * 8.0)),
                hoverinfo="none", showlegend=False,
            ))
        node_x = [net.nodes[i].pos_x for i in range(net.num_nodes)]
        node_y = [net.nodes[i].pos_y for i in range(net.num_nodes)]
        peak_v = [np.max(V_net[:, i]) for i in range(net.num_nodes)]
        fig_g.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=36, color=["#10b981" if p > 0 else "#94a3b8" for p in peak_v], line=dict(color="#1e293b", width=2)),
            text=[net.nodes[i].name for i in range(net.num_nodes)],
            textposition="middle center", textfont=dict(color="white", size=11, family="Arial Black"),
            showlegend=False,
        ))
        fig_g.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20), xaxis=dict(visible=False), yaxis=dict(visible=False), template="plotly_white")
        st.plotly_chart(fig_g, use_container_width=True)

    with col_traces:
        st.subheader("Synchronized Wave Propagation Traces")
        fig_w = go.Figure()
        colors = ["#2563eb", "#0284c7", "#d97706", "#7c3aed", "#ec4899", "#dc2626"]
        for i in range(net.num_nodes):
            fig_w.add_trace(go.Scatter(x=t_net, y=V_net[:, i], mode="lines", name=net.nodes[i].name, line=dict(color=colors[i % len(colors)], width=2.0)))
        fig_w.update_layout(height=450, margin=dict(l=40, r=20, t=20, b=20), xaxis_title="Time t", yaxis_title="Voltage v_i", hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_w, use_container_width=True)


# =============================================================================
# WORKSPACE 1: STANDALONE SINGLE-CELL ENGINE (STAGE 1)
# =============================================================================
else:
    st.title("⚡ Standalone FitzHugh–Nagumo Mathematical Engine (Stage 1)")
    st.caption("Rigorous Phase-Plane, Runge–Kutta 4 Integration, Jacobian Stability, and Bifurcation Analysis")

    st.sidebar.subheader("Single-Cell Presets")
    sc_preset = st.sidebar.selectbox(
        "Load Canonical Parameter Preset:",
        [
            "Custom",
            "Limit Cycle / Oscillatory Firing (I=0.5)",
            "Polarized Resting State (I=0.0)",
            "Near Hopf Bifurcation (I=0.331)",
            "High Current Excitation (I=1.2)",
        ],
        index=1,
    )

    if sc_preset == "Limit Cycle / Oscillatory Firing (I=0.5)":
        def_a, def_b, def_eps, def_I = 0.7, 0.8, 0.08, 0.5
        def_v0, def_w0 = -1.0, 0.2
    elif sc_preset == "Polarized Resting State (I=0.0)":
        def_a, def_b, def_eps, def_I = 0.7, 0.8, 0.08, 0.0
        def_v0, def_w0 = -1.2, -0.6
    elif sc_preset == "Near Hopf Bifurcation (I=0.331)":
        def_a, def_b, def_eps, def_I = 0.7, 0.8, 0.08, 0.331
        def_v0, def_w0 = -0.967, -0.334
    elif sc_preset == "High Current Excitation (I=1.2)":
        def_a, def_b, def_eps, def_I = 0.7, 0.8, 0.08, 1.2
        def_v0, def_w0 = 0.5, 0.8
    else:
        def_a, def_b, def_eps, def_I = 0.7, 0.8, 0.08, 0.5
        def_v0, def_w0 = -1.0, 0.2

    st.sidebar.subheader("Single-Cell Model Parameters")
    I_val = st.sidebar.slider("Applied Current (I)", -0.5, 2.0, float(def_I), 0.01)
    eps_val = st.sidebar.slider("Timescale Ratio (ε)", 0.005, 0.4, float(def_eps), 0.005)
    a_val = st.sidebar.slider("Intercept Parameter (a)", 0.0, 1.5, float(def_a), 0.05)
    b_val = st.sidebar.slider("Slope Parameter (b)", 0.1, 1.5, float(def_b), 0.05)

    params = FHNParameters(a=a_val, b=b_val, epsilon=eps_val, I=I_val)
    equilibria = find_equilibria(params)
    t, v, w = rk4_solve((def_v0, def_w0), (0.0, 50.0), 0.02, params)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Current (I)", f"{params.I:.4f}")
    with cols[1]:
        st.metric("Timescale (ε)", f"{params.epsilon:.4f}")
    with cols[2]:
        st.metric("Equilibria Found", len(equilibria))
    with cols[3]:
        st.metric("Stability", "Stable" if (equilibria and equilibria[0].is_stable) else "Unstable")

    c_ts, c_pp = st.columns([1, 1])
    with c_ts:
        st.subheader("Time Series v(t) & w(t)")
        fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("v(t) [Fast]", "w(t) [Slow]"))
        fig_ts.add_trace(go.Scatter(x=t, y=v, mode="lines", name="v(t)", line=dict(color="#2563eb", width=2)), row=1, col=1)
        fig_ts.add_trace(go.Scatter(x=t, y=w, mode="lines", name="w(t)", line=dict(color="#f97316", width=2)), row=2, col=1)
        fig_ts.update_layout(height=480, margin=dict(l=40, r=20, t=30, b=30), template="plotly_white", showlegend=False)
        st.plotly_chart(fig_ts, use_container_width=True)

    with c_pp:
        st.subheader("Phase Plane & Nullclines")
        v_grid, w_vn, w_wn = compute_nullclines((-2.5, 2.5), params)
        fig_pp = go.Figure()
        fig_pp.add_trace(go.Scatter(x=v_grid, y=w_vn, mode="lines", name="v-nullcline", line=dict(color="#1d4ed8", width=2.5)))
        fig_pp.add_trace(go.Scatter(x=v_grid, y=w_wn, mode="lines", name="w-nullcline", line=dict(color="#ea580c", width=2.5, dash="dash")))
        fig_pp.add_trace(go.Scatter(x=v, y=w, mode="lines", name="Trajectory", line=dict(color="#059669", width=2)))
        for eq in equilibria:
            fig_pp.add_trace(go.Scatter(
                x=[eq.v_star], y=[eq.w_star], mode="markers",
                marker=dict(size=12, color="#10b981" if eq.is_stable else "#ef4444", line=dict(color="black", width=1.5)),
                name=f"E: {eq.stability_class.value}"
            ))
        fig_pp.update_layout(height=480, margin=dict(l=40, r=20, t=30, b=30), xaxis=dict(range=[-2.5, 2.5]), yaxis=dict(range=[-2.0, 2.5]), template="plotly_white")
        st.plotly_chart(fig_pp, use_container_width=True)
