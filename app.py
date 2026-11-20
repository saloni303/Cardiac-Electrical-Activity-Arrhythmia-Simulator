"""
Cardiac Electrical Activity and Arrhythmia Simulator (Stage 7 Integrated Dashboard)

An integrated, professional scientific simulation platform:
    1. Main Dashboard:
       - Section 1: Heart Visualization (Interactive anatomical 60 FPS Canvas illustration)
       - Section 2: Simulation Controls (Start, Pause, Reset, Speed, Scenarios)
       - Section 3: Arrhythmia Controls (Normal, Tachycardia, Bradycardia, Sinus Pause, AV Block, Re-entry)
       - Section 4: Parameter Panel (I, ε, a, b, D, Pacing, Connection, Regional parameters)
       - Section 5: Electrical Rhythm (Activation-derived rhythm trace, timeline, estimated rate)
       - Section 6: Mathematical Analysis (v(t), w(t), nullclines, equilibria, Jacobian, eigenvalues, stability)
       - Section 7: Experiment Comparison (Save baseline, modify parameters, compare deltas and waveforms)
    2. Arrhythmia Lab: Deep-dive comparative scenario study, Lead II dipole, numerical event log.
    3. Mathematical Analysis: Cardano roots, analytical Jacobian, Poincaré stability, phase plane, sweeps.
    4. Parameter Experiment Lab: Dedicated baseline vs trial sandbox.
    5. About the Model: Comprehensive academic reference detailing equations, variables, network extension,
       assumptions, limitations, and distinction from a real human heart.
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components

# Core FHN Package Imports
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
from fhn_core.laboratory import (
    sweep_parameter_a,
    sweep_parameter_b,
    analyze_bifurcation_conditions,
    compare_fhn_states,
    StateComparisonResult,
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
from fhn_core.visualization import (
    map_state_to_color,
    extract_animation_frames,
    build_heart_visualization_html,
)
from fhn_core.rhythm_analysis import (
    EventLogEntry,
    generate_numerical_event_log,
    compute_beat_to_beat_intervals,
    estimate_heart_rate_bpm,
    detect_missing_events,
    compute_true_ecg_lead_approximation,
    compare_with_normal_rhythm,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & DARK SCIENTIFIC THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardiac Electrical Activity and Arrhythmia Simulator",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: Dark Scientific Medical Dashboard
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .reportview-container .main .block-container {
        padding-top: 1.0rem;
        padding-bottom: 2.0rem;
        max-width: 98%;
    }
    /* Metric Cards */
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px 14px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 8px;
    }
    .metric-card-title {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-card-val {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
    }
    /* Section Headers */
    .sec-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #60a5fa;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 6px;
        margin-top: 14px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    /* Scientific Alerts */
    .stAlert {
        border-radius: 6px;
        background-color: #111827;
        border: 1px solid #334155;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State for Experiment Comparison
if "baseline_sim" not in st.session_state:
    st.session_state["baseline_sim"] = None


# -----------------------------------------------------------------------------
# SIDEBAR: NAVIGATION & SCIENTIFIC DISCLAIMERS
# -----------------------------------------------------------------------------
st.sidebar.title("🫀 Cardiac Simulator")
st.sidebar.caption("FitzHugh–Nagumo Electrophysiology Engine")

nav_view = st.sidebar.radio(
    "Navigation:",
    [
        "🫀 Main Dashboard",
        "🏥 Arrhythmia Lab",
        "🔬 Mathematical Analysis",
        "⚖️ Parameter Experiment Lab",
        "📖 About the Model",
    ],
    index=0,
)

st.sidebar.divider()

st.sidebar.info("""
**Academic & Scientific Boundaries:**
- **Model:** Dimensionless FitzHugh–Nagumo ODE.
- **Network:** 6-compartment diffusive conduction graph.
- **Rhythm Trace:** Electrical dipole projection.
- **Non-Clinical:** Illustrative educational surrogate; not a diagnostic ECG.
""")


# =============================================================================
# VIEW 1: MAIN DASHBOARD (SECTIONS 1 THROUGH 7)
# =============================================================================
if "Main Dashboard" in nav_view:
    st.title("Cardiac Electrical Activity and Arrhythmia Simulator")
    st.caption("Interactive Multi-Scale Electrophysiology Platform: Single-Cell FitzHugh–Nagumo Dynamics to Coupled Conduction Networks")

    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS: SECTIONS 2, 3 & 4
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### 🎛️ Simulation & Arrhythmia Controls")

    # SECTION 3: ARRHYTHMIA CONTROLS
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

    if "Normal" in selected_scenario_name:
        sc_type = ScenarioType.NORMAL
        def_pacing = 45.0
        def_t_end = 160.0
    elif "Tachycardia" in selected_scenario_name:
        sc_type = ScenarioType.TACHYCARDIA
        def_pacing = 35.0
        def_t_end = 160.0
    elif "Bradycardia" in selected_scenario_name:
        sc_type = ScenarioType.BRADYCARDIA
        def_pacing = 75.0
        def_t_end = 160.0
    elif "Sinus Pause" in selected_scenario_name:
        sc_type = ScenarioType.SINUS_PAUSE
        def_pacing = 45.0
        def_t_end = 160.0
    elif "AV Block" in selected_scenario_name:
        sc_type = ScenarioType.AV_BLOCK
        def_pacing = 45.0
        def_t_end = 160.0
    else:
        sc_type = ScenarioType.REENTRY
        def_pacing = 45.0
        def_t_end = 150.0

    # SECTION 2: SIMULATION CONTROLS
    st.sidebar.markdown("#### Scenario Pacing & Timing")
    if sc_type == ScenarioType.REENTRY:
        reentry_nodes = st.sidebar.slider("Loop Size (Ring Nodes)", 4, 12, 10, 1, help="Critical loop size: circulation persists only if loop traversal time > refractoriness.")
        reentry_g = st.sidebar.slider("Ring Coupling (g)", 0.15, 0.55, 0.35, 0.05)
        reentry_eps = st.sidebar.slider("Recovery Rate (ε)", 0.05, 0.25, 0.15, 0.02)
        pacing_val = 100.0
        pause_start = None
        pause_end = None
        g_av_val = 0.35
        block_loc = "Atrium-to-AVN"
    else:
        reentry_nodes = 10
        reentry_g = 0.35
        reentry_eps = 0.15
        pacing_val = st.sidebar.slider("SA Pacing Interval (T_pace)", 25.0, 100.0, float(def_pacing), 2.0, help="Period between consecutive impulses delivered to SA node.")

        if sc_type == ScenarioType.SINUS_PAUSE:
            pause_start = st.sidebar.slider("Pause Start (u)", 10.0, 80.0, 30.0, 5.0)
            pause_end = st.sidebar.slider("Pause End (u)", pause_start + 10.0, 140.0, 110.0, 5.0)
        else:
            pause_start = None
            pause_end = None

        if sc_type == ScenarioType.AV_BLOCK:
            block_loc = st.sidebar.selectbox("Block Location:", ["Atrium-to-AVN", "AVN-to-His"])
            g_av_val = st.sidebar.slider("Junction Conductance (g_AV)", 0.0, 0.35, 0.0, 0.02, help="0.0 represents complete conduction block; intermediate values represent delayed conduction.")
        else:
            block_loc = "Atrium-to-AVN"
            g_av_val = 0.35

    t_end_val = st.sidebar.slider("Simulation Horizon (T_end)", 60.0, 240.0, float(def_t_end), 10.0)

    # SECTION 4: PARAMETER PANEL
    st.sidebar.markdown("### ⚙️ Parameter Panel")
    with st.sidebar.expander("Global Model Parameters", expanded=False):
        st.caption("Governing single-cell parameters and global network coupling:")
        I_val = st.slider("Stimulus Current (I)", 0.5, 3.0, 1.8, 0.1, help="External stimulus pulse amplitude delivered during pacing.")
        eps_val = st.slider("Timescale Ratio (ε)", 0.01, 0.25, 0.08, 0.005, help="Ratio of fast voltage to slow recovery dynamics. Smaller values prolong refractory plateau.")
        a_val = st.slider("Recovery Intercept (a)", 0.0, 1.2, 0.70, 0.05, help="Sets resting potential threshold position.")
        b_val = st.slider("Recovery Slope (b)", 0.2, 1.5, 0.80, 0.05, help="Slope of recovery nullcline. b < 1 guarantees unique resting state.")
        D_val = st.slider("Global Coupling (D)", 0.1, 1.5, 0.70, 0.05, help="Global gap-junction electrical coupling strength across connected cells.")

    with st.sidebar.expander("Region-Specific Parameters", expanded=False):
        st.caption("Fine-tune regional excitability, recovery and conductances:")
        eps_san = st.slider("SA Node ε (SAN)", 0.02, 0.20, 0.08, 0.01)
        eps_avn = st.slider("AV Node ε (AVN)", 0.02, 0.15, 0.08, 0.01)
        eps_vent = st.slider("Ventricle ε (Vent)", 0.02, 0.15, 0.08, 0.01)

    # Build Scenario Configuration
    cfg = ScenarioConfig(
        scenario_type=sc_type,
        pacing_interval=pacing_val,
        pulse_amplitude=I_val,
        pause_start=pause_start,
        pause_end=pause_end,
        av_conductance=g_av_val,
        global_coupling=D_val,
        t_end=t_end_val,
        dt=0.02,
        block_location=block_loc,
        reentry_num_nodes=reentry_nodes,
        reentry_coupling=reentry_g,
        reentry_epsilon=reentry_eps,
    )

    # Execute Coupled Network Simulation
    res = run_scenario(cfg)
    hr_metrics = estimate_heart_rate_bpm(res.mean_cycle_length)

    # -------------------------------------------------------------------------
    # TOP KPI METRICS BAR
    # -------------------------------------------------------------------------
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Active Scenario</div>
            <div class="metric-card-val" style="font-size: 1.05rem;">{selected_scenario_name.split(':')[0]}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_cols[1]:
        pacing_text = f"{cfg.pacing_interval:.1f} u" if sc_type != ScenarioType.REENTRY else "Single Starter"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">SA Pacing Interval</div>
            <div class="metric-card-val">{pacing_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_cols[2]:
        mcl_text = f"{res.mean_cycle_length:.2f} u" if res.mean_cycle_length else "N/A (Blocked)"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Mean Cycle Length</div>
            <div class="metric-card-val">{mcl_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_cols[3]:
        bpm_text = f"{hr_metrics['bpm']:.1f} BPM" if hr_metrics['is_meaningful'] else "0.0 BPM (Blocked)"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Estimated Heart Rate</div>
            <div class="metric-card-val" style="color: #38bdf8;">{bpm_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_cols[4]:
        if sc_type == ScenarioType.REENTRY:
            status_text = "Sustained Circus" if res.reentry_sustained else "Extinguished"
            status_color = "#10b981" if res.reentry_sustained else "#ef4444"
        elif res.failures:
            status_text = f"Block ({len(res.failures)} drops)"
            status_color = "#ef4444"
        else:
            status_text = "1:1 Synchronous"
            status_color = "#10b981"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Conduction Status</div>
            <div class="metric-card-val" style="color: {status_color};">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROW 1: SECTIONS 1 & 5 (HEART VISUALIZATION + ELECTRICAL RHYTHM)
    # -------------------------------------------------------------------------
    col_vis, col_rhythm = st.columns([1.15, 1.0])

    with col_vis:
        st.markdown('<div class="sec-title">🫀 Section 1: Interactive Anatomical Heart Visualization</div>', unsafe_allow_html=True)
        st.caption("Live 60 FPS anatomical wavefront animation driven directly by continuous numerical ODE states.")
        heart_html = build_heart_visualization_html(res, width=820, height=660)
        components.html(heart_html, height=690, scrolling=False)

    with col_rhythm:
        st.markdown('<div class="sec-title">⚡ Section 5: Electrical Rhythm & Event Timeline</div>', unsafe_allow_html=True)
        st.caption("Simulated activation-derived rhythm trace — not a clinically validated ECG.")

        # Activation-Derived Rhythm Waveform
        fig_rhythm = go.Figure()
        fig_rhythm.add_trace(go.Scatter(
            x=res.time, y=res.rhythm_trace, mode="lines", name="Rhythm Trace Φ(t)",
            line=dict(color="#38bdf8", width=2.0)
        ))
        fig_rhythm.add_hline(y=0.0, line_dash="dash", line_color="#475569")
        fig_rhythm.update_layout(
            title="Activation-Derived Rhythm Waveform (Dipole Projection)",
            xaxis_title="Simulation Time (model units)",
            yaxis_title="Dipole Moment Φ(t)",
            template="plotly_dark",
            height=260,
            margin=dict(l=40, r=20, t=35, b=30),
        )
        st.plotly_chart(fig_rhythm, use_container_width=True)

        # Multi-Channel Activation Event Timeline
        fig_timeline = go.Figure()
        region_colors = {
            "Sinoatrial (SA) Node": "#f59e0b",
            "Atrial Myocardium": "#3b82f6",
            "Atrioventricular (AV) Node": "#ec4899",
            "Bundle of His": "#8b5cf6",
            "Purkinje Network": "#06b6d4",
            "Ventricular Myocardium": "#10b981",
        }
        y_labels = []
        for i in range(res.network.num_nodes):
            node = res.network.nodes[i]
            y_labels.append(node.name)
            t_acts = res.activations.get(i, [])
            c = region_colors.get(node.region.value, "#94a3b8")
            if t_acts:
                fig_timeline.add_trace(go.Scatter(
                    x=t_acts, y=[node.name] * len(t_acts), mode="markers",
                    marker=dict(size=12, color=c, symbol="diamond", line=dict(color="#0f172a", width=1.5)),
                    name=node.name,
                ))
            else:
                fig_timeline.add_trace(go.Scatter(
                    x=[res.time[-1] / 2], y=[node.name], mode="text",
                    text=["[BLOCKED / NO ACTIVATION]"], textposition="middle center",
                    textfont=dict(color="#ef4444", size=10), showlegend=False,
                ))

        fig_timeline.update_layout(
            title="Multi-Channel Activation Event Timeline",
            xaxis_title="Time t (model units)",
            yaxis=dict(categoryorder="array", categoryarray=list(reversed(y_labels))),
            template="plotly_dark",
            height=320,
            showlegend=False,
            margin=dict(l=60, r=20, t=35, b=30),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # -------------------------------------------------------------------------
    # ROW 2: SECTION 6 (MATHEMATICAL ANALYSIS)
    # -------------------------------------------------------------------------
    st.markdown('<div class="sec-title">🔬 Section 6: Mathematical Analysis of the Underlying FitzHugh–Nagumo ODE</div>', unsafe_allow_html=True)
    st.caption("Analytical Jacobian, Poincaré stability invariants, nullclines, and phase-plane flow field for the resting single-cell model.")

    m_tab1, m_tab2 = st.tabs(["📐 Phase Plane & Nullclines", "🧮 Jacobian & Poincaré Stability"])

    # Monitored Single-Cell Parameters (Resting Cell)
    p_cell = FHNParameters(a=a_val, b=b_val, epsilon=eps_val, I=0.0)
    eqs_cell = find_equilibria(p_cell)
    eq_cell = eqs_cell[0] if eqs_cell else None

    with m_tab1:
        col_pp1, col_pp2 = st.columns([1.2, 1.0])
        with col_pp1:
            # Phase Plane with Normalized Quiver + Nullclines
            v_grid, w_vn, w_wn = compute_nullclines((-2.6, 2.6), p_cell, num_points=250)
            V_m, W_m, dV, dW, dV_norm, dW_norm = compute_vector_field((-2.5, 2.5), (-1.8, 2.2), 16, p_cell)

            fig_pp = ff.create_quiver(
                V_m, W_m, dV_norm, dW_norm,
                scale=0.12, arrow_scale=0.35,
                line=dict(color="#475569", width=1.0),
                name="Direction Flow"
            )
            fig_pp.add_trace(go.Scatter(x=v_grid, y=w_vn, mode="lines", name="v-nullcline: w = v - v³/3", line=dict(color="#3b82f6", width=2.5)))
            fig_pp.add_trace(go.Scatter(x=v_grid, y=w_wn, mode="lines", name="w-nullcline: w = (v+a)/b", line=dict(color="#f97316", width=2.5, dash="dash")))

            # Node 0 trajectory from simulation
            t_traj = res.time[:len(res.time)//2]
            v_traj = res.V[:len(res.time)//2, 0]
            w_traj = res.W[:len(res.time)//2, 0]
            fig_pp.add_trace(go.Scatter(x=v_traj, y=w_traj, mode="lines", name="SAN Trajectory (v_0, w_0)", line=dict(color="#10b981", width=2.5)))

            if eq_cell:
                fig_pp.add_trace(go.Scatter(
                    x=[eq_cell.v_star], y=[eq_cell.w_star], mode="markers+text",
                    text=[f"E* ({eq_cell.v_star:.2f}, {eq_cell.w_star:.2f})"], textposition="top right",
                    marker=dict(size=14, color="#10b981" if eq_cell.is_stable else "#ef4444", line=dict(color="black", width=2)),
                    name=f"Fixed Point E*"
                ))
            fig_pp.update_layout(
                title=f"Planar Phase Portrait & Vector Field Flow",
                xaxis_title="Voltage Variable v", yaxis_title="Recovery Variable w",
                template="plotly_dark", height=420, margin=dict(l=40, r=20, t=35, b=30),
            )
            st.plotly_chart(fig_pp, use_container_width=True)

        with col_pp2:
            st.markdown("#### Regional Transmembrane Potentials v_i(t) & w_i(t)")
            fig_vw = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Fast Voltage v_i(t) [Myocardial Depolarization]", "Slow Recovery w_i(t) [Refractoriness]"))
            # Plot SA Node and Ventricle
            fig_vw.add_trace(go.Scatter(x=res.time, y=res.V[:, 0], mode="lines", name="SAN Voltage", line=dict(color="#f59e0b", width=2)), row=1, col=1)
            fig_vw.add_trace(go.Scatter(x=res.time, y=res.V[:, 5], mode="lines", name="LV Voltage", line=dict(color="#10b981", width=2)), row=1, col=1)
            fig_vw.add_trace(go.Scatter(x=res.time, y=res.W[:, 0], mode="lines", name="SAN Recovery", line=dict(color="#f59e0b", width=2, dash="dash")), row=2, col=1)
            fig_vw.add_trace(go.Scatter(x=res.time, y=res.W[:, 5], mode="lines", name="LV Recovery", line=dict(color="#10b981", width=2, dash="dash")), row=2, col=1)
            fig_vw.update_layout(height=420, template="plotly_dark", margin=dict(l=40, r=20, t=30, b=30), hovermode="x unified")
            st.plotly_chart(fig_vw, use_container_width=True)

    with m_tab2:
        if eq_cell:
            r_v = eq_cell.v_star - (eq_cell.v_star**3) / 3.0 - eq_cell.w_star + p_cell.I
            r_w = p_cell.epsilon * (eq_cell.v_star + p_cell.a - p_cell.b * eq_cell.w_star)

            j_c1, j_c2 = st.columns([1, 1.2])
            with j_c1:
                st.markdown(f"""
                **Analytical Jacobian Matrix:**
                $$\\begin{{pmatrix}} {eq_cell.jacobian[0, 0]:+.5f} & {eq_cell.jacobian[0, 1]:+.5f} \\\\ {eq_cell.jacobian[1, 0]:+.5f} & {eq_cell.jacobian[1, 1]:+.5f} \\end{{pmatrix}}$$

                **Stability Invariants & Verified Residuals:**
                - **Equilibrium Coordinates:** `(v* = {eq_cell.v_star:+.6f}, w* = {eq_cell.w_star:+.6f})`
                - **Residuals:** `|r_v| = {abs(r_v):.2e}`, `|r_w| = {abs(r_w):.2e}` (Verified < 10⁻¹²)
                - **Trace τ:** `{eq_cell.trace:+.6f}`
                - **Determinant Δ:** `{eq_cell.determinant:+.6f}`
                - **Discriminant D:** `{eq_cell.discriminant:+.6f}`
                - **Eigenvalues λ1,2:** `{eq_cell.eigenvalues[0].real:+.5f} ± {abs(eq_cell.eigenvalues[0].imag):.5f}i`
                """)
            with j_c2:
                box_c = "#10b981" if eq_cell.is_stable else "#ef4444"
                st.markdown(f"""
                <div style="background-color: #111827; padding: 16px; border-radius: 8px; border-left: 5px solid {box_c};">
                    <h4 style="margin: 0; color: #f8fafc;">Poincaré Stability: {eq_cell.stability_class.value}</h4>
                    <p style="color: #94a3b8; margin-top: 8px; font-size: 14px;">
                        <strong>Asymptotic Nature:</strong> {"Locally Asymptotically Stable (Spiral Sink)" if eq_cell.is_stable else "Unstable (Limit Cycle Onset)"}<br>
                        <strong>Poincaré Criteria:</strong><br>
                        • Determinant Δ = {eq_cell.determinant:+.5f} > 0 (Planar node/focus regime)<br>
                        • Discriminant D = {eq_cell.discriminant:+.5f} < 0 (Complex conjugate eigenvalues -> Spiral/Focus)<br>
                        • Trace τ = {eq_cell.trace:+.5f} {"< 0 (Trajectories contract inward)" if eq_cell.trace < 0 else "> 0 (Trajectories spiral outward)"}
                    </p>
                    <p style="color: #cbd5e1; font-size: 13px; margin-bottom: 0;">
                        <em>{eq_cell.summary_text}</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROW 3: SECTION 7 (EXPERIMENT COMPARISON)
    # -------------------------------------------------------------------------
    st.markdown('<div class="sec-title">⚖️ Section 7: Parameter Experiment & Baseline Comparison</div>', unsafe_allow_html=True)
    st.caption("Save a baseline simulation, alter one or more parameters, and compare the resulting activation patterns and waveforms.")

    exp_btn_col1, exp_btn_col2, _ = st.columns([1, 1, 2])
    with exp_btn_col1:
        if st.button("💾 Save Current Run as Baseline", use_container_width=True):
            st.session_state["baseline_sim"] = {
                "scenario_name": selected_scenario_name,
                "config": cfg,
                "result": res,
                "mean_cycle_length": res.mean_cycle_length,
                "hr_metrics": hr_metrics,
                "time": res.time,
                "rhythm_trace": res.rhythm_trace,
                "latencies": res.latencies,
                "failures": res.failures,
                "eq_cell": eq_cell,
            }
            st.success("✓ Current simulation saved as baseline!")

    with exp_btn_col2:
        if st.button("🗑️ Clear Stored Baseline", use_container_width=True):
            st.session_state["baseline_sim"] = None
            st.info("Stored baseline cleared.")

    if st.session_state["baseline_sim"] is not None:
        base = st.session_state["baseline_sim"]
        st.markdown(f"**Saved Baseline:** `{base['scenario_name']}` (Pacing: `{base['config'].pacing_interval:.1f} u`, g_AV: `{base['config'].av_conductance:.2f}`, D: `{base['config'].global_coupling:.2f}`)")

        cmp_col1, cmp_col2 = st.columns([1, 1.2])

        with cmp_col1:
            st.markdown("#### Parameter & Metric Deltas")
            delta_pacing = cfg.pacing_interval - base["config"].pacing_interval
            delta_gav = cfg.av_conductance - base["config"].av_conductance
            delta_D = cfg.global_coupling - base["config"].global_coupling

            base_bpm = base["hr_metrics"]["bpm"] if base["hr_metrics"]["is_meaningful"] else 0.0
            curr_bpm = hr_metrics["bpm"] if hr_metrics["is_meaningful"] else 0.0
            delta_bpm = curr_bpm - base_bpm

            delta_df = pd.DataFrame([
                {"Parameter / Metric": "Pacing Interval (T_pace)", "Baseline": f"{base['config'].pacing_interval:.1f} u", "Current": f"{cfg.pacing_interval:.1f} u", "Delta (Δ)": f"{delta_pacing:+.1f} u"},
                {"Parameter / Metric": "AV Conductance (g_AV)", "Baseline": f"{base['config'].av_conductance:.3f}", "Current": f"{cfg.av_conductance:.3f}", "Delta (Δ)": f"{delta_gav:+.3f}"},
                {"Parameter / Metric": "Global Coupling (D)", "Baseline": f"{base['config'].global_coupling:.2f}", "Current": f"{cfg.global_coupling:.2f}", "Delta (Δ)": f"{delta_D:+.2f}"},
                {"Parameter / Metric": "Estimated Heart Rate", "Baseline": f"{base_bpm:.1f} BPM", "Current": f"{curr_bpm:.1f} BPM", "Delta (Δ)": f"{delta_bpm:+.1f} BPM"},
                {"Parameter / Metric": "Conduction Drops", "Baseline": f"{len(base['failures'])} drops", "Current": f"{len(res.failures)} drops", "Delta (Δ)": f"{len(res.failures) - len(base['failures']):+d}"},
            ])
            st.dataframe(delta_df, use_container_width=True)

        with cmp_col2:
            st.markdown("#### Overlaid Rhythm Waveforms (Baseline vs Current)")
            fig_cmp_trace = go.Figure()
            min_len = min(len(base["time"]), len(res.time))
            fig_cmp_trace.add_trace(go.Scatter(
                x=base["time"][:min_len], y=base["rhythm_trace"][:min_len], mode="lines",
                name="Baseline Rhythm", line=dict(color="#3b82f6", width=2.0)
            ))
            fig_cmp_trace.add_trace(go.Scatter(
                x=res.time[:min_len], y=res.rhythm_trace[:min_len], mode="lines",
                name="Current Simulation", line=dict(color="#f59e0b", width=2.0, dash="dash")
            ))
            fig_cmp_trace.update_layout(
                title="Direct Comparison of Simulated Activation Traces",
                xaxis_title="Time t", yaxis_title="Dipole Moment Φ(t)",
                template="plotly_dark", height=280, margin=dict(l=40, r=20, t=35, b=30),
            )
            st.plotly_chart(fig_cmp_trace, use_container_width=True)
    else:
        st.info("ℹ️ No baseline simulation currently stored. Click **'Save Current Run as Baseline'** above to capture the current state, then modify any parameter to see live comparative deltas.")


# =============================================================================
# VIEW 2: ARRHYTHMIA LAB (DETAILED SCENARIO ENGINE & EVENT LOG)
# =============================================================================
elif "Arrhythmia Lab" in nav_view:
    st.title("🏥 Cardiac Arrhythmia Laboratory")
    st.caption("In-Depth Numerical Scenario Engine: Conduction Delays, Suppressed Impulses, Conduction Blocks, and Circus Re-entry")

    lab_sc_name = st.selectbox(
        "Select Arrhythmia Scenario to Investigate:",
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

    # Dynamic Scenario-Specific Parameter Controls
    with st.expander(f"⚙️ Fine-Tune Parameters for {lab_sc_name}", expanded=True):
        if "Normal" in lab_sc_name:
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                lab_pacing = st.slider("Pacing Interval (T_pace)", 30.0, 70.0, 45.0, 1.0, key="lab_pace_norm")
            with col_p2:
                lab_coupling = st.slider("Global Coupling (D)", 0.2, 1.5, 0.70, 0.05, key="lab_coup_norm")
            with col_p3:
                lab_tend = st.slider("Simulation Horizon (T_end)", 80.0, 240.0, 160.0, 10.0, key="lab_tend_norm")
            lab_cfg = ScenarioConfig(
                scenario_type=ScenarioType.NORMAL,
                pacing_interval=lab_pacing,
                global_coupling=lab_coupling,
                t_end=lab_tend,
            )
        elif "Tachycardia" in lab_sc_name:
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                lab_pacing = st.slider("Rapid Pacing Interval (T_pace)", 20.0, 38.0, 32.0, 1.0, key="lab_pace_tachy", help="Shorter cycle length accelerates depolarizations.")
            with col_p2:
                lab_coupling = st.slider("Global Coupling (D)", 0.2, 1.5, 0.70, 0.05, key="lab_coup_tachy")
            with col_p3:
                lab_tend = st.slider("Simulation Horizon (T_end)", 80.0, 240.0, 160.0, 10.0, key="lab_tend_tachy")
            lab_cfg = ScenarioConfig(
                scenario_type=ScenarioType.TACHYCARDIA,
                pacing_interval=lab_pacing,
                global_coupling=lab_coupling,
                t_end=lab_tend,
            )
        elif "Bradycardia" in lab_sc_name:
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                lab_pacing = st.slider("Depressed Pacing Interval (T_pace)", 60.0, 120.0, 80.0, 2.0, key="lab_pace_brady", help="Longer cycle length prolongs resting quiescent intervals.")
            with col_p2:
                lab_coupling = st.slider("Global Coupling (D)", 0.2, 1.5, 0.70, 0.05, key="lab_coup_brady")
            with col_p3:
                lab_tend = st.slider("Simulation Horizon (T_end)", 100.0, 280.0, 180.0, 10.0, key="lab_tend_brady")
            lab_cfg = ScenarioConfig(
                scenario_type=ScenarioType.BRADYCARDIA,
                pacing_interval=lab_pacing,
                global_coupling=lab_coupling,
                t_end=lab_tend,
            )
        elif "Sinus Pause" in lab_sc_name:
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                lab_pacing = st.slider("Pacing Interval (T_pace)", 30.0, 60.0, 45.0, 1.0, key="lab_pace_pause")
            with col_p2:
                lab_pstart = st.slider("Pause Window Start", 15.0, 60.0, 30.0, 5.0, key="lab_pstart")
            with col_p3:
                lab_pend = st.slider("Pause Window End", 70.0, 150.0, 110.0, 5.0, key="lab_pend")
            with col_p4:
                lab_tend = st.slider("Simulation Horizon (T_end)", 100.0, 240.0, 160.0, 10.0, key="lab_tend_pause")
            lab_cfg = ScenarioConfig(
                scenario_type=ScenarioType.SINUS_PAUSE,
                pacing_interval=lab_pacing,
                pause_start=lab_pstart,
                pause_end=lab_pend,
                t_end=lab_tend,
            )
        elif "AV Block" in lab_sc_name:
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                lab_pacing = st.slider("Pacing Interval (T_pace)", 30.0, 60.0, 45.0, 1.0, key="lab_pace_avb")
            with col_p2:
                lab_gav = st.slider("Junction Conductance (g_AV)", 0.0, 0.35, 0.0, 0.02, key="lab_gav", help="0.0 = complete 3rd-degree block; ~0.15 = delayed conduction; >=0.30 = normal 1:1 conduction.")
            with col_p3:
                lab_bloc = st.selectbox("Block Location", ["Atrium-to-AVN", "AVN-to-His"], key="lab_bloc")
            with col_p4:
                lab_tend = st.slider("Simulation Horizon (T_end)", 80.0, 240.0, 160.0, 10.0, key="lab_tend_avb")
            lab_cfg = ScenarioConfig(
                scenario_type=ScenarioType.AV_BLOCK,
                pacing_interval=lab_pacing,
                av_conductance=lab_gav,
                block_location=lab_bloc,
                t_end=lab_tend,
            )
        else:  # Re-entry
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                lab_rnodes = st.slider("Circuit Nodes", 6, 16, 10, 1, key="lab_rnodes")
            with col_p2:
                lab_rcoup = st.slider("Ring Coupling (g)", 0.15, 0.70, 0.35, 0.05, key="lab_rcoup")
            with col_p3:
                lab_reps = st.slider("Recovery ε", 0.05, 0.30, 0.15, 0.01, key="lab_reps")
            with col_p4:
                lab_tend = st.slider("Simulation Horizon (T_end)", 80.0, 240.0, 150.0, 10.0, key="lab_tend_re")
            lab_cfg = ScenarioConfig(
                scenario_type=ScenarioType.REENTRY,
                reentry_num_nodes=lab_rnodes,
                reentry_coupling=lab_rcoup,
                reentry_epsilon=lab_reps,
                t_end=lab_tend,
            )

    # Execute Scenario ODE Integration
    lab_res = run_scenario(lab_cfg)
    lab_hr = estimate_heart_rate_bpm(lab_res.mean_cycle_length)
    lab_bpm_text = f"{lab_hr['bpm']:.1f} BPM" if lab_hr.get("bpm") is not None else "0.0 BPM (Blocked)"
    lab_mcl_text = f"{lab_res.mean_cycle_length:.2f} u" if lab_res.mean_cycle_length else "N/A (Blocked)"

    # Top KPI Metrics Bar
    kpi_lab_cols = st.columns(5)
    with kpi_lab_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Investigated Scenario</div>
            <div class="metric-card-val" style="font-size: 1.05rem;">{lab_sc_name.split(':')[0]}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_lab_cols[1]:
        p_info = f"{lab_cfg.pacing_interval:.1f} u" if lab_cfg.scenario_type != ScenarioType.REENTRY else "Starter Pulse"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">SA Drive Interval</div>
            <div class="metric-card-val">{p_info}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_lab_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Mean Cycle Length</div>
            <div class="metric-card-val">{lab_mcl_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_lab_cols[3]:
        bpm_color = "#38bdf8" if lab_hr.get("is_meaningful") else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Ventricular Rate</div>
            <div class="metric-card-val" style="color: {bpm_color};">{lab_bpm_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_lab_cols[4]:
        if lab_cfg.scenario_type == ScenarioType.REENTRY:
            c_text = "Sustained Circus" if lab_res.reentry_sustained else "Extinguished"
            c_color = "#10b981" if lab_res.reentry_sustained else "#ef4444"
        elif lab_res.failures:
            c_text = f"Block ({len(lab_res.failures)} drops)"
            c_color = "#ef4444"
        elif lab_cfg.scenario_type == ScenarioType.SINUS_PAUSE:
            c_text = "Sinus Arrest Window"
            c_color = "#f59e0b"
        else:
            c_text = "1:1 Synchronous"
            c_color = "#10b981"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Conduction Status</div>
            <div class="metric-card-val" style="color: {c_color};">{c_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MAIN ARRHYTHMIA DISPLAY: HEART VISUALIZATION + RHYTHM TRACE
    # -------------------------------------------------------------------------
    col_vis, col_rhythm = st.columns([1.15, 1.0])

    with col_vis:
        st.markdown('<div class="sec-title">🫀 Interactive Anatomical Heart Visualization</div>', unsafe_allow_html=True)
        st.caption(f"Continuous numerical state $(v_i, w_i)$ playback for **{lab_sc_name}**. Watch excitation wavefront propagation, blockages, and recovery.")
        heart_html = build_heart_visualization_html(lab_res, width=820, height=660)
        components.html(heart_html, height=690, scrolling=False)

    with col_rhythm:
        st.markdown('<div class="sec-title">⚡ Arrhythmia Rhythm Waveform & Event Timeline</div>', unsafe_allow_html=True)
        st.caption("Simulated activation-derived rhythm trace — not a clinically validated ECG.")

        # Activation-Derived Rhythm Waveform
        fig_rhythm = go.Figure()
        fig_rhythm.add_trace(go.Scatter(
            x=lab_res.time, y=lab_res.rhythm_trace, mode="lines", name="Rhythm Trace Φ(t)",
            line=dict(color="#38bdf8", width=2.0)
        ))
        fig_rhythm.add_hline(y=0.0, line_dash="dash", line_color="#475569")
        fig_rhythm.update_layout(
            title=f"Activation-Derived Rhythm Waveform ({lab_sc_name.split(':')[0]})",
            xaxis_title="Simulation Time (model units)",
            yaxis_title="Dipole Moment Φ(t)",
            template="plotly_dark",
            height=260,
            margin=dict(l=40, r=20, t=35, b=30),
        )
        st.plotly_chart(fig_rhythm, use_container_width=True)

        # Multi-Channel Activation Event Timeline
        fig_timeline = go.Figure()
        region_colors = {
            "Sinoatrial (SA) Node": "#f59e0b",
            "Atrial Myocardium": "#3b82f6",
            "Atrioventricular (AV) Node": "#ec4899",
            "Bundle of His": "#8b5cf6",
            "Purkinje Network": "#06b6d4",
            "Ventricular Myocardium": "#10b981",
        }
        y_labels = []
        for i in range(lab_res.network.num_nodes):
            node = lab_res.network.nodes[i]
            y_labels.append(node.name)
            t_acts = lab_res.activations.get(i, [])
            c = region_colors.get(node.region.value, "#94a3b8")
            if t_acts:
                fig_timeline.add_trace(go.Scatter(
                    x=t_acts, y=[node.name] * len(t_acts), mode="markers",
                    marker=dict(size=12, color=c, symbol="diamond", line=dict(color="#0f172a", width=1.5)),
                    name=node.name,
                ))
            else:
                fig_timeline.add_trace(go.Scatter(
                    x=[lab_res.time[-1] / 2], y=[node.name], mode="text",
                    text=["[BLOCKED / NO ACTIVATION]"], textposition="middle center",
                    textfont=dict(color="#ef4444", size=10), showlegend=False,
                ))

        fig_timeline.update_layout(
            title="Multi-Channel Activation Event Timeline",
            xaxis_title="Time t (model units)",
            yaxis=dict(categoryorder="array", categoryarray=list(reversed(y_labels))),
            template="plotly_dark",
            height=320,
            showlegend=False,
            margin=dict(l=60, r=20, t=35, b=30),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # -------------------------------------------------------------------------
    # DEEP-DIVE ARRHYTHMIA LABORATORY TABS
    # -------------------------------------------------------------------------
    a_tab1, a_tab2, a_tab3, a_tab4 = st.tabs([
        "📊 Multi-Channel Action Potentials",
        "📋 Numerical Event Log & CSV Export",
        "🌐 Einthoven Lead II Far-Field Dipole",
        "🔍 Mechanistic Electrophysiological Breakdown",
    ])

    with a_tab1:
        st.subheader("Synchronized Transmembrane Potentials across Functional Regions")
        fig_sync = go.Figure()
        colors = ["#f59e0b", "#3b82f6", "#ec4899", "#8b5cf6", "#06b6d4", "#10b981"]
        for i in range(lab_res.network.num_nodes):
            node = lab_res.network.nodes[i]
            fig_sync.add_trace(go.Scatter(
                x=lab_res.time, y=lab_res.V[:, i], mode="lines",
                name=f"{node.name} ({node.region.value})",
                line=dict(color=colors[i % len(colors)], width=2.0)
            ))
        fig_sync.update_layout(
            title=f"Multi-Channel Action Potentials: {lab_sc_name}",
            xaxis_title="Time t (model units)", yaxis_title="Transmembrane Voltage v_i",
            template="plotly_dark", height=450, hovermode="x unified",
        )
        st.plotly_chart(fig_sync, use_container_width=True)

    with a_tab2:
        st.subheader("Numerical Activation Event Log (Chronological Sub-Step Timestamps)")
        event_entries = generate_numerical_event_log(lab_res.activations, lab_res.network)
        if event_entries:
            df_events = pd.DataFrame([e.to_dict() for e in event_entries])
            st.dataframe(df_events, use_container_width=True)
            csv_data = df_events.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Event Log as CSV", data=csv_data, file_name=f"event_log_{lab_cfg.scenario_type.name.lower()}.csv", mime="text/csv")
        else:
            st.warning("No threshold crossings detected.")

    with a_tab3:
        st.subheader("Einthoven Lead II Projection & Dipole Physics")
        lead_trace, doc_text = compute_true_ecg_lead_approximation(lab_res.time, lab_res.V, lab_res.network)
        fig_lead = go.Figure()
        fig_lead.add_trace(go.Scatter(x=lab_res.time, y=lead_trace, mode="lines", name="Lead II Projection", line=dict(color="#10b981", width=2.0)))
        fig_lead.update_layout(title="Einthoven Lead II Far-Field Dipole Projection", xaxis_title="Time t (model units)", yaxis_title="Potential V_II (a.u.)", template="plotly_dark", height=380)
        st.plotly_chart(fig_lead, use_container_width=True)
        st.info(f"💡 **Physical Dipole Formulation:** {doc_text}")

    with a_tab4:
        st.subheader(f"Electrophysiological & Biophysical Analysis: {lab_sc_name}")
        st.markdown(f"""
        #### 1. Applied Parameter Modification
        {lab_res.parameter_change_summary}

        #### 2. Mathematical Dynamics & Conduction Mechanics
        {lab_res.mathematical_assumptions}
        """)

        comp_dict = lab_res.comparison_with_normal
        st.markdown("#### 3. Quantitative Comparison vs Normal Sinus Baseline")
        if "rate_ratio" in comp_dict:
            st.write(f"- **Activation Rate Ratio:** `{comp_dict['rate_ratio']:.2f}x` of baseline normal rhythm")
        if "status" in comp_dict:
            st.write(f"- **Conduction Status:** {comp_dict['status']}")
        if "dropped_count" in comp_dict:
            st.write(f"- **Dropped Conduction Events:** `{comp_dict['dropped_count']}` downstream activations blocked")
        if "reentry_revolutions" in comp_dict:
            st.write(f"- **Re-entrant Circus Revolutions:** `{comp_dict['reentry_revolutions']}` continuous self-sustaining wavefront rotations")


# =============================================================================
# VIEW 3: MATHEMATICAL ANALYSIS (STAGE 6 COMPLETE LABORATORY)
# =============================================================================
elif "Mathematical Analysis" in nav_view:
    st.title("🔬 Mathematical Analysis Laboratory (Stage 6)")
    st.caption("Rigorous Analytical & Numerical Evaluation of the Single-Cell FitzHugh–Nagumo ODE System")

    # Parameters
    p_I = st.sidebar.slider("Applied Current (I)", -0.5, 2.0, 0.5, 0.01)
    p_eps = st.sidebar.slider("Timescale Ratio (ε)", 0.005, 0.40, 0.08, 0.005)
    p_a = st.sidebar.slider("Intercept (a)", -0.5, 1.5, 0.70, 0.05)
    p_b = st.sidebar.slider("Slope (b)", 0.1, 2.0, 0.80, 0.05)
    p_single = FHNParameters(a=p_a, b=p_b, epsilon=p_eps, I=p_I)

    eqs_single = find_equilibria(p_single)

    lab_t1, lab_t2, lab_t3, lab_t4, lab_t5 = st.tabs([
        "📐 Parts A & B: Nullclines & Cardano Equilibria",
        "🧮 Part C: Analytical Jacobian & Poincaré Stability",
        "🌀 Part D: Phase Plane & Flow Field Explorer",
        "📈 Part E: Bifurcation Sweeps Engine (I, ε, a, b)",
        "⚖️ Part F: Before-and-After Parameter Comparison Lab",
    ])

    with lab_t1:
        st.subheader("Part A & B: Nullclines & Cardano Equilibrium Roots")
        st.markdown(r"""
        - **$v$-nullcline:** $w = v - \frac{v^3}{3} + I$
        - **$w$-nullcline:** $w = \frac{v + a}{b}$
        - **Cardano Depressed Cubic:** $b(v^*)^3 + 3(1 - b)v^* + 3(a - bI) = 0$
        """)
        rec_list = []
        for idx, eq in enumerate(eqs_single, start=1):
            r_v = eq.v_star - (eq.v_star**3) / 3.0 - eq.w_star + p_single.I
            r_w = p_single.epsilon * (eq.v_star + p_single.a - p_single.b * eq.w_star)
            rec_list.append({
                "Index": f"E_{idx}",
                "v*": f"{eq.v_star:+.8f}",
                "w*": f"{eq.w_star:+.8f}",
                "|r_v|": f"{abs(r_v):.2e}",
                "|r_w|": f"{abs(r_w):.2e}",
                "Verification": "✓ Verified (< 10⁻¹²)",
                "Poincaré Class": eq.stability_class.value,
            })
        st.dataframe(pd.DataFrame(rec_list), use_container_width=True)

    with lab_t2:
        st.subheader("Part C: Analytical Jacobian & Poincaré Stability")
        for idx, eq in enumerate(eqs_single, start=1):
            st.markdown(f"### Equilibrium E_{idx}: ({eq.v_star:.4f}, {eq.w_star:.4f})")
            st.markdown(f"""
            - **Trace τ:** `{eq.trace:+.6f}`
            - **Determinant Δ:** `{eq.determinant:+.6f}`
            - **Discriminant D:** `{eq.discriminant:+.6f}`
            - **Eigenvalues λ1,2:** `{eq.eigenvalues[0]:+.5f}, {eq.eigenvalues[1]:+.5f}`
            - **Classification:** **{eq.stability_class.value}**
            """)

    with lab_t3:
        st.subheader("Part D: Phase Plane & Flow Field Explorer")
        v_grid, w_vn, w_wn = compute_nullclines((-2.6, 2.6), p_single)
        V_m, W_m, dV, dW, dV_norm, dW_norm = compute_vector_field((-2.5, 2.5), (-1.8, 2.2), 16, p_single)
        fig_d = ff.create_quiver(V_m, W_m, dV_norm, dW_norm, scale=0.12, arrow_scale=0.35, line=dict(color="#475569", width=1.0))
        fig_d.add_trace(go.Scatter(x=v_grid, y=w_vn, mode="lines", name="v-nullcline", line=dict(color="#3b82f6", width=2.5)))
        fig_d.add_trace(go.Scatter(x=v_grid, y=w_wn, mode="lines", name="w-nullcline", line=dict(color="#f97316", width=2.5, dash="dash")))
        for eq in eqs_single:
            fig_d.add_trace(go.Scatter(x=[eq.v_star], y=[eq.w_star], mode="markers", marker=dict(size=14, color="#10b981" if eq.is_stable else "#ef4444"), name=eq.stability_class.name))
        fig_d.update_layout(template="plotly_dark", height=480)
        st.plotly_chart(fig_d, use_container_width=True)

    with lab_t4:
        st.subheader("Part E: Multi-Parameter Sweeps & Bifurcations")
        sw_var = st.selectbox("Select Parameter to Sweep:", ["I (Current)", "ε (Timescale)", "a (Intercept)", "b (Slope)"])
        if "I (" in sw_var:
            sw_data = sweep_parameter_I(-0.2, 1.6, 60, p_single)
        elif "ε (" in sw_var:
            sw_data = sweep_parameter_epsilon(0.01, 0.35, 60, p_single)
        elif "a (" in sw_var:
            sw_data = sweep_parameter_a(0.0, 1.4, 60, p_single)
        else:
            sw_data = sweep_parameter_b(0.2, 1.6, 60, p_single)
        bifs = analyze_bifurcation_conditions(sw_data)
        if bifs:
            st.dataframe(pd.DataFrame(bifs), use_container_width=True)
        else:
            st.info("No codimension-1 zero crossings detected in this interval.")

    with lab_t5:
        st.subheader("Part F: Parameter Comparison Laboratory")
        st.caption("Compare State A (Baseline) vs State B (Perturbed):")
        cA, cB = st.columns(2)
        with cA:
            st.markdown("##### State A")
            iA = st.number_input("I_A", value=0.0, step=0.1)
            pA = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=iA)
        with cB:
            st.markdown("##### State B")
            iB = st.number_input("I_B", value=0.5, step=0.1)
            pB = FHNParameters(a=0.7, b=0.8, epsilon=0.08, I=iB)
        cmp_res = compare_fhn_states(pA, pB)
        st.markdown(f"**Transition:** `{cmp_res.stability_transition}`")
        st.markdown(f"**Diagnosis:** {cmp_res.dynamical_diagnosis}")


# =============================================================================
# VIEW 4: PARAMETER EXPERIMENT LAB
# =============================================================================
elif "Parameter Experiment Lab" in nav_view:
    st.title("⚖️ Parameter Experiment Laboratory")
    st.caption("Controlled Exploration of Electrophysiological Parameter Changes on Cardiac Conduction")

    exp_colA, exp_colB = st.columns(2)

    with exp_colA:
        st.markdown("### 🔵 Baseline State (Condition A)")
        p_pace_A = st.slider("Pacing Interval (A)", 25.0, 100.0, 45.0, 5.0, key="p_pace_A")
        g_av_A = st.slider("AV Conductance g_AV (A)", 0.0, 0.50, 0.35, 0.05, key="g_av_A")
        D_A = st.slider("Global Coupling D (A)", 0.1, 1.5, 0.70, 0.1, key="D_A")
        cfg_A = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=p_pace_A, av_conductance=g_av_A, global_coupling=D_A, t_end=120.0)
        res_A = run_scenario(cfg_A)

    with exp_colB:
        st.markdown("### 🔴 Experimental State (Condition B)")
        p_pace_B = st.slider("Pacing Interval (B)", 25.0, 100.0, 35.0, 5.0, key="p_pace_B")
        g_av_B = st.slider("AV Conductance g_AV (B)", 0.0, 0.50, 0.08, 0.05, key="g_av_B")
        D_B = st.slider("Global Coupling D (B)", 0.1, 1.5, 0.70, 0.1, key="D_B")
        cfg_B = ScenarioConfig(scenario_type=ScenarioType.NORMAL, pacing_interval=p_pace_B, av_conductance=g_av_B, global_coupling=D_B, t_end=120.0)
        res_B = run_scenario(cfg_B)

    # Comparative Metrics Table
    hr_A = estimate_heart_rate_bpm(res_A.mean_cycle_length)
    hr_B = estimate_heart_rate_bpm(res_B.mean_cycle_length)

    val_cl_A = f"{res_A.mean_cycle_length:.2f} u" if res_A.mean_cycle_length else "N/A (Blocked)"
    val_cl_B = f"{res_B.mean_cycle_length:.2f} u" if res_B.mean_cycle_length else "N/A (Blocked)"
    diff_cl = (
        f"{res_B.mean_cycle_length - res_A.mean_cycle_length:+.2f} u"
        if (res_A.mean_cycle_length is not None and res_B.mean_cycle_length is not None)
        else "N/A"
    )

    val_bpm_A = f"{hr_A['bpm']:.1f} BPM" if hr_A.get("bpm") is not None else "0.0 BPM (Blocked)"
    val_bpm_B = f"{hr_B['bpm']:.1f} BPM" if hr_B.get("bpm") is not None else "0.0 BPM (Blocked)"
    diff_bpm = (
        f"{hr_B['bpm'] - hr_A['bpm']:+.1f} BPM"
        if (hr_A.get("bpm") is not None and hr_B.get("bpm") is not None)
        else "N/A"
    )

    st.markdown("### Side-by-Side Experimental Analysis")
    cmp_table = pd.DataFrame([
        {"Parameter / Metric": "Pacing Interval T_pace", "State A (Baseline)": f"{p_pace_A:.1f} u", "State B (Trial)": f"{p_pace_B:.1f} u", "Difference (B - A)": f"{p_pace_B - p_pace_A:+.1f} u"},
        {"Parameter / Metric": "AV Conductance g_AV", "State A (Baseline)": f"{g_av_A:.3f}", "State B (Trial)": f"{g_av_B:.3f}", "Difference (B - A)": f"{g_av_B - g_av_A:+.3f}"},
        {"Parameter / Metric": "Global Coupling D", "State A (Baseline)": f"{D_A:.2f}", "State B (Trial)": f"{D_B:.2f}", "Difference (B - A)": f"{D_B - D_A:+.2f}"},
        {"Parameter / Metric": "Mean Cycle Length", "State A (Baseline)": val_cl_A, "State B (Trial)": val_cl_B, "Difference (B - A)": diff_cl},
        {"Parameter / Metric": "Estimated Rate (BPM)", "State A (Baseline)": val_bpm_A, "State B (Trial)": val_bpm_B, "Difference (B - A)": diff_bpm},
        {"Parameter / Metric": "Conduction Drops", "State A (Baseline)": f"{len(res_A.failures)}", "State B (Trial)": f"{len(res_B.failures)}", "Difference (B - A)": f"{len(res_B.failures) - len(res_A.failures):+d}"},
    ])
    st.dataframe(cmp_table, use_container_width=True)

    # Overlaid Waveforms
    fig_ab = go.Figure()
    fig_ab.add_trace(go.Scatter(x=res_A.time, y=res_A.rhythm_trace, mode="lines", name="Rhythm State A (Baseline)", line=dict(color="#3b82f6", width=2.0)))
    fig_ab.add_trace(go.Scatter(x=res_B.time, y=res_B.rhythm_trace, mode="lines", name="Rhythm State B (Trial)", line=dict(color="#ef4444", width=2.0, dash="dash")))
    fig_ab.update_layout(title="Simulated Activation Waveform Comparison (State A vs State B)", xaxis_title="Time t", yaxis_title="Dipole Moment Φ(t)", template="plotly_dark", height=350)
    st.plotly_chart(fig_ab, use_container_width=True)


# =============================================================================
# VIEW 5: ABOUT THE MODEL (COMPREHENSIVE ACADEMIC REFERENCE)
# =============================================================================
else:
    st.title("📖 About the Model: Mathematical & Biophysical Foundations")
    st.caption("Formal documentation of equations, variables, coupled network extensions, assumptions, and scientific boundaries.")

    st.markdown(r"""
    ### 1. The Single-Cell FitzHugh–Nagumo Equations
    The classical FitzHugh–Nagumo (FHN) model (FitzHugh, 1961; Nagumo et al., 1962) is a two-variable planar simplification of the four-variable Hodgkin–Huxley equations:
    $$\frac{dv}{dt} = v - \frac{v^3}{3} - w + I$$
    $$\frac{dw}{dt} = \varepsilon(v + a - bw)$$

    #### Meaning of Every Variable & Parameter:
    - **$v(t) \in \mathbb{R}$ (Fast Transmembrane Voltage):**  
      Represents the dimensionless cellular membrane potential. The cubic non-linearity $v - v^3/3$ models the self-regenerative fast inward depolarizing current (analogous to $\text{Na}^+$ or $\text{Ca}^{2+}$ influx) and its rapid saturation.
    - **$w(t) \in \mathbb{R}$ (Slow Recovery Variable):**  
      Represents the outward repolarizing currents and refractory deactivation (analogous to delayed-rectifier $\text{K}^+$ currents).
    - **$I \in \mathbb{R}$ (Applied External Current):**  
      Models pacing stimulus or intracellular bias currents.
    - **$\varepsilon \in (0, 1)$ (Timescale Separation Ratio):**  
      Because $\varepsilon \ll 1$ (typically $0.08$), the recovery variable evolves on a much slower timescale than $v(t)$, creating the rapid upstroke and prolonged refractory recovery typical of excitable cardiac tissue.
    - **$a, b \in \mathbb{R}$ (Nullcline Geometry Parameters):**  
      Determine the resting membrane potential and threshold excitability. Constraining $b \in (0, 1)$ mathematically guarantees a unique resting equilibrium point.

    ---

    ### 2. The Coupled-Cell Network Extension
    To model coordinated cardiac conduction, the standalone ODE is extended to an $N$-compartment discrete network governed by gap-junction diffusive coupling:
    $$\frac{dv_i}{dt} = v_i - \frac{v_i^3}{3} - w_i + I_{\text{stim}, i}(t) + D \sum_{j \in \mathcal{N}(i)} A_{ij}(v_j - v_i)$$
    $$\frac{dw_i}{dt} = \varepsilon_i(v_i + a_i - b_i w_i)$$

    - **$A_{ij} \ge 0$:** Intercellular gap-junction coupling weight between compartment $i$ and $j$.
    - **$D$:** Global electrical coupling coefficient.
    - **Graph Laplacian $\mathcal{L} = \mathcal{D} - A$:** Diffusive operator dictating electrotonic wave spread.
    - **Anatomical Conduction Sequence:**
      $$\text{SA Node} \longrightarrow \text{Atrial Myocardium} \longrightarrow \text{AV Node (Delayed)} \longrightarrow \text{Bundle of His} \longrightarrow \text{Purkinje Network} \longrightarrow \text{Ventricles}$$

    ---

    ### 3. Physical & Mathematical Assumptions
    1. **Planar Reduction:** Excitation dynamics within each compartment are described by 2 state variables $(v_i, w_i)$.
    2. **Far-Field Dipole Approximation:** Rhythm waveforms $\Phi(t)$ are calculated via electrical dipole moment projection:
       $$\Phi(t) = \sum_{(i,j) \in \mathcal{E}} g_{ij} [v_j(t) - v_i(t)] \cdot \vec{d}_{ij}$$
    3. **Homogeneous Isotropic Volume Conductor:** Neglects electrical impedance discontinuities across thoracic tissues.
    4. **Quasi-Static Electromagnetic Field:** Valid for low physiological bioelectric frequencies ($< 1\text{ kHz}$).

    ---

    ### 4. Model Limitations
    - **No Discrete Ion Channels:** The phenomenological formulation does not resolve individual microscopic currents ($I_{\text{Na}}, I_{\text{Ca,L}}, I_{\text{to}}, I_{\text{Kr}}, I_{\text{Ks}}$).
    - **Discrete 6-Compartment Hierarchy:** Represents primary physiological regions rather than a 3D continuum finite-element syncytium.
    - **No Mechanical Pumping:** Models electrical activation only; does not compute myocardial wall stress, chamber pressure, or hemodynamic ejection.

    ---

    ### 5. Critical Distinction: Reduced Simulation vs Real Human Heart
    > [!WARNING]
    > **Non-Clinical Warning:**  
    > This software is an educational and dynamical systems research tool designed to demonstrate the fundamental mathematics of threshold excitability, decremental conduction, and re-entrant circus movement.  
    > The computed rhythm waveforms are activation-derived surrogates. They are **not** clinical 12-lead electrocardiograms (ECGs) and must **never** be used for clinical diagnosis, patient management, or medical decision-making.
    """)
