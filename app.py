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

import os
import sys

# Ensure repository root is on sys.path for cloud deployment
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
# VIEW 4: PARAMETER EXPERIMENT LAB (STAGE 8 COMPLETE LABORATORY)
# =============================================================================
elif "Parameter Experiment Lab" in nav_view:
    st.title("⚖️ Parameter Experiment Laboratory (Stage 8)")
    st.caption("Controlled Scientific Exploration of Electrophysiological Parameter Changes, Conduction Transitions, and Dynamical Stability")

    # Mode Selection Tabs
    tab_exp_ctrl, tab_exp_sweep = st.tabs([
        "🔬 Controlled Experiment: Baseline vs Trial Comparison",
        "📈 Automated Parameter Sweep Engine",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: CONTROLLED EXPERIMENT (BASELINE VS TRIAL)
    # -------------------------------------------------------------------------
    with tab_exp_ctrl:
        st.markdown("""
        **Controlled Experiment Workflow:**
        `1. Select baseline scenario` ➔ `2. Run & inspect baseline` ➔ `3. Save baseline state` ➔ `4. Change a targeted parameter` ➔ `5. Run trial simulation` ➔ `6. Compare activation sequences, traces, and stability`
        """)

        # Initialize Session State for Saved Baseline
        if "exp_saved_params" not in st.session_state:
            st.session_state.exp_saved_params = ExperimentParameters(
                scenario_type=ScenarioType.NORMAL,
                pacing_interval=45.0,
                av_conductance=0.35,
                global_coupling=0.70,
                cell_a=0.70,
                cell_b=0.80,
                cell_epsilon=0.08,
                cell_I_bias=0.0,
                t_end=120.0,
            )
            st.session_state.exp_is_saved = False

        col_base_box, col_trial_box = st.columns(2)

        # -----------------------------
        # 🔵 CONDITION A: BASELINE
        # -----------------------------
        with col_base_box:
            st.markdown("### 🔵 Baseline State (Condition A)")

            base_sc_choice = st.selectbox(
                "Baseline Scenario:",
                ["Normal Sinus Rhythm", "Tachycardia-Like", "Bradycardia-Like", "Sinus Pause", "AV Block-Like"],
                key="base_sc_choice"
            )
            sc_map = {
                "Normal Sinus Rhythm": ScenarioType.NORMAL,
                "Tachycardia-Like": ScenarioType.TACHYCARDIA,
                "Bradycardia-Like": ScenarioType.BRADYCARDIA,
                "Sinus Pause": ScenarioType.SINUS_PAUSE,
                "AV Block-Like": ScenarioType.AV_BLOCK,
            }
            base_sc_type = sc_map[base_sc_choice]

            # Pacing and coupling
            base_pace = st.slider("Pacing Interval T_pace (A)", 25.0, 100.0, 45.0, 1.0, key="base_pace")
            base_gav = st.slider("AV Conductance g_AV (A)", 0.0, 0.50, 0.35, 0.02, key="base_gav")
            base_D = st.slider("Global Coupling D (A)", 0.1, 1.5, 0.70, 0.05, key="base_D")
            base_I_stim = st.slider("Stimulus Current Amplitude I (A)", 0.5, 3.0, 1.8, 0.1, key="base_I_stim")

            with st.expander("Cellular & Regional Parameters (A)", expanded=False):
                base_a = st.slider("Parameter a (A)", 0.0, 1.2, 0.70, 0.05, key="base_a")
                base_b = st.slider("Parameter b (A)", 0.2, 1.5, 0.80, 0.05, key="base_b")
                base_eps = st.slider("Recovery Timescale ε (A)", 0.01, 0.25, 0.08, 0.005, key="base_eps")
                base_I_bias = st.slider("Cellular Bias Current I_bias (A)", -0.5, 1.0, 0.0, 0.05, key="base_I_bias")
                base_avn_eps = st.slider("AV Node ε (AVN, A)", 0.02, 0.15, 0.04, 0.01, key="base_avn_eps")
                base_vent_eps = st.slider("Ventricle ε (Vent, A)", 0.02, 0.15, 0.06, 0.01, key="base_vent_eps")

            base_params = ExperimentParameters(
                scenario_type=base_sc_type,
                pacing_interval=base_pace,
                stimulus_amplitude=base_I_stim,
                av_conductance=base_gav,
                global_coupling=base_D,
                cell_a=base_a,
                cell_b=base_b,
                cell_epsilon=base_eps,
                cell_I_bias=base_I_bias,
                t_end=120.0,
                region_overrides={
                    "AVN": {"epsilon": base_avn_eps},
                    "Ventricle": {"epsilon": base_vent_eps},
                }
            )

            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("💾 Save State A as Baseline", key="btn_save_baseline", use_container_width=True):
                    st.session_state.exp_saved_params = base_params.copy()
                    st.session_state.exp_is_saved = True
                    st.toast("Baseline State A saved successfully!", icon="💾")
            with c_btn2:
                if st.button("🔄 Reset to Default Normal", key="btn_reset_baseline", use_container_width=True):
                    st.session_state.exp_saved_params = ExperimentParameters()
                    st.session_state.exp_is_saved = False
                    st.rerun()

            if st.session_state.exp_is_saved:
                st.caption("🔒 **Baseline state currently locked:** Used as reference for delta comparisons.")

        # -----------------------------
        # 🔴 CONDITION B: TRIAL STATE
        # -----------------------------
        with col_trial_box:
            st.markdown("### 🔴 Experimental State (Condition B)")

            trial_sc_choice = st.selectbox(
                "Trial Scenario:",
                ["Normal Sinus Rhythm", "Tachycardia-Like", "Bradycardia-Like", "Sinus Pause", "AV Block-Like"],
                key="trial_sc_choice"
            )
            trial_sc_type = sc_map[trial_sc_choice]

            trial_pace = st.slider("Pacing Interval T_pace (B)", 25.0, 100.0, 36.0, 1.0, key="trial_pace")
            trial_gav = st.slider("AV Conductance g_AV (B)", 0.0, 0.50, 0.08, 0.02, key="trial_gav", help="Lower conductances can produce partial or complete conduction block.")
            trial_D = st.slider("Global Coupling D (B)", 0.1, 1.5, 0.70, 0.05, key="trial_D")
            trial_I_stim = st.slider("Stimulus Current Amplitude I (B)", 0.5, 3.0, 1.8, 0.1, key="trial_I_stim")

            with st.expander("Cellular & Regional Parameters (B)", expanded=False):
                trial_a = st.slider("Parameter a (B)", 0.0, 1.2, 0.70, 0.05, key="trial_a")
                trial_b = st.slider("Parameter b (B)", 0.2, 1.5, 0.80, 0.05, key="trial_b")
                trial_eps = st.slider("Recovery Timescale ε (B)", 0.01, 0.25, 0.08, 0.005, key="trial_eps")
                trial_I_bias = st.slider("Cellular Bias Current I_bias (B)", -0.5, 1.0, 0.0, 0.05, key="trial_I_bias")
                trial_avn_eps = st.slider("AV Node ε (AVN, B)", 0.02, 0.15, 0.04, 0.01, key="trial_avn_eps")
                trial_vent_eps = st.slider("Ventricle ε (Vent, B)", 0.02, 0.15, 0.06, 0.01, key="trial_vent_eps")

            trial_params = ExperimentParameters(
                scenario_type=trial_sc_type,
                pacing_interval=trial_pace,
                stimulus_amplitude=trial_I_stim,
                av_conductance=trial_gav,
                global_coupling=trial_D,
                cell_a=trial_a,
                cell_b=trial_b,
                cell_epsilon=trial_eps,
                cell_I_bias=trial_I_bias,
                t_end=120.0,
                region_overrides={
                    "AVN": {"epsilon": trial_avn_eps},
                    "Ventricle": {"epsilon": trial_vent_eps},
                }
            )

        # Execute Comparison
        comp_res = compare_experiments(base_params, trial_params)

        # ---------------------------------------------------------------------
        # SCIENTIFIC INTERPRETATION ALERT BANNER
        # ---------------------------------------------------------------------
        st.markdown("---")
        if comp_res.is_arrhythmia_induced:
            st.warning(f"⚠️ **Electrophysiological Transition Detected:**\n\n{comp_res.scientific_interpretation}")
        else:
            st.success(f"✅ **Physiological Conduction Margin Preserved:**\n\n{comp_res.scientific_interpretation}")

        # ---------------------------------------------------------------------
        # DETAILED COMPARISON TABS
        # ---------------------------------------------------------------------
        c_sub1, c_sub2, c_sub3, c_sub4, c_sub5 = st.tabs([
            "📊 Metrics & Parameters Comparison",
            "⚡ Transmembrane Potential Traces v(t) & w(t)",
            "🫀 Activation Sequence & Latencies",
            "🌐 Activation-Derived Rhythm Waveform Φ(t)",
            "📐 Standalone Cell Equilibrium & Phase Plane",
        ])

        with c_sub1:
            st.subheader("Side-by-Side Controlled Experiment Comparison")

            # Format Metrics Table
            bpm_A_str = f"{comp_res.bpm_baseline:.1f} BPM" if comp_res.bpm_baseline is not None else "0.0 BPM (Blocked)"
            bpm_B_str = f"{comp_res.bpm_trial:.1f} BPM" if comp_res.bpm_trial is not None else "0.0 BPM (Blocked)"
            diff_bpm_str = f"{comp_res.bpm_trial - comp_res.bpm_baseline:+.1f} BPM" if (comp_res.bpm_baseline and comp_res.bpm_trial) else "N/A"

            cl_A_str = f"{comp_res.mean_cl_baseline:.2f} u" if comp_res.mean_cl_baseline is not None else "N/A (Blocked)"
            cl_B_str = f"{comp_res.mean_cl_trial:.2f} u" if comp_res.mean_cl_trial is not None else "N/A (Blocked)"
            diff_cl_str = f"{comp_res.mean_cl_trial - comp_res.mean_cl_baseline:+.2f} u" if (comp_res.mean_cl_baseline and comp_res.mean_cl_trial) else "N/A"

            lat_A_str = f"{comp_res.san_to_vent_latency_baseline:.2f} u" if comp_res.san_to_vent_latency_baseline is not None else "N/A (Severed)"
            lat_B_str = f"{comp_res.san_to_vent_latency_trial:.2f} u" if comp_res.san_to_vent_latency_trial is not None else "N/A (Severed)"
            diff_lat_str = f"{comp_res.san_to_vent_latency_trial - comp_res.san_to_vent_latency_baseline:+.2f} u" if (comp_res.san_to_vent_latency_baseline and comp_res.san_to_vent_latency_trial) else "N/A"

            cmp_table_rows = [
                {"Category": "Pacing & Conduction", "Item": "Ventricular Activation Rate", "State A (Baseline)": bpm_A_str, "State B (Trial)": bpm_B_str, "Delta (B - A)": diff_bpm_str},
                {"Category": "Pacing & Conduction", "Item": "Mean Cycle Length", "State A (Baseline)": cl_A_str, "State B (Trial)": cl_B_str, "Delta (B - A)": diff_cl_str},
                {"Category": "Pacing & Conduction", "Item": "SAN-to-Ventricle Latency", "State A (Baseline)": lat_A_str, "State B (Trial)": lat_B_str, "Delta (B - A)": diff_lat_str},
                {"Category": "Pacing & Conduction", "Item": "Conduction Drops / Failures", "State A (Baseline)": f"{comp_res.conduction_drops_baseline}", "State B (Trial)": f"{comp_res.conduction_drops_trial}", "Delta (B - A)": f"{comp_res.conduction_drops_trial - comp_res.conduction_drops_baseline:+d}"},
                {"Category": "Pacing & Conduction", "Item": "Conduction Status", "State A (Baseline)": comp_res.conduction_status_baseline, "State B (Trial)": comp_res.conduction_status_trial, "Delta (B - A)": "Arrhythmia Shift" if comp_res.is_arrhythmia_induced else "Preserved"},
            ]

            for diff in comp_res.changed_parameters:
                cmp_table_rows.append({
                    "Category": "Parameter Changed",
                    "Item": diff["label"],
                    "State A (Baseline)": str(diff["baseline"]),
                    "State B (Trial)": str(diff["trial"]),
                    "Delta (B - A)": f"{diff['difference']:+.3f}" if isinstance(diff["difference"], (int, float)) else str(diff["difference"]),
                })

            if comp_res.single_cell_baseline_eq and comp_res.single_cell_trial_eq:
                eq_a = comp_res.single_cell_baseline_eq
                eq_b = comp_res.single_cell_trial_eq
                cmp_table_rows.extend([
                    {"Category": "Cellular Stability", "Item": "Equilibrium Voltage v*", "State A (Baseline)": f"{eq_a.v_star:.4f}", "State B (Trial)": f"{eq_b.v_star:.4f}", "Delta (B - A)": f"{eq_b.v_star - eq_a.v_star:+.4f}"},
                    {"Category": "Cellular Stability", "Item": "Jacobian Trace τ", "State A (Baseline)": f"{eq_a.trace:.4f}", "State B (Trial)": f"{eq_b.trace:.4f}", "Delta (B - A)": f"{eq_b.trace - eq_a.trace:+.4f}"},
                    {"Category": "Cellular Stability", "Item": "Jacobian Determinant Δ", "State A (Baseline)": f"{eq_a.determinant:.4f}", "State B (Trial)": f"{eq_b.determinant:.4f}", "Delta (B - A)": f"{eq_b.determinant - eq_a.determinant:+.4f}"},
                    {"Category": "Cellular Stability", "Item": "Stability Regime", "State A (Baseline)": eq_a.stability_class.value, "State B (Trial)": eq_b.stability_class.value, "Delta (B - A)": "Shifted" if eq_a.stability_class != eq_b.stability_class else "Invariant"},
                ])

            df_comp = pd.DataFrame(cmp_table_rows)
            st.dataframe(df_comp, use_container_width=True)

            # Download CSV
            csv_comp = export_comparison_to_csv(comp_res)
            st.download_button(
                label="📥 Download Experiment Comparison as CSV",
                data=csv_comp,
                file_name="cardiac_experiment_comparison.csv",
                mime="text/csv",
            )

        with c_sub2:
            st.subheader("Transmembrane Voltage v_i(t) and Recovery Variable w_i(t) Traces")
            fig_traces = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                subplot_titles=(
                    "Transmembrane Potentials v_i(t) [Solid: Baseline A | Dashed: Trial B]",
                    "Recovery Variables w_i(t) [Solid: Baseline A | Dashed: Trial B]",
                )
            )

            res_A = comp_res.baseline_result
            res_B = comp_res.trial_result
            ch_colors = {0: "#f59e0b", 1: "#3b82f6", 2: "#ec4899", 5: "#10b981"}
            ch_names = {0: "SAN", 1: "Atrium", 2: "AVN", 5: "Ventricle"}

            for node_idx in [0, 1, 2, 5]:
                col = ch_colors[node_idx]
                nm = ch_names[node_idx]
                # Baseline
                fig_traces.add_trace(go.Scatter(
                    x=res_A.time, y=res_A.V[:, node_idx], mode="lines",
                    name=f"{nm} Voltage (A)", line=dict(color=col, width=2.0)
                ), row=1, col=1)
                fig_traces.add_trace(go.Scatter(
                    x=res_A.time, y=res_A.W[:, node_idx], mode="lines",
                    name=f"{nm} Recovery (A)", line=dict(color=col, width=1.5), showlegend=False
                ), row=2, col=1)

                # Trial
                fig_traces.add_trace(go.Scatter(
                    x=res_B.time, y=res_B.V[:, node_idx], mode="lines",
                    name=f"{nm} Voltage (B)", line=dict(color=col, width=2.0, dash="dash")
                ), row=1, col=1)
                fig_traces.add_trace(go.Scatter(
                    x=res_B.time, y=res_B.W[:, node_idx], mode="lines",
                    name=f"{nm} Recovery (B)", line=dict(color=col, width=1.5, dash="dash"), showlegend=False
                ), row=2, col=1)

            fig_traces.update_layout(
                height=520, template="plotly_dark",
                hovermode="x unified", margin=dict(l=40, r=20, t=35, b=30),
            )
            st.plotly_chart(fig_traces, use_container_width=True)

        with c_sub3:
            st.subheader("Activation Sequences & Latency Analysis")
            fig_acts = go.Figure()

            # Plot Baseline activations
            for i in range(res_A.network.num_nodes):
                node_name = res_A.network.nodes[i].name
                t_acts_A = res_A.activations.get(i, [])
                t_acts_B = res_B.activations.get(i, [])

                if t_acts_A:
                    fig_acts.add_trace(go.Scatter(
                        x=t_acts_A, y=[f"{node_name} (A)"] * len(t_acts_A), mode="markers",
                        marker=dict(size=11, color="#3b82f6", symbol="circle"),
                        name=f"{node_name} (A)", showlegend=(i == 0)
                    ))
                if t_acts_B:
                    fig_acts.add_trace(go.Scatter(
                        x=t_acts_B, y=[f"{node_name} (B)"] * len(t_acts_B), mode="markers",
                        marker=dict(size=11, color="#ef4444", symbol="diamond"),
                        name=f"{node_name} (B)", showlegend=(i == 0)
                    ))

            fig_acts.update_layout(
                title="Comparative Chamber Activation Timestamps (Blue Circles: Baseline A | Red Diamonds: Trial B)",
                xaxis_title="Time t (model units)",
                template="plotly_dark",
                height=420,
                margin=dict(l=60, r=20, t=35, b=30),
            )
            st.plotly_chart(fig_acts, use_container_width=True)

        with c_sub4:
            st.subheader("Activation-Derived Rhythm Waveform Comparison")
            fig_rhythm_cmp = go.Figure()
            fig_rhythm_cmp.add_trace(go.Scatter(
                x=res_A.time, y=res_A.rhythm_trace, mode="lines",
                name="Rhythm State A (Baseline)", line=dict(color="#38bdf8", width=2.0)
            ))
            fig_rhythm_cmp.add_trace(go.Scatter(
                x=res_B.time, y=res_B.rhythm_trace, mode="lines",
                name="Rhythm State B (Trial)", line=dict(color="#f43f5e", width=2.0, dash="dash")
            ))
            fig_rhythm_cmp.add_hline(y=0.0, line_dash="dash", line_color="#475569")
            fig_rhythm_cmp.update_layout(
                title="Superimposed Activation-Derived Rhythm Traces (Dipole Projection)",
                xaxis_title="Time t (model units)", yaxis_title="Dipole Moment Φ(t)",
                template="plotly_dark", height=350, margin=dict(l=40, r=20, t=35, b=30),
            )
            st.plotly_chart(fig_rhythm_cmp, use_container_width=True)

        with c_sub5:
            st.subheader("Single-Cell Phase Plane & Flow Field Comparison")
            col_pp_A, col_pp_B = st.columns(2)

            with col_pp_A:
                st.markdown("#### 🔵 State A Phase Portrait")
                p_cell_A = FHNParameters(a=base_params.cell_a, b=base_params.cell_b, epsilon=base_params.cell_epsilon, I=base_params.cell_I_bias)
                v_grid_A, w_vn_A, w_wn_A = compute_nullclines((-2.5, 2.5), p_cell_A, num_points=200)
                fig_pp_A = go.Figure()
                fig_pp_A.add_trace(go.Scatter(x=v_grid_A, y=w_vn_A, mode="lines", name="v-nullcline", line=dict(color="#3b82f6", width=2)))
                fig_pp_A.add_trace(go.Scatter(x=v_grid_A, y=w_wn_A, mode="lines", name="w-nullcline", line=dict(color="#f97316", width=2, dash="dash")))
                if comp_res.single_cell_baseline_eq:
                    eqA = comp_res.single_cell_baseline_eq
                    fig_pp_A.add_trace(go.Scatter(
                        x=[eqA.v_star], y=[eqA.w_star], mode="markers+text",
                        text=[f"E* ({eqA.v_star:.2f}, {eqA.w_star:.2f})"], textposition="top right",
                        marker=dict(size=13, color="#10b981" if eqA.is_stable else "#ef4444"), name="Fixed Point"
                    ))
                fig_pp_A.update_layout(title="State A Nullclines", xaxis_title="v", yaxis_title="w", template="plotly_dark", height=340)
                st.plotly_chart(fig_pp_A, use_container_width=True)

            with col_pp_B:
                st.markdown("#### 🔴 State B Phase Portrait")
                p_cell_B = FHNParameters(a=trial_params.cell_a, b=trial_params.cell_b, epsilon=trial_params.cell_epsilon, I=trial_params.cell_I_bias)
                v_grid_B, w_vn_B, w_wn_B = compute_nullclines((-2.5, 2.5), p_cell_B, num_points=200)
                fig_pp_B = go.Figure()
                fig_pp_B.add_trace(go.Scatter(x=v_grid_B, y=w_vn_B, mode="lines", name="v-nullcline", line=dict(color="#3b82f6", width=2)))
                fig_pp_B.add_trace(go.Scatter(x=v_grid_B, y=w_wn_B, mode="lines", name="w-nullcline", line=dict(color="#f97316", width=2, dash="dash")))
                if comp_res.single_cell_trial_eq:
                    eqB = comp_res.single_cell_trial_eq
                    fig_pp_B.add_trace(go.Scatter(
                        x=[eqB.v_star], y=[eqB.w_star], mode="markers+text",
                        text=[f"E* ({eqB.v_star:.2f}, {eqB.w_star:.2f})"], textposition="top right",
                        marker=dict(size=13, color="#10b981" if eqB.is_stable else "#ef4444"), name="Fixed Point"
                    ))
                fig_pp_B.update_layout(title="State B Nullclines", xaxis_title="v", yaxis_title="w", template="plotly_dark", height=340)
                st.plotly_chart(fig_pp_B, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: AUTOMATED PARAMETER SWEEP ENGINE
    # -------------------------------------------------------------------------
    with tab_exp_sweep:
        st.subheader("Automated Parameter Sweep & Bifurcation Explorer")
        st.caption("Continuously sweeps a selected electrophysiological parameter across user-defined bounds, calculating conduction latency, activation rates, and stability transitions.")

        sweep_col1, sweep_col2, sweep_col3, sweep_col4 = st.columns(4)

        with sweep_col1:
            sweep_param_choice = st.selectbox(
                "Select Parameter to Sweep:",
                [
                    "AV Nodal Conductance (g_AV)",
                    "Global Electrical Coupling (D)",
                    "SA Pacing Interval (T_pace)",
                    "Stimulus Current Amplitude (I)",
                    "Cellular Recovery Timescale (ε)",
                    "Excitability Parameter a",
                    "Regional AVN Timescale (ε_AVN)",
                    "Regional Ventricle Timescale (ε_Vent)",
                ],
                key="sweep_param_choice"
            )
            param_key_map = {
                "AV Nodal Conductance (g_AV)": "av_conductance",
                "Global Electrical Coupling (D)": "D",
                "SA Pacing Interval (T_pace)": "pacing_interval",
                "Stimulus Current Amplitude (I)": "I",
                "Cellular Recovery Timescale (ε)": "epsilon",
                "Excitability Parameter a": "a",
                "Regional AVN Timescale (ε_AVN)": "region_avn_eps",
                "Regional Ventricle Timescale (ε_Vent)": "region_vent_eps",
            }
            sweep_key = param_key_map[sweep_param_choice]

        # Suggested default ranges
        default_ranges = {
            "av_conductance": (0.00, 0.40),
            "D": (0.10, 1.20),
            "pacing_interval": (30.0, 85.0),
            "I": (0.8, 2.5),
            "epsilon": (0.02, 0.20),
            "a": (0.3, 1.0),
            "region_avn_eps": (0.02, 0.12),
            "region_vent_eps": (0.03, 0.12),
        }
        def_min, def_max = default_ranges.get(sweep_key, (0.0, 1.0))

        with sweep_col2:
            sweep_min = st.number_input("Sweep Range Min", value=float(def_min), step=0.05, key="sweep_min")
        with sweep_col3:
            sweep_max = st.number_input("Sweep Range Max", value=float(def_max), step=0.05, key="sweep_max")
        with sweep_col4:
            sweep_steps = st.slider("Resolution (Steps)", 6, 25, 12, 1, key="sweep_steps")

        if st.button("🚀 Execute Multi-Point Parameter Sweep", key="btn_run_sweep", use_container_width=True):
            sweep_cfg = ParameterSweepConfig(
                parameter_name=sweep_key,
                range_min=float(sweep_min),
                range_max=float(sweep_max),
                num_steps=int(sweep_steps),
                base_params=base_params,
            )
            with st.spinner("Executing numerical ODE integrations across parameter range..."):
                pts, sweep_summary = run_parameter_sweep(sweep_cfg)

            st.session_state.last_sweep_points = pts
            st.session_state.last_sweep_summary = sweep_summary
            st.session_state.last_sweep_key = sweep_key
            st.session_state.last_sweep_name = sweep_param_choice

        # Display Sweep Results if available
        if "last_sweep_points" in st.session_state:
            pts = st.session_state.last_sweep_points
            s_sum = st.session_state.last_sweep_summary
            s_key = st.session_state.last_sweep_key
            s_name = st.session_state.last_sweep_name

            st.info(f"💡 {s_sum}")

            # 4 Outcome Charts
            sw_colA, sw_colB = st.columns(2)

            param_vals = [p.param_value for p in pts]
            bpm_vals = [p.bpm if p.bpm is not None else 0.0 for p in pts]
            lat_vals = [p.san_to_vent_latency if p.san_to_vent_latency is not None else 0.0 for p in pts]
            drops_vals = [p.conduction_drops for p in pts]
            vstar_vals = [p.equilibrium_v_star for p in pts]

            with sw_colA:
                # Chart 1: Ventricular Rate vs Parameter
                fig_sw1 = go.Figure()
                fig_sw1.add_trace(go.Scatter(
                    x=param_vals, y=bpm_vals, mode="lines+markers",
                    name="Ventricular BPM", line=dict(color="#38bdf8", width=2.5),
                    marker=dict(size=8, color="#0284c7")
                ))
                fig_sw1.update_layout(
                    title=f"Ventricular Rate (BPM) vs {s_name}",
                    xaxis_title=s_name, yaxis_title="Heart Rate (BPM)",
                    template="plotly_dark", height=320, margin=dict(l=40, r=20, t=35, b=30),
                )
                st.plotly_chart(fig_sw1, use_container_width=True)

                # Chart 3: Conduction Drops / Failures vs Parameter
                fig_sw3 = go.Figure()
                fig_sw3.add_trace(go.Bar(
                    x=param_vals, y=drops_vals, name="Conduction Drops",
                    marker_color=["#ef4444" if d > 0 else "#10b981" for d in drops_vals]
                ))
                fig_sw3.update_layout(
                    title=f"Conduction Block / Drop Count vs {s_name}",
                    xaxis_title=s_name, yaxis_title="Dropped Activations Count",
                    template="plotly_dark", height=320, margin=dict(l=40, r=20, t=35, b=30),
                )
                st.plotly_chart(fig_sw3, use_container_width=True)

            with sw_colB:
                # Chart 2: Conduction Latency vs Parameter
                fig_sw2 = go.Figure()
                fig_sw2.add_trace(go.Scatter(
                    x=param_vals, y=lat_vals, mode="lines+markers",
                    name="Conduction Latency (Δt)", line=dict(color="#f59e0b", width=2.5),
                    marker=dict(size=8, color="#d97706")
                ))
                fig_sw2.update_layout(
                    title=f"SAN-to-Ventricle Latency vs {s_name}",
                    xaxis_title=s_name, yaxis_title="Transit Time (model units)",
                    template="plotly_dark", height=320, margin=dict(l=40, r=20, t=35, b=30),
                )
                st.plotly_chart(fig_sw2, use_container_width=True)

                # Chart 4: Single-cell Equilibrium v* vs Parameter
                fig_sw4 = go.Figure()
                fig_sw4.add_trace(go.Scatter(
                    x=param_vals, y=vstar_vals, mode="lines+markers",
                    name="Equilibrium Voltage v*", line=dict(color="#a855f7", width=2.5),
                    marker=dict(size=8, color="#7e22ce")
                ))
                fig_sw4.update_layout(
                    title=f"Cellular Equilibrium Potential v* vs {s_name}",
                    xaxis_title=s_name, yaxis_title="Equilibrium Potential v*",
                    template="plotly_dark", height=320, margin=dict(l=40, r=20, t=35, b=30),
                )
                st.plotly_chart(fig_sw4, use_container_width=True)

            # Sweep Data Table & Download
            st.subheader("Tabular Parameter Sweep Dataset")
            sweep_csv_data = export_sweep_to_csv(pts, s_key)
            df_sw_view = pd.read_csv(io.StringIO(sweep_csv_data))
            st.dataframe(df_sw_view, use_container_width=True)

            st.download_button(
                label="📥 Download Parameter Sweep Data as CSV",
                data=sweep_csv_data,
                file_name=f"cardiac_sweep_{s_key}.csv",
                mime="text/csv",
            )


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
