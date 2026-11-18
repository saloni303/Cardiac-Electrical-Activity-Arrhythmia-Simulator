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
import streamlit.components.v1 as components

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
        "🏥 Arrhythmia Scenarios & Rhythm Analysis (Stages 3, 4 & 5)",
        "🌐 Coupled Conduction Network & Heart Visualizer (Stage 2 & 4)",
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
# WORKSPACE 3: ARRHYTHMIA SCENARIO ENGINE & RHYTHM ANALYSIS (STAGES 3-5)
# =============================================================================
if "Arrhythmia" in app_mode:
    st.title("🏥 Cardiac Arrhythmia Simulation & Rhythm Analysis Panel (Stages 3–5)")
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

    # Summary metric bar (Stage 5 Enhanced)
    hr_metrics = estimate_heart_rate_bpm(res.mean_cycle_length)
    m_cols = st.columns(5)
    with m_cols[0]:
        st.metric("Pacing Interval (T_pace)", f"{cfg.pacing_interval:.1f} u" if sc_type != ScenarioType.REENTRY else "Single Starter")
    with m_cols[1]:
        st.metric("Mean Cycle Length", f"{res.mean_cycle_length:.2f} u" if res.mean_cycle_length else "N/A (Blocked)")
    with m_cols[2]:
        st.metric("Estimated Heart Rate", f"{hr_metrics['bpm']:.1f} BPM" if hr_metrics['is_meaningful'] else "0.0 BPM (Blocked)")
    with m_cols[3]:
        rate_ratio = res.comparison_with_normal.get("rate_ratio", 1.0)
        st.metric("Relative to Normal", f"{rate_ratio:.2f}x" if rate_ratio > 0 else "0.0x (Blocked)")
    with m_cols[4]:
        if sc_type == ScenarioType.REENTRY:
            status_label = "Sustained Circus" if res.reentry_sustained else "Extinguished"
            st.metric("Re-entry Status", status_label)
        else:
            status_label = "1:1 Conduction" if res.comparison_with_normal.get("conduction_success") else "Conduction Block"
            st.metric("Conduction Status", status_label)

    # -------------------------------------------------------------------------
    # STAGE 4: INTERACTIVE ANATOMICAL CARDIAC VISUALIZATION
    # -------------------------------------------------------------------------
    st.markdown("### 🫀 Interactive Anatomical Conduction Visualization (Stage 4)")
    st.caption("Synchronized 60 FPS anatomical wavefront animation driven directly by continuous numerical ODE integration. Features Play/Pause/Reset, Scrubber, Speed controls, and Real-time Rhythm Cursor.")
    heart_html = build_heart_visualization_html(res, width=950, height=720)
    components.html(heart_html, height=740, scrolling=False)

    # -------------------------------------------------------------------------
    # STAGE 5: RHYTHM VISUALIZATION & ACTIVATION ANALYSIS PANELS
    # -------------------------------------------------------------------------
    sc_tab_rhythm, sc_tab_timeline, sc_tab_log, sc_tab_comp, sc_tab_math = st.tabs([
        "📊 Stage 5: Rhythm Analysis Panel & Lead Traces",
        "⏱️ Stage 5: Activation Timeline & Conduction Integrity",
        "📋 Stage 5: Numerical Activation Event Log",
        "⚖️ Comparative Analysis vs Normal Rhythm",
        "📐 Mathematical Formulation & Automated Tests",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: RHYTHM ANALYSIS PANEL & LEAD TRACES (STAGE 5)
    # -------------------------------------------------------------------------
    with sc_tab_rhythm:
        st.subheader("1. Activation-Derived Rhythm Trace & Multi-Lead Field Approximation")
        st.warning("⚠️ **Simulated activation-derived rhythm trace — not a clinically validated ECG.** Generated algorithmically from net intercellular electrical dipole moments of the coupled FitzHugh–Nagumo network.")

        # Rate & Interval cards
        b2b_vent = compute_beat_to_beat_intervals(res.activations, monitored_node_id=(0 if sc_type == ScenarioType.REENTRY else 5))
        r_cols = st.columns(4)
        with r_cols[0]:
            st.metric("Estimated Activation Rate", f"{hr_metrics['bpm']:.1f} BPM" if hr_metrics['is_meaningful'] else "0.0 BPM")
        with r_cols[1]:
            st.metric("Mean Interval (CL)", f"{b2b_vent['mean_interval']:.2f} u" if b2b_vent['mean_interval'] else "N/A")
        with r_cols[2]:
            st.metric("Beat-to-Beat SD (SDNN)", f"{b2b_vent['std_interval']:.3f} u" if b2b_vent['std_interval'] is not None else "N/A")
        with r_cols[3]:
            st.metric("Rhythm Regularity", "Regular (SD < 5%)" if b2b_vent['is_regular'] else "Irregular / Blocked")

        # Primary Activation-Derived Rhythm Waveform
        fig_rhythm = go.Figure()
        fig_rhythm.add_trace(go.Scatter(
            x=res.time,
            y=res.rhythm_trace,
            mode="lines",
            name="Activation-Derived Rhythm Φ(t)",
            line=dict(color="#8b5cf6", width=2.2),
            hovertemplate="Time: %{x:.2f} u<br>Dipole Potential: %{y:.4f}<extra></extra>",
        ))
        fig_rhythm.update_layout(
            title="Algorithmically Projected Dipole Potential: Φ(t) = Σ g_ij (v_j - v_i) · Δy_ij",
            height=280,
            margin=dict(l=40, r=20, t=35, b=20),
            xaxis_title="Simulation Time t (model units)",
            yaxis_title="Dipole Potential (arb. units)",
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig_rhythm, use_container_width=True)

        # Einthoven Lead II Dipole Field Approximation (Separated & Explained)
        with st.expander("📐 Optional: Einthoven Limb Lead Approximation (Lead II) & Source-to-Electrode Model"):
            st.markdown("""
            **Source-to-Electrode Biophysical Assumptions:**
            1. **Far-Field Dipole Approximation:** Assumes the cardiac electrical generator acts as an equivalent current dipole $\\vec{P}(t) = \\sum g_{ij}(v_j - v_i) \\vec{d}_{ij}$ viewed from distant surface electrodes.
            2. **Homogeneous Isotropic Volume Conductor:** Neglects electrical conductivity boundaries between intracardiac blood ($0.6\\text{ S/m}$), myocardial wall ($0.2\\text{ S/m}$), lungs ($0.05\\text{ S/m}$), and rib cage.
            3. **Quasi-Static Approximation:** Neglects tissue capacitive displacement currents (valid for electrophysiological frequencies $< 1\\text{ kHz}$).
            4. **Einthoven Lead II Projection:** Standard limb lead vector directed at $60^\\circ$ from right shoulder to left lower abdomen:
               $$V_{\\text{Lead II}}(t) = \\vec{P}(t) \\cdot [\\cos(60^\\circ), \\sin(60^\\circ)] = 0.5 P_x(t) + 0.866 P_y(t)$$
            """)
            lead2_trace, lead2_doc = compute_true_ecg_lead_approximation(res.time, res.V, res.network, "Lead_II")
            fig_lead = go.Figure()
            fig_lead.add_trace(go.Scatter(
                x=res.time,
                y=lead2_trace,
                mode="lines",
                name="Einthoven Lead II Approximation",
                line=dict(color="#06b6d4", width=2.0),
            ))
            fig_lead.update_layout(
                height=240,
                margin=dict(l=40, r=20, t=20, b=20),
                xaxis_title="Time t (model units)",
                yaxis_title="Lead II Potential (arb. units)",
                hovermode="x unified",
                template="plotly_white",
            )
            st.plotly_chart(fig_lead, use_container_width=True)

        # Regional Voltage v(t) and Recovery w(t) Probe
        st.subheader("2. Regional Action Potential v(t) & Recovery Variable w(t) Probe")
        probe_node_names = [f"Node {i}: {res.network.nodes[i].name} ({res.network.nodes[i].region.value})" for i in range(res.network.num_nodes)]
        selected_probe_idx = st.selectbox(
            "Select Model Compartment for In-Depth State Inspection:",
            range(res.network.num_nodes),
            format_func=lambda idx: probe_node_names[idx],
            index=min(5, res.network.num_nodes - 1),
        )

        col_v, col_w = st.columns(2)
        with col_v:
            st.markdown(f"**Fast Voltage Variable $v(t)$ — {res.network.nodes[selected_probe_idx].name}**")
            fig_pv = go.Figure()
            fig_pv.add_trace(go.Scatter(
                x=res.time,
                y=res.V[:, selected_probe_idx],
                mode="lines",
                name="Voltage v(t)",
                line=dict(color="#3b82f6", width=2.2),
            ))
            # Depolarization threshold line
            fig_pv.add_hline(y=0.0, line_dash="dash", line_color="#ef4444", annotation_text="Threshold (v=0.0)", annotation_position="top left")
            fig_pv.update_layout(
                height=280,
                margin=dict(l=40, r=20, t=20, b=20),
                xaxis_title="Time t",
                yaxis_title="Voltage-Like State v",
                template="plotly_white",
            )
            st.plotly_chart(fig_pv, use_container_width=True)

        with col_w:
            st.markdown(f"**Slow Recovery Variable $w(t)$ — {res.network.nodes[selected_probe_idx].name}**")
            fig_pw = go.Figure()
            fig_pw.add_trace(go.Scatter(
                x=res.time,
                y=res.W[:, selected_probe_idx],
                mode="lines",
                name="Recovery w(t)",
                line=dict(color="#8b5cf6", width=2.2),
            ))
            # Refractory threshold indicator
            fig_pw.add_hline(y=-0.35, line_dash="dot", line_color="#a855f7", annotation_text="Refractory Boundary (w = -0.35)", annotation_position="bottom left")
            fig_pw.update_layout(
                height=280,
                margin=dict(l=40, r=20, t=20, b=20),
                xaxis_title="Time t",
                yaxis_title="Recovery State w",
                template="plotly_white",
            )
            st.plotly_chart(fig_pw, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: ACTIVATION TIMELINE & CONDUCTION INTEGRITY (STAGE 5)
    # -------------------------------------------------------------------------
    with sc_tab_timeline:
        st.subheader("Activation-Event Timeline & Conduction Integrity Analysis")
        st.caption("Chronological event progression from SA node through atria, AV node, His bundle, Purkinje network, to ventricles.")

        # Detect missing events
        missing_data = detect_missing_events(res)

        # Highlight missing events if present
        if sc_type == ScenarioType.SINUS_PAUSE and missing_data["missing_sa_impulses"]:
            st.error(f"⚠️ **Sinus Pause Impulses Suppressed:** {len(missing_data['missing_sa_impulses'])} expected SA-node pacing impulse(s) were suppressed during the pause window $[{cfg.pause_start:.1f}, {cfg.pause_end:.1f}]$.")
            for m in missing_data["missing_sa_impulses"]:
                st.caption(f"- Expected pulse at **t = {m['expected_time']:.1f} u** was suppressed by the pacing arrest protocol.")

        if sc_type == ScenarioType.AV_BLOCK and missing_data["missing_downstream_events"]:
            st.error(f"⚠️ **Downstream Conduction Failures Detected:** {len(missing_data['missing_downstream_events'])} atrial activation(s) failed to propagate to the AV Node and Ventricles. Conduction Ratio: **{missing_data['conduction_ratio']}**.")
            for m in missing_data["missing_downstream_events"]:
                st.caption(f"- Atrial Beat #{m['atrial_beat_index']} at **t = {m['atrial_time']:.1f} u** was blocked before reaching downstream chambers.")

        # Multi-Channel Timeline Chart
        fig_timeline = go.Figure()
        y_labels = [res.network.nodes[i].name for i in range(res.network.num_nodes)]

        # Plot activation scatter dots for each compartment
        for i in range(res.network.num_nodes):
            acts = res.activations.get(i, [])
            if acts:
                fig_timeline.add_trace(go.Scatter(
                    x=acts,
                    y=[res.network.nodes[i].name] * len(acts),
                    mode="markers",
                    name=res.network.nodes[i].name,
                    marker=dict(size=14, symbol="circle", line=dict(color="#0f172a", width=2)),
                    hovertemplate=f"Region: {res.network.nodes[i].name}<br>Activation Time: %{{x:.2f}} u<extra></extra>",
                ))

        # Annotate Sinus Pause window
        if sc_type == ScenarioType.SINUS_PAUSE and cfg.pause_start and cfg.pause_end:
            fig_timeline.add_vrect(
                x0=cfg.pause_start, x1=cfg.pause_end,
                fillcolor="#fef08a", opacity=0.3, line_width=1.5, line_dash="dash", line_color="#ca8a04",
                annotation_text="Sinus Pause Window (Stimulus Suppressed)", annotation_position="top left",
            )

        fig_timeline.update_layout(
            title="Multi-Compartment Depolarization Event Timeline",
            height=340,
            margin=dict(l=80, r=20, t=40, b=20),
            xaxis_title="Simulation Time t (model units)",
            yaxis_title="Anatomical Compartment",
            yaxis=dict(categoryorder="array", categoryarray=list(reversed(y_labels))),
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Beat-to-Beat Interval Tachogram
        st.subheader("Beat-to-Beat Interval (Tachogram) Sequence")
        if b2b_vent["intervals"]:
            fig_tach = go.Figure()
            fig_tach.add_trace(go.Scatter(
                x=b2b_vent["beat_indices"],
                y=b2b_vent["intervals"],
                mode="lines+markers",
                name="RR Interval (Δt)",
                marker=dict(size=10, color="#ec4899"),
                line=dict(color="#db2777", width=2),
                hovertemplate="Beat: %{x}<br>Cycle Length: %{y:.2f} u<extra></extra>",
            ))
            if b2b_vent["mean_interval"]:
                fig_tach.add_hline(
                    y=b2b_vent["mean_interval"], line_dash="dash", line_color="#475569",
                    annotation_text=f"Mean CL = {b2b_vent['mean_interval']:.2f} u",
                )
            fig_tach.update_layout(
                height=260,
                margin=dict(l=40, r=20, t=20, b=20),
                xaxis_title="Consecutive Beat Index",
                yaxis_title="Cycle Length Δt (model units)",
                template="plotly_white",
            )
            st.plotly_chart(fig_tach, use_container_width=True)
        else:
            st.info("ℹ️ Fewer than 2 ventricular activation events occurred; beat-to-beat tachogram requires at least two consecutive beats.")

    # -------------------------------------------------------------------------
    # TAB 3: NUMERICAL ACTIVATION EVENT LOG (STAGE 5)
    # -------------------------------------------------------------------------
    with sc_tab_log:
        st.subheader("Numerical Activation Event Log")
        st.caption("Chronological record of sub-step interpolated threshold crossings (v = 0.0) across all network compartments.")

        event_log = generate_numerical_event_log(res.activations, res.network, threshold=0.0)
        df_log = pd.DataFrame([e.to_dict() for e in event_log])

        if not df_log.empty:
            # Region filter control
            reg_options = ["All Compartments"] + list(df_log["Region"].unique())
            selected_filter = st.selectbox("Filter Event Log by Region:", reg_options)

            if selected_filter != "All Compartments":
                filtered_df = df_log[df_log["Region"] == selected_filter]
            else:
                filtered_df = df_log

            st.dataframe(filtered_df, use_container_width=True, height=360)

            # Download CSV button
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Numerical Event Log as CSV",
                data=csv_data,
                file_name=f"fhn_event_log_{sc_type.name.lower()}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No activation events detected in the simulation run.")

    # -------------------------------------------------------------------------
    # TAB 4: COMPARATIVE ANALYSIS VS NORMAL RHYTHM
    # -------------------------------------------------------------------------
    with sc_tab_comp:
        st.subheader("Comparative Analysis: Scenario vs. Normal Sinus Rhythm")

        comp_stage5 = compare_with_normal_rhythm(res)
        comp_data = [
            {"Metric": "Scenario Type", "Normal Baseline": "Normal Sinus Rhythm", f"{res.scenario_type.value}": res.scenario_type.value},
            {"Metric": "SA Pacing Interval", "Normal Baseline": "45.0 u", f"{res.scenario_type.value}": f"{cfg.pacing_interval:.1f} u" if sc_type != ScenarioType.REENTRY else "Single starter beat"},
            {"Metric": "Ventricular Mean Cycle Length", "Normal Baseline": f"{comp_stage5['baseline_cycle_length']:.2f} u", f"{res.scenario_type.value}": f"{comp_stage5['scenario_cycle_length']:.2f} u" if comp_stage5['scenario_cycle_length'] else "N/A (Blocked)"},
            {"Metric": "Estimated Heart Rate", "Normal Baseline": f"{comp_stage5['baseline_bpm']:.1f} BPM", f"{res.scenario_type.value}": f"{comp_stage5['scenario_bpm']:.1f} BPM" if comp_stage5['scenario_bpm'] else "0.0 BPM (Asystole)"},
            {"Metric": "Ventricular Beats Count", "Normal Baseline": f"{comp_stage5['baseline_vent_beats']} beats", f"{res.scenario_type.value}": f"{comp_stage5['scenario_vent_beats']} beats"},
            {"Metric": "Rate Acceleration Ratio", "Normal Baseline": "1.00x", f"{res.scenario_type.value}": f"{comp_stage5['rate_ratio']:.2f}x"},
            {"Metric": "Conduction Integrity", "Normal Baseline": "100% (1:1 Conduction)", f"{res.scenario_type.value}": "1:1 Forward" if comp_stage5['conduction_success'] else "Conduction Loss / Block"},
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

    # -------------------------------------------------------------------------
    # TAB 5: MATHEMATICAL ASSUMPTIONS & TESTS
    # -------------------------------------------------------------------------
    with sc_tab_math:
        st.subheader("Mathematical Assumptions, Limitations & Automated Verification")

        st.markdown(f"""
        **Mathematical Assumptions:**  
        {res.mathematical_assumptions}

        **Governing Coupled Network Equations:**
        $$\\frac{{dv_i}}{{dt}} = v_i - \\frac{{v_i^3}}{{3}} - w_i + I_i(t) + D \\sum_{{j \\in \\mathcal{{N}}(i)}} A_{{ij}} (v_j - v_i)$$
        $$\\frac{{dw_i}}{{dt}} = \\varepsilon_i(v_i + a_i - b_i w_i)$$

        **Refractoriness and Conduction Wavelength:**
        - Refractory state is represented by the slow recovery variable $w_i(t)$.
        - Conduction wavelength $\\lambda = \\theta \\times \\text{{ERP}}$, where $\\theta$ is conduction velocity and $\\text{{ERP}}$ is effective refractory period.
        - Conduction block occurs when incoming activation arrives while $w_i$ is elevated above resting potential ($w_i > -0.35$).
        """)

        st.divider()
        st.markdown("### Automated Test Suite Execution")
        if st.button("▶ Run All Automated Unit Tests (Stages 1-5)"):
            with st.spinner("Executing pytest suite across tests/..."):
                import subprocess
                p_res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"], capture_output=True, text=True)
                if p_res.returncode == 0:
                    st.success("All Unit Tests Passed Successfully!")
                    st.code(p_res.stdout)
                else:
                    st.error("Some tests failed:")
                    st.code(p_res.stdout + "\n" + p_res.stderr)
            with st.spinner("Executing pytest suite across tests/..."):
                import subprocess
                p_res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"], capture_output=True, text=True)
                if p_res.returncode == 0:
                    st.success("All Unit Tests Passed Successfully!")
                    st.code(p_res.stdout)
                else:
                    st.error("Some tests failed:")
                    st.code(p_res.stdout + "\n" + p_res.stderr)


# =============================================================================
# WORKSPACE 2: COUPLED CARDIAC NETWORK & HEART VISUALIZER (STAGE 2 & 4)
# =============================================================================
elif "Coupled Conduction" in app_mode:
    st.title("🌐 Coupled Cardiac-Cell Conduction Network & Interactive Heart (Stage 2 & 4)")
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

    # Stage 4: Interactive Anatomical Conduction Visualization for Stage 2
    st.markdown("### 🫀 Interactive Anatomical Conduction Visualization (Stage 4)")
    st.caption("Live 60 FPS anatomical wavefront animation driven directly by the coupled ODE network solver.")
    net_sim_dict = {
        "network": net,
        "time": t_net,
        "V": V_net,
        "W": W_net,
        "name": f"Coupled Conduction ({net_preset})",
    }
    net_heart_html = build_heart_visualization_html(net_sim_dict, width=950, height=720)
    components.html(net_heart_html, height=740, scrolling=False)


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
