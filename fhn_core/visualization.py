"""
Stage 4: Interactive Cardiac Visualization Module.

Generates an anatomical-style scientific illustration of the human heart
connected directly to the coupled FitzHugh-Nagumo numerical ODE network.

Features:
    - Anatomically organized heart regions:
        * SA Node
        * Right Atrial region (RA)
        * Left Atrial region (LA)
        * AV Node (central junction)
        * Bundle of His (septal conduction highway)
        * Purkinje Network (ventricular arborization)
        * Right Ventricular region (RV)
        * Left Ventricular region (LV)
        * Apex
    - Physiologically grounded 3-state color mapping:
        * Resting: Polarized excitable state (v ≈ -1.2, w ≈ -0.6)
        * Activating / Depolarized: High-intensity electric gold (v > 0.0)
        * Recovering / Refractory: Amethyst purple (w > -0.2, v < 0.0)
    - 60 FPS interactive HTML5/SVG/Canvas component with:
        * Play, Pause, Reset controls
        * Speed multiplier (0.25x, 0.5x, 1x, 2x, 4x)
        * Interactive time scrubber
        * Live digital simulation time display
        * Synchronized mini-electrogram strip with moving time cursor
        * Explicit color legend and scientific disclaimers
"""

import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from .network import CardiacNetwork, RegionType
from .scenarios import ScenarioResult, ScenarioType


def map_state_to_color(v: float, w: float) -> Tuple[str, str, float]:
    """
    Maps a cell's instantaneous state (v, w) to a visual color representation.

    Returns:
        (hex_color, state_name, activation_intensity)
    """
    # 1. Active / Depolarized: v > 0.0
    if v > 0.0:
        # Scale intensity with peak voltage up to +1.8
        intensity = min(1.0, max(0.2, v / 1.8))
        # Warm golden-amber to bright yellow
        r = int(245 + 10 * intensity)
        g = int(158 + 60 * intensity)
        b = int(11 + 30 * intensity)
        return f"rgb({r},{g},{b})", "Activating / Depolarized", intensity

    # 2. Refractory Recovery: w is still high while v is below resting
    elif w > -0.35:
        # Recovery phase: w decaying from ~ +1.4 down to -0.4
        rec_intensity = min(1.0, max(0.15, (w + 0.35) / 1.7))
        # Amethyst violet / purple
        r = int(139 * rec_intensity + 40 * (1 - rec_intensity))
        g = int(92 * rec_intensity + 35 * (1 - rec_intensity))
        b = int(246 * rec_intensity + 65 * (1 - rec_intensity))
        return f"rgb({r},{g},{b})", "Refractory Recovery", rec_intensity

    # 3. Polarized Resting State: v ≈ -1.2, w ≈ -0.6
    else:
        # Deep cardiac slate tone
        return "rgb(30, 41, 59)", "Resting / Polarized", 0.0


def extract_animation_frames(
    scenario_result: ScenarioResult,
    target_fps: int = 30,
) -> Dict[str, Any]:
    """
    Subsamples the continuous RK4 ODE trajectory into time-synchronized animation frames.

    Ensures the simulation state is the single source of truth without random flashes.
    """
    time = scenario_result.time
    V = scenario_result.V
    W = scenario_result.W
    rhythm = scenario_result.rhythm_trace
    t_end = float(time[-1])

    # Determine frame spacing
    dt_sim = time[1] - time[0] if len(time) > 1 else 0.02
    step_interval = max(1, int(round((1.0 / target_fps) / dt_sim)))

    sampled_indices = list(range(0, len(time), step_interval))
    if sampled_indices[-1] != len(time) - 1:
        sampled_indices.append(len(time) - 1)

    sampled_times = [round(float(time[idx]), 3) for idx in sampled_indices]
    sampled_rhythm = [round(float(rhythm[idx]), 4) for idx in sampled_indices]

    # Map node indices to regional identifiers
    net = scenario_result.network
    node_metadata = []
    for i in range(net.num_nodes):
        node = net.nodes[i]
        node_metadata.append({
            "id": i,
            "name": node.name,
            "region": node.region.value,
            "x": node.pos_x,
            "y": node.pos_y,
        })

    # Pack state arrays per frame: shape (num_frames, num_nodes)
    sampled_V = [[round(float(V[idx, i]), 3) for i in range(net.num_nodes)] for idx in sampled_indices]
    sampled_W = [[round(float(W[idx, i]), 3) for i in range(net.num_nodes)] for idx in sampled_indices]

    return {
        "scenario_name": scenario_result.scenario_type.value,
        "t_end": t_end,
        "num_frames": len(sampled_indices),
        "times": sampled_times,
        "rhythm": sampled_rhythm,
        "nodes": node_metadata,
        "V": sampled_V,
        "W": sampled_W,
        "is_reentry": (scenario_result.scenario_type == ScenarioType.REENTRY),
        "reentry_sustained": scenario_result.reentry_sustained,
    }


def build_heart_visualization_html(
    scenario_result: ScenarioResult,
    width: int = 800,
    height: int = 700,
) -> str:
    """
    Generates a standalone, responsive, high-performance HTML5/SVG/Canvas interactive visualizer.
    """
    anim_data = extract_animation_frames(scenario_result, target_fps=30)
    json_payload = json.dumps(anim_data)

    html_code = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    body {{
        background: #0f172a;
        color: #f8fafc;
        padding: 12px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    .vis-container {{
        width: 100%;
        max-width: {width}px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }}
    .vis-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
    }}
    .vis-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .time-badge {{
        background: #0f172a;
        border: 1px solid #475569;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.9rem;
        font-family: monospace;
        color: #fbbf24;
        font-weight: 700;
    }}
    .stage-canvas-wrap {{
        position: relative;
        width: 100%;
        height: 420px;
        background: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #334155;
    }}
    svg.heart-svg {{
        width: 100%;
        height: 100%;
        display: block;
    }}
    /* Anatomical chambers */
    .chamber-path {{
        transition: fill 0.08s ease, stroke 0.08s ease;
        stroke: #475569;
        stroke-width: 2.0;
        cursor: pointer;
    }}
    .chamber-path:hover {{
        stroke: #38bdf8;
        stroke-width: 2.5;
    }}
    /* Conduction fibers */
    .conduction-line {{
        stroke: #64748b;
        stroke-width: 3.0;
        stroke-linecap: round;
        transition: stroke 0.06s ease, stroke-width 0.06s ease;
    }}
    .conduction-node {{
        stroke: #0f172a;
        stroke-width: 2.0;
        transition: fill 0.06s ease, r 0.06s ease;
    }}
    /* Controls Bar */
    .controls-bar {{
        display: flex;
        align-items: center;
        gap: 12px;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 12px;
    }}
    .btn {{
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 4px;
        transition: background 0.15s ease;
    }}
    .btn:hover {{
        background: #2563eb;
    }}
    .btn-secondary {{
        background: #334155;
    }}
    .btn-secondary:hover {{
        background: #475569;
    }}
    .scrubber {{
        flex: 1;
        accent-color: #38bdf8;
        cursor: pointer;
    }}
    .speed-select {{
        background: #1e293b;
        color: #f8fafc;
        border: 1px solid #475569;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.8rem;
    }}
    /* Rhythm Waveform Canvas */
    .wave-wrap {{
        height: 60px;
        width: 100%;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        position: relative;
    }}
    canvas#rhythmCanvas {{
        width: 100%;
        height: 100%;
        display: block;
    }}
    /* Legend Bar */
    .legend-bar {{
        display: flex;
        justify-content: space-around;
        align-items: center;
        font-size: 0.78rem;
        color: #cbd5e1;
        padding: 4px 8px;
        background: #0f172a;
        border-radius: 6px;
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .legend-dot {{
        width: 12px;
        height: 12px;
        border-radius: 3px;
    }}
    .disclaimer-note {{
        font-size: 0.72rem;
        color: #94a3b8;
        text-align: center;
        line-height: 1.3;
    }}
</style>
</head>
<body>

<div class="vis-container">
    <div class="vis-header">
        <div class="vis-title">
            <span>❤️</span>
            <span id="scenarioTitle">{scenario_result.scenario_type.value}</span>
        </div>
        <div class="time-badge">t = <span id="timeVal">0.00</span> / {scenario_result.time[-1]:.1f} u</div>
    </div>

    <!-- Main Anatomical SVG -->
    <div class="stage-canvas-wrap">
        <svg class="heart-svg" id="heartSvg" viewBox="0 0 600 420">
            <defs>
                <!-- Glow Filters -->
                <filter id="glowGold" x="-30%" y="-30%" width="160%" height="160%">
                    <feGaussianBlur stdDeviation="6" result="blur"/>
                    <feMerge>
                        <feMergeNode in="blur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>

            <!-- Anatomical Chambers Background Contours -->
            <g id="chambersGroup">
                <!-- Right Atrium (Anatomical Right = Left side of diagram) -->
                <path id="chamber_RA" class="chamber-path" d="M 170 120 C 130 110, 100 150, 110 200 C 120 230, 160 230, 200 220 C 220 180, 210 130, 170 120 Z" fill="#1e293b"/>
                <!-- Left Atrium (Anatomical Left = Right side of diagram) -->
                <path id="chamber_LA" class="chamber-path" d="M 430 120 C 470 110, 500 150, 490 200 C 480 230, 440 230, 400 220 C 380 180, 390 130, 430 120 Z" fill="#1e293b"/>

                <!-- Fibrous Insulating Boundary (AV barrier) -->
                <path d="M 120 225 C 250 240, 350 240, 480 225" stroke="#0f172a" stroke-width="8" fill="none"/>

                <!-- Right Ventricle -->
                <path id="chamber_RV" class="chamber-path" d="M 170 235 C 150 280, 180 340, 270 385 C 285 340, 290 290, 290 240 C 240 240, 190 235, 170 235 Z" fill="#1e293b"/>
                <!-- Left Ventricle (Thicker myocardium, apex formation) -->
                <path id="chamber_LV" class="chamber-path" d="M 430 235 C 450 280, 420 340, 330 385 C 315 340, 310 290, 310 240 C 360 240, 410 235, 430 235 Z" fill="#1e293b"/>
                <!-- Interventricular Septum (Center) -->
                <path id="chamber_IVS" class="chamber-path" d="M 290 240 L 310 240 L 330 385 L 270 385 Z" fill="#1e293b"/>
            </g>

            <!-- Chamber Scientific Labels -->
            <g font-size="11" font-weight="700" fill="#94a3b8" text-anchor="middle">
                <text x="155" y="175">Right Atrium (RA)</text>
                <text x="445" y="175">Left Atrium (LA)</text>
                <text x="225" y="295">Right Ventricle (RV)</text>
                <text x="375" y="295">Left Ventricle (LV)</text>
                <text x="300" y="398">Apex</text>
            </g>

            <!-- Electrical Conduction System Pathways (Fibers) -->
            <g id="conductionFibers">
                <!-- Internodal pathways: SA to AV and Bachmann to LA -->
                <path id="fiber_sa_avn" class="conduction-line" d="M 160 145 C 200 150, 270 180, 300 215" fill="none"/>
                <path id="fiber_sa_la" class="conduction-line" d="M 160 145 C 240 120, 360 120, 420 150" fill="none"/>

                <!-- His Bundle (Septal Trunk) -->
                <path id="fiber_his" class="conduction-line" d="M 300 215 L 300 270" fill="none"/>

                <!-- Bundle Branches & Purkinje Network -->
                <path id="fiber_rbb" class="conduction-line" d="M 300 270 C 280 290, 240 330, 230 350 C 210 330, 190 290, 190 260" fill="none"/>
                <path id="fiber_lbb" class="conduction-line" d="M 300 270 C 320 290, 360 330, 370 350 C 390 330, 410 290, 410 260" fill="none"/>
            </g>

            <!-- Conduction Nodes Markers -->
            <g id="conductionNodes">
                <!-- SA Node -->
                <circle id="nodeMarker_0" class="conduction-node" cx="160" cy="145" r="9" fill="#3b82f6"/>
                <text x="160" y="130" font-size="10" font-weight="700" fill="#fbbf24" text-anchor="middle">SA Node</text>

                <!-- AV Node -->
                <circle id="nodeMarker_2" class="conduction-node" cx="300" cy="215" r="8" fill="#3b82f6"/>
                <text x="300" y="205" font-size="10" font-weight="700" fill="#fbbf24" text-anchor="middle">AV Node</text>

                <!-- His Marker -->
                <circle id="nodeMarker_3" class="conduction-node" cx="300" cy="270" r="6" fill="#3b82f6"/>
                <text x="330" y="274" font-size="9" font-weight="600" fill="#94a3b8">Bundle of His</text>

                <!-- Purkinje Marker Left & Right -->
                <circle id="nodeMarker_4" class="conduction-node" cx="370" cy="350" r="6" fill="#3b82f6"/>
                <circle class="conduction-node" cx="230" cy="350" r="6" fill="#3b82f6"/>
                <text x="375" y="365" font-size="9" font-weight="600" fill="#94a3b8">Purkinje Network</text>
            </g>
        </svg>
    </div>

    <!-- Mini Rhythm Waveform Strip -->
    <div class="wave-wrap">
        <canvas id="rhythmCanvas" width="{width}" height="60"></canvas>
    </div>

    <!-- Playback Controls Bar -->
    <div class="controls-bar">
        <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play</button>
        <button class="btn btn-secondary" onclick="resetSim()">↺ Reset</button>
        <input type="range" class="scrubber" id="timeScrubber" min="0" max="{anim_data['num_frames'] - 1}" value="0" oninput="onScrub(this.value)"/>
        <select class="speed-select" id="speedSelect" onchange="setSpeed(this.value)">
            <option value="0.25">0.25x Speed</option>
            <option value="0.5">0.5x Speed</option>
            <option value="1.0" selected>1.0x (Realtime)</option>
            <option value="2.0">2.0x Fast</option>
        </select>
    </div>

    <!-- Color Legend -->
    <div class="legend-bar">
        <div class="legend-item">
            <div class="legend-dot" style="background: #1e293b; border: 1px solid #475569;"></div>
            <span><b>Resting / Polarized:</b> v ≈ -1.2, w ≈ -0.6</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #fbbf24;"></div>
            <span><b>Activating / Depolarized:</b> v > 0.0 (Impulse Upstroke)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #8b5cf6;"></div>
            <span><b>Refractory Recovery:</b> w > -0.35 (Repolarizing)</span>
        </div>
    </div>

    <div class="disclaimer-note">
        <b>Simplified Educational Model:</b> Displays multi-compartment electrical wavefront propagation driven strictly by numerical integration of coupled FitzHugh–Nagumo ODEs. Not a diagnostic 3D electrophysiology simulation.
    </div>
</div>

<script>
    const simData = {json_payload};
    let currentFrame = 0;
    let isPlaying = false;
    let playSpeed = 1.0;
    let lastTimestamp = 0;
    let frameDurationMs = 33.33; // 30 FPS base

    // DOM Elements
    const playBtn = document.getElementById("playBtn");
    const timeVal = document.getElementById("timeVal");
    const scrubber = document.getElementById("timeScrubber");
    const rhythmCanvas = document.getElementById("rhythmCanvas");
    const ctx = rhythmCanvas.getContext("2d");

    // Color interpolation function matching Python map_state_to_color
    function getColorForState(v, w) {{
        if (v > 0.0) {{
            const intensity = Math.min(1.0, Math.max(0.2, v / 1.8));
            const r = Math.round(245 + 10 * intensity);
            const g = Math.round(158 + 60 * intensity);
            const b = Math.round(11 + 30 * intensity);
            return `rgb(${{r}},${{g}},${{b}})`;
        }} else if (w > -0.35) {{
            const rec = Math.min(1.0, Math.max(0.15, (w + 0.35) / 1.7));
            const r = Math.round(139 * rec + 40 * (1 - rec));
            const g = Math.round(92 * rec + 35 * (1 - rec));
            const b = Math.round(246 * rec + 65 * (1 - rec));
            return `rgb(${{r}},${{g}},${{b}})`;
        }} else {{
            return "#1e293b";
        }}
    }}

    function renderFrame(frameIdx) {{
        if (frameIdx < 0 || frameIdx >= simData.num_frames) return;

        const t = simData.times[frameIdx];
        const V = simData.V[frameIdx];
        const W = simData.W[frameIdx];

        timeVal.textContent = t.toFixed(2);
        scrubber.value = frameIdx;

        // In standard 6-region model:
        // Node 0: SAN, Node 1: Atrium, Node 2: AVN, Node 3: His, Node 4: Purkinje, Node 5: Ventricle
        const isReentry = simData.is_reentry;

        if (!isReentry) {{
            const v_san = V[0]; const w_san = W[0];
            const v_atr = V[1]; const w_atr = W[1];
            const v_avn = V[2]; const w_avn = W[2];
            const v_his = V[3]; const w_his = W[3];
            const v_purk = V[4]; const w_purk = W[4];
            const v_vent = V[5]; const w_vent = W[5];

            // 1. Color SA Node
            const sanColor = getColorForState(v_san, w_san);
            const node0 = document.getElementById("nodeMarker_0");
            if (node0) {{
                node0.setAttribute("fill", sanColor);
                node0.setAttribute("r", v_san > 0 ? "11" : "8");
                if (v_san > 0) node0.setAttribute("filter", "url(#glowGold)");
                else node0.removeAttribute("filter");
            }}

            // 2. Color Atrial Chambers (RA and LA)
            const atrColor = getColorForState(v_atr, w_atr);
            const ra = document.getElementById("chamber_RA");
            const la = document.getElementById("chamber_LA");
            if (ra) ra.setAttribute("fill", atrColor);
            if (la) la.setAttribute("fill", atrColor);

            // Conduction line from SA to Atria
            const fiber_sa_avn = document.getElementById("fiber_sa_avn");
            const fiber_sa_la = document.getElementById("fiber_sa_la");
            if (fiber_sa_avn) fiber_sa_avn.setAttribute("stroke", (v_san > 0 || v_atr > 0) ? "#fbbf24" : "#475569");
            if (fiber_sa_la) fiber_sa_la.setAttribute("stroke", v_atr > 0 ? "#fbbf24" : "#475569");

            // 3. Color AV Node
            const avnColor = getColorForState(v_avn, w_avn);
            const node2 = document.getElementById("nodeMarker_2");
            if (node2) {{
                node2.setAttribute("fill", avnColor);
                node2.setAttribute("r", v_avn > 0 ? "10" : "7");
                if (v_avn > 0) node2.setAttribute("filter", "url(#glowGold)");
                else node2.removeAttribute("filter");
            }}

            // 4. Color His Bundle
            const hisColor = getColorForState(v_his, w_his);
            const fiber_his = document.getElementById("fiber_his");
            const node3 = document.getElementById("nodeMarker_3");
            if (fiber_his) fiber_his.setAttribute("stroke", v_his > 0 ? "#fbbf24" : "#475569");
            if (node3) node3.setAttribute("fill", hisColor);

            // 5. Color Purkinje Network
            const purkColor = getColorForState(v_purk, w_purk);
            const fiber_rbb = document.getElementById("fiber_rbb");
            const fiber_lbb = document.getElementById("fiber_lbb");
            const node4 = document.getElementById("nodeMarker_4");
            if (fiber_rbb) fiber_rbb.setAttribute("stroke", v_purk > 0 ? "#fbbf24" : "#475569");
            if (fiber_lbb) fiber_lbb.setAttribute("stroke", v_purk > 0 ? "#fbbf24" : "#475569");
            if (node4) node4.setAttribute("fill", purkColor);

            // 6. Color Ventricular Chambers (RV, LV, Septum)
            const ventColor = getColorForState(v_vent, w_vent);
            const rv = document.getElementById("chamber_RV");
            const lv = document.getElementById("chamber_LV");
            const ivs = document.getElementById("chamber_IVS");
            if (rv) rv.setAttribute("fill", ventColor);
            if (lv) lv.setAttribute("fill", ventColor);
            if (ivs) ivs.setAttribute("fill", (v_his > 0 || v_vent > 0) ? (v_his > 0 ? hisColor : ventColor) : "#1e293b");

        }} else {{
            // Reentry Ring visualization: map active ring nodes
            for (let i = 0; i < V.length; i++) {{
                const col = getColorForState(V[i], W[i]);
                // Map node i to ventricular or atrial chamber
                if (i === 0) {{
                    const node0 = document.getElementById("nodeMarker_0");
                    if (node0) node0.setAttribute("fill", col);
                }}
                if (i >= 5) {{
                    const lv = document.getElementById("chamber_LV");
                    if (lv && V[i] > 0) lv.setAttribute("fill", col);
                }}
            }}
        }}

        // Render Rhythm Strip on Canvas
        drawRhythmStrip(frameIdx);
    }}

    function drawRhythmStrip(currentIdx) {{
        const w = rhythmCanvas.width;
        const h = rhythmCanvas.height;
        ctx.clearRect(0, 0, w, h);

        const rhythm = simData.rhythm;
        const n = rhythm.length;
        if (n < 2) return;

        // Draw baseline
        ctx.strokeStyle = "#334155";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();

        // Find min/max for scaling
        let rMin = -0.6, rMax = 0.6;
        for (let i = 0; i < n; i++) {{
            if (rhythm[i] < rMin) rMin = rhythm[i];
            if (rhythm[i] > rMax) rMax = rhythm[i];
        }}
        const range = (rMax - rMin) || 1.0;

        // Plot rhythm trace
        ctx.strokeStyle = "#a855f7";
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {{
            const x = (i / (n - 1)) * w;
            const y = h - ((rhythm[i] - rMin) / range) * (h - 8) - 4;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }}
        ctx.stroke();

        // Draw current time cursor
        const cursorX = (currentIdx / (n - 1)) * w;
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cursorX, 0);
        ctx.lineTo(cursorX, h);
        ctx.stroke();
    }}

    function animate(timestamp) {{
        if (!isPlaying) return;

        if (!lastTimestamp) lastTimestamp = timestamp;
        const elapsed = timestamp - lastTimestamp;

        if (elapsed >= (frameDurationMs / playSpeed)) {{
            lastTimestamp = timestamp;
            currentFrame++;
            if (currentFrame >= simData.num_frames) {{
                currentFrame = 0; // Loop playback
            }}
            renderFrame(currentFrame);
        }}

        requestAnimationFrame(animate);
    }}

    function togglePlay() {{
        isPlaying = !isPlaying;
        playBtn.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
        if (isPlaying) {{
            lastTimestamp = 0;
            requestAnimationFrame(animate);
        }}
    }}

    function resetSim() {{
        isPlaying = false;
        playBtn.textContent = "▶ Play";
        currentFrame = 0;
        renderFrame(0);
    }}

    function onScrub(val) {{
        currentFrame = parseInt(val);
        renderFrame(currentFrame);
    }}

    function setSpeed(val) {{
        playSpeed = parseFloat(val);
    }}

    // Initial render
    renderFrame(0);
</script>
</body>
</html>
"""
    return html_code

