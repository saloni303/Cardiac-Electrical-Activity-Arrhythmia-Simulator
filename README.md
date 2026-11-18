# ODE-Based Simulation of Cardiac Electrical Activity and Arrhythmia Dynamics Using the FitzHugh–Nagumo Model

> **Cardiac-Electrical-Activity-Arrhythmia-Simulator**  
> A mathematical simulation of heart cell electrical activity using the FitzHugh–Nagumo (FHN) model. Explore how changing parameters affects cell behaviour, conduction latencies, and arrhythmia-like patterns through interactive graphs and an anatomical heart visualizer. Focused on single-cell and coupled-network mathematical modelling, not clinical diagnosis or medical ECG analysis.

**Stages 1, 2, 3, 4 & 5: Standalone FHN Mathematical Engine, Conduction Network, Arrhythmia Scenario Suite, Interactive Anatomical Heart & Rhythm Analysis Panel**

---

## 1. Overview & Scientific Boundaries
This repository contains an academic research simulation suite developed to investigate cardiac electrophysiology, wave propagation, and cardiac rhythm disorders using the **FitzHugh–Nagumo (FHN)** ordinary differential equation (ODE) system.

- **Stage 1 (Standalone Mathematical Engine):** Complete analytical phase-plane analysis, cubic nullcline geometry, Cardano root solving, Jacobian matrix evaluations, Poincaré stability classification, and automated parameter sweeps ($I, \varepsilon$).
- **Stage 2 (Coupled Cardiac Conduction Network):** Explicit multi-compartment discrete conduction network extending the single-cell model through gap-junction diffusion across six primary functional regions: **Sinoatrial (SA) node, Atrial myocardium, Atrioventricular (AV) node, Bundle of His, Purkinje network, and Ventricular myocardium**.
- **Stage 3 (Arrhythmia Scenario Engine):** Numerically integrated simulation suite modeling six distinct physiological and pathological rhythm scenarios with live dipole rhythm trace projection.
- **Stage 4 (Interactive Anatomical Heart Visualization):** High-performance 60 FPS HTML5/SVG/Canvas visualizer rendering the anatomical human heart, driven directly by continuous numerical ODE states ($v_i, w_i$) without pre-recorded animations.
- **Stage 5 (Rhythm Visualization & Activation Analysis):** Full rhythm analysis panel featuring multi-lead field projections, separate $v(t)$ and $w(t)$ regional plots, multi-channel activation timelines with missing-event detection (Sinus Pause & AV Block), beat-to-beat tachograms, estimated BPM conversion, and numerical activation event logs with CSV export.
- **Deliverable 1 (Mathematical Research Report):** Complete formal documentation and analytical derivations located in [docs/MATHEMATICAL_RESEARCH_REPORT.md](file:///c:/Users/Saloni/Desktop/Cardiac%20Electrical%20Activity%20&%20Arrhythmia%20Simulator/docs/MATHEMATICAL_RESEARCH_REPORT.md).

> **Scientific & Medical Boundaries (Non-Clinical Disclaimers):**
> 1. **Model Scope:** The FHN system is a dimensionless, two-variable phenomenological excitable model. It does **not** simulate an anatomical 3D human heart or individual ionic currents ($I_{\text{Na}}, I_{\text{Ca,L}}, I_{\text{K}}$).
> 2. **Rhythm Traces:** The computed rhythm waveforms are activation-derived signals generated via electrical dipole moment projection ($\Phi(t) = \sum g_{ij}(v_j - v_i) \cdot \vec{d}_{ij}$) and idealized Einthoven limb lead projections. They are **not** clinical 12-lead electrocardiograms (ECGs) and cannot be used for medical diagnosis or clinical decision-making.
> 3. **Emergent Physics:** Conduction delays and blocks emerge dynamically from non-linear threshold dynamics, intercellular coupling ($D \cdot A_{ij}$), and recovery variable refractoriness ($w_i$). Zero artificial or pre-recorded animation delays are used.
> 4. **Honest Numerical Re-entry:** The re-entry module simulates an actual closed-loop circuit and evaluates whether circulation persists based on the relationship between loop traversal time $T_{\text{loop}}$ and refractory recovery duration $T_{\text{refractory}}$.

---

## 2. Project File Structure

```
Cardiac Electrical Activity & Arrhythmia Simulator/
├── docs/                               # Academic Deliverable 1
│   └── MATHEMATICAL_RESEARCH_REPORT.md # Comprehensive Mathematical & Dynamical Report
├── fhn_core/                           # Standalone Mathematical Engine (Zero UI code)
│   ├── __init__.py                     # Package interface & unified exports
│   ├── model.py                        # Single-cell FHN equations, parameter dataclass, vector field
│   ├── solver.py                       # Single-cell 4th-order Runge-Kutta (RK4) integrator
│   ├── analysis.py                     # Nullclines, cubic roots, Jacobian, eigenvalues, stability
│   ├── sweeps.py                       # Automated sweeps (I, ε), Hopf & fold checks
│   ├── network.py                      # Stage 2: Graph topology, network RK4 integrator
│   ├── activation.py                   # Stage 2: Depolarization detector, latency & block analyzer
│   ├── scenarios.py                    # Stage 3: Arrhythmia scenario engine & dipole rhythm projection
│   ├── visualization.py                # Stage 4: Anatomical heart SVG, 3-state color mapper, 60 FPS Canvas
│   └── rhythm_analysis.py              # Stage 5: Event logs, BPM conversion, tachogram, Lead II projection
├── tests/                              # Automated Verification Suite (47 Unit Tests)
│   ├── __init__.py
│   ├── test_rk4_solver.py              # RK4 4th-order convergence on known problem
│   ├── test_equilibrium.py             # Cubic roots & residual verification (< 10⁻¹⁰)
│   ├── test_jacobian.py                # Analytical vs finite-difference Jacobian, trace/det/λ
│   ├── test_sweeps.py                  # Parameter sweeps, analytical Hopf locus, fold non-existence
│   ├── test_network.py                 # Isolated cells, connected cells, emergent wave, AV block
│   ├── test_scenarios.py               # Stage 3: Normal, Tachycardia, Bradycardia, Pause, AV Block, Re-entry
│   ├── test_visualization.py          # Stage 4: Color mapping, frame packing, HTML generation, fidelity
│   ├── test_audit_stage4.py           # Technical validation audit suite (7 formal criteria)
│   └── test_rhythm_analysis.py        # Stage 5: Event log, tachogram, BPM, missing events, Lead II
├── app.py                              # Interactive Streamlit Web Application (Stages 1-5)
├── run_cli.py                          # CLI runner for headless analysis, testing & scenario figures/HTML
├── scenario_normal.png                 # Generated Normal Rhythm multi-channel & dipole trace figure
├── scenario_reentry.png                # Generated Re-entry multi-channel & circus trace figure
├── scenario_normal_vis.html            # Exported standalone interactive anatomical visualizer (Normal)
├── scenario_reentry_vis.html           # Exported standalone interactive anatomical visualizer (Re-entry)
├── scenario_normal_event_log.csv       # Exported numerical activation event log (Normal)
├── scenario_pause_event_log.csv        # Exported numerical activation event log (Sinus Pause)
├── scenario_av_block_event_log.csv     # Exported numerical activation event log (AV Block)
└── README.md                           # Documentation & test manual
```
│   └── test_visualization.py          # Stage 4: Color mapping, frame packing, HTML generation, fidelity
├── app.py                              # Interactive Streamlit Web Application (Stages 1-4)
├── run_cli.py                          # CLI runner for headless analysis, testing & scenario figures/HTML
├── scenario_normal.png                 # Generated Normal Rhythm multi-channel & dipole trace figure
├── scenario_reentry.png                # Generated Re-entry multi-channel & circus trace figure
├── scenario_normal_vis.html            # Exported standalone interactive anatomical visualizer (Normal)
├── scenario_reentry_vis.html           # Exported standalone interactive anatomical visualizer (Re-entry)
└── README.md                           # Documentation & test manual
```

---

## 3. Mathematical Formulations

### 3.1 Base Single-Cell Model (Stage 1)
$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + I$$
$$\frac{dw}{dt} = \varepsilon(v + a - bw)$$

### 3.2 Coupled Conduction Network Model (Stages 2 & 3)
$$\frac{dv_i}{dt} = v_i - \frac{v_i^3}{3} - w_i + I_{\text{stim}, i}(t) + D \sum_{j \in \mathcal{N}(i)} A_{ij}(v_j - v_i)$$
$$\frac{dw_i}{dt} = \varepsilon_i(v_i + a_i - b_i w_i)$$

- $i \in \{0, \dots, N-1\}$: Network node compartments.
- $A_{ij}$: Intercellular gap-junction coupling conductance matrix ($g_{ij} \ge 0$).
- $D$: Global electrical coupling strength ($D = 0.70$).
- $I_{\text{stim}, i}(t)$: External stimulus current delivered to target pacing node (e.g. SA node).
- $w_i(t)$: Recovery variable governing refractoriness.

### 3.3 Dipole-Projected Rhythm Waveform
The simulated rhythm trace $\Phi(t)$ projects the network's instantaneous electrical dipole moment onto a base-to-apex vertical lead axis:
$$\Phi(t) = \sum_{(i,j) \in \text{Edges}} g_{ij} \big( v_j(t) - v_i(t) \big) \cdot (y_i - y_j)$$

---

## 4. Stage 3 Arrhythmia Scenario Suite & Stage 4 Heart Visualizer

| Scenario | Primary Parameter / Mechanism Altered | Resulting Electrophysiological Behavior | Stage 4 Anatomical Heart Visualizer Behavior |
|:---|:---|:---|:---|
| **1. Normal Sinus Rhythm** | Baseline regular SA pacing: $T_{\text{pace}} = 45.0$, conductances intact ($g_{\text{AV}} = 0.35$). | Synchronous 1:1 forward propagation: SA $\to$ Atria $\to$ AVN delay $\to$ His $\to$ Purkinje $\to$ Ventricles. | Regular golden depolarization waves originate in SA node, sweep through atria, pause at AV node, fire through ventricles, and repolarize in violet. |
| **2. Tachycardia-Like Rhythm** | Shortened SA pacing interval: $T_{\text{pace}} = 35.0$ time units. | Accelerated pacing frequency; shorter diastolic interval; 1:1 rapid ventricular activation (**1.28x** rate). | Rapid golden excitation pulses flash frequently across myocardium with abbreviated resting intervals. |
| **3. Bradycardia-Like Rhythm** | Lengthened SA pacing interval: $T_{\text{pace}} = 75.0$ time units. | Depressed pacing frequency; prolonged resting recovery between beats (**0.60x** rate). | Slow, infrequent activation waves with extended dark slate polarized resting periods. |
| **4. Sinus Pause** | Pacing stimulus clamped to zero during pause window $[30.0, 110.0]$. | Complete cessation of activations during pause interval; resumption of regular beats upon unclamp. | Golden waves abruptly cease at $t = 30$; heart rests silently in dark slate until regular pacing resumes at $t = 110$. |
| **5. AV Block-Like Condition** | Junctional coupling weakened or severed ($g_{\text{AV}} = 0.00$). | Upstream SA node and Atrium fire regularly (4 beats); downstream Ventricles experience complete conduction failure (0 beats). | SA node and atria flash gold repeatedly; excitation hits the severed AV junction and fails completely; ventricles remain dark slate. |
| **6. Re-entry-Like Activation** | Closed-loop circuit (10 nodes) with single starter pulse at $t = 0$. | Circulating excitation wave continuously revolves around the ring if loop traversal time $T_{\text{loop}} > T_{\text{refractory}}$. | Dedicated ring circuit displays a self-sustaining golden wavefront orbiting continuously around the obstacle core with a trailing purple refractory wake. |

---

## 5. Automated Verification Suite (47 Tests)

All 47 automated unit tests pass with zero warnings across all 5 stages:

```
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Saloni\Desktop\Cardiac Electrical Activity & Arrhythmia Simulator
collected 47 items

tests/test_audit_stage4.py (7 tests) ................................... PASSED [ 14%]
tests/test_equilibrium.py (4 tests) .................................... PASSED [ 23%]
tests/test_jacobian.py (3 tests) ....................................... PASSED [ 29%]
tests/test_network.py (5 tests) ........................................ PASSED [ 40%]
tests/test_rhythm_analysis.py (7 tests) ................................ PASSED [ 55%]
tests/test_rk4_solver.py (4 tests) ..................................... PASSED [ 63%]
tests/test_scenarios.py (6 tests) ...................................... PASSED [ 76%]
tests/test_sweeps.py (4 tests) ......................................... PASSED [ 85%]
tests/test_visualization.py (7 tests) .................................. PASSED [100%]

============================= 47 passed in 57.04s =============================
```

---

## 6. How to Run the Application

### 6.1 Interactive Streamlit Web Prototype
To launch the complete interactive web dashboard:

```powershell
python -m streamlit run app.py
```
*(Available in your web browser at `http://localhost:8501`)*

**Available Workspaces:**
1. **🏥 Arrhythmia Scenario Engine & Heart Visualizer (Stage 3 & 4):**
   - Select any of the 6 clinical/mathematical scenarios.
   - Watch the interactive anatomical heart visualization update live at 60 FPS.
   - Use Play, Pause, Reset, interactive time scrubbers, and playback speed multipliers (0.25x to 2.0x).
   - Inspect synchronized membrane voltage traces $v_i(t)$ and live dipole rhythm waveforms.
   - Review side-by-side comparative analysis tables and parameter change explanations.
2. **🌐 Coupled Conduction Network & Heart Visualizer (Stage 2 & 4):**
   - Graph visualizer with nodes colored by dynamic voltage and edges scaled by conductance.
   - Interactive anatomical heart animation responding directly to network coupling changes ($g_{\text{AV}}, D, \varepsilon_{\text{AV}}$).
   - Conduction failure alerts and single-cell refractory probe.
3. **⚡ Standalone Single-Cell Engine (Stage 1):**
   - Full phase-plane analysis, nullclines, Jacobian stability, and automated bifurcation sweeps.

### 6.2 Command-Line Interface (CLI)

1. **Run Normal Sinus Rhythm Scenario (Exports PNG and Interactive HTML):**
   ```powershell
   python run_cli.py --scenario normal
   ```
2. **Run Tachycardia-Like Scenario:**
   ```powershell
   python run_cli.py --scenario tachycardia --pacing-interval 35.0
   ```
3. **Run Bradycardia-Like Scenario:**
   ```powershell
   python run_cli.py --scenario bradycardia --pacing-interval 75.0
   ```
4. **Run Sinus Pause Scenario:**
   ```powershell
   python run_cli.py --scenario pause
   ```
5. **Run AV Block Scenario:**
   ```powershell
   python run_cli.py --scenario av_block
   ```
6. **Run Re-entry Circus Movement Scenario:**
   ```powershell
   python run_cli.py --scenario reentry
   ```
7. **Run All 33 Automated Unit Tests:**
   ```powershell
   python -m pytest tests -v
   ```
