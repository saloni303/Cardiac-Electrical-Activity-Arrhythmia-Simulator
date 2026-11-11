# ODE-Based Simulation of Cardiac Electrical Activity and Arrhythmia Dynamics Using the FitzHugh–Nagumo Model

> **Cardiac-Electrical-Activity-Arrhythmia-Simulator**  
> A mathematical simulation of heart cell electrical activity using the FitzHugh–Nagumo (FHN) model. Explore how changing parameters affects cell behaviour, conduction latencies, and arrhythmia-like patterns through interactive graphs. Focused on single-cell and coupled-network mathematical modelling, not clinical diagnosis or medical ECG analysis.

**Stages 1, 2 & 3: Standalone FHN Mathematical Engine, Conduction Network & Arrhythmia Scenario Suite**

---

## 1. Overview & Scientific Boundaries
This repository contains an academic research simulation suite developed to investigate cardiac electrophysiology, wave propagation, and cardiac rhythm disorders using the **FitzHugh–Nagumo (FHN)** ordinary differential equation (ODE) system.

- **Stage 1 (Standalone Mathematical Engine):** Complete analytical phase-plane analysis, cubic nullcline geometry, Cardano root solving, Jacobian matrix evaluations, Poincaré stability classification, and automated parameter sweeps ($I, \varepsilon$).
- **Stage 2 (Coupled Cardiac Conduction Network):** Explicit multi-compartment discrete conduction network extending the single-cell model through gap-junction diffusion across six primary functional regions: **Sinoatrial (SA) node, Atrial myocardium, Atrioventricular (AV) node, Bundle of His, Purkinje network, and Ventricular myocardium**.
- **Stage 3 (Arrhythmia Scenario Engine):** Numerically integrated simulation suite modeling six distinct physiological and pathological rhythm scenarios with live dipole rhythm trace projection.

> **Scientific & Medical Boundaries (Non-Clinical Disclaimers):**
> 1. **Model Scope:** The FHN system is a dimensionless, two-variable phenomenological excitable model. It does **not** simulate an anatomical 3D human heart or individual ionic currents ($I_{\text{Na}}, I_{\text{Ca,L}}, I_{\text{K}}$).
> 2. **Rhythm Traces:** The computed rhythm waveforms are activation-derived signals generated via electrical dipole moment projection ($\Phi(t) = \sum g_{ij}(v_j - v_i) \cdot \vec{d}_{ij}$). They are **not** clinical 12-lead electrocardiograms (ECGs) and cannot be used for medical diagnosis or clinical decision-making.
> 3. **Emergent Physics:** Conduction delays and blocks emerge dynamically from non-linear threshold dynamics, intercellular coupling ($D \cdot A_{ij}$), and recovery variable refractoriness ($w_i$). Zero artificial or pre-recorded animation delays are used.
> 4. **Honest Numerical Re-entry:** The re-entry module simulates an actual closed-loop circuit and evaluates whether circulation persists based on the relationship between loop traversal time $T_{\text{loop}}$ and refractory recovery duration $T_{\text{refractory}}$.

---

## 2. Project File Structure

```
Cardiac Electrical Activity & Arrhythmia Simulator/
├── fhn_core/                           # Standalone Mathematical Engine (Zero UI code)
│   ├── __init__.py                     # Package interface & unified exports
│   ├── model.py                        # Single-cell FHN equations, parameter dataclass, vector field
│   ├── solver.py                       # Single-cell 4th-order Runge-Kutta (RK4) integrator
│   ├── analysis.py                     # Nullclines, cubic roots, Jacobian, eigenvalues, stability
│   ├── sweeps.py                       # Automated sweeps (I, ε), Hopf & fold checks
│   ├── network.py                      # Stage 2: Graph topology, network RK4 integrator
│   ├── activation.py                   # Stage 2: Depolarization detector, latency & block analyzer
│   └── scenarios.py                    # Stage 3: Arrhythmia scenario engine & dipole rhythm projection
├── tests/                              # Automated Verification Suite (26 Unit Tests)
│   ├── __init__.py
│   ├── test_rk4_solver.py              # RK4 4th-order convergence on known problem
│   ├── test_equilibrium.py             # Cubic roots & residual verification (< 10⁻¹⁰)
│   ├── test_jacobian.py                # Analytical vs finite-difference Jacobian, trace/det/λ
│   ├── test_sweeps.py                  # Parameter sweeps, analytical Hopf locus, fold non-existence
│   ├── test_network.py                 # Isolated cells, connected cells, emergent wave, AV block
│   └── test_scenarios.py               # Stage 3: Normal, Tachycardia, Bradycardia, Pause, AV Block, Re-entry
├── app.py                              # Interactive Streamlit Web Application (Stages 1, 2 & 3)
├── run_cli.py                          # CLI runner for headless analysis, testing & scenario figures
├── scenario_normal.png                 # Generated Normal Rhythm multi-channel & dipole trace figure
├── scenario_reentry.png                # Generated Re-entry multi-channel & circus trace figure
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

## 4. Stage 3 Arrhythmia Scenario Suite

| Scenario | Primary Parameter / Mechanism Altered | Resulting Electrophysiological Behavior | Cycle Length & Ventricular Rate |
|:---|:---|:---|:---|
| **1. Normal Sinus Rhythm** | Baseline regular SA pacing: $T_{\text{pace}} = 45.0$, conductances intact ($g_{\text{AV}} = 0.35$). | Synchronous 1:1 forward propagation: SA $\to$ Atria $\to$ AVN delay $\to$ His $\to$ Purkinje $\to$ Ventricles. | $T_{\text{cycle}} \approx 44.8$ units, Rate $\approx 2.23$ beats / 100 units (1.00x). |
| **2. Tachycardia-Like Rhythm** | Shortened SA pacing interval: $T_{\text{pace}} = 35.0$ time units. | Accelerated pacing frequency; shorter diastolic interval; 1:1 rapid ventricular activation. | $T_{\text{cycle}} \approx 35.0$ units, Rate $\approx 2.86$ beats / 100 units (**1.28x**). |
| **3. Bradycardia-Like Rhythm** | Lengthened SA pacing interval: $T_{\text{pace}} = 75.0$ time units. | Depressed pacing frequency; prolonged resting recovery between beats. | $T_{\text{cycle}} \approx 74.5$ units, Rate $\approx 1.33$ beats / 100 units (**0.60x**). |
| **4. Sinus Pause** | Pacing stimulus clamped to zero during pause window $[30.0, 110.0]$. | Complete cessation of activations during pause interval; resumption of regular beats upon unclamp. | Quiescent interval $> 80$ units; 0 beats during suppression window. |
| **5. AV Block-Like Condition** | Junctional coupling weakened or severed ($g_{\text{AV}} = 0.00$). | Upstream SA node and Atrium fire regularly (4 beats); downstream Ventricles experience complete conduction failure (0 beats). | Ventricular rate = 0.0 beats / 100 units (**Complete Block**). |
| **6. Re-entry-Like Activation** | Closed-loop circuit (10 nodes) with single starter pulse at $t = 0$. | Circulating excitation wave continuously revolves around the ring if loop traversal time $T_{\text{loop}} > T_{\text{refractory}}$. | Sustained circus movement: **6 spontaneous revolutions**, $T_{\text{cycle}} \approx 23.9$ units. |

---

## 5. Automated Verification Suite (26 Tests)

All 26 automated unit tests pass with zero warnings:

```
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Saloni\Desktop\Cardiac Electrical Activity & Arrhythmia Simulator
collected 26 items

tests/test_equilibrium.py::test_standard_cardiac_equilibrium PASSED      [  3%]
tests/test_equilibrium.py::test_resting_state_equilibrium PASSED         [  7%]
tests/test_equilibrium.py::test_nullcline_intersection_coincidence PASSED [ 11%]
tests/test_equilibrium.py::test_multiple_equilibria_regime PASSED        [ 15%]
tests/test_jacobian.py::test_analytical_vs_finite_difference_jacobian PASSED [ 19%]
tests/test_jacobian.py::test_trace_determinant_eigenvalue_consistency PASSED [ 23%]
tests/test_jacobian.py::test_stability_classification_regimes PASSED     [ 26%]
tests/test_network.py::test_isolated_cells_uncoupled_dynamics PASSED     [ 30%]
tests/test_network.py::test_connected_cells_diffusive_equilibration PASSED [ 34%]
tests/test_network.py::test_sequential_excitation_propagation PASSED     [ 38%]
tests/test_network.py::test_conduction_block_detection_on_severed_pathway PASSED [ 42%]
tests/test_network.py::test_refractory_suppression_of_premature_stimulus PASSED [ 46%]
tests/test_rk4_solver.py::test_rk4_order_of_accuracy_on_exponential PASSED [ 50%]
tests/test_rk4_solver.py::test_fhn_rk4_solve_structure PASSED            [ 53%]
tests/test_rk4_solver.py::test_fhn_step_refinement_convergence PASSED    [ 57%]
tests/test_rk4_solver.py::test_invalid_parameters_handling PASSED        [ 61%]
tests/test_scenarios.py::test_scenario_normal_rhythm_conduction PASSED   [ 65%]
tests/test_scenarios.py::test_scenario_tachycardia_frequency_increase PASSED [ 69%]
tests/test_scenarios.py::test_scenario_bradycardia_frequency_decrease PASSED [ 73%]
tests/test_scenarios.py::test_scenario_sinus_pause_suppression_window PASSED [ 76%]
tests/test_scenarios.py::test_scenario_av_block_conduction_failure PASSED [ 80%]
tests/test_scenarios.py::test_scenario_reentry_numerical_honesty PASSED  [ 84%]
tests/test_sweeps.py::test_sweep_parameter_I_execution PASSED            [ 88%]
tests/test_sweeps.py::test_hopf_candidates_accuracy PASSED               [ 92%]
tests/test_sweeps.py::test_fold_nonexistence_for_cardiac_b PASSED        [ 96%]
tests/test_sweeps.py::test_sweep_parameter_epsilon_execution PASSED      [100%]

============================= 26 passed in 8.54s ==============================
```

---

## 6. How to Run the Application

### 6.1 Interactive Streamlit Web Prototype
To launch the complete dual-mode interactive web dashboard:

```powershell
python -m streamlit run app.py
```
*(Available in your web browser at `http://localhost:8501`)*

**Available Workspaces:**
1. **🏥 Arrhythmia Scenario Engine (Stage 3):**
   - Select any of the 6 clinical/mathematical scenarios.
   - Adjust pacing intervals, pause windows, block conductances, or re-entry loop sizes dynamically.
   - Inspect multi-channel action potential traces and live dipole-projected rhythm waveforms.
   - Review side-by-side comparative analysis tables and parameter change explanations.
2. **🌐 Coupled Conduction Network (Stage 2):**
   - Graph visualizer with nodes colored by dynamic voltage and edges scaled by conductance.
   - Conduction failure alerts and single-cell refractory probe.
3. **⚡ Standalone Single-Cell Engine (Stage 1):**
   - Full phase-plane analysis, nullclines, Jacobian stability, and automated bifurcation sweeps.

### 6.2 Command-Line Interface (CLI)

1. **Run Normal Sinus Rhythm Scenario:**
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
7. **Run All 26 Automated Unit Tests:**
   ```powershell
   python -m pytest tests -v
   ```
