"""
Command-line interface and benchmark utility for the FitzHugh-Nagumo mathematical engine.

Usage:
    python run_cli.py --help
    python run_cli.py --analyze --I 0.5 --eps 0.08 --a 0.7 --b 0.8
    python run_cli.py --solve --v0 -1.0 --w0 0.2 --tend 40.0
    python run_cli.py --sweep
    python run_cli.py --test
    python run_cli.py --plot-figures
"""

import argparse
import sys
import numpy as np
import matplotlib
# Use non-interactive backend for headless plotting
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fhn_core.model import FHNParameters, compute_vector_field
from fhn_core.solver import rk4_solve
from fhn_core.analysis import find_equilibria, compute_nullclines
from fhn_core.sweeps import sweep_parameter_I, sweep_parameter_epsilon, find_hopf_candidates, check_fold_existence
from fhn_core.network import create_default_cardiac_network, solve_network
from fhn_core.activation import detect_activations, compute_propagation_latencies, detect_conduction_failures
from fhn_core.scenarios import ScenarioType, ScenarioConfig, run_scenario


def print_banner():
    print("=" * 80)
    print("   FITZHUGH-NAGUMO STANDALONE MATHEMATICAL ENGINE (STAGE 1)")
    print("   ODE System:")
    print("       dv/dt = v - v³/3 - w + I")
    print("       dw/dt = ε(v + a - bw)")
    print("   Scientific Note: Dimensionless single-cell nonlinear excitable model.")
    print("   Not a whole-heart model; not for clinical diagnosis.")
    print("=" * 80)


def run_analysis(params: FHNParameters):
    print(f"\n--- MODEL PARAMETERS ---")
    print(f"  a = {params.a:.4f},  b = {params.b:.4f},  ε = {params.epsilon:.4f},  I = {params.I:.4f}")

    equilibria = find_equilibria(params)
    print(f"\n--- EQUILIBRIA & STABILITY ANALYSIS ({len(equilibria)} found) ---")

    for idx, eq in enumerate(equilibria, start=1):
        print(f"\n[Equilibrium #{idx}]")
        print(f"  Coordinates:           v* = {eq.v_star:+.6f},  w* = {eq.w_star:+.6f}")
        print(f"  Residuals:             |r_v| = {abs(eq.residual_v):.2e},  |r_w| = {abs(eq.residual_w):.2e}")
        print(f"  Jacobian J*:           [[{eq.jacobian[0, 0]:+.6f}, {eq.jacobian[0, 1]:+.6f}],")
        print(f"                          [{eq.jacobian[1, 0]:+.6f}, {eq.jacobian[1, 1]:+.6f}]]")
        print(f"  Trace τ:               {eq.trace:+.6f}")
        print(f"  Determinant Δ:         {eq.determinant:+.6f}")
        print(f"  Discriminant D:        {eq.discriminant:+.6f}")
        print(f"  Eigenvalues:           λ1 = {eq.eigenvalues[0]:+.6f}")
        print(f"                         λ2 = {eq.eigenvalues[1]:+.6f}")
        print(f"  Stability Class:       {eq.stability_class.value}")
        print(f"  Asymptotically Stable: {eq.is_stable}")
        print(f"  Explanation:           {eq.summary_text}")


def run_solve(params: FHNParameters, v0: float, w0: float, tend: float, dt: float):
    print(f"\n--- NUMERICAL RK4 INTEGRATION ---")
    print(f"  Initial State:  v(0) = {v0:.4f},  w(0) = {w0:.4f}")
    print(f"  Time Span:      t in [0.0, {tend:.2f}],  dt = {dt:.4f}")

    t, v, w = rk4_solve((v0, w0), (0.0, tend), dt, params)

    print(f"  Integration completed successfully across {len(t)} steps.")
    print(f"  Final State:    v({tend:.1f}) = {v[-1]:+.6f},  w({tend:.1f}) = {w[-1]:+.6f}")
    print(f"  Voltage Range:  v_min = {np.min(v):+.4f},  v_max = {np.max(v):+.4f}")
    print(f"  Recovery Range: w_min = {np.min(w):+.4f},  w_max = {np.max(w):+.4f}")


def run_sweep(params: FHNParameters):
    print(f"\n--- BIFURCATION & PARAMETER SWEEP ANALYSIS ---")

    # 1. Hopf candidate check
    hopf_cands = find_hopf_candidates(params)
    print(f"\nAnalytical Hopf candidates (where τ = 0, Δ > 0):")
    if hopf_cands:
        for c in hopf_cands:
            print(f"  v* = {c['v_star']:+.5f},  w* = {c['w_star']:+.5f},  I_Hopf = {c['I_hopf']:+.5f},  "
                  f"det = {c['determinant']:.5f},  transverse = {c['is_transverse']}")
    else:
        print("  None detected for this parameter combination.")

    # 2. Fold check
    fold_info = check_fold_existence(params)
    print(f"\nSaddle-Node Fold Check:")
    print(f"  {fold_info['explanation']}")

    # 3. Sweep over I
    print(f"\nRunning fine numerical sweep over I in [-0.2, 1.8] (50 points)...")
    res_I = sweep_parameter_I(-0.2, 1.8, 50, params)
    stable_count = sum(1 for r in res_I["records"] if r["is_stable"])
    unstable_count = len(res_I["records"]) - stable_count
    print(f"  Evaluated {len(res_I['records'])} equilibrium states:")
    print(f"    - Stable states:   {stable_count}")
    print(f"    - Unstable states: {unstable_count}")


def generate_figures(params: FHNParameters, v0: float = 1.2, w0: float = -0.5, tend: float = 50.0):
    print(f"\n--- GENERATING PUBLICATION FIGURES ---")

    # 1. Time Series Plot
    t, v, w = rk4_solve((v0, w0), (0.0, tend), 0.02, params)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(t, v, color="#1f77b4", lw=2, label="v(t) [Voltage-like]")
    ax1.set_ylabel("v (Dimensionless)")
    ax1.set_title(f"FitzHugh-Nagumo Time Series (a={params.a}, b={params.b}, ε={params.epsilon}, I={params.I})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(t, w, color="#ff7f0e", lw=2, label="w(t) [Recovery]")
    ax2.set_xlabel("Time (model units)")
    ax2.set_ylabel("w (Dimensionless)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig_time = "fhn_time_series.png"
    plt.savefig(fig_time, dpi=150)
    plt.close()
    print(f"  Saved: {fig_time}")

    # 2. Phase Plane with Nullclines, Direction Field, and Trajectory
    fig, ax = plt.subplots(figsize=(8, 7))

    v_bounds = (-2.5, 2.5)
    w_bounds = (-2.0, 2.5)

    # Vector Field
    V, W, dV, dW, dV_norm, dW_norm = compute_vector_field(v_bounds, w_bounds, 22, params)
    speed = np.sqrt(dV**2 + dW**2)
    ax.quiver(V, W, dV_norm, dW_norm, speed, cmap="Blues", alpha=0.55, pivot="mid")

    # Nullclines
    v_grid, w_v_null, w_w_null = compute_nullclines(v_bounds, params)
    ax.plot(v_grid, w_v_null, color="#1f77b4", lw=2.5, label="v-nullcline: w = v - v³/3 + I")
    ax.plot(v_grid, w_w_null, color="#ff7f0e", lw=2.5, label="w-nullcline: w = (v + a)/b")

    # Numerical RK4 Trajectory
    ax.plot(v, w, color="#2ca02c", lw=2.0, label=f"RK4 Trajectory (v0={v0}, w0={w0})")
    ax.plot(v[0], w[0], "go", markersize=8, label="Initial State (v0, w0)")

    # Equilibria
    equilibria = find_equilibria(params)
    for eq in equilibria:
        marker = "ro" if not eq.is_stable else "bo"
        label = f"Equilibrium ({eq.v_star:.2f}, {eq.w_star:.2f}) [{eq.stability_class.name}]"
        ax.plot(eq.v_star, eq.w_star, marker, markersize=10, markeredgecolor="black", label=label)

    ax.set_xlim(v_bounds)
    ax.set_ylim(w_bounds)
    ax.set_xlabel("v (Voltage-like variable)")
    ax.set_ylabel("w (Recovery variable)")
    ax.set_title("FitzHugh-Nagumo Phase Plane: Nullclines, Direction Field & Trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig_phase = "fhn_phase_portrait.png"
    plt.savefig(fig_phase, dpi=150)
    plt.close()
    print(f"  Saved: {fig_phase}")


def run_network_demo(av_conductance: float = 0.15, global_coupling: float = 0.7, t_end: float = 60.0):
    print("\n" + "=" * 80)
    print("   STAGE 2: COUPLED CARDIAC CONDUCTION NETWORK SIMULATION")
    print("=" * 80)
    print(f"Network Parameters: AV Conductance g_AV = {av_conductance:.3f}, Global Coupling D = {global_coupling:.2f}")

    net = create_default_cardiac_network(av_conductance=av_conductance, global_coupling=global_coupling)

    print("\n--- Network Nodes & Regional Parameters ---")
    for i in range(net.num_nodes):
        node = net.nodes[i]
        p = node.params
        print(f"  Node {node.node_id:2d}: [{node.region.name:<12s}] {node.name:<10s} | a={p.a:.2f}, b={p.b:.2f}, ε={p.epsilon:.3f}, I_bias={p.I:.2f}")

    print("\n--- Inter-Node Conduction Pathways (Edges) ---")
    for edge in net.edges:
        src = net.nodes[edge.source_id].name
        tgt = net.nodes[edge.target_id].name
        status = "ACTIVE" if edge.is_active else "SEVERED"
        print(f"  {src:<10s} ---> {tgt:<10s} | Conductance g = {edge.conductance:.3f} | [{status}]")

    print(f"\nIntegrating network ODE system via RK4 over t in [0.0, {t_end:.1f}]...")
    sim = solve_network(net, t_span=(0.0, t_end), dt=0.02)
    time = sim["time"]
    V = sim["V"]

    print("Detecting depolarization activations...")
    activations = detect_activations(time, V, threshold=0.0)

    print("\n--- Activation Timestamps ---")
    for i in range(net.num_nodes):
        node_name = net.nodes[i].name
        acts = activations[i]
        acts_str = ", ".join(f"{t:.2f}" for t in acts[:4]) + ("..." if len(acts) > 4 else "")
        print(f"  {node_name:<10s}: {len(acts)} beats detected at t = [{acts_str}]")

    # Propagation Latencies
    print("\n--- Regional Conduction Latencies (Beat #1) ---")
    lat_info = compute_propagation_latencies(activations, net, beat_index=0)
    reg_sum = lat_info["regional_summary"]

    for segment, latency in reg_sum.items():
        if latency is not None:
            print(f"  {segment:<25s}: Δt = {latency:.3f} time units")
        else:
            print(f"  {segment:<25s}: CONDUCTION FAILED / NOT REACHED")

    # Conduction Failures / Blocks
    failures = detect_conduction_failures(activations, net)
    print("\n--- Conduction Failure & Block Detection ---")
    if not failures:
        print("  ✓ Normal Propagation: Excitation wavefront successfully traversed all regions without block.")
    else:
        print(f"  ⚠ {len(failures)} Conduction Failure(s) Detected:")
        for f in failures:
            print(f"    - [{f['failure_type']}] between {f['source_name']} and {f['target_name']}")
            print(f"      Details: {f['reason']}")


def run_scenario_cli(scenario_name: str, pacing_interval: float = None):
    print("\n" + "=" * 80)
    print("   STAGE 3: ARRHYTHMIA SCENARIO SIMULATION ENGINE")
    print("=" * 80)

    s_map = {
        "normal": ScenarioType.NORMAL,
        "tachycardia": ScenarioType.TACHYCARDIA,
        "bradycardia": ScenarioType.BRADYCARDIA,
        "pause": ScenarioType.SINUS_PAUSE,
        "av_block": ScenarioType.AV_BLOCK,
        "reentry": ScenarioType.REENTRY,
    }

    norm_name = scenario_name.lower().strip()
    if norm_name not in s_map:
        print(f"Unknown scenario '{scenario_name}'. Available: {list(s_map.keys())}")
        return

    stype = s_map[norm_name]
    print(f"Selected Scenario: {stype.value.upper()}")

    # Configure scenario
    if stype == ScenarioType.NORMAL:
        cfg = ScenarioConfig(scenario_type=stype, pacing_interval=pacing_interval or 45.0, t_end=160.0)
    elif stype == ScenarioType.TACHYCARDIA:
        cfg = ScenarioConfig(scenario_type=stype, pacing_interval=pacing_interval or 35.0, t_end=160.0)
    elif stype == ScenarioType.BRADYCARDIA:
        cfg = ScenarioConfig(scenario_type=stype, pacing_interval=pacing_interval or 75.0, t_end=160.0)
    elif stype == ScenarioType.SINUS_PAUSE:
        cfg = ScenarioConfig(scenario_type=stype, pacing_interval=pacing_interval or 45.0,
                             pause_start=30.0, pause_end=110.0, t_end=160.0)
    elif stype == ScenarioType.AV_BLOCK:
        cfg = ScenarioConfig(scenario_type=stype, pacing_interval=pacing_interval or 45.0,
                             av_conductance=0.0, block_location="Atrium-to-AVN", t_end=160.0)
    elif stype == ScenarioType.REENTRY:
        cfg = ScenarioConfig(scenario_type=stype, reentry_num_nodes=10, reentry_coupling=0.35,
                             reentry_epsilon=0.15, t_end=150.0)

    print(f"\n1. Parameter Changes: {cfg.scenario_type.value}")
    print(f"   Pacing Interval: {cfg.pacing_interval} units | Duration: {cfg.t_end} units")

    res = run_scenario(cfg)

    print(f"\n2. Explanatory Documentation:")
    print(f"   - Mechanism:   {res.parameter_change_summary}")
    print(f"   - Assumptions: {res.mathematical_assumptions}")

    print(f"\n3. Activation Timing & Detection:")
    for nid in range(res.network.num_nodes):
        name = res.network.nodes[nid].name
        acts = res.activations.get(nid, [])
        acts_str = ", ".join(f"{t:.2f}" for t in acts[:5]) + ("..." if len(acts) > 5 else "")
        print(f"   Node {nid:2d} ({name:<10s}): {len(acts):2d} beats at t = [{acts_str}]")

    print(f"\n4. Rhythm Metrics & Comparison against Normal:")
    if res.mean_cycle_length is not None:
        print(f"   - Mean Cycle Length:    {res.mean_cycle_length:.2f} time units")
    else:
        print(f"   - Mean Cycle Length:    N/A (insufficient beats)")
    print(f"   - Activation Frequency: {res.activation_frequency:.2f} beats / 100 units")
    print(f"   - Relative to Normal:   {res.comparison_with_normal.get('rate_ratio', 1.0):.2f}x")
    print(f"   - Status Description:   {res.comparison_with_normal.get('status')}")

    if stype == ScenarioType.REENTRY:
        print(f"\n5. Re-entry Validation:")
        print(f"   - Sustained Circus Movement: {res.reentry_sustained}")
        print(f"   - Spontaneous Revolutions:   {res.reentry_revolutions}")

    # Plot scenario traces
    fig_name = f"scenario_{norm_name}.png"
    plt.figure(figsize=(9, 6))

    plt.subplot(2, 1, 1)
    for nid in range(min(res.network.num_nodes, 6)):
        plt.plot(res.time, res.V[:, nid], lw=1.5, label=res.network.nodes[nid].name)
    plt.ylabel("Voltage v_i")
    plt.title(f"{stype.value} - Voltage-Like Traces (RK4 Simulation)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)

    plt.subplot(2, 1, 2)
    plt.plot(res.time, res.rhythm_trace, color="#7c3aed", lw=1.8, label="Simulated Rhythm Trace (Dipole-Derived)")
    plt.xlabel("Time (model units)")
    plt.ylabel("Projected Dipole Potential")
    plt.title("Simulated ECG-Like Rhythm Waveform (Illustrative, Non-Diagnostic)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_name, dpi=150)
    plt.close()
    print(f"\n6. Rhythm & Action Potential Plot Saved: {fig_name}")


def main():
    parser = argparse.ArgumentParser(description="Standalone FitzHugh-Nagumo Mathematical Engine (Stage 1)")
    parser.add_argument("--a", type=float, default=0.7, help="Recovery nullcline intercept parameter a (default: 0.7)")
    parser.add_argument("--b", type=float, default=0.8, help="Recovery nullcline slope parameter b (default: 0.8)")
    parser.add_argument("--eps", type=float, default=0.08, help="Timescale separation parameter ε (default: 0.08)")
    parser.add_argument("--I", type=float, default=0.5, help="Applied external current I (default: 0.5)")
    parser.add_argument("--v0", type=float, default=-1.0, help="Initial voltage v(0) (default: -1.0)")
    parser.add_argument("--w0", type=float, default=0.2, help="Initial recovery w(0) (default: 0.2)")
    parser.add_argument("--tend", type=float, default=50.0, help="Simulation end time (default: 50.0)")
    parser.add_argument("--dt", type=float, default=0.02, help="RK4 step size (default: 0.02)")

    parser.add_argument("--analyze", action="store_true", help="Run analytical equilibrium and Jacobian analysis")
    parser.add_argument("--solve", action="store_true", help="Run RK4 numerical trajectory integration")
    parser.add_argument("--sweep", action="store_true", help="Run automated parameter sweeps and bifurcation checks")
    parser.add_argument("--plot-figures", action="store_true", help="Generate and save PNG figures")
    parser.add_argument("--test", action="store_true", help="Execute unit test suite")
    parser.add_argument("--network", action="store_true", help="Run Stage 2 coupled cardiac network demonstration")
    parser.add_argument("--av-conductance", type=float, default=0.15, help="AV nodal coupling conductance (default: 0.15)")
    parser.add_argument("--global-coupling", type=float, default=0.7, help="Global electrical coupling strength D (default: 0.7)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run Stage 3 arrhythmia scenario: normal, tachycardia, bradycardia, pause, av_block, reentry")
    parser.add_argument("--pacing-interval", type=float, default=None, help="Custom pacing interval for scenario")

    args = parser.parse_args()

    print_banner()

    params = FHNParameters(a=args.a, b=args.b, epsilon=args.eps, I=args.I)

    # If no specific action specified, run default demonstration
    if not (args.analyze or args.solve or args.sweep or args.plot_figures or args.test or args.network or args.scenario):
        run_analysis(params)
        run_solve(params, args.v0, args.w0, args.tend, args.dt)
        run_sweep(params)
        return

    if args.analyze:
        run_analysis(params)

    if args.solve:
        run_solve(params, args.v0, args.w0, args.tend, args.dt)

    if args.sweep:
        run_sweep(params)

    if args.plot_figures:
        generate_figures(params, args.v0, args.w0, args.tend)

    if args.scenario:
        run_scenario_cli(args.scenario, args.pacing_interval)
        return

    if args.network:
        run_network_demo(av_conductance=args.av_conductance, global_coupling=args.global_coupling, t_end=args.tend)
        return

    if args.test:
        import pytest
        sys.exit(pytest.main(["tests", "-v"]))


if __name__ == "__main__":
    main()

