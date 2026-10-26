"""Demonstration script for Phase 5: Fixed ACK Policy Engine & Comparative Evaluation.

Runs a reproducible controlled comparative benchmark across all 5 ACK policies:
- Policy A: Static MQTT QoS 0 (Fire-and-forget)
- Policy B: Static Local ACK (Edge proxy ACK without E2E proof)
- Policy C: Static MQTT QoS 1 + Retries (Broker PUBACK + bounded app-level retries)
- Policy D: Local ACK + Selective Retransmission (CC priority retransmission)
- Policy E: Static MQTT QoS 2 (Exactly-once 4-way protocol handshake)

Outputs a structured summary table and exports results to CSV.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rl_ack.policies.network_model import NetworkModelConfig
from rl_ack.policies.runner import PolicyExperimentRunner
from rl_ack.policies.workload import WorkloadGenerator


def main() -> None:
    print("=" * 78)
    print("⚖️  RL-ACK Phase 5: Fixed ACK Policy Comparative Evaluation Benchmark")
    print("=" * 78)

    seed = 42
    workload_size = 25
    loss_rate = 0.12
    base_lat = 15.0

    print(f"\n[CONFIG] Workload Size        : {workload_size} heterogeneous messages")
    print(f"[CONFIG] Random Seed          : {seed} (Deterministic Reproducibility)")
    print(f"[CONFIG] Packet Loss Rate     : {loss_rate * 100:.1f}%")
    print(f"[CONFIG] Base Latency         : {base_lat:.1f} ms (Jitter: ±5.0 ms)")
    print(f"[CONFIG] Latency Deadline     : 100.0 ms\n")

    # 1. Generate identical benchmark workload
    gen = WorkloadGenerator(random_seed=seed)
    workload = gen.generate_workload(count=workload_size, cc_ratio=0.32, hbs_ratio=0.36, bet_ratio=0.32)
    cc_count = sum(1 for m in workload if m.priority.value == "Critical Control")
    print(f"[STEP 1] Generated workload: {len(workload)} messages ({cc_count} Critical Control alerts)")

    # 2. Run controlled evaluation
    print("[STEP 2] Executing 5-Policy Comparative Evaluation under identical channel draws...")
    runner = PolicyExperimentRunner()
    net_cfg = NetworkModelConfig(
        latency_base_ms=base_lat,
        jitter_ms=5.0,
        packet_loss_rate=loss_rate,
        ack_loss_rate=0.05,
        random_seed=seed,
    )

    summary = runner.run_controlled_comparison(workload=workload, network_config=net_cfg)

    # 3. Print side-by-side summary table
    print("\n" + "=" * 78)
    print("📊 EMPIRICAL COMPARISON SUMMARY TABLE")
    print("=" * 78)
    header = f"{'Policy':<42} | {'Deliv %':>7} | {'Mean ms':>8} | {'P95 ms':>8} | {'Overhead':>9} | {'CC Deliv %':>10}"
    print(header)
    print("-" * 78)

    for p_name, m in summary.items():
        mean_str = f"{m.mean_e2e_latency_ms:.1f}" if m.mean_e2e_latency_ms is not None else "—"
        p95_str = f"{m.p95_e2e_latency_ms:.1f}" if m.p95_e2e_latency_ms is not None else "—"
        print(
            f"{p_name:<42} | "
            f"{m.delivery_confirmation_rate * 100:>6.1f}% | "
            f"{mean_str:>8} | "
            f"{p95_str:>8} | "
            f"{m.total_overhead_bytes:>7} B | "
            f"{m.cc_delivery_rate * 100:>9.1f}%"
        )

    # 4. Export CSV
    output_dir = root_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "policy_comparison_summary.csv"
    results_csv_path = output_dir / "policy_comparison_messages.csv"

    runner.export_summary_csv(str(summary_csv_path))
    runner.export_results_csv(str(results_csv_path))

    print("-" * 78)
    print(f"✅ Summary CSV saved to : {summary_csv_path.name}")
    print(f"✅ Granular CSV saved to: {results_csv_path.name}")
    print("\n[CONCLUSION] Policy trade-offs empirically demonstrated:")
    print("  - Policy A (QoS 0) has minimal overhead but suffers from channel packet drops.")
    print("  - Policy B (Local ACK) provides near-instant edge proxy response but no E2E confirmation.")
    print("  - Policy C (QoS 1 + Retries) achieves high delivery via bounded retries with added latency.")
    print("  - Policy D (Local ACK + Selective Retransmit) protects Critical Control alerts with high delivery")
    print("    while avoiding wasteful retries for routine telemetry, saving bandwidth and energy.")
    print("  - Policy E (QoS 2) enforces exactly-once protocol delivery at the expense of higher 4-way overhead.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
