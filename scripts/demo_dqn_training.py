"""RL-ACK Phase 6: Deep Q-Network Training and Evaluation CLI Benchmark.

Demonstrates:
1. Training the 8 -> 64 -> 32 -> 5 DQN agent in the controlled simulated environment.
2. Saving checkpoint to checkpoints/dqn_latest.pt and checkpoints/dqn_best.pt.
3. Evaluating the trained agent without exploration across test workloads.
4. Comparing Adaptive RL-ACK DQN against all 5 fixed policies from Phase 5 on identical workloads.
5. Exporting training and evaluation metrics to CSV.

Usage:
    python scripts/demo_dqn_training.py --episodes 15 --steps 30 --eval-samples 50
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure src/ is on sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rl_ack.config import DQNConfig
from rl_ack.dqn.agent import DQNAgent
from rl_ack.dqn.environment import EnvConfig, SmartKitchenIoTEnv
from rl_ack.dqn.evaluator import DQNEvaluator
from rl_ack.dqn.trainer import DQNTrainer
from rl_ack.policies.network_model import NetworkModelConfig
from rl_ack.policies.workload import WorkloadGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-ACK Phase 6 DQN Training & Comparative Benchmark")
    parser.add_argument("--episodes", type=int, default=15, help="Number of training episodes (default: 15)")
    parser.add_argument("--steps", type=int, default=30, help="Steps per episode (default: 30)")
    parser.add_argument("--eval-samples", type=int, default=40, help="Number of test messages for evaluation (default: 40)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for CSV export reports")

    args = parser.parse_args()

    print("=" * 80)
    print("RL-ACK MAJOR PROJECT — PHASE 6: DQN AGENT TRAINING & EVALUATION")
    print("Architecture: 8 Features -> Linear(64) -> Linear(32) -> 5 Q-Values")
    print("=" * 80)
    print(f"Episodes: {args.episodes} | Steps/Ep: {args.steps} | Seed: {args.seed} | LR: {args.lr}")
    print(f"Checkpoints: {args.checkpoint_dir} | Output: {args.output_dir}\n")

    # 1. Setup Configuration & Environment
    dqn_cfg = DQNConfig(
        state_dim=8,
        hidden_dim_1=64,
        hidden_dim_2=32,
        action_dim=5,
        learning_rate=args.lr,
        batch_size=32,
        memory_capacity=5000,
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    env_cfg = EnvConfig(
        steps_per_episode=args.steps,
        random_seed=args.seed,
    )

    agent = DQNAgent(config=dqn_cfg, random_seed=args.seed)
    env = SmartKitchenIoTEnv(config=env_cfg)
    trainer = DQNTrainer(agent=agent, env=env, checkpoint_dir=args.checkpoint_dir)

    # 2. Run Training Loop
    print(">>> Starting DQN Training Loop...")
    start_time = time.time()

    def on_progress(ep: int, total: int, m) -> None:
        loss_str = f"{m.avg_loss:.4f}" if m.avg_loss is not None else "N/A"
        print(
            f"  [Ep {ep:02d}/{total:02d}] "
            f"Reward: {m.total_reward:+6.1f} | "
            f"Delivery: {m.delivery_rate * 100:5.1f}% | "
            f"CC Del: {m.cc_delivery_rate * 100:5.1f}% | "
            f"Loss: {loss_str} | "
            f"Eps: {m.epsilon:.3f} | "
            f"Overhead: {m.total_overhead_bytes} B"
        )

    history = trainer.train(
        num_episodes=args.episodes,
        progress_callback=on_progress,
        save_checkpoints=True,
    )
    train_dur = time.time() - start_time
    print(f"\nTraining completed in {train_dur:.2f}s.")

    # Export training metrics CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "dqn_training_metrics.csv"
    trainer.export_metrics_csv(train_csv)
    print(f"Exported training metrics: {train_csv}")

    # 3. Comparative Evaluation Against Phase 5 Baselines
    print("\n" + "=" * 80)
    print(">>> Running Comparative Benchmark: Adaptive DQN vs. 5 Fixed Policies")
    print(f"Workload: {args.eval_samples} messages across CC (Emergency), HBS, and BET")
    print("Channel: Controlled Simulation with identical loss and latency conditions")
    print("=" * 80)

    workload_gen = WorkloadGenerator(random_seed=args.seed + 999)
    test_workload = workload_gen.generate_workload(count=args.eval_samples)

    evaluator = DQNEvaluator(agent=agent)
    net_cfg = NetworkModelConfig(
        latency_base_ms=20.0,
        packet_loss_rate=0.12,
        ack_loss_rate=0.05,
        random_seed=args.seed + 999,
    )

    report = evaluator.run_comparative_benchmark(
        workload=test_workload,
        network_config=net_cfg,
    )

    summary_df = report.to_summary_df()
    print("\nBenchmark Summary Table:")
    print(
        summary_df[
            [
                "Policy",
                "Delivery Confirmed Rate (%)",
                "CC Delivery Rate (%)",
                "Mean E2E Latency (ms)",
                "Deadline Compliance (%)",
                "Avg Overhead / Msg (B)",
                "Retransmissions",
            ]
        ].to_string(index=False)
    )

    print("\nAdaptive DQN Action Distribution by Traffic Class:")
    for tc, dist in report.dqn_action_distribution_by_class.items():
        print(f"  [{tc}]: {dist}")

    eval_csv = out_dir / "dqn_comparative_benchmark.csv"
    summary_df.to_csv(str(eval_csv), index=False)
    print(f"\nExported comparative benchmark CSV: {eval_csv}")
    print("\nPhase 6 DQN verification demonstration complete.")


if __name__ == "__main__":
    main()
