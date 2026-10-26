"""Comparative Evaluation Engine for the RL-ACK DQN Agent.

Evaluates trained DQN agents strictly greedily (without epsilon exploration) and benchmarks
them side-by-side against the five Phase 5 fixed policies under identical channel conditions.
"""

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from rl_ack.common.enums import AckAction, EvaluationPolicy, TrafficClass
from rl_ack.common.models import AlertMessage, NetworkStateVector
from rl_ack.dqn.actions import ACTION_DIM, get_action_metadata, instantiate_policy_for_action
from rl_ack.dqn.agent import DQNAgent
from rl_ack.dqn.state import NetworkStateTracker, validate_and_normalize_state
from rl_ack.policies.base import BaseAckPolicy, PolicyConfig, PolicyEvaluationResult
from rl_ack.policies.network_model import NetworkModelConfig, SimulatedNetworkModel
from rl_ack.policies.runner import PolicyExperimentRunner, PolicySummaryMetrics
from rl_ack.policies.workload import WorkloadGenerator


@dataclass
class DQNEvaluationReport:
    """Consolidated comparative report contrasting DQN against the 5 fixed baselines."""

    summary_metrics: Dict[str, PolicySummaryMetrics]
    all_results: Dict[str, List[PolicyEvaluationResult]]
    dqn_action_distribution_by_class: Dict[str, Dict[str, int]]
    total_messages: int
    random_seed: int

    def to_summary_df(self) -> pd.DataFrame:
        """Convert policy summaries to comparative DataFrame."""
        rows = [metric.to_dict() for metric in self.summary_metrics.values()]
        return pd.DataFrame(rows)


class DQNEvaluator:
    """Evaluates trained DQN policies against reproducible smart kitchen IoT workloads."""

    def __init__(
        self,
        agent: Optional[DQNAgent] = None,
        policy_config: Optional[PolicyConfig] = None,
    ) -> None:
        self.agent = agent or DQNAgent()
        self.policy_config = policy_config or PolicyConfig()
        self.runner = PolicyExperimentRunner(policy_config=self.policy_config)

    def evaluate_dqn_policy(
        self,
        workload: List[AlertMessage],
        network_config: Optional[NetworkModelConfig] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[List[PolicyEvaluationResult], Dict[str, Dict[str, int]]]:
        """Evaluate the DQN agent strictly greedily (explore=False) across a benchmark workload.

        Args:
            workload: Benchmark list of AlertMessage instances.
            network_config: Channel simulation configuration.
            progress_callback: Optional progress indicator callback.

        Returns:
            Tuple of (dqn_evaluation_results, action_distribution_by_traffic_class).
        """
        net_cfg = network_config or NetworkModelConfig()
        sim_network = SimulatedNetworkModel(config=net_cfg)
        sim_network.set_seed(net_cfg.random_seed)

        # Separate policies for execution
        policies: Dict[int, BaseAckPolicy] = {
            a_idx: instantiate_policy_for_action(a_idx, config=self.policy_config)
            for a_idx in range(ACTION_DIM)
        }
        for pol in policies.values():
            pol.reset()

        state_tracker = NetworkStateTracker()
        dqn_results: List[PolicyEvaluationResult] = []
        action_dist_by_class: Dict[str, Dict[str, int]] = {
            tc.value: {get_action_metadata(a).label: 0 for a in range(ACTION_DIM)}
            for tc in TrafficClass
        }

        for m_idx, message in enumerate(workload):
            # 1. Construct current 8-feature state
            state_tracker.enqueue_message(message.priority)
            state_vec = state_tracker.build_state(message.priority)

            # 2. Greedy action selection (no exploration)
            chosen_action = self.agent.select_action(state_vec, explore=False)
            act_idx = int(chosen_action)
            action_label = get_action_metadata(act_idx).label
            action_dist_by_class[message.priority.value][action_label] += 1

            # 3. Execute chosen policy under identical channel simulation
            policy = policies[act_idx]
            result = policy.evaluate_controlled(message=message, network=sim_network)

            # Override policy name to reflect adaptive DQN selection
            dqn_result = PolicyEvaluationResult(
                alert_id=result.alert_id,
                attempt_id=result.attempt_id,
                policy_name=EvaluationPolicy.ADAPTIVE_RL_ACK_DQN.value,
                evaluation_mode="controlled_simulation",
                sensor_id=result.sensor_id,
                traffic_class=result.traffic_class,
                severity=result.severity,
                configured_qos=result.configured_qos,
                publication_attempts=result.publication_attempts,
                retry_count=result.retry_count,
                broker_acknowledged=result.broker_acknowledged,
                local_acknowledged=result.local_acknowledged,
                subscriber_received=result.subscriber_received,
                delivery_confirmed=result.delivery_confirmed,
                delivery_status=result.delivery_status,
                publish_latency_ms=result.publish_latency_ms,
                local_ack_latency_ms=result.local_ack_latency_ms,
                subscriber_receipt_latency_ms=result.subscriber_receipt_latency_ms,
                end_to_end_latency_ms=result.end_to_end_latency_ms,
                message_overhead_bytes=result.message_overhead_bytes,
                deadline_met=result.deadline_met,
                error_message=result.error_message,
                simulated=True,
                simulation_timestamp=result.simulation_timestamp,
            )
            dqn_results.append(dqn_result)

            # 4. Update rolling history
            state_tracker.record_outcome(
                confirmed=result.delivery_confirmed,
                latency_ms=result.end_to_end_latency_ms,
                overhead_bytes=result.message_overhead_bytes,
            )

            if progress_callback:
                progress = (m_idx + 1) / len(workload)
                progress_callback(progress, f"Evaluating Adaptive RL-ACK DQN ({m_idx + 1}/{len(workload)})")

        return dqn_results, action_dist_by_class

    def run_comparative_benchmark(
        self,
        workload: List[AlertMessage],
        network_config: Optional[NetworkModelConfig] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> DQNEvaluationReport:
        """Run all five Phase 5 fixed policies plus Adaptive RL-ACK DQN on identical workloads.

        Args:
            workload: Benchmark list of messages.
            network_config: Network conditions.
            progress_callback: UI progress reporter.

        Returns:
            DQNEvaluationReport containing comprehensive comparative tables.
        """
        net_cfg = network_config or NetworkModelConfig()

        # Step A: Run 5 fixed baseline policies
        baseline_summaries = self.runner.run_controlled_comparison(
            workload=workload,
            network_config=net_cfg,
            progress_callback=lambda p, desc: progress_callback(p * 0.8, desc) if progress_callback else None,
        )

        # Step B: Run Adaptive RL-ACK DQN
        dqn_results, dqn_action_dist = self.evaluate_dqn_policy(
            workload=workload,
            network_config=net_cfg,
            progress_callback=lambda p, desc: progress_callback(0.8 + p * 0.2, desc) if progress_callback else None,
        )

        # Aggregate DQN metrics
        dqn_summary = PolicyExperimentRunner._aggregate_metrics(
            policy_name=EvaluationPolicy.ADAPTIVE_RL_ACK_DQN.value,
            mode="controlled_simulation",
            results=dqn_results,
        )

        all_summaries = dict(baseline_summaries)
        all_summaries[EvaluationPolicy.ADAPTIVE_RL_ACK_DQN.value] = dqn_summary

        all_results = dict(self.runner.last_results)
        all_results[EvaluationPolicy.ADAPTIVE_RL_ACK_DQN.value] = dqn_results

        return DQNEvaluationReport(
            summary_metrics=all_summaries,
            all_results=all_results,
            dqn_action_distribution_by_class=dqn_action_dist,
            total_messages=len(workload),
            random_seed=net_cfg.random_seed,
        )
