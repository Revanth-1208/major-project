"""Reproducible Experiment Runner for Phase 5 ACK Policy Comparative Evaluation.

Coordinates controlled simulations and live MQTT comparative runs across all 5 policies,
guaranteeing identical workload conditions, aggregating performance metrics,
and exporting results to CSV.
"""

from dataclasses import dataclass, field
import io
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from rl_ack.common.enums import (
    EvaluationPolicy,
    TrafficClass,
)
from rl_ack.common.interfaces import IMqttManager
from rl_ack.common.models import AlertMessage
from rl_ack.policies.base import BaseAckPolicy, PolicyConfig, PolicyEvaluationResult
from rl_ack.policies.local_ack_retry import LocalAckSelectiveRetryPolicy
from rl_ack.policies.network_model import NetworkModelConfig, SimulatedNetworkModel
from rl_ack.policies.static_local_ack import StaticLocalAckPolicy
from rl_ack.policies.static_qos0 import StaticQos0Policy
from rl_ack.policies.static_qos1 import StaticQos1RetryPolicy
from rl_ack.policies.static_qos2 import StaticQos2Policy
from rl_ack.policies.workload import WorkloadGenerator


@dataclass
class PolicySummaryMetrics:
    """Aggregated statistical performance metrics for a single ACK policy."""

    policy_name: str
    evaluation_mode: str
    total_messages: int = 0
    publication_attempts: int = 0
    retry_count: int = 0
    broker_ack_rate: float = 0.0
    local_ack_rate: float = 0.0
    subscriber_receipt_rate: float = 0.0
    delivery_confirmation_rate: float = 0.0
    mean_e2e_latency_ms: Optional[float] = None
    median_e2e_latency_ms: Optional[float] = None
    p95_e2e_latency_ms: Optional[float] = None
    deadline_compliance_rate: float = 0.0
    total_overhead_bytes: int = 0
    avg_overhead_bytes: float = 0.0
    failure_count: int = 0
    # Traffic class specific breakdowns
    cc_count: int = 0
    cc_delivery_rate: float = 0.0
    cc_mean_latency_ms: Optional[float] = None
    hbs_count: int = 0
    hbs_delivery_rate: float = 0.0
    bet_count: int = 0
    bet_delivery_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "Policy": self.policy_name,
            "Mode": self.evaluation_mode,
            "Total Messages": self.total_messages,
            "Attempts": self.publication_attempts,
            "Retransmissions": self.retry_count,
            "Broker ACK Rate (%)": round(self.broker_ack_rate * 100.0, 1),
            "Local ACK Rate (%)": round(self.local_ack_rate * 100.0, 1),
            "Subscriber Receipt Rate (%)": round(self.subscriber_receipt_rate * 100.0, 1),
            "Delivery Confirmed Rate (%)": round(self.delivery_confirmation_rate * 100.0, 1),
            "Mean E2E Latency (ms)": round(self.mean_e2e_latency_ms, 2) if self.mean_e2e_latency_ms is not None else None,
            "Median E2E Latency (ms)": round(self.median_e2e_latency_ms, 2) if self.median_e2e_latency_ms is not None else None,
            "P95 E2E Latency (ms)": round(self.p95_e2e_latency_ms, 2) if self.p95_e2e_latency_ms is not None else None,
            "Deadline Compliance (%)": round(self.deadline_compliance_rate * 100.0, 1),
            "Total Overhead (Bytes)": self.total_overhead_bytes,
            "Avg Overhead / Msg (B)": round(self.avg_overhead_bytes, 1),
            "Failures": self.failure_count,
            "CC Total": self.cc_count,
            "CC Delivery Rate (%)": round(self.cc_delivery_rate * 100.0, 1),
            "CC Mean Latency (ms)": round(self.cc_mean_latency_ms, 2) if self.cc_mean_latency_ms is not None else None,
            "HBS Delivery Rate (%)": round(self.hbs_delivery_rate * 100.0, 1),
            "BET Delivery Rate (%)": round(self.bet_delivery_rate * 100.0, 1),
        }


class PolicyExperimentRunner:
    """Orchestrates five-policy comparative evaluations under identical conditions."""

    def __init__(self, policy_config: Optional[PolicyConfig] = None) -> None:
        self.default_policy_config = policy_config or PolicyConfig()
        self.policies: Dict[EvaluationPolicy, BaseAckPolicy] = {
            EvaluationPolicy.STATIC_MQTT_QOS_0: StaticQos0Policy(config=self.default_policy_config),
            EvaluationPolicy.STATIC_LOCAL_ACK: StaticLocalAckPolicy(config=self.default_policy_config),
            EvaluationPolicy.STATIC_MQTT_QOS_1: StaticQos1RetryPolicy(config=self.default_policy_config),
            EvaluationPolicy.LOCAL_ACK_SELECTIVE_RETRANSMIT: LocalAckSelectiveRetryPolicy(config=self.default_policy_config),
            EvaluationPolicy.STATIC_MQTT_QOS_2: StaticQos2Policy(config=self.default_policy_config),
        }

        self.last_results: Dict[str, List[PolicyEvaluationResult]] = {}
        self.last_summary: Dict[str, PolicySummaryMetrics] = {}
        self.last_evaluation_mode: str = "controlled_simulation"

    def run_controlled_comparison(
        self,
        workload: List[AlertMessage],
        network_config: Optional[NetworkModelConfig] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, PolicySummaryMetrics]:
        """Execute controlled evaluation across all 5 policies with identical channel conditions.

        Args:
            workload: Benchmark list of AlertMessage instances.
            network_config: Simulation parameters (latency, jitter, loss).
            progress_callback: Optional callback for UI progress reporting.

        Returns:
            Dict mapping policy name to PolicySummaryMetrics.
        """
        net_cfg = network_config or NetworkModelConfig()
        self.last_evaluation_mode = "controlled_simulation"
        self.last_results.clear()
        self.last_summary.clear()

        total_policies = len(self.policies)

        for p_idx, (policy_type, policy) in enumerate(self.policies.items()):
            policy.reset()
            # Initialize network simulator with fixed seed for identical channel conditions per policy
            sim_network = SimulatedNetworkModel(config=net_cfg)
            sim_network.set_seed(net_cfg.random_seed)

            policy_results: List[PolicyEvaluationResult] = []

            for m_idx, message in enumerate(workload):
                res = policy.evaluate_controlled(message=message, network=sim_network)
                policy_results.append(res)

                if progress_callback:
                    progress = (p_idx + (m_idx + 1) / len(workload)) / total_policies
                    progress_callback(min(1.0, progress), f"Evaluating {policy_type.value} ({m_idx + 1}/{len(workload)})")

            self.last_results[policy_type.value] = policy_results
            self.last_summary[policy_type.value] = self._aggregate_metrics(
                policy_name=policy_type.value,
                mode="controlled_simulation",
                results=policy_results,
            )

        return self.last_summary

    def run_live_comparison(
        self,
        workload: List[AlertMessage],
        mqtt_manager: IMqttManager,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, PolicySummaryMetrics]:
        """Execute comparative evaluation against live Eclipse Mosquitto broker.

        Args:
            workload: Benchmark list of AlertMessage instances.
            mqtt_manager: Live MQTT manager connection.
            progress_callback: Optional callback for UI progress reporting.

        Returns:
            Dict mapping policy name to PolicySummaryMetrics.
        """
        self.last_evaluation_mode = "live_mqtt"
        self.last_results.clear()
        self.last_summary.clear()

        total_policies = len(self.policies)

        for p_idx, (policy_type, policy) in enumerate(self.policies.items()):
            policy.reset()
            policy_results: List[PolicyEvaluationResult] = []

            for m_idx, message in enumerate(workload):
                res = policy.evaluate_live(
                    message=message,
                    mqtt_manager=mqtt_manager,
                    timeout_s=self.default_policy_config.ack_timeout_ms / 1000.0,
                )
                policy_results.append(res)

                if progress_callback:
                    progress = (p_idx + (m_idx + 1) / len(workload)) / total_policies
                    progress_callback(min(1.0, progress), f"Testing {policy_type.value} over Live MQTT ({m_idx + 1}/{len(workload)})")

            self.last_results[policy_type.value] = policy_results
            self.last_summary[policy_type.value] = self._aggregate_metrics(
                policy_name=policy_type.value,
                mode="live_mqtt",
                results=policy_results,
            )

        return self.last_summary

    @staticmethod
    def _aggregate_metrics(
        policy_name: str,
        mode: str,
        results: List[PolicyEvaluationResult],
    ) -> PolicySummaryMetrics:
        """Derive rigorous aggregate statistical metrics from a list of policy results."""
        total = len(results)
        if total == 0:
            return PolicySummaryMetrics(policy_name=policy_name, evaluation_mode=mode)

        attempts = sum(r.publication_attempts for r in results)
        retries = sum(r.retry_count for r in results)
        broker_acked = sum(1 for r in results if r.broker_acknowledged)
        local_acked = sum(1 for r in results if r.local_acknowledged)
        sub_received = sum(1 for r in results if r.subscriber_received)
        confirmed = sum(1 for r in results if r.delivery_confirmed)
        deadlines = sum(1 for r in results if r.deadline_met is True)
        overhead = sum(r.message_overhead_bytes for r in results)
        failures = sum(1 for r in results if not r.delivery_confirmed)

        # Confirmed latencies only (never substitute zero for unknown/unconfirmed latencies)
        confirmed_latencies = [
            r.end_to_end_latency_ms
            for r in results
            if r.delivery_confirmed and r.end_to_end_latency_ms is not None
        ]

        mean_lat: Optional[float] = None
        median_lat: Optional[float] = None
        p95_lat: Optional[float] = None

        if confirmed_latencies:
            mean_lat = float(np.mean(confirmed_latencies))
            median_lat = float(np.median(confirmed_latencies))
            if len(confirmed_latencies) >= 2:
                p95_lat = float(np.percentile(confirmed_latencies, 95))
            else:
                p95_lat = confirmed_latencies[0]

        # Traffic class breakdowns
        cc_results = [r for r in results if r.traffic_class == TrafficClass.CRITICAL_CONTROL]
        hbs_results = [r for r in results if r.traffic_class == TrafficClass.HIGH_BANDWIDTH_STREAMING]
        bet_results = [r for r in results if r.traffic_class == TrafficClass.BEST_EFFORT]

        cc_confirmed = sum(1 for r in cc_results if r.delivery_confirmed)
        cc_lats = [r.end_to_end_latency_ms for r in cc_results if r.delivery_confirmed and r.end_to_end_latency_ms is not None]

        return PolicySummaryMetrics(
            policy_name=policy_name,
            evaluation_mode=mode,
            total_messages=total,
            publication_attempts=attempts,
            retry_count=retries,
            broker_ack_rate=broker_acked / total,
            local_ack_rate=local_acked / total,
            subscriber_receipt_rate=sub_received / total,
            delivery_confirmation_rate=confirmed / total,
            mean_e2e_latency_ms=mean_lat,
            median_e2e_latency_ms=median_lat,
            p95_e2e_latency_ms=p95_lat,
            deadline_compliance_rate=deadlines / total,
            total_overhead_bytes=overhead,
            avg_overhead_bytes=overhead / total,
            failure_count=failures,
            cc_count=len(cc_results),
            cc_delivery_rate=cc_confirmed / len(cc_results) if cc_results else 0.0,
            cc_mean_latency_ms=float(np.mean(cc_lats)) if cc_lats else None,
            hbs_count=len(hbs_results),
            hbs_delivery_rate=(sum(1 for r in hbs_results if r.delivery_confirmed) / len(hbs_results)) if hbs_results else 0.0,
            bet_count=len(bet_results),
            bet_delivery_rate=(sum(1 for r in bet_results if r.delivery_confirmed) / len(bet_results)) if bet_results else 0.0,
        )

    def to_summary_dataframe(self) -> pd.DataFrame:
        """Format aggregate metrics as a pandas DataFrame for UI table presentation."""
        rows = [metric.to_dict() for metric in self.last_summary.values()]
        return pd.DataFrame(rows)

    def to_results_dataframe(self) -> pd.DataFrame:
        """Format individual message evaluation audit records as a pandas DataFrame."""
        rows = []
        for pol_results in self.last_results.values():
            for r in pol_results:
                rows.append(r.to_dict())
        return pd.DataFrame(rows)

    def export_summary_csv(self, file_path: Optional[str] = None) -> str:
        """Export aggregate summary metrics table to CSV string or file."""
        df = self.to_summary_dataframe()
        csv_str = df.to_csv(index=False)
        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(csv_str)
        return csv_str

    def export_results_csv(self, file_path: Optional[str] = None) -> str:
        """Export granular per-message audit records to CSV string or file."""
        df = self.to_results_dataframe()
        csv_str = df.to_csv(index=False)
        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(csv_str)
        return csv_str
