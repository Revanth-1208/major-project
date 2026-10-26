"""Automated test suite for Phase 5: Fixed ACK Policy Engine and Comparative Evaluation.

Covers:
1. Interface conformance and fixed action mapping for all 5 policies.
2. Policy A (Static QoS 0) fire-and-forget behavior without retransmissions.
3. Policy B (Static Local ACK) edge proxy acknowledgment and separation from E2E delivery.
4. Policy C (Static QoS 1) broker PUBACK tracking, timeout-driven bounded retries, and deduplication.
5. Policy D (Local ACK + Selective Retransmission) traffic-class prioritization (CC vs HBS/BET).
6. Policy E (Static QoS 2) 4-way protocol handshake modeling and exactly-once semantics.
7. SimulatedNetworkModel deterministic reproducibility, loss, and jitter behavior.
8. WorkloadGenerator traffic-class proportions and message validation.
9. PolicyExperimentRunner controlled comparative execution and metric aggregation.
10. Latency metrics handling (no zero substitution for missing confirmations).
11. Summary metrics and granular audit trail CSV export.
12. Live Mosquitto integration with policies (conditionally executed if broker listening).
"""

import json
from pathlib import Path
import socket
import time
import pytest

from rl_ack.common.enums import (
    AckAction,
    AlertSeverity,
    DeliveryStatus,
    EvaluationPolicy,
    TrafficClass,
)
from rl_ack.common.interfaces import IAckPolicy
from rl_ack.common.models import AlertMessage
from rl_ack.policies.base import (
    BaseAckPolicy,
    PolicyConfig,
    PolicyEvaluationResult,
)
from rl_ack.policies.local_ack_retry import LocalAckSelectiveRetryPolicy
from rl_ack.policies.network_model import (
    NetworkModelConfig,
    SimulatedNetworkModel,
)
from rl_ack.policies.runner import (
    PolicyExperimentRunner,
    PolicySummaryMetrics,
)
from rl_ack.policies.static_local_ack import StaticLocalAckPolicy
from rl_ack.policies.static_qos0 import StaticQos0Policy
from rl_ack.policies.static_qos1 import StaticQos1RetryPolicy
from rl_ack.policies.static_qos2 import StaticQos2Policy
from rl_ack.policies.workload import WorkloadGenerator


def _create_test_message(
    alert_id: str = "TEST-001",
    sensor_id: str = "GAS-001",
    traffic_class: TrafficClass = TrafficClass.CRITICAL_CONTROL,
    severity: AlertSeverity = AlertSeverity.CRITICAL,
    val: float = 450.0,
) -> AlertMessage:
    """Helper to construct sample AlertMessage."""
    return AlertMessage(
        alert_id=alert_id,
        sensor_id=sensor_id,
        severity=severity,
        reading_value=val,
        threshold_value=400.0,
        unit="ppm",
        incident_type="gas_leak",
        message="Critical Gas Alert Test",
        simulation_time_s=10.0,
        priority=traffic_class,
    )


# -----------------------------------------------------------------------------
# 1. Interface Conformance and Action Mapping
# -----------------------------------------------------------------------------

def test_five_policies_implement_iack_policy():
    """Verify all five policies implement the IAckPolicy interface and action space."""
    policies = [
        (StaticQos0Policy(), EvaluationPolicy.STATIC_MQTT_QOS_0, AckAction.MQTT_QOS_0),
        (StaticLocalAckPolicy(), EvaluationPolicy.STATIC_LOCAL_ACK, AckAction.LOCAL_ACK),
        (StaticQos1RetryPolicy(), EvaluationPolicy.STATIC_MQTT_QOS_1, AckAction.MQTT_QOS_1),
        (LocalAckSelectiveRetryPolicy(), EvaluationPolicy.LOCAL_ACK_SELECTIVE_RETRANSMIT, AckAction.LOCAL_ACK_SELECTIVE_RETRANSMIT),
        (StaticQos2Policy(), EvaluationPolicy.STATIC_MQTT_QOS_2, AckAction.MQTT_QOS_2),
    ]

    for pol, expected_type, expected_action in policies:
        assert isinstance(pol, IAckPolicy)
        assert isinstance(pol, BaseAckPolicy)
        assert pol.policy_type == expected_type
        assert pol.corresponding_action == expected_action
        assert pol.select_action(None) == expected_action  # Fixed policy action mapping


# -----------------------------------------------------------------------------
# 2. Policy A: Static QoS 0 Fire-and-Forget
# -----------------------------------------------------------------------------

def test_policy_a_qos0_success_and_packet_loss():
    """Verify Policy A transmits once, does not retry, and records loss as failure."""
    pol = StaticQos0Policy()
    msg = _create_test_message("ALT-QOS0-01")

    # Case 1: Channel delivers packet
    net_ok = SimulatedNetworkModel(NetworkModelConfig(packet_loss_rate=0.0, ack_loss_rate=0.0))
    res_ok = pol.evaluate_controlled(msg, net_ok)
    assert res_ok.publication_attempts == 1
    assert res_ok.retry_count == 0
    assert res_ok.broker_acknowledged is False  # QoS 0 has no PUBACK
    assert res_ok.local_acknowledged is False
    assert res_ok.subscriber_received is True
    assert res_ok.delivery_confirmed is True
    assert res_ok.delivery_status == DeliveryStatus.DELIVERED
    assert res_ok.end_to_end_latency_ms is not None

    # Case 2: Channel drops packet (100% loss)
    net_loss = SimulatedNetworkModel(NetworkModelConfig(packet_loss_rate=1.0))
    res_loss = pol.evaluate_controlled(msg, net_loss)
    assert res_loss.publication_attempts == 1
    assert res_loss.retry_count == 0  # No retransmission
    assert res_loss.subscriber_received is False
    assert res_loss.delivery_confirmed is False
    assert res_loss.delivery_status == DeliveryStatus.FAILED
    assert res_loss.end_to_end_latency_ms is None  # Unknown latency is None, not 0.0
    assert "Packet lost" in (res_loss.error_message or "")


# -----------------------------------------------------------------------------
# 3. Policy B: Static Local ACK vs End-to-End Delivery
# -----------------------------------------------------------------------------

def test_policy_b_local_ack_distinct_from_e2e_confirmation():
    """Verify Local ACK acknowledges proximate proxy but distinguishes downstream loss."""
    pol = StaticLocalAckPolicy(PolicyConfig(local_ack_delay_ms=2.5))
    msg = _create_test_message("ALT-LACK-01")

    # Case 1: Downstream packet lost, but Local Proxy ACK succeeds
    net_drop_downstream = SimulatedNetworkModel(NetworkModelConfig(packet_loss_rate=1.0))
    res = pol.evaluate_controlled(msg, net_drop_downstream)

    assert res.local_acknowledged is True
    assert res.local_ack_latency_ms is not None
    assert res.local_ack_latency_ms < 10.0  # Low local latency
    assert res.subscriber_received is False
    assert res.delivery_confirmed is False  # CRITICAL: Local ACK is NOT E2E delivery!
    assert res.delivery_status == DeliveryStatus.ACKNOWLEDGED  # Local ACK state
    assert res.end_to_end_latency_ms is None

    # Case 2: Full round-trip downstream success
    net_ok = SimulatedNetworkModel(NetworkModelConfig(packet_loss_rate=0.0, ack_loss_rate=0.0))
    res_ok = pol.evaluate_controlled(msg, net_ok)
    assert res_ok.local_acknowledged is True
    assert res_ok.subscriber_received is True
    assert res_ok.delivery_confirmed is True
    assert res_ok.delivery_status == DeliveryStatus.DELIVERY_CONFIRMED
    assert res_ok.end_to_end_latency_ms is not None
    assert res_ok.end_to_end_latency_ms > res_ok.local_ack_latency_ms


# -----------------------------------------------------------------------------
# 4. Policy C: Static QoS 1 with Bounded Retries
# -----------------------------------------------------------------------------

def test_policy_c_qos1_bounded_retries_and_attempt_ids():
    """Verify Policy C triggers bounded retries on missing receipts and updates attempt IDs."""
    cfg = PolicyConfig(max_retries=3, ack_timeout_ms=100.0)
    pol = StaticQos1RetryPolicy(config=cfg)
    msg = _create_test_message("ALT-QOS1-01")

    # Case 1: 100% loss exhausts all retries
    net_loss = SimulatedNetworkModel(NetworkModelConfig(packet_loss_rate=1.0))
    res_loss = pol.evaluate_controlled(msg, net_loss)

    assert res_loss.publication_attempts == 4  # 1 initial + 3 retries
    assert res_loss.retry_count == 3
    assert res_loss.broker_acknowledged is True  # Protocol PUBACK tracked
    assert res_loss.subscriber_received is False
    assert res_loss.delivery_confirmed is False
    assert res_loss.delivery_status == DeliveryStatus.TIMEOUT
    assert res_loss.attempt_id == "ALT-QOS1-01_att4"

    # Case 2: 0% loss delivers on attempt 1
    net_ok = SimulatedNetworkModel(NetworkModelConfig(packet_loss_rate=0.0, ack_loss_rate=0.0))
    res_ok = pol.evaluate_controlled(msg, net_ok)
    assert res_ok.publication_attempts == 1
    assert res_ok.retry_count == 0
    assert res_ok.delivery_confirmed is True
    assert res_ok.delivery_status == DeliveryStatus.DELIVERY_CONFIRMED
    assert res_ok.attempt_id == "ALT-QOS1-01_att1"


# -----------------------------------------------------------------------------
# 5. Policy D: Local ACK + Selective Retransmission (Traffic Class Prioritization)
# -----------------------------------------------------------------------------

def test_policy_d_selective_retransmission_by_traffic_class():
    """Verify Policy D retries Critical Control alerts but suppresses retries for BET."""
    cfg = PolicyConfig(max_retries=3, ack_timeout_ms=120.0)
    pol = LocalAckSelectiveRetryPolicy(config=cfg)

    # Message 1: Critical Control (Emergency Gas)
    msg_cc = _create_test_message("ALT-D-CC", traffic_class=TrafficClass.CRITICAL_CONTROL)
    # Message 2: Best Effort Traffic (Routine Humidity)
    msg_bet = _create_test_message("ALT-D-BET", sensor_id="HUM-001", traffic_class=TrafficClass.BEST_EFFORT)

    # Channel with 100% packet loss
    net_loss = SimulatedNetworkModel(NetworkModelConfig(packet_loss_rate=1.0))

    # Critical Control must undergo active retransmissions
    res_cc = pol.evaluate_controlled(msg_cc, net_loss)
    assert res_cc.local_acknowledged is True
    assert res_cc.publication_attempts == 4  # Retries executed
    assert res_cc.retry_count == 3
    assert res_cc.delivery_confirmed is False

    # Best Effort Traffic must suppress retransmissions to save energy/bandwidth
    res_bet = pol.evaluate_controlled(msg_bet, net_loss)
    assert res_bet.local_acknowledged is True
    assert res_bet.publication_attempts == 1  # Selective retransmission suppressed!
    assert res_bet.retry_count == 0
    assert res_bet.delivery_confirmed is False
    assert "selective retransmission suppressed" in (res_bet.error_message or "").lower()


# -----------------------------------------------------------------------------
# 6. Policy E: Static QoS 2 Four-Way Handshake
# -----------------------------------------------------------------------------

def test_policy_e_qos2_protocol_handshake_and_overhead():
    """Verify Policy E models the 4-way protocol handshake and tracks higher overhead."""
    pol = StaticQos2Policy()
    msg = _create_test_message("ALT-QOS2-01")

    net_ok = SimulatedNetworkModel(NetworkModelConfig(packet_loss_rate=0.0, ack_loss_rate=0.0))
    res_ok = pol.evaluate_controlled(msg, net_ok)

    assert res_ok.configured_qos == 2
    assert res_ok.broker_acknowledged is True
    assert res_ok.subscriber_received is True
    assert res_ok.delivery_confirmed is True
    assert res_ok.delivery_status == DeliveryStatus.DELIVERY_CONFIRMED
    # QoS 2 overhead includes PUBLISH + PUBREC + PUBREL + PUBCOMP
    assert res_ok.message_overhead_bytes > len(msg.to_json()) + 35


# -----------------------------------------------------------------------------
# 7. Deterministic Network Model Reproducibility
# -----------------------------------------------------------------------------

def test_network_model_deterministic_seed_reproducibility():
    """Verify that identical random seeds produce identical channel draw sequences."""
    net1 = SimulatedNetworkModel(NetworkModelConfig(random_seed=123, packet_loss_rate=0.15))
    net2 = SimulatedNetworkModel(NetworkModelConfig(random_seed=123, packet_loss_rate=0.15))
    net3 = SimulatedNetworkModel(NetworkModelConfig(random_seed=999, packet_loss_rate=0.15))

    seq1 = [net1.transmit_packet() for _ in range(20)]
    seq2 = [net2.transmit_packet() for _ in range(20)]
    seq3 = [net3.transmit_packet() for _ in range(20)]

    assert seq1 == seq2, "Identical seeds must produce identical sequences"
    assert seq1 != seq3, "Different seeds must produce different sequences"


# -----------------------------------------------------------------------------
# 8. Workload Generator Validation
# -----------------------------------------------------------------------------

def test_workload_generator_traffic_class_proportions():
    """Verify WorkloadGenerator creates correct mixture of CC, HBS, and BET messages."""
    gen = WorkloadGenerator(random_seed=42)
    workload = gen.generate_workload(count=50, cc_ratio=0.30, hbs_ratio=0.40, bet_ratio=0.30)

    assert len(workload) == 50
    cc_msgs = [m for m in workload if m.priority == TrafficClass.CRITICAL_CONTROL]
    hbs_msgs = [m for m in workload if m.priority == TrafficClass.HIGH_BANDWIDTH_STREAMING]
    bet_msgs = [m for m in workload if m.priority == TrafficClass.BEST_EFFORT]

    assert len(cc_msgs) > 0
    assert len(hbs_msgs) > 0
    assert len(bet_msgs) > 0
    assert len(cc_msgs) + len(hbs_msgs) + len(bet_msgs) == 50

    # Verify critical alerts have valid thresholds and alarms
    for cc in cc_msgs:
        assert cc.severity == AlertSeverity.CRITICAL
        assert cc.reading_value >= cc.threshold_value


# -----------------------------------------------------------------------------
# 9. Experiment Runner Controlled Comparison
# -----------------------------------------------------------------------------

def test_policy_experiment_runner_controlled_comparison():
    """Verify runner evaluates all five policies on identical workload and aggregates metrics."""
    runner = PolicyExperimentRunner()
    gen = WorkloadGenerator(random_seed=100)
    workload = gen.generate_workload(count=15)

    net_cfg = NetworkModelConfig(packet_loss_rate=0.10, random_seed=42)
    summary = runner.run_controlled_comparison(workload=workload, network_config=net_cfg)

    # Must contain all five policies
    assert len(summary) == 5
    expected_policies = [
        EvaluationPolicy.STATIC_MQTT_QOS_0.value,
        EvaluationPolicy.STATIC_LOCAL_ACK.value,
        EvaluationPolicy.STATIC_MQTT_QOS_1.value,
        EvaluationPolicy.LOCAL_ACK_SELECTIVE_RETRANSMIT.value,
        EvaluationPolicy.STATIC_MQTT_QOS_2.value,
    ]
    for p_name in expected_policies:
        assert p_name in summary
        metric = summary[p_name]
        assert metric.total_messages == 15
        assert 0.0 <= metric.delivery_confirmation_rate <= 1.0
        assert metric.total_overhead_bytes > 0


# -----------------------------------------------------------------------------
# 10. Metric Aggregation Handling (No Fake Zeroes for Missing Latencies)
# -----------------------------------------------------------------------------

def test_metric_aggregation_never_replaces_unknown_latency_with_zero():
    """Verify unconfirmed messages leave mean/median latency as None, not 0.0."""
    runner = PolicyExperimentRunner()
    workload = [_create_test_message(f"ALT-{i}") for i in range(5)]

    # 100% loss channel ensures 0 confirmations under Policy A
    net_cfg = NetworkModelConfig(packet_loss_rate=1.0, random_seed=42)
    summary = runner.run_controlled_comparison(workload=workload, network_config=net_cfg)

    metric_qos0 = summary[EvaluationPolicy.STATIC_MQTT_QOS_0.value]
    assert metric_qos0.delivery_confirmation_rate == 0.0
    assert metric_qos0.mean_e2e_latency_ms is None  # Must NOT be 0.0!
    assert metric_qos0.median_e2e_latency_ms is None
    assert metric_qos0.p95_e2e_latency_ms is None
    assert metric_qos0.failure_count == 5


# -----------------------------------------------------------------------------
# 11. CSV Export Functions
# -----------------------------------------------------------------------------

def test_csv_export_summary_and_results(tmp_path: Path):
    """Verify summary and granular results export valid, non-empty CSV files."""
    runner = PolicyExperimentRunner()
    gen = WorkloadGenerator(random_seed=77)
    workload = gen.generate_workload(count=10)

    runner.run_controlled_comparison(workload=workload)

    # 1. Summary CSV
    summary_path = tmp_path / "summary.csv"
    csv_str = runner.export_summary_csv(file_path=str(summary_path))
    assert "Policy" in csv_str
    assert "Delivery Confirmed Rate" in csv_str
    assert summary_path.exists()
    assert summary_path.stat().st_size > 0

    # 2. Results CSV
    results_path = tmp_path / "results.csv"
    res_str = runner.export_results_csv(file_path=str(results_path))
    assert "alert_id" in res_str
    assert "attempt_id" in res_str
    assert results_path.exists()
    assert results_path.stat().st_size > 0


# -----------------------------------------------------------------------------
# 12. Live Mosquitto Integration
# -----------------------------------------------------------------------------

def _is_local_mosquitto_listening(host="127.0.0.1", port=1883) -> bool:
    """Check if Mosquitto broker is accepting TCP connections."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        s.connect((host, port))
        s.close()
        return True
    except (socket.error, ConnectionRefusedError, OSError):
        return False


@pytest.mark.skipif(
    not _is_local_mosquitto_listening(),
    reason="Local Mosquitto broker (127.0.0.1:1883) is not currently running",
)
def test_live_mosquitto_policy_evaluation():
    """Live integration test evaluating policies against running Eclipse Mosquitto broker."""
    from rl_ack.mqtt.config import MqttConfig
    from rl_ack.mqtt.manager import MqttManager

    config = MqttConfig(
        host="127.0.0.1",
        port=1883,
        alert_topic="rlack/policy/alerts",
        ack_topic="rlack/policy/alerts/ack",
    )
    manager = MqttManager(config=config)
    connected = manager.connect(timeout_s=3.0)
    assert connected is True, "Failed to connect to live Mosquitto broker"

    try:
        runner = PolicyExperimentRunner()
        workload = [
            _create_test_message("ALT-LIVE-POL-01", "GAS-001", TrafficClass.CRITICAL_CONTROL),
            _create_test_message("ALT-LIVE-POL-02", "TEMP-001", TrafficClass.BEST_EFFORT),
        ]

        summary = runner.run_live_comparison(workload=workload, mqtt_manager=manager)
        assert len(summary) == 5
        for p_name, metrics in summary.items():
            assert metrics.total_messages == 2
            assert metrics.evaluation_mode == "live_mqtt"
    finally:
        manager.disconnect()
