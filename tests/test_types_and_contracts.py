"""Tests for domain models, enums, and data contracts."""

import numpy as np
import pytest
from rl_ack.common.enums import (
    AckAction,
    AlertSeverity,
    DeliveryStatus,
    EvaluationPolicy,
    NodeOperationalStatus,
    SensorType,
    TrafficClass,
)
from rl_ack.common.models import (
    AckEvent,
    AlertMessage,
    ExperimentRecord,
    NetworkStateVector,
    SensorReading,
)


def test_four_virtual_sensor_types():
    """Verify all four required virtual sensors exist with proper units."""
    expected = {
        SensorType.GAS: "ppm",
        SensorType.TEMPERATURE: "°C",
        SensorType.HUMIDITY: "% RH",
        SensorType.SMOKE: "% obs/m",
    }
    assert len(SensorType) == 4
    for st, unit in expected.items():
        assert st in SensorType
        assert st.default_unit == unit


def test_six_evaluation_policies():
    """Verify exactly six comparative evaluation policies are defined."""
    assert len(EvaluationPolicy) == 6
    expected_policies = [
        "Static MQTT QoS 0",
        "Static Local ACK",
        "Static MQTT QoS 1",
        "Local ACK plus Selective Retransmission",
        "Static MQTT QoS 2",
        "Adaptive RL-ACK DQN",
    ]
    actual_policies = [p.value for p in EvaluationPolicy]
    for expected in expected_policies:
        assert expected in actual_policies, f"Policy {expected} missing"


def test_five_dqn_actions():
    """Verify strictly five DQN action outputs matching the underlying ACK strategies."""
    assert len(AckAction) == 5
    action_values = [a.value for a in AckAction]
    assert sorted(action_values) == [0, 1, 2, 3, 4]

    expected_labels = [
        "MQTT QoS 0",
        "Local ACK",
        "MQTT QoS 1",
        "Local ACK plus Selective Retransmission",
        "MQTT QoS 2",
    ]
    for action in AckAction:
        assert action.label in expected_labels


def test_traffic_class_one_hot_encoding():
    """Verify traffic class to one-hot encoding converts accurately."""
    assert TrafficClass.CRITICAL_CONTROL.to_one_hot() == (1.0, 0.0, 0.0)
    assert TrafficClass.HIGH_BANDWIDTH_STREAMING.to_one_hot() == (0.0, 1.0, 0.0)
    assert TrafficClass.BEST_EFFORT.to_one_hot() == (0.0, 0.0, 1.0)


def test_network_state_vector_shape_and_values():
    """Verify eight-feature state vector contract."""
    state = NetworkStateVector.from_features(
        network_utilization=0.65,
        rolling_ack_rate=0.92,
        transmit_queue_occupancy=0.40,
        traffic_class=TrafficClass.CRITICAL_CONTROL,
        remaining_node_energy=0.85,
        rolling_latency=35.0,
    )

    arr = state.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (8,), f"State shape must be exactly (8,), got {arr.shape}"
    assert arr.dtype == np.float32

    # Check one-hot feature values
    assert arr[3] == 1.0  # Critical Control
    assert arr[4] == 0.0  # High Bandwidth
    assert arr[5] == 0.0  # Best Effort


def test_delivery_status_lifecycle():
    """Verify distinct lifecycle stages for delivery verification."""
    expected_stages = [
        "generated",
        "published",
        "received",
        "acknowledged",
        "delivered",
        "failed",
        "timeout",
    ]
    for stage in expected_stages:
        assert any(s.value == stage for s in DeliveryStatus)


def test_sensor_reading_serialization():
    """Verify SensorReading dataclass serialization."""
    reading = SensorReading(
        sensor_id="gas_01",
        host_node="node_alpha",
        sensor_type=SensorType.GAS,
        value=350.5,
        unit="ppm",
        timestamp=100.0,
        operational_status=NodeOperationalStatus.NORMAL,
        is_threshold_exceeded=False,
    )
    d = reading.to_dict()
    assert d["sensor_id"] == "gas_01"
    assert d["sensor_type"] == "gas"
    assert d["value"] == 350.5
    assert d["unit"] == "ppm"


def test_alert_message_serialization():
    """Verify AlertMessage dataclass serialization."""
    alert = AlertMessage(
        alert_id="alt-001",
        sensor_id="smoke_01",
        host_node="node_beta",
        sensor_type=SensorType.SMOKE,
        severity=AlertSeverity.CRITICAL,
        traffic_class=TrafficClass.CRITICAL_CONTROL,
        reading_value=4.8,
        threshold_value=3.5,
        timestamp=102.5,
        status=DeliveryStatus.GENERATED,
        chosen_action=AckAction.MQTT_QOS_2,
    )
    d = alert.to_dict()
    assert d["alert_id"] == "alt-001"
    assert d["severity"] == "critical"
    assert d["traffic_class"] == "Critical Control"
    assert d["chosen_action"] == AckAction.MQTT_QOS_2.value


def test_ack_event_serialization():
    """Verify AckEvent lifecycle record."""
    event = AckEvent(
        event_id="evt-001",
        alert_id="alt-001",
        event_type="app_delivery",
        timestamp=102.8,
        latency_ms=30.0,
        success=True,
    )
    d = event.to_dict()
    assert d["event_id"] == "evt-001"
    assert d["latency_ms"] == 30.0
    assert d["success"] is True


def test_experiment_record_serialization():
    """Verify ExperimentRecord summary serialization."""
    rec = ExperimentRecord(
        experiment_id="exp-01",
        simulation_id="sim-01",
        policy_name=EvaluationPolicy.ADAPTIVE_RL_ACK_DQN.value,
        timestamp=1000.0,
        total_alerts=100,
        delivered_alerts=98,
        failed_alerts=2,
        delivery_rate=0.98,
        avg_latency_ms=22.4,
        avg_energy_consumed=1.85,
        total_retransmissions=4,
    )
    d = rec.to_dict()
    assert d["experiment_id"] == "exp-01"
    assert d["delivery_rate"] == 0.98
