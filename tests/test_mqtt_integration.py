"""Automated test suite for Phase 4: MQTT Alert Transmission and Delivery Verification.

Covers:
1. MQTT configuration validation and environment overrides.
2. Alert JSON serialization, schema validation, and error handling.
3. Publisher connection management and lifecycle states.
4. Broker-level publication confirmation (PUBACK / mid mapping).
5. Subscriber message ingestion and field reconstruction.
6. Application-level receipt acknowledgement generation.
7. End-to-end receipt matching by alert ID.
8. Duplicate alert handling and deduplication metrics.
9. Formal delivery state progression (GENERATED -> PENDING -> PUBLISHED -> RECEIVED -> CONFIRMED).
10. Delivery timeout detection and state transitions.
11. Broker disconnection and publish failure handling.
12. Malformed JSON handling without subscriber crashing.
13. Latency calculations (publication, subscriber receipt, E2E round trip).
14. Delivery metrics derivation and DataFrame formatting.
15. Integration with Phase 3 IncidentManager and virtual sensors.
16. Live Mosquitto broker integration test (conditionally executed if broker is listening).
"""

from dataclasses import dataclass
import json
import socket
import time
from unittest.mock import MagicMock, patch
import pytest

from rl_ack.common.enums import (
    AlertSeverity,
    DeliveryStatus,
    NodeOperationalStatus,
    SensorType,
    TrafficClass,
)
from rl_ack.common.exceptions import ConfigurationError, MqttPayloadError
from rl_ack.common.models import AlertMessage, SensorReading
from rl_ack.monitoring.detector import IncidentDetector
from rl_ack.monitoring.manager import IncidentManager
from rl_ack.mqtt.config import MqttConfig
from rl_ack.mqtt.delivery_tracker import DeliveryRecord, DeliveryTracker
from rl_ack.mqtt.manager import MqttManager
from rl_ack.mqtt.publisher import MqttAlertPublisher
from rl_ack.mqtt.subscriber import MqttAlertSubscriber


@dataclass
class MockMqttMessage:
    """Mock MQTT message container mimicking paho.mqtt.client.MQTTMessage."""
    topic: str
    payload: bytes
    qos: int = 1
    mid: int = 1


def _create_sample_alert(
    alert_id: str = "ALT-0001",
    sensor_id: str = "GAS-001",
    severity: AlertSeverity = AlertSeverity.CRITICAL,
    val: float = 480.0,
) -> AlertMessage:
    """Helper to construct a valid AlertMessage for tests."""
    return AlertMessage(
        alert_id=alert_id,
        sensor_id=sensor_id,
        severity=severity,
        reading_value=val,
        threshold_value=400.0,
        unit="ppm",
        incident_type="gas_leak",
        message="CRITICAL Gas leak emergency",
        simulation_time_s=10.0,
        priority=TrafficClass.CRITICAL_CONTROL,
    )


# -----------------------------------------------------------------------------
# 1. Configuration Validation & Environment Overrides
# -----------------------------------------------------------------------------

def test_mqtt_config_defaults_and_validation():
    """Verify default configuration parameters and constraint validation."""
    cfg = MqttConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 1883
    assert cfg.default_alert_qos == 1
    assert "alerts" in cfg.alert_topic
    assert "ack" in cfg.ack_topic

    # Invalid host
    with pytest.raises(ConfigurationError):
        MqttConfig(host="").validate()

    # Invalid port
    with pytest.raises(ConfigurationError):
        MqttConfig(port=99999).validate()

    # Invalid QoS
    with pytest.raises(ConfigurationError):
        MqttConfig(default_alert_qos=3).validate()

    # Same alert and ack topic
    with pytest.raises(ConfigurationError):
        MqttConfig(alert_topic="same/topic", ack_topic="same/topic").validate()


def test_mqtt_config_env_overrides(monkeypatch):
    """Verify configuration loads environment variables when provided."""
    monkeypatch.setenv("RLACK_MQTT_HOST", "192.168.1.100")
    monkeypatch.setenv("RLACK_MQTT_PORT", "1884")
    monkeypatch.setenv("RLACK_MQTT_USERNAME", "operator1")
    monkeypatch.setenv("RLACK_MQTT_PASSWORD", "secret")

    cfg = MqttConfig()
    assert cfg.host == "192.168.1.100"
    assert cfg.port == 1884
    assert cfg.username == "operator1"
    assert cfg.password == "secret"


# -----------------------------------------------------------------------------
# 2. JSON Serialization & Payload Validation
# -----------------------------------------------------------------------------

def test_alert_json_serialization_and_reconstruction():
    """Verify alert serializes to JSON and is reconstructed identically by subscriber."""
    alert = _create_sample_alert("ALT-0005", "TEMP-001", AlertSeverity.CRITICAL, 65.5)
    payload_str = alert.to_json()

    # Parse and validate reconstructed object
    reconstructed = AlertMessage.from_json(payload_str)
    assert reconstructed.alert_id == "ALT-0005"
    assert reconstructed.sensor_id == "TEMP-001"
    assert reconstructed.severity == AlertSeverity.CRITICAL
    assert reconstructed.reading_value == 65.5
    assert reconstructed.priority == TrafficClass.CRITICAL_CONTROL


def test_alert_validation_rejects_malformed_payloads():
    """Verify AlertMessage.from_json and from_dict reject missing or invalid fields."""
    # Malformed JSON string
    with pytest.raises(MqttPayloadError):
        AlertMessage.from_json("invalid-json{not:closed")

    # Missing alert_id
    with pytest.raises(MqttPayloadError):
        AlertMessage.from_dict({"sensor_id": "GAS-001", "value": 100.0})

    # Missing sensor_id
    with pytest.raises(MqttPayloadError):
        AlertMessage.from_dict({"alert_id": "ALT-0001", "value": 100.0})


# -----------------------------------------------------------------------------
# 3-4. Publisher Connection and Broker ACK Callback
# -----------------------------------------------------------------------------

def test_publisher_connection_and_disconnect():
    """Verify publisher connects, sets is_connected, and disconnects cleanly."""
    tracker = DeliveryTracker()
    publisher = MqttAlertPublisher(tracker=tracker)

    with patch.object(publisher.client, "connect_async") as mock_conn, \
         patch.object(publisher.client, "loop_start") as mock_start, \
         patch.object(publisher.client, "loop_stop") as mock_stop, \
         patch.object(publisher.client, "disconnect") as mock_disc:

        # Simulate on_connect callback firing
        def fake_connect(*args, **kwargs):
            publisher._on_connect(publisher.client, None, None, 0)

        mock_conn.side_effect = fake_connect
        connected = publisher.connect(timeout_s=0.5)

        assert connected is True
        assert publisher.is_connected is True
        mock_start.assert_called_once()

        publisher.disconnect()
        assert publisher.is_connected is False
        mock_stop.assert_called_once()
        mock_disc.assert_called_once()


def test_publisher_records_publish_attempt_and_broker_ack():
    """Verify publishing queues message and on_publish callback records broker ACK."""
    tracker = DeliveryTracker()
    publisher = MqttAlertPublisher(tracker=tracker)
    publisher._is_connected = True

    mock_msg_info = MagicMock()
    mock_msg_info.mid = 42

    with patch.object(publisher.client, "publish", return_value=mock_msg_info) as mock_pub:
        alert = _create_sample_alert("ALT-0010")
        tracker.record_created(alert)

        pub_ok = publisher.publish_alert(alert, qos=1)
        assert pub_ok is True
        mock_pub.assert_called_once()

        # Check status is PUBLISH_PENDING
        rec = tracker.records["ALT-0010"]
        assert rec.status == DeliveryStatus.PUBLISH_PENDING
        assert rec.publish_attempt_time is not None

        # Simulate broker PUBACK callback
        publisher._on_publish(publisher.client, None, mid=42)

        # Check status is updated to PUBLISHED
        assert rec.status == DeliveryStatus.PUBLISHED
        assert rec.broker_ack_time is not None
        assert rec.publication_latency_ms is not None


# -----------------------------------------------------------------------------
# 5-8. Subscriber Ingestion, Receipt ACKs and Deduplication
# -----------------------------------------------------------------------------

def test_subscriber_message_parsing_and_receipt_ack():
    """Verify subscriber processes valid alert and transmits application-level receipt ACK."""
    tracker = DeliveryTracker()
    received_alerts = []

    subscriber = MqttAlertSubscriber(
        tracker=tracker,
        on_alert_received=lambda a: received_alerts.append(a),
    )
    subscriber._is_connected = True

    alert = _create_sample_alert("ALT-0020", "SMOKE-001", AlertSeverity.CRITICAL, 4.5)
    raw_payload = alert.to_json().encode("utf-8")
    mock_msg = MockMqttMessage(topic=subscriber.config.alert_topic, payload=raw_payload)

    with patch.object(subscriber.client, "publish") as mock_ack_pub:
        subscriber._on_message(subscriber.client, None, mock_msg)

        # Check subscriber store
        assert len(received_alerts) == 1
        assert received_alerts[0].alert_id == "ALT-0020"

        # Check tracker recorded subscriber receipt
        rec = tracker.records.get("ALT-0020")
        assert rec is not None
        assert rec.status == DeliveryStatus.RECEIVED
        assert rec.subscriber_receipt_time is not None

        # Check application-level receipt ACK was transmitted to ack_topic
        mock_ack_pub.assert_called_once()
        call_kwargs = mock_ack_pub.call_args.kwargs
        call_topic = call_kwargs.get("topic") if "topic" in call_kwargs else mock_ack_pub.call_args.args[0]
        assert call_topic == subscriber.config.ack_topic


def test_subscriber_duplicate_detection_and_metrics():
    """Verify subscriber detects repeated alert IDs and increments duplicate count."""
    tracker = DeliveryTracker()
    subscriber = MqttAlertSubscriber(tracker=tracker)
    subscriber._is_connected = True

    alert = _create_sample_alert("ALT-0030")
    raw_payload = alert.to_json().encode("utf-8")
    mock_msg = MockMqttMessage(topic=subscriber.config.alert_topic, payload=raw_payload)

    with patch.object(subscriber.client, "publish"):
        # First arrival
        subscriber._on_message(subscriber.client, None, mock_msg)
        assert len(subscriber.get_received_alerts()) == 1
        assert subscriber.duplicate_count == 0

        # Duplicate arrival
        subscriber._on_message(subscriber.client, None, mock_msg)
        assert len(subscriber.get_received_alerts()) == 1  # Not duplicated in store
        assert subscriber.duplicate_count == 1


def test_subscriber_malformed_message_graceful_handling():
    """Verify subscriber discards malformed payloads without crashing."""
    tracker = DeliveryTracker()
    subscriber = MqttAlertSubscriber(tracker=tracker)
    subscriber._is_connected = True

    mock_msg = MockMqttMessage(topic=subscriber.config.alert_topic, payload=b"not-json-data")
    subscriber._on_message(subscriber.client, None, mock_msg)

    assert subscriber.malformed_count == 1
    assert len(subscriber.get_received_alerts()) == 0


# -----------------------------------------------------------------------------
# 9-10. End-to-End Delivery Confirmation & Latency Tracking
# -----------------------------------------------------------------------------

def test_full_delivery_state_transitions_and_receipt_matching():
    """Verify complete 5-stage progression from CREATED to DELIVERY_CONFIRMED."""
    tracker = DeliveryTracker()
    publisher = MqttAlertPublisher(tracker=tracker)
    subscriber = MqttAlertSubscriber(tracker=tracker)

    alert = _create_sample_alert("ALT-0040")

    # Stage 1: Created locally
    rec = tracker.record_created(alert)
    assert rec.status == DeliveryStatus.GENERATED

    # Stage 2: Publish attempted
    tracker.record_publish_attempt(alert.alert_id, qos=1, topic="rlack/kitchen/alerts", timestamp=100.0)
    assert rec.status == DeliveryStatus.PUBLISH_PENDING

    # Stage 3: Broker PUBACK received
    tracker.record_broker_ack(alert.alert_id, timestamp=100.02)
    assert rec.status == DeliveryStatus.PUBLISHED
    assert pytest.approx(rec.publication_latency_ms, 0.1) == 20.0  # 20ms

    # Stage 4: Subscriber message received
    tracker.record_subscriber_receipt(alert.alert_id, timestamp=100.05)
    assert rec.status == DeliveryStatus.RECEIVED
    assert pytest.approx(rec.subscriber_receipt_latency_ms, 0.1) == 50.0  # 50ms

    # Stage 5: Application Receipt Confirmed (Publisher receives App-Ack)
    ack_payload = json.dumps({"alert_id": "ALT-0040", "receipt_status": "received"}).encode("utf-8")
    mock_ack_msg = MockMqttMessage(topic="rlack/kitchen/alerts/ack", payload=ack_payload)

    with patch("time.time", return_value=100.08):
        publisher._on_message(publisher.client, None, mock_ack_msg)

    assert rec.status == DeliveryStatus.DELIVERY_CONFIRMED
    assert pytest.approx(rec.end_to_end_latency_ms, 0.1) == 80.0  # 80ms round-trip


def test_delivery_timeout_detection():
    """Verify check_timeouts identifies overdue pending alerts and updates status to TIMED_OUT."""
    tracker = DeliveryTracker()
    alert = _create_sample_alert("ALT-0050")
    tracker.record_created(alert)

    # Recorded 10 seconds ago
    old_time = time.time() - 10.0
    tracker.record_publish_attempt("ALT-0050", qos=1, topic="rlack/kitchen/alerts", timestamp=old_time)

    timed_out = tracker.check_timeouts(timeout_s=5.0)
    assert "ALT-0050" in timed_out
    assert tracker.records["ALT-0050"].status == DeliveryStatus.TIMED_OUT


def test_publish_failure_when_disconnected():
    """Verify publish_alert records failure when client is offline."""
    tracker = DeliveryTracker()
    publisher = MqttAlertPublisher(tracker=tracker)
    publisher._is_connected = False

    alert = _create_sample_alert("ALT-0060")
    tracker.record_created(alert)

    success = publisher.publish_alert(alert)
    assert success is False
    assert tracker.records["ALT-0060"].status == DeliveryStatus.FAILED
    assert "Broker not connected" in tracker.records["ALT-0060"].error_message


# -----------------------------------------------------------------------------
# 11-14. Metrics, DataFrame & Reset
# -----------------------------------------------------------------------------

def test_delivery_tracker_metrics_aggregation():
    """Verify get_metrics computes accurate aggregate delivery counts."""
    tracker = DeliveryTracker()

    a1 = _create_sample_alert("ALT-01")
    a2 = _create_sample_alert("ALT-02")
    a3 = _create_sample_alert("ALT-03")

    tracker.record_created(a1)
    tracker.record_created(a2)
    tracker.record_created(a3)

    tracker.record_publish_attempt("ALT-01", 1, "topic", 1.0)
    tracker.record_broker_ack("ALT-01", 1.02)
    tracker.record_subscriber_receipt("ALT-01", 1.05)
    tracker.record_delivery_confirmed("ALT-01", 1.08)

    tracker.record_publish_attempt("ALT-02", 1, "topic", 1.0)
    tracker.record_failure("ALT-03", "Connection lost")

    metrics = tracker.get_metrics()
    assert metrics["total_alerts"] == 3
    assert metrics["total_published"] == 2
    assert metrics["confirmed_deliveries"] == 1
    assert metrics["pending_receipts"] == 1
    assert metrics["failed"] == 1
    assert metrics["avg_e2e_latency_ms"] is not None


def test_delivery_dataframe_formatting():
    """Verify to_dataframe produces table with expected column schema."""
    tracker = DeliveryTracker()
    alert = _create_sample_alert("ALT-0070")
    tracker.record_created(alert)
    tracker.record_publish_attempt("ALT-0070", 1, "topic", 10.0)
    tracker.record_delivery_confirmed("ALT-0070", 10.05)

    df = tracker.to_dataframe()
    assert not df.empty
    assert "Alert ID" in df.columns
    assert "Status" in df.columns
    assert "E2E Latency" in df.columns
    assert df.iloc[0]["Alert ID"] == "ALT-0070"
    assert df.iloc[0]["Status"] == "DELIVERY_CONFIRMED"


def test_delivery_tracker_reset():
    """Verify reset clears all records and events."""
    tracker = DeliveryTracker()
    alert = _create_sample_alert("ALT-0080")
    tracker.record_created(alert)
    assert len(tracker.records) == 1

    tracker.reset()
    assert len(tracker.records) == 0
    assert len(tracker.events) == 0


# -----------------------------------------------------------------------------
# 15. Phase 3 IncidentManager Integration
# -----------------------------------------------------------------------------

def test_incident_manager_alert_to_mqtt_manager_pipeline():
    """Verify alerts produced by Phase 3 IncidentManager flow directly into MqttManager."""
    detector = IncidentDetector()
    incident_mgr = IncidentManager(detector=detector)
    mqtt_mgr = MqttManager()

    # Synthetic emergency reading
    reading = SensorReading(
        sensor_id="GAS-001",
        sensor_type=SensorType.GAS,
        value=500.0,
        unit="ppm",
        simulation_time_s=5.0,
        host_node_id="NODE-001",
        status=NodeOperationalStatus.CRITICAL,
        sequence_number=5,
        is_threshold_exceeded=True,
    )

    alert = incident_mgr.process_reading(reading)
    assert alert is not None
    assert alert.alert_id == "ALT-0001"

    # Mock publish operation on publisher client
    mqtt_mgr.publisher._is_connected = True
    mock_msg_info = MagicMock()
    mock_msg_info.mid = 101

    with patch.object(mqtt_mgr.publisher.client, "publish", return_value=mock_msg_info):
        pub_ok = mqtt_mgr.publish_alert(alert, qos=1)
        assert pub_ok is True

    # Validate tracking record
    rec = mqtt_mgr.tracker.records.get("ALT-0001")
    assert rec is not None
    assert rec.status == DeliveryStatus.PUBLISH_PENDING
    assert rec.sensor_id == "GAS-001"


# -----------------------------------------------------------------------------
# 16. Optional Live Mosquitto Integration Test
# -----------------------------------------------------------------------------

def _is_local_mosquitto_listening(host="127.0.0.1", port=1883) -> bool:
    """Check if Mosquitto broker is actively accepting TCP connections."""
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
def test_live_mosquitto_broker_alert_roundtrip():
    """Live end-to-end integration test against running Eclipse Mosquitto broker."""
    config = MqttConfig(
        host="127.0.0.1",
        port=1883,
        alert_topic="rlack/test/alerts",
        ack_topic="rlack/test/alerts/ack",
    )
    manager = MqttManager(config=config)

    connected = manager.connect(timeout_s=3.0)
    assert connected is True, "Failed to connect to live Mosquitto broker"

    try:
        alert = _create_sample_alert("ALT-LIVE-01", "GAS-001", AlertSeverity.CRITICAL, 499.0)
        pub_ok = manager.publish_alert(alert, qos=1)
        assert pub_ok is True

        # Wait for round-trip receipt and app-ack
        confirmed = False
        start_t = time.time()
        while (time.time() - start_t) < 4.0:
            rec = manager.tracker.records.get("ALT-LIVE-01")
            if rec and rec.status == DeliveryStatus.DELIVERY_CONFIRMED:
                confirmed = True
                break
            time.sleep(0.05)

        assert confirmed is True, "Did not receive delivery confirmation within 4 seconds"
        assert rec.end_to_end_latency_ms is not None
        assert rec.end_to_end_latency_ms > 0
    finally:
        manager.disconnect()


# -----------------------------------------------------------------------------
# 17. Focused Connection Lifecycle, Callbacks, and Status Reporting Tests
# -----------------------------------------------------------------------------

def test_client_id_uniqueness_validation():
    """Verify identical publisher and subscriber client IDs are rejected."""
    with pytest.raises(ConfigurationError, match="must be distinct"):
        MqttConfig(publisher_client_id="same_client_id", subscriber_client_id="same_client_id").validate()


def test_publisher_and_subscriber_subscription_callbacks():
    """Verify on_subscribe callbacks confirm topic subscriptions."""
    publisher = MqttAlertPublisher()
    subscriber = MqttAlertSubscriber()

    assert publisher.is_subscribed is False
    assert subscriber.is_subscribed is False

    # Simulate SUBACK callbacks
    publisher._on_subscribe(publisher.client, None, mid=1)
    subscriber._on_subscribe(subscriber.client, None, mid=2)

    assert publisher.is_subscribed is True
    assert subscriber.is_subscribed is True

    # Disconnect clears subscription status
    publisher._on_disconnect(publisher.client, None)
    subscriber._on_disconnect(subscriber.client, None)

    assert publisher.is_subscribed is False
    assert subscriber.is_subscribed is False


def test_connection_failure_handling_and_error_capture():
    """Verify connection errors are captured and do not raise unhandled exceptions."""
    cfg = MqttConfig(host="127.0.0.1", port=19999)  # Closed port
    publisher = MqttAlertPublisher(config=cfg)

    # Patch client.connect to simulate socket failure
    with patch.object(publisher.client, "connect", side_effect=ConnectionRefusedError("Connection refused")):
        ok = publisher.connect(timeout_s=0.2)
        assert ok is False
        assert publisher.is_connected is False
        assert publisher.is_connecting is False
        assert "Connection refused" in (publisher.connection_error or "")

    # Repeated calls should remain safe
    with patch.object(publisher.client, "connect", side_effect=ConnectionRefusedError("Connection refused")):
        ok = publisher.connect(timeout_s=0.2)
        assert ok is False


def test_on_connect_fail_callback():
    """Verify on_connect_fail callback resets connection flags and records error."""
    publisher = MqttAlertPublisher()
    publisher._is_connecting = True

    publisher._on_connect_fail(publisher.client, None)
    assert publisher.is_connected is False
    assert publisher.is_connecting is False
    assert publisher.connection_error is not None
    assert "Broker unreachable" in publisher.connection_error


def test_on_connect_rejection_reason_code():
    """Verify on_connect handles rejection reason codes."""
    subscriber = MqttAlertSubscriber()

    # Rejection code (e.g. 5 = Not authorized)
    subscriber._on_connect(subscriber.client, None, None, rc=5)
    assert subscriber.is_connected is False
    assert subscriber.is_connecting is False
    assert "rejected" in (subscriber.connection_error or "").lower()


def test_mqtt_manager_connection_status_reporting():
    """Verify MqttManager reports detailed human-readable status from callbacks."""
    manager = MqttManager()

    # Initial state
    assert manager.is_connected is False
    assert manager.broker_status == "Disconnected"

    # Publisher connected only
    manager.publisher._is_connected = True
    assert manager.broker_status == "Publisher Connected (Subscriber Offline)"

    # Subscriber connected only
    manager.publisher._is_connected = False
    manager.subscriber._is_connected = True
    assert manager.broker_status == "Subscriber Connected (Publisher Offline)"

    # Both connected, subscriber still subscribing
    manager.publisher._is_connected = True
    manager.subscriber._is_connected = True
    manager.subscriber._is_subscribed = False
    assert manager.is_connected is True
    assert manager.broker_status == "Connected (Subscribing...)"

    # Both connected and subscriber confirmed subscribed
    manager.subscriber._is_subscribed = True
    assert manager.broker_status == "Connected (Subscribed)"

    # Disconnected with specific error
    manager.publisher._is_connected = False
    manager.subscriber._is_connected = False
    manager.publisher._connection_error = "Connection refused"
    assert manager.broker_status == "Disconnected (Connection refused)"


def test_dashboard_simulation_manager_mqtt_integration():
    """Verify DashboardSimulationManager connects to MQTT and triggers test alerts."""
    from rl_ack.dashboard.state import DashboardSimulationManager

    sim_mgr = DashboardSimulationManager()
    assert sim_mgr.mqtt_manager is not None

    # Test trigger_test_mqtt_alert when disconnected
    sim_mgr.mqtt_manager.publisher._is_connected = False
    alt = sim_mgr.trigger_test_mqtt_alert()
    assert alt is not None
    rec = sim_mgr.mqtt_manager.tracker.records.get(alt.alert_id)
    assert rec is not None
    assert rec.status == DeliveryStatus.FAILED

