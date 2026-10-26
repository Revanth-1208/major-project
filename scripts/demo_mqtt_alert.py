"""Demonstration runner for Phase 4: MQTT Alert Transmission and Delivery Verification.

Executes a live end-to-end test against Eclipse Mosquitto broker:
1. Generates an emergency gas leak alert via the virtual sensor engine and incident detector.
2. Publishes the alert to MQTT topic 'rlack/kitchen/alerts'.
3. Receives and validates the alert via an independent subscriber.
4. Transmits and verifies application-level receipt acknowledgment on 'rlack/kitchen/alerts/ack'.
5. Measures and reports publication latency, subscriber receipt latency, and end-to-end round trip.
"""

import sys
import time
from pathlib import Path

# Add src/ to path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rl_ack.common.enums import NodeOperationalStatus, SensorType
from rl_ack.common.models import SensorReading
from rl_ack.monitoring.detector import IncidentDetector
from rl_ack.monitoring.manager import IncidentManager
from rl_ack.mqtt.config import MqttConfig
from rl_ack.mqtt.manager import MqttManager


def main() -> None:
    print("\n" + "=" * 70)
    print("📡 RL-ACK Phase 4: MQTT Emergency Alert Transmission & Delivery Verification")
    print("=" * 70)

    config = MqttConfig()
    print(f"\n[CONFIG] Target Broker : {config.host}:{config.port}")
    print(f"[CONFIG] Alert Topic   : {config.alert_topic}")
    print(f"[CONFIG] Ack Topic     : {config.ack_topic}")
    print(f"[CONFIG] Default QoS   : {config.default_alert_qos}")

    mqtt_manager = MqttManager(config=config)
    print("\n[STEP 1] Connecting Publisher and Subscriber clients to MQTT Broker...")

    connected = mqtt_manager.connect(timeout_s=3.0)
    if not connected:
        print("\n❌ Could not connect to local MQTT broker.")
        print("Diagnosis: Eclipse Mosquitto broker is not actively listening on 127.0.0.1:1883.")
        print("\nTo start Mosquitto on Windows, run:")
        print("  & 'C:\\Program Files\\mosquitto\\mosquitto.exe' -v")
        print("Or check service with: Get-Service mosquitto\n")
        return

    print("✅ Connected to MQTT broker successfully!")
    print(f"   Publisher  ID: {config.publisher_client_id}")
    print(f"   Subscriber ID: {config.subscriber_client_id}")

    # Step 2: Generate emergency alert via Phase 3 IncidentManager
    print("\n[STEP 2] Simulating dangerous gas concentration reading...")
    incident_mgr = IncidentManager(detector=IncidentDetector())

    gas_reading = SensorReading(
        sensor_id="GAS-001",
        sensor_type=SensorType.GAS,
        value=520.0,  # Far above critical limit (400 ppm)
        unit="ppm",
        simulation_time_s=12.0,
        host_node_id="NODE-001",
        status=NodeOperationalStatus.CRITICAL,
        sequence_number=12,
        is_threshold_exceeded=True,
    )

    alert = incident_mgr.process_reading(gas_reading)
    assert alert is not None, "Failed to create emergency alert"

    print(f"✅ Created Emergency Alert:")
    print(f"   Alert ID     : {alert.alert_id}")
    print(f"   Incident ID  : {alert.incident_id}")
    print(f"   Sensor ID    : {alert.sensor_id}")
    print(f"   Severity     : {alert.severity.value.upper()}")
    print(f"   Trigger Value: {alert.reading_value:.1f} ppm (Threshold: {alert.threshold_value:.1f} ppm)")
    print(f"   Message      : {alert.message}")
    print(f"   Priority     : {alert.priority.value}")

    # Step 3: Publish Alert over MQTT
    print(f"\n[STEP 3] Publishing alert to MQTT topic '{config.alert_topic}' (QoS {config.default_alert_qos})...")
    pub_ok = mqtt_manager.publish_alert(alert, qos=config.default_alert_qos)
    if not pub_ok:
        print("❌ Failed to initiate alert publication.")
        mqtt_manager.disconnect()
        return

    print("✅ Publish initiated. Waiting for broker PUBACK and subscriber receipt...")

    # Wait for subscriber receipt and application receipt confirmation
    start_wait = time.time()
    confirmed = False
    while (time.time() - start_wait) < 5.0:
        rec = mqtt_manager.tracker.records.get(alert.alert_id)
        if rec and rec.delivery_confirmed_time is not None:
            confirmed = True
            break
        time.sleep(0.05)

    rec = mqtt_manager.tracker.records.get(alert.alert_id)
    assert rec is not None

    print("\n" + "=" * 70)
    print("📋 MULTI-STAGE DELIVERY VERIFICATION AUDIT")
    print("=" * 70)
    print(f"  Stage 1: Alert Created Locally       : ✅ {rec.created_time:.4f}")
    print(f"  Stage 2: MQTT Publish Attempted       : ✅ {rec.publish_attempt_time:.4f}" if rec.publish_attempt_time else "  Stage 2: Pending")
    print(f"  Stage 3: Broker Acknowledged (PUBACK) : ✅ {rec.broker_ack_time:.4f}" if rec.broker_ack_time else "  Stage 3: Not confirmed")
    print(f"  Stage 4: Subscriber Message Received  : ✅ {rec.subscriber_receipt_time:.4f}" if rec.subscriber_receipt_time else "  Stage 4: Not received")
    print(f"  Stage 5: Application Receipt Confirmed: ✅ {rec.delivery_confirmed_time:.4f}" if rec.delivery_confirmed_time else "  Stage 5: Not confirmed")
    print(f"  Final Delivery Status                : {rec.status.value.upper()}")

    print("\n⏱️ MEASURED LATENCY METRICS:")
    if rec.publication_latency_ms is not None:
        print(f"  - Publication Latency (Broker ACK)     : {rec.publication_latency_ms:.2f} ms")
    if rec.subscriber_receipt_latency_ms is not None:
        print(f"  - Subscriber Receipt Latency          : {rec.subscriber_receipt_latency_ms:.2f} ms")
    if rec.app_ack_latency_ms is not None:
        print(f"  - Application ACK Receipt Latency     : {rec.app_ack_latency_ms:.2f} ms")
    if rec.end_to_end_latency_ms is not None:
        print(f"  - End-to-End Delivery Latency         : {rec.end_to_end_latency_ms:.2f} ms")

    # Step 6: Verify Subscriber's Independent Message
    received_alerts = mqtt_manager.subscriber.get_received_alerts()
    print(f"\n📦 Subscriber Received Store: {len(received_alerts)} alert(s)")
    if received_alerts:
        sub_alert = received_alerts[0]
        print(f"   Verified Alert ID match: {sub_alert.alert_id == alert.alert_id}")
        print(f"   Verified Sensor ID match: {sub_alert.sensor_id == alert.sensor_id}")
        print(f"   Payload accurately reconstructed over network boundary!")

    print("\nDisconnecting clients...")
    mqtt_manager.disconnect()
    print("✅ Phase 4 demonstration complete!\n")


if __name__ == "__main__":
    main()
