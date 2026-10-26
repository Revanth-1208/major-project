"""Standalone CLI MQTT Alert Receiver / Subscriber for RL-ACK Phase 4.

Usage:
    python -m rl_ack.mqtt.receiver
    python -m rl_ack.mqtt.receiver --host 127.0.0.1 --port 1883
"""

import argparse
import logging
import signal
import sys
import time

from rl_ack.common.models import AlertMessage
from rl_ack.mqtt.config import MqttConfig
from rl_ack.mqtt.subscriber import MqttAlertSubscriber

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MQTT-Receiver")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-ACK Standalone MQTT Alert Receiver")
    parser.add_argument("--host", default=None, help="MQTT broker host (default: 127.0.0.1 or RLACK_MQTT_HOST)")
    parser.add_argument("--port", type=int, default=None, help="MQTT broker port (default: 1883 or RLACK_MQTT_PORT)")
    parser.add_argument("--topic", default=None, help="Alert subscription topic")
    args = parser.parse_args()

    cfg_kwargs = {}
    if args.host:
        cfg_kwargs["host"] = args.host
    if args.port:
        cfg_kwargs["port"] = args.port
    if args.topic:
        cfg_kwargs["alert_topic"] = args.topic

    config = MqttConfig(**cfg_kwargs)

    def on_alert(alert: AlertMessage) -> None:
        print("\n" + "=" * 65)
        print(f"🚨 EMERGENCY ALERT RECEIVED via MQTT: {alert.alert_id}")
        print("=" * 65)
        print(f"  Incident ID   : {alert.incident_id or 'N/A'}")
        print(f"  Sensor ID     : {alert.sensor_id}")
        print(f"  Severity      : {alert.severity.value.upper()}")
        print(f"  Message       : {alert.message}")
        print(f"  Trigger Value : {alert.reading_value:.2f} {alert.unit} (Threshold: {alert.threshold_value:.2f} {alert.unit})")
        print(f"  Sim Time      : {alert.simulation_time_s:.1f}s")
        print(f"  Created (UTC) : {alert.created_at_utc}")
        print(f"  Priority      : {alert.priority.value}")
        print("  Status        : Verified & Application Receipt Transmitted (ack_topic)")
        print("=" * 65 + "\n")

    subscriber = MqttAlertSubscriber(config=config, on_alert_received=on_alert)

    print(f"Connecting to MQTT Broker at {config.host}:{config.port}...")
    if not subscriber.connect(timeout_s=5.0):
        print(f"\n❌ Error: Could not connect to MQTT broker at {config.host}:{config.port}.")
        print("Please ensure Mosquitto broker is running (e.g. 'mosquitto -v').")
        sys.exit(1)

    print(f"✅ Connected! Subscribed to alert topic: '{config.alert_topic}'")
    print(f"Listening for emergency alerts... (Press Ctrl+C to exit)\n")

    def handle_sigint(sig, frame):
        print("\nStopping MQTT receiver...")
        subscriber.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
