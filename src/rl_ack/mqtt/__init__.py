"""Module 5: MQTT Alert Transmission and Delivery Verification.

Responsible for:
- Eclipse Mosquitto broker integration and Paho MQTT clients
- Topic separation (telemetry vs alerts vs acks)
- Distinct lifecycle event tracking:
    1. Alert Generation
    2. MQTT Publication Attempt
    3. Broker Acknowledgement (PUBACK)
    4. Subscriber Receipt
    5. Application-level Receipt Acknowledgement (Delivery Confirmation)
- Strict constraint: Never mark an alert as delivered merely because a publication was attempted.
"""

from rl_ack.mqtt.config import MqttConfig
from rl_ack.mqtt.delivery_tracker import DeliveryRecord, DeliveryTracker
from rl_ack.mqtt.manager import MqttManager
from rl_ack.mqtt.publisher import MqttAlertPublisher
from rl_ack.mqtt.subscriber import MqttAlertSubscriber

MODULE_NAME = "mqtt"
MODULE_ID = 5
IS_IMPLEMENTED = True

__all__ = [
    "MODULE_NAME",
    "MODULE_ID",
    "IS_IMPLEMENTED",
    "MqttConfig",
    "MqttManager",
    "MqttAlertPublisher",
    "MqttAlertSubscriber",
    "DeliveryRecord",
    "DeliveryTracker",
]
