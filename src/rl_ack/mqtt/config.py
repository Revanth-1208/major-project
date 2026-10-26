"""Centralized MQTT configuration for RL-ACK Phase 4.

Supports environment variable overrides, configurable timeouts, topics,
and credentials for local demonstration and remote brokers.
"""

from dataclasses import dataclass, field
import os
import uuid
from typing import Optional

from rl_ack.common.exceptions import ConfigurationError


@dataclass
class MqttConfig:
    """MQTT connection, topic, and delivery verification settings."""

    host: str = field(
        default_factory=lambda: os.getenv("RLACK_MQTT_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("RLACK_MQTT_PORT", "1883"))
    )
    keepalive: int = 60
    publisher_client_id: str = field(
        default_factory=lambda: f"rlack_pub_{uuid.uuid4().hex[:8]}"
    )
    subscriber_client_id: str = field(
        default_factory=lambda: f"rlack_sub_{uuid.uuid4().hex[:8]}"
    )
    clean_session: bool = True
    username: Optional[str] = field(
        default_factory=lambda: os.getenv("RLACK_MQTT_USERNAME", None)
    )
    password: Optional[str] = field(
        default_factory=lambda: os.getenv("RLACK_MQTT_PASSWORD", None)
    )

    # Topic taxonomy
    alert_topic: str = "rlack/kitchen/alerts"
    ack_topic: str = "rlack/kitchen/alerts/ack"
    telemetry_topic: str = "rlack/kitchen/telemetry"

    # Default QoS for alert messages in Phase 4 (0, 1, or 2)
    default_alert_qos: int = 1

    # Timeouts in seconds
    connect_timeout_s: float = 3.0
    publish_timeout_s: float = 3.0
    receipt_timeout_s: float = 5.0

    # Exponential reconnect limits
    reconnect_delay_min_s: float = 1.0
    reconnect_delay_max_s: float = 10.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration settings, raising ConfigurationError on violations."""
        if not self.host or not isinstance(self.host, str) or not self.host.strip():
            raise ConfigurationError("MQTT host must be a non-empty string.")

        if not (1 <= self.port <= 65535):
            raise ConfigurationError(f"MQTT port must be between 1 and 65535, got {self.port}")

        if not (0 <= self.default_alert_qos <= 2):
            raise ConfigurationError(
                f"MQTT default_alert_qos must be 0, 1, or 2, got {self.default_alert_qos}"
            )

        if self.connect_timeout_s <= 0:
            raise ConfigurationError("connect_timeout_s must be positive.")

        if self.receipt_timeout_s <= 0:
            raise ConfigurationError("receipt_timeout_s must be positive.")

        if not self.alert_topic or not self.ack_topic:
            raise ConfigurationError("alert_topic and ack_topic must be specified.")

        if self.alert_topic == self.ack_topic:
            raise ConfigurationError(
                "alert_topic and ack_topic must be distinct to prevent message loopback."
            )
