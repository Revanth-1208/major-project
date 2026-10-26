"""Central MQTT Manager orchestrating publisher, subscriber, and delivery tracking.

Fulfills IMqttManager interface and encapsulates the complete Phase 4 pipeline.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from rl_ack.common.interfaces import IMqttManager
from rl_ack.common.models import AckEvent, AlertMessage, SensorReading
from rl_ack.mqtt.config import MqttConfig
from rl_ack.mqtt.delivery_tracker import DeliveryRecord, DeliveryTracker
from rl_ack.mqtt.publisher import MqttAlertPublisher
from rl_ack.mqtt.subscriber import MqttAlertSubscriber


class MqttManager(IMqttManager):
    """Module 5 coordinator managing MQTT connections, message flows, and verification."""

    def __init__(
        self,
        config: Optional[MqttConfig] = None,
        tracker: Optional[DeliveryTracker] = None,
    ) -> None:
        self.config = config or MqttConfig()
        self.tracker = tracker or DeliveryTracker()

        self.publisher = MqttAlertPublisher(config=self.config, tracker=self.tracker)
        self.subscriber = MqttAlertSubscriber(config=self.config, tracker=self.tracker)

    @property
    def is_connected(self) -> bool:
        """True if both publisher and subscriber are actively connected to broker."""
        return self.publisher.is_connected and self.subscriber.is_connected

    @property
    def is_subscribed(self) -> bool:
        """True if subscriber has confirmed alert topic subscription."""
        return self.subscriber.is_subscribed

    @property
    def is_connecting(self) -> bool:
        """True if either client is actively attempting connection."""
        return self.publisher.is_connecting or self.subscriber.is_connecting

    @property
    def connection_error(self) -> Optional[str]:
        """Most recent connection failure reason reported by publisher or subscriber."""
        return self.publisher.connection_error or self.subscriber.connection_error

    @property
    def broker_status(self) -> str:
        """Human-readable connection status string."""
        if self.is_connected:
            sub_status = "Subscribed" if self.subscriber.is_subscribed else "Subscribing..."
            return f"Connected ({sub_status})"
        elif self.is_connecting:
            return "Connecting..."
        elif self.publisher.is_connected:
            return "Publisher Connected (Subscriber Offline)"
        elif self.subscriber.is_connected:
            return "Subscriber Connected (Publisher Offline)"
        elif self.connection_error:
            return f"Disconnected ({self.connection_error})"
        return "Disconnected"

    def connect(self, timeout_s: Optional[float] = None) -> bool:
        """Connect both publisher and subscriber components to the MQTT broker."""
        timeout = timeout_s if timeout_s is not None else self.config.connect_timeout_s
        pub_ok = self.publisher.connect(timeout_s=timeout)
        sub_ok = self.subscriber.connect(timeout_s=timeout)
        return pub_ok and sub_ok

    def disconnect(self) -> None:
        """Gracefully disconnect all MQTT components."""
        self.publisher.disconnect()
        self.subscriber.disconnect()

    def publish_alert(self, alert: AlertMessage, qos: Optional[int] = None) -> bool:
        """Publish an emergency alert message and track its lifecycle.

        Args:
            alert: The AlertMessage instance.
            qos: MQTT QoS level (0, 1, or 2). Defaults to config.default_alert_qos.

        Returns:
            True if publication was queued successfully, False on error.
        """
        # Ensure alert is registered in delivery tracker
        if alert.alert_id not in self.tracker.records:
            self.tracker.record_created(alert)

        return self.publisher.publish_alert(alert, qos=qos)

    def publish_telemetry(self, reading: SensorReading) -> bool:
        """Publish sensor telemetry to the telemetry topic at QoS 0."""
        return self.publisher.publish_telemetry(reading)

    def track_event(self, event: AckEvent) -> None:
        """Track explicit delivery lifecycle acknowledgment event."""
        self.tracker.events.append(event)

    def check_timeouts(self, timeout_s: Optional[float] = None) -> List[str]:
        """Check and transition pending deliveries that exceeded timeout."""
        t = timeout_s or self.config.receipt_timeout_s
        return self.tracker.check_timeouts(timeout_s=t)

    def get_delivery_metrics(self) -> Dict[str, Any]:
        """Derive aggregate delivery verification statistics."""
        metrics = self.tracker.get_metrics()
        metrics["broker_status"] = self.broker_status
        metrics["duplicate_receipts"] = self.subscriber.duplicate_count
        metrics["malformed_receipts"] = self.subscriber.malformed_count
        metrics["subscriber_subscribed"] = self.subscriber.is_subscribed
        metrics["publisher_subscribed"] = self.publisher.is_subscribed
        metrics["connection_error"] = self.connection_error
        return metrics

    def get_delivery_records(self) -> List[DeliveryRecord]:
        """Return all tracked delivery records."""
        return list(self.tracker.records.values())

    def get_delivery_df(self, limit: int = 50) -> pd.DataFrame:
        """Format delivery records for Streamlit table presentation."""
        return self.tracker.to_dataframe(limit=limit)

    def reset(self) -> None:
        """Reset internal delivery state and counters."""
        self.tracker.reset()
        self.publisher.reset()
        self.subscriber.reset()
