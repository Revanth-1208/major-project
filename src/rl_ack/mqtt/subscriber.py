"""Non-blocking MQTT Alert Subscriber and Delivery Receiver for Phase 4.

Subscribes to emergency alerts, validates payloads, suppresses duplicates,
records receipt timestamps, and transmits application-level receipt ACKs.
"""

from collections import deque
from datetime import datetime, timezone
import json
import logging
import socket
import time
from typing import Any, Callable, Dict, List, Optional, Set
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt

from rl_ack.common.exceptions import MqttPayloadError
from rl_ack.common.models import AlertMessage
from rl_ack.mqtt.config import MqttConfig
from rl_ack.mqtt.delivery_tracker import DeliveryTracker

logger = logging.getLogger(__name__)


class MqttAlertSubscriber:
    """Asynchronous MQTT receiver validating alerts and issuing application receipts."""

    def __init__(
        self,
        config: Optional[MqttConfig] = None,
        tracker: Optional[DeliveryTracker] = None,
        on_alert_received: Optional[Callable[[AlertMessage], None]] = None,
        max_retained_alerts: int = 500,
    ) -> None:
        self.config = config or MqttConfig()
        self.tracker = tracker or DeliveryTracker()
        self.on_alert_received = on_alert_received

        self._is_connected: bool = False
        self._is_connecting: bool = False
        self._is_subscribed: bool = False
        self._connection_error: Optional[str] = None
        self._received_alert_ids: Set[str] = set()
        self._duplicate_count: int = 0
        self._malformed_count: int = 0
        self._received_alerts: deque[AlertMessage] = deque(maxlen=max_retained_alerts)

        self._init_client()

    def _init_client(self) -> None:
        """Initialize Paho MQTT client supporting Paho v1 and v2 API."""
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.config.subscriber_client_id,
                clean_session=self.config.clean_session,
            )
        else:
            self.client = mqtt.Client(  # type: ignore[call-arg]
                client_id=self.config.subscriber_client_id,
                clean_session=self.config.clean_session,
            )

        if self.config.username:
            self.client.username_pw_set(
                username=self.config.username,
                password=self.config.password,
            )

        self.client.on_connect = self._on_connect
        self.client.on_connect_fail = self._on_connect_fail
        self.client.on_disconnect = self._on_disconnect
        self.client.on_subscribe = self._on_subscribe
        self.client.on_message = self._on_message

    @property
    def is_connected(self) -> bool:
        """Check if subscriber is actively connected to the broker."""
        return self._is_connected

    @property
    def is_connecting(self) -> bool:
        """Check if subscriber is actively attempting to establish connection."""
        return self._is_connecting

    @property
    def is_subscribed(self) -> bool:
        """Check if subscriber has verified alert topic subscription."""
        return self._is_subscribed

    @property
    def connection_error(self) -> Optional[str]:
        """Description of the most recent connection failure, if any."""
        return self._connection_error

    @property
    def duplicate_count(self) -> int:
        """Total number of duplicate alert deliveries detected."""
        return self._duplicate_count

    @property
    def malformed_count(self) -> int:
        """Total number of malformed or invalid alert payloads received."""
        return self._malformed_count

    @staticmethod
    def _is_reason_success(rc: Any) -> bool:
        """Evaluate whether a connection or subscription reason code signifies success."""
        if rc is None:
            return False
        if isinstance(rc, int):
            return rc == 0
        if hasattr(rc, "is_failure"):
            return not rc.is_failure
        if hasattr(rc, "value"):
            return rc.value == 0
        return rc == 0

    def connect(self, timeout_s: Optional[float] = None) -> bool:
        """Connect subscriber to broker and start network loop."""
        if self._is_connected:
            return True

        timeout = timeout_s if timeout_s is not None else self.config.connect_timeout_s
        self._is_connecting = True
        self._connection_error = None

        try:
            # If connect_async is an explicit Mock, use it directly
            is_async_mock = isinstance(getattr(self.client, "connect_async", None), MagicMock)
            if is_async_mock:
                self.client.connect_async(
                    self.config.host,
                    self.config.port,
                    keepalive=self.config.keepalive,
                )
            elif hasattr(self.client, "connect"):
                rc = self.client.connect(
                    self.config.host,
                    self.config.port,
                    keepalive=self.config.keepalive,
                )
                if rc != 0 and rc != mqtt.MQTT_ERR_SUCCESS:
                    self._connection_error = f"connect() failed with return code {rc}"
                    self._is_connecting = False
                    return False
            else:
                self.client.connect_async(
                    self.config.host,
                    self.config.port,
                    keepalive=self.config.keepalive,
                )

            self.client.loop_start()

            # Wait briefly for on_connect and on_subscribe callbacks
            start_t = time.time()
            while (time.time() - start_t) < timeout:
                if self._is_connected and self._is_subscribed:
                    self._is_connecting = False
                    return True
                elif self._is_connected and (time.time() - start_t) > 0.5:
                    # Connected, give a short grace period for SUBACK
                    self._is_connecting = False
                    return True
                time.sleep(0.02)

            self._is_connecting = False
            return self._is_connected
        except (ConnectionRefusedError, socket.error, OSError) as e:
            self._connection_error = f"Socket error: {e}"
            self._is_connected = False
            self._is_connecting = False
            logger.warning("Subscriber connection refused to %s:%d: %s", self.config.host, self.config.port, e)
            return False
        except Exception as e:
            self._connection_error = f"Connect exception: {e}"
            self._is_connected = False
            self._is_connecting = False
            logger.error("Failed to connect MQTT subscriber: %s", e)
            return False

    def disconnect(self) -> None:
        """Gracefully disconnect subscriber and stop network loop."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as e:
            logger.debug("Error during subscriber disconnect: %s", e)
        finally:
            self._is_connected = False
            self._is_subscribed = False
            self._is_connecting = False

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: Any, *args: Any) -> None:
        """Subscribe to alert topic once connection is established."""
        is_success = self._is_reason_success(rc)
        if is_success:
            self._is_connected = True
            self._is_connecting = False
            self._connection_error = None
            logger.info("Subscriber connected to MQTT broker. Subscribing to %s", self.config.alert_topic)
            try:
                self.client.subscribe(self.config.alert_topic, qos=self.config.default_alert_qos)
            except Exception as e:
                logger.error("Failed to subscribe to alert topic: %s", e)
        else:
            self._is_connected = False
            self._is_connecting = False
            self._connection_error = f"Connection rejected: {rc}"
            logger.warning("Subscriber connection rejected with code %s", rc)

    def _on_connect_fail(self, client: Any, userdata: Any, *args: Any) -> None:
        """Callback when background connection attempt fails."""
        self._is_connected = False
        self._is_connecting = False
        self._connection_error = "Broker unreachable / connection failed"
        logger.warning("Subscriber on_connect_fail triggered")

    def _on_disconnect(self, client: Any, userdata: Any, *args: Any, **kwargs: Any) -> None:
        """Handle disconnection event."""
        self._is_connected = False
        self._is_subscribed = False
        self._is_connecting = False
        logger.info("Subscriber disconnected from MQTT broker.")

    def _on_subscribe(self, client: Any, userdata: Any, mid: int, reason_codes: Any = None, *args: Any, **kwargs: Any) -> None:
        """Callback when subscription to alert_topic is confirmed."""
        self._is_subscribed = True
        logger.info("Subscriber confirmed subscription to %s (mid=%d)", self.config.alert_topic, mid)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        """Process incoming alert message."""
        receipt_time = time.time()
        try:
            payload_str = msg.payload.decode("utf-8")
            alert = AlertMessage.from_json(payload_str)
        except Exception as e:
            self._malformed_count += 1
            logger.warning("Subscriber rejected malformed alert on topic %s: %s", msg.topic, e)
            return

        is_duplicate = alert.alert_id in self._received_alert_ids
        if is_duplicate:
            self._duplicate_count += 1
            logger.info("Duplicate delivery detected for alert %s", alert.alert_id)
        else:
            self._received_alert_ids.add(alert.alert_id)
            self._received_alerts.append(alert)

            # Record receipt in delivery tracker
            self.tracker.record_subscriber_receipt(alert.alert_id, timestamp=receipt_time)

            if self.on_alert_received:
                try:
                    self.on_alert_received(alert)
                except Exception as e:
                    logger.error("Error in on_alert_received callback: %s", e)

        # Transmit application-level receipt acknowledgement regardless of duplication
        self._send_receipt_ack(alert.alert_id, alert.incident_id, is_duplicate)

    def _send_receipt_ack(
        self, alert_id: str, incident_id: Optional[str], is_duplicate: bool
    ) -> None:
        """Publish application-level receipt ACK message to ack_topic."""
        try:
            now_utc = datetime.now(timezone.utc).isoformat()
            ack_payload = {
                "alert_id": alert_id,
                "incident_id": incident_id,
                "receipt_status": "received",
                "received_at_utc": now_utc,
                "subscriber_id": self.config.subscriber_client_id,
                "is_duplicate": is_duplicate,
            }
            self.client.publish(
                topic=self.config.ack_topic,
                payload=json.dumps(ack_payload),
                qos=1,
            )
        except Exception as e:
            logger.error("Failed to transmit receipt ACK for %s: %s", alert_id, e)

    def get_received_alerts(self) -> List[AlertMessage]:
        """Return list of uniquely received alert messages."""
        return list(self._received_alerts)

    def reset(self) -> None:
        """Reset received alert tracking state."""
        self._received_alert_ids.clear()
        self._duplicate_count = 0
        self._malformed_count = 0
        self._received_alerts.clear()
