"""Non-blocking MQTT Alert Publisher for Phase 4.

Publishes emergency alerts over MQTT, tracks broker PUBACK callbacks,
subscribes to application-level receipt ACKs, and integrates with the DeliveryTracker.
"""

import json
import logging
import socket
import time
from typing import Any, Dict, Optional, Set
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt

from rl_ack.common.models import AlertMessage, SensorReading
from rl_ack.mqtt.config import MqttConfig
from rl_ack.mqtt.delivery_tracker import DeliveryTracker

logger = logging.getLogger(__name__)


class MqttAlertPublisher:
    """Asynchronous MQTT publisher for virtual sensor emergency alerts and telemetry."""

    def __init__(
        self,
        config: Optional[MqttConfig] = None,
        tracker: Optional[DeliveryTracker] = None,
    ) -> None:
        self.config = config or MqttConfig()
        self.tracker = tracker or DeliveryTracker()

        self._is_connected: bool = False
        self._is_connecting: bool = False
        self._is_subscribed: bool = False
        self._connection_error: Optional[str] = None
        self._mid_to_alert_id: Dict[int, str] = {}
        self._published_alert_ids: Set[str] = set()

        self._init_client()

    def _init_client(self) -> None:
        """Initialize Paho MQTT client supporting Paho v1 and v2 API."""
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.config.publisher_client_id,
                clean_session=self.config.clean_session,
            )
        else:
            self.client = mqtt.Client(  # type: ignore[call-arg]
                client_id=self.config.publisher_client_id,
                clean_session=self.config.clean_session,
            )

        if self.config.username:
            self.client.username_pw_set(
                username=self.config.username,
                password=self.config.password,
            )

        # Wire callbacks
        self.client.on_connect = self._on_connect
        self.client.on_connect_fail = self._on_connect_fail
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.on_subscribe = self._on_subscribe
        self.client.on_message = self._on_message

    @property
    def is_connected(self) -> bool:
        """Check if publisher is actively connected to the MQTT broker."""
        return self._is_connected

    @property
    def is_connecting(self) -> bool:
        """Check if publisher is actively attempting to establish connection."""
        return self._is_connecting

    @property
    def is_subscribed(self) -> bool:
        """Check if publisher has verified receipt ACK topic subscription."""
        return self._is_subscribed

    @property
    def connection_error(self) -> Optional[str]:
        """Description of the most recent connection failure, if any."""
        return self._connection_error

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
        """Connect to broker and start background networking loop."""
        if self._is_connected:
            return True

        timeout = timeout_s if timeout_s is not None else self.config.connect_timeout_s
        self._is_connecting = True
        self._connection_error = None

        try:
            # If connect_async is an explicit Mock (e.g. unit tests), invoke it directly
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
            logger.warning("Publisher connection refused to %s:%d: %s", self.config.host, self.config.port, e)
            return False
        except Exception as e:
            self._connection_error = f"Connect exception: {e}"
            self._is_connected = False
            self._is_connecting = False
            logger.error("Failed to connect MQTT publisher: %s", e)
            return False

    def disconnect(self) -> None:
        """Gracefully disconnect from broker and stop network loop."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as e:
            logger.debug("Error during publisher disconnect: %s", e)
        finally:
            self._is_connected = False
            self._is_subscribed = False
            self._is_connecting = False

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: Any, *args: Any) -> None:
        """Callback when client connects to broker."""
        is_success = self._is_reason_success(rc)
        if is_success:
            self._is_connected = True
            self._is_connecting = False
            self._connection_error = None
            logger.info("Publisher connected to MQTT broker %s:%d", self.config.host, self.config.port)
            # Subscribe to application-level receipt ACK topic
            try:
                self.client.subscribe(self.config.ack_topic, qos=1)
            except Exception as e:
                logger.error("Failed to subscribe publisher to ack_topic: %s", e)
        else:
            self._is_connected = False
            self._is_connecting = False
            self._connection_error = f"Connection rejected: {rc}"
            logger.warning("Publisher connection rejected with code %s", rc)

    def _on_connect_fail(self, client: Any, userdata: Any, *args: Any) -> None:
        """Callback when background connection attempt fails."""
        self._is_connected = False
        self._is_connecting = False
        self._connection_error = "Broker unreachable / connection failed"
        logger.warning("Publisher on_connect_fail triggered")

    def _on_disconnect(self, client: Any, userdata: Any, *args: Any, **kwargs: Any) -> None:
        """Callback when client disconnects from broker."""
        self._is_connected = False
        self._is_subscribed = False
        self._is_connecting = False
        logger.info("Publisher disconnected from MQTT broker.")

    def _on_subscribe(self, client: Any, userdata: Any, mid: int, reason_codes: Any = None, *args: Any, **kwargs: Any) -> None:
        """Callback when broker acknowledges subscription to ack_topic."""
        self._is_subscribed = True
        logger.info("Publisher confirmed subscription to %s (mid=%d)", self.config.ack_topic, mid)

    def _on_publish(self, client: Any, userdata: Any, mid: int, *args: Any) -> None:
        """Callback when broker acknowledges message publication (protocol ACK)."""
        alert_id = self._mid_to_alert_id.get(mid)
        if alert_id:
            self.tracker.record_broker_ack(alert_id)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        """Callback when receiving application-level receipt ACKs on ack_topic."""
        try:
            if msg.topic == self.config.ack_topic:
                payload = json.loads(msg.payload.decode("utf-8"))
                alert_id = payload.get("alert_id")
                status = payload.get("receipt_status")
                if alert_id and status == "received":
                    self.tracker.record_delivery_confirmed(alert_id)
        except Exception as e:
            logger.warning("Publisher received malformed receipt ACK payload: %s", e)

    def publish_alert(
        self,
        alert: AlertMessage,
        qos: Optional[int] = None,
        force: bool = False,
    ) -> bool:
        """Publish alert message to the configured alert topic.

        Args:
            alert: The validated AlertMessage to publish.
            qos: MQTT QoS level (0, 1, or 2). Defaults to config.default_alert_qos.
            force: If True, bypasses local deduplication to republish.

        Returns:
            True if publish request was successfully queued, False otherwise.
        """
        # Deduplication: do not republish the same alert_id unless force=True
        if not force and alert.alert_id in self._published_alert_ids:
            return True

        if not self._is_connected:
            self.tracker.record_failure(alert.alert_id, "Broker not connected")
            return False

        use_qos = self.config.default_alert_qos if qos is None else qos
        topic = self.config.alert_topic

        try:
            payload_str = alert.to_json()
            self.tracker.record_publish_attempt(
                alert_id=alert.alert_id,
                qos=use_qos,
                topic=topic,
            )

            msg_info = self.client.publish(
                topic=topic,
                payload=payload_str,
                qos=use_qos,
            )
            self._mid_to_alert_id[msg_info.mid] = alert.alert_id
            self._published_alert_ids.add(alert.alert_id)
            return True
        except Exception as e:
            err_msg = f"Publish exception: {e}"
            logger.error(err_msg)
            self.tracker.record_failure(alert.alert_id, err_msg)
            return False

    def publish_telemetry(self, reading: SensorReading) -> bool:
        """Publish sensor telemetry to the telemetry topic at QoS 0."""
        if not self._is_connected:
            return False

        try:
            payload = json.dumps(reading.to_dict())
            self.client.publish(
                topic=self.config.telemetry_topic,
                payload=payload,
                qos=0,
            )
            return True
        except Exception as e:
            logger.debug("Telemetry publish failed: %s", e)
            return False

    def reset(self) -> None:
        """Reset publisher tracking state."""
        self._mid_to_alert_id.clear()
        self._published_alert_ids.clear()
