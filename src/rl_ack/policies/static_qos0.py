"""Policy A — Static MQTT QoS 0.

Fire-and-forget delivery strategy without protocol acknowledgments or application retries.
Publication success merely indicates local socket dispatch, not subscriber delivery.
"""

import time
from typing import Optional

from rl_ack.common.enums import (
    AckAction,
    DeliveryStatus,
    EvaluationPolicy,
)
from rl_ack.common.interfaces import IMqttManager
from rl_ack.common.models import AlertMessage
from rl_ack.policies.base import BaseAckPolicy, PolicyConfig, PolicyEvaluationResult
from rl_ack.policies.network_model import SimulatedNetworkModel


class StaticQos0Policy(BaseAckPolicy):
    """Evaluates unconfirmed fire-and-forget transmission using MQTT QoS 0."""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        super().__init__(config=config)
        self.config.configured_qos = 0

    @property
    def policy_type(self) -> EvaluationPolicy:
        return EvaluationPolicy.STATIC_MQTT_QOS_0

    @property
    def corresponding_action(self) -> AckAction:
        return AckAction.MQTT_QOS_0

    def evaluate_controlled(
        self,
        message: AlertMessage,
        network: SimulatedNetworkModel,
    ) -> PolicyEvaluationResult:
        """Evaluate QoS 0 under deterministic channel simulation."""
        attempt_id = f"{message.alert_id}_att1"
        payload_bytes = len(message.to_json().encode("utf-8"))

        # Single fire-and-forget forward attempt
        delivered, latency_ms = network.transmit_packet(
            traffic_class=message.priority,
            attempt=1,
        )

        overhead = self.calculate_overhead_bytes(
            attempts=1,
            payload_bytes=payload_bytes,
            qos=0,
            has_local_ack=False,
        )

        deadline_met = (latency_ms <= self.config.deadline_ms) if delivered else False
        error_msg = None if delivered else "Packet lost in forward channel (QoS 0 fire-and-forget)"

        result = PolicyEvaluationResult(
            alert_id=message.alert_id,
            attempt_id=attempt_id,
            policy_name=self.policy_type.value,
            evaluation_mode="controlled_simulation",
            sensor_id=message.sensor_id,
            traffic_class=message.priority,
            severity=message.severity,
            configured_qos=0,
            publication_attempts=1,
            retry_count=0,
            broker_acknowledged=False,  # QoS 0 has no protocol PUBACK
            local_acknowledged=False,
            subscriber_received=delivered,
            delivery_confirmed=delivered,
            delivery_status=DeliveryStatus.DELIVERED if delivered else DeliveryStatus.FAILED,
            publish_latency_ms=1.5,  # Nominal local enqueue delay
            local_ack_latency_ms=None,
            subscriber_receipt_latency_ms=latency_ms if delivered else None,
            end_to_end_latency_ms=latency_ms if delivered else None,
            message_overhead_bytes=overhead,
            deadline_met=deadline_met,
            error_message=error_msg,
            simulated=True,
            simulation_timestamp=time.time(),
        )

        self._results_history.append(result)
        return result

    def evaluate_live(
        self,
        message: AlertMessage,
        mqtt_manager: IMqttManager,
        timeout_s: float = 2.0,
    ) -> PolicyEvaluationResult:
        """Evaluate QoS 0 against live Mosquitto broker."""
        attempt_id = f"{message.alert_id}_att1"
        payload_bytes = len(message.to_json().encode("utf-8"))

        t_start = time.time()
        pub_ok = mqtt_manager.publish_alert(message, qos=0)
        pub_latency_ms = (time.time() - t_start) * 1000.0

        # In QoS 0 live mode, check if subscriber recorded receipt
        time.sleep(0.08)  # Brief delay for loop delivery
        subscriber_rec = False
        e2e_lat: Optional[float] = None

        if hasattr(mqtt_manager, "tracker"):
            rec = mqtt_manager.tracker.records.get(message.alert_id)
            if rec and rec.subscriber_receipt_time:
                subscriber_rec = True
                e2e_lat = rec.end_to_end_latency_ms

        overhead = self.calculate_overhead_bytes(
            attempts=1,
            payload_bytes=payload_bytes,
            qos=0,
            has_local_ack=False,
        )

        deadline_met = (e2e_lat <= self.config.deadline_ms) if e2e_lat is not None else False

        result = PolicyEvaluationResult(
            alert_id=message.alert_id,
            attempt_id=attempt_id,
            policy_name=self.policy_type.value,
            evaluation_mode="live_mqtt",
            sensor_id=message.sensor_id,
            traffic_class=message.priority,
            severity=message.severity,
            configured_qos=0,
            publication_attempts=1,
            retry_count=0,
            broker_acknowledged=False,
            local_acknowledged=False,
            subscriber_received=subscriber_rec,
            delivery_confirmed=subscriber_rec,
            delivery_status=DeliveryStatus.DELIVERED if subscriber_rec else (DeliveryStatus.PUBLISHED if pub_ok else DeliveryStatus.FAILED),
            publish_latency_ms=round(pub_latency_ms, 2),
            local_ack_latency_ms=None,
            subscriber_receipt_latency_ms=e2e_lat,
            end_to_end_latency_ms=e2e_lat,
            message_overhead_bytes=overhead,
            deadline_met=deadline_met,
            error_message=None if pub_ok else "Publish failed",
            simulated=False,
            simulation_timestamp=time.time(),
        )

        self._results_history.append(result)
        return result
