"""Policy C — Static MQTT QoS 1 with Application-Level Retries.

Tracks transport-layer broker PUBACK separately from end-to-end subscriber receipts.
When application receipt confirmation is not received within ack_timeout_ms, bounded
retransmissions are executed up to max_retries. Deduplication prevents multiple receipts
from distorting logical alert counts.
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


class StaticQos1RetryPolicy(BaseAckPolicy):
    """Evaluates MQTT QoS 1 with bounded application-level retransmissions."""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        super().__init__(config=config)
        self.config.configured_qos = 1

    @property
    def policy_type(self) -> EvaluationPolicy:
        return EvaluationPolicy.STATIC_MQTT_QOS_1

    @property
    def corresponding_action(self) -> AckAction:
        return AckAction.MQTT_QOS_1

    def evaluate_controlled(
        self,
        message: AlertMessage,
        network: SimulatedNetworkModel,
    ) -> PolicyEvaluationResult:
        """Evaluate QoS 1 with bounded retransmissions under controlled network simulation."""
        payload_bytes = len(message.to_json().encode("utf-8"))

        attempts = 0
        subscriber_received = False
        delivery_confirmed = False
        broker_acked = False
        total_latency_ms = 0.0
        final_fwd_latency: Optional[float] = None
        attempt_id = f"{message.alert_id}_att1"

        max_attempts = max(1, 1 + self.config.max_retries)

        for attempt in range(1, max_attempts + 1):
            attempts += 1
            attempt_id = f"{message.alert_id}_att{attempt}"

            # Broker PUBACK at transport layer (QoS 1)
            broker_acked = True

            # Forward transmission to subscriber
            delivered, fwd_lat = network.transmit_packet(
                traffic_class=message.priority,
                attempt=attempt,
            )

            # Reverse application receipt ACK
            if delivered:
                subscriber_received = True
                final_fwd_latency = fwd_lat
                ack_returned, rev_lat = network.transmit_ack(attempt=attempt)
                if ack_returned:
                    # Successful delivery and application ACK confirmed
                    delivery_confirmed = True
                    total_latency_ms += (fwd_lat + rev_lat)
                    break

            # If unconfirmed and more attempts remain, add retry timeout delay
            if attempt < max_attempts:
                total_latency_ms += self.config.ack_timeout_ms

        retry_count = attempts - 1
        overhead = self.calculate_overhead_bytes(
            attempts=attempts,
            payload_bytes=payload_bytes,
            qos=1,
            has_local_ack=False,
        )

        e2e_lat = round(total_latency_ms, 2) if delivery_confirmed else None
        deadline_met = (e2e_lat <= self.config.deadline_ms) if e2e_lat is not None else False

        error_msg = None
        if not delivery_confirmed:
            error_msg = f"Delivery unconfirmed after {attempts} attempts ({retry_count} retries exhausted)"

        status = DeliveryStatus.DELIVERY_CONFIRMED if delivery_confirmed else (
            DeliveryStatus.RECEIVED if subscriber_received else DeliveryStatus.TIMEOUT
        )

        result = PolicyEvaluationResult(
            alert_id=message.alert_id,
            attempt_id=attempt_id,
            policy_name=self.policy_type.value,
            evaluation_mode="controlled_simulation",
            sensor_id=message.sensor_id,
            traffic_class=message.priority,
            severity=message.severity,
            configured_qos=1,
            publication_attempts=attempts,
            retry_count=retry_count,
            broker_acknowledged=broker_acked,
            local_acknowledged=False,
            subscriber_received=subscriber_received,
            delivery_confirmed=delivery_confirmed,
            delivery_status=status,
            publish_latency_ms=2.0,  # Nominal broker PUBACK latency
            local_ack_latency_ms=None,
            subscriber_receipt_latency_ms=final_fwd_latency,
            end_to_end_latency_ms=e2e_lat,
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
        """Evaluate QoS 1 strategy in live MQTT mode."""
        payload_bytes = len(message.to_json().encode("utf-8"))
        attempts = 0
        confirmed = False
        sub_received = False
        broker_acked = False
        e2e_lat: Optional[float] = None
        attempt_id = f"{message.alert_id}_att1"

        max_attempts = max(1, 1 + self.config.max_retries)
        t_start_e2e = time.time()

        for attempt in range(1, max_attempts + 1):
            attempts += 1
            attempt_id = f"{message.alert_id}_att{attempt}"

            pub_ok = mqtt_manager.publish_alert(message, qos=1)
            time.sleep(0.08)

            if hasattr(mqtt_manager, "tracker"):
                rec = mqtt_manager.tracker.records.get(message.alert_id)
                if rec:
                    if rec.broker_ack_time:
                        broker_acked = True
                    if rec.subscriber_receipt_time:
                        sub_received = True
                    if rec.delivery_confirmed_time or rec.status == DeliveryStatus.DELIVERY_CONFIRMED:
                        confirmed = True
                        e2e_lat = rec.end_to_end_latency_ms
                        break

            if attempt < max_attempts and not confirmed:
                time.sleep(self.config.ack_timeout_ms / 1000.0)

        retry_count = attempts - 1
        overhead = self.calculate_overhead_bytes(
            attempts=attempts,
            payload_bytes=payload_bytes,
            qos=1,
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
            configured_qos=1,
            publication_attempts=attempts,
            retry_count=retry_count,
            broker_acknowledged=broker_acked,
            local_acknowledged=False,
            subscriber_received=sub_received,
            delivery_confirmed=confirmed,
            delivery_status=DeliveryStatus.DELIVERY_CONFIRMED if confirmed else DeliveryStatus.TIMEOUT,
            publish_latency_ms=2.0,
            local_ack_latency_ms=None,
            subscriber_receipt_latency_ms=e2e_lat,
            end_to_end_latency_ms=e2e_lat,
            message_overhead_bytes=overhead,
            deadline_met=deadline_met,
            error_message=None if confirmed else "Live receipt ACK timeout",
            simulated=False,
            simulation_timestamp=time.time(),
        )

        self._results_history.append(result)
        return result
