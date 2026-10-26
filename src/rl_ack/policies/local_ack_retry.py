"""Policy D — Local ACK plus Selective Retransmission.

A proximate edge proxy provides immediate local acknowledgment to minimize latency.
Downstream delivery is continuously monitored, and bounded retransmission is applied
selectively: Critical Control (CC) emergency alerts undergo active retransmission,
whereas routine streaming (HBS) and best-effort (BET) telemetry avoid wasteful retries.
"""

import time
from typing import Optional

from rl_ack.common.enums import (
    AckAction,
    DeliveryStatus,
    EvaluationPolicy,
    TrafficClass,
)
from rl_ack.common.interfaces import IMqttManager
from rl_ack.common.models import AlertMessage
from rl_ack.policies.base import BaseAckPolicy, PolicyConfig, PolicyEvaluationResult
from rl_ack.policies.network_model import SimulatedNetworkModel


class LocalAckSelectiveRetryPolicy(BaseAckPolicy):
    """Evaluates Local ACK with selective retransmissions prioritized by traffic class."""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        super().__init__(config=config)
        self.config.configured_qos = 1

    @property
    def policy_type(self) -> EvaluationPolicy:
        return EvaluationPolicy.LOCAL_ACK_SELECTIVE_RETRANSMIT

    @property
    def corresponding_action(self) -> AckAction:
        return AckAction.LOCAL_ACK_SELECTIVE_RETRANSMIT

    def evaluate_controlled(
        self,
        message: AlertMessage,
        network: SimulatedNetworkModel,
    ) -> PolicyEvaluationResult:
        """Evaluate Local ACK plus selective retransmissions under network simulation."""
        payload_bytes = len(message.to_json().encode("utf-8"))

        # 1. Immediate local proxy ACK
        local_acked, local_ack_lat = network.query_local_proxy()

        # 2. Determine retransmission policy by traffic class
        is_critical = (message.priority == TrafficClass.CRITICAL_CONTROL)
        max_attempts = max(1, 1 + self.config.max_retries) if is_critical else 1

        attempts = 0
        subscriber_received = False
        delivery_confirmed = False
        total_latency_ms = 0.0
        final_fwd_latency: Optional[float] = None
        attempt_id = f"{message.alert_id}_att1"

        for attempt in range(1, max_attempts + 1):
            attempts += 1
            attempt_id = f"{message.alert_id}_att{attempt}"

            # Forward packet transmission
            delivered, fwd_lat = network.transmit_packet(
                traffic_class=message.priority,
                attempt=attempt,
            )

            if delivered:
                subscriber_received = True
                final_fwd_latency = fwd_lat
                ack_returned, rev_lat = network.transmit_ack(attempt=attempt)
                if ack_returned:
                    delivery_confirmed = True
                    total_latency_ms += (fwd_lat + rev_lat)
                    break

            if attempt < max_attempts:
                total_latency_ms += self.config.ack_timeout_ms

        retry_count = attempts - 1
        overhead = self.calculate_overhead_bytes(
            attempts=attempts,
            payload_bytes=payload_bytes,
            qos=1 if is_critical else 0,
            has_local_ack=True,
        )

        e2e_lat = round(total_latency_ms, 2) if delivery_confirmed else None
        deadline_met = (e2e_lat <= self.config.deadline_ms) if e2e_lat is not None else False

        error_msg = None
        if not delivery_confirmed:
            if is_critical:
                error_msg = f"Critical alert unconfirmed after {attempts} attempts ({retry_count} retries exhausted)"
            else:
                error_msg = "Non-critical telemetry lost; selective retransmission suppressed"

        if delivery_confirmed:
            status = DeliveryStatus.DELIVERY_CONFIRMED
        elif subscriber_received:
            status = DeliveryStatus.RECEIVED
        elif local_acked:
            status = DeliveryStatus.ACKNOWLEDGED  # Local proxy ACK only
        else:
            status = DeliveryStatus.FAILED

        result = PolicyEvaluationResult(
            alert_id=message.alert_id,
            attempt_id=attempt_id,
            policy_name=self.policy_type.value,
            evaluation_mode="controlled_simulation",
            sensor_id=message.sensor_id,
            traffic_class=message.priority,
            severity=message.severity,
            configured_qos=1 if is_critical else 0,
            publication_attempts=attempts,
            retry_count=retry_count,
            broker_acknowledged=False,
            local_acknowledged=local_acked,
            subscriber_received=subscriber_received,
            delivery_confirmed=delivery_confirmed,
            delivery_status=status,
            publish_latency_ms=1.5,
            local_ack_latency_ms=local_ack_lat if local_acked else None,
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
        """Evaluate Local ACK plus selective retransmissions in live MQTT mode."""
        payload_bytes = len(message.to_json().encode("utf-8"))
        is_critical = (message.priority == TrafficClass.CRITICAL_CONTROL)
        max_attempts = max(1, 1 + self.config.max_retries) if is_critical else 1

        local_acked = True
        local_ack_lat_ms = round(self.config.local_ack_delay_ms, 2)

        attempts = 0
        confirmed = False
        sub_received = False
        e2e_lat: Optional[float] = None
        attempt_id = f"{message.alert_id}_att1"

        for attempt in range(1, max_attempts + 1):
            attempts += 1
            attempt_id = f"{message.alert_id}_att{attempt}"

            pub_ok = mqtt_manager.publish_alert(message, qos=1 if is_critical else 0)
            time.sleep(0.08)

            if hasattr(mqtt_manager, "tracker"):
                rec = mqtt_manager.tracker.records.get(message.alert_id)
                if rec:
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
            qos=1 if is_critical else 0,
            has_local_ack=True,
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
            configured_qos=1 if is_critical else 0,
            publication_attempts=attempts,
            retry_count=retry_count,
            broker_acknowledged=False,
            local_acknowledged=local_acked,
            subscriber_received=sub_received,
            delivery_confirmed=confirmed,
            delivery_status=DeliveryStatus.DELIVERY_CONFIRMED if confirmed else (DeliveryStatus.ACKNOWLEDGED if local_acked else DeliveryStatus.FAILED),
            publish_latency_ms=1.5,
            local_ack_latency_ms=local_ack_lat_ms,
            subscriber_receipt_latency_ms=e2e_lat,
            end_to_end_latency_ms=e2e_lat,
            message_overhead_bytes=overhead,
            deadline_met=deadline_met,
            error_message=None if confirmed else "Live receipt ACK unconfirmed",
            simulated=False,
            simulation_timestamp=time.time(),
        )

        self._results_history.append(result)
        return result
