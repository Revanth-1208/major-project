"""Policy B — Static Local ACK.

Simulates immediate local acknowledgment from a proximate edge proxy node.
The local proxy acknowledges transmission locally, decoupling transmitter wait-time
from downstream subscriber reception. Local ACK is explicitly distinct from end-to-end confirmation.
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


class StaticLocalAckPolicy(BaseAckPolicy):
    """Evaluates edge proxy local acknowledgment without retransmissions."""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        super().__init__(config=config)
        self.config.configured_qos = 0

    @property
    def policy_type(self) -> EvaluationPolicy:
        return EvaluationPolicy.STATIC_LOCAL_ACK

    @property
    def corresponding_action(self) -> AckAction:
        return AckAction.LOCAL_ACK

    def evaluate_controlled(
        self,
        message: AlertMessage,
        network: SimulatedNetworkModel,
    ) -> PolicyEvaluationResult:
        """Evaluate Local ACK under controlled network conditions."""
        attempt_id = f"{message.alert_id}_att1"
        payload_bytes = len(message.to_json().encode("utf-8"))

        # 1. Local Edge Proxy Acknowledgment
        local_acked, local_ack_lat = network.query_local_proxy()

        # 2. Downstream network transmission to subscriber
        delivered, fwd_lat = network.transmit_packet(
            traffic_class=message.priority,
            attempt=1,
        )

        # 3. Downstream receipt acknowledgment return
        ack_returned, rev_lat = network.transmit_ack(attempt=1) if delivered else (False, 0.0)

        # End-to-end delivery confirmed ONLY when downstream subscriber receipt is confirmed
        e2e_confirmed = delivered and ack_returned
        e2e_lat = round(fwd_lat + rev_lat, 2) if e2e_confirmed else None

        overhead = self.calculate_overhead_bytes(
            attempts=1,
            payload_bytes=payload_bytes,
            qos=0,
            has_local_ack=True,
        )

        deadline_met = (e2e_lat <= self.config.deadline_ms) if e2e_lat is not None else False
        error_msg = None
        if not delivered:
            error_msg = "Packet lost in transit; local proxy ACK recorded but downstream delivery failed"
        elif not ack_returned:
            error_msg = "Subscriber received message but reverse ACK was lost in transit"

        # Determine exact delivery status
        if e2e_confirmed:
            status = DeliveryStatus.DELIVERY_CONFIRMED
        elif delivered:
            status = DeliveryStatus.RECEIVED
        elif local_acked:
            status = DeliveryStatus.ACKNOWLEDGED  # Local ACK only
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
            configured_qos=0,
            publication_attempts=1,
            retry_count=0,
            broker_acknowledged=False,
            local_acknowledged=local_acked,
            subscriber_received=delivered,
            delivery_confirmed=e2e_confirmed,
            delivery_status=status,
            publish_latency_ms=1.5,
            local_ack_latency_ms=local_ack_lat if local_acked else None,
            subscriber_receipt_latency_ms=fwd_lat if delivered else None,
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
        """Evaluate Local ACK strategy in live MQTT mode."""
        attempt_id = f"{message.alert_id}_att1"
        payload_bytes = len(message.to_json().encode("utf-8"))

        # Immediate local proxy ACK emulation
        t_local_start = time.time()
        local_ack_lat_ms = round(self.config.local_ack_delay_ms, 2)
        local_acked = True

        # Publish over MQTT
        pub_ok = mqtt_manager.publish_alert(message, qos=0)

        # Await subscriber receipt
        time.sleep(0.08)
        subscriber_rec = False
        e2e_confirmed = False
        e2e_lat: Optional[float] = None

        if hasattr(mqtt_manager, "tracker"):
            rec = mqtt_manager.tracker.records.get(message.alert_id)
            if rec and rec.subscriber_receipt_time:
                subscriber_rec = True
                if rec.delivery_confirmed_time or rec.status == DeliveryStatus.DELIVERY_CONFIRMED:
                    e2e_confirmed = True
                e2e_lat = rec.end_to_end_latency_ms

        overhead = self.calculate_overhead_bytes(
            attempts=1,
            payload_bytes=payload_bytes,
            qos=0,
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
            configured_qos=0,
            publication_attempts=1,
            retry_count=0,
            broker_acknowledged=False,
            local_acknowledged=local_acked,
            subscriber_received=subscriber_rec,
            delivery_confirmed=e2e_confirmed,
            delivery_status=DeliveryStatus.DELIVERY_CONFIRMED if e2e_confirmed else (DeliveryStatus.ACKNOWLEDGED if local_acked else DeliveryStatus.FAILED),
            publish_latency_ms=1.5,
            local_ack_latency_ms=local_ack_lat_ms,
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
