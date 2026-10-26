"""Policy E — Static MQTT QoS 2.

Implements MQTT QoS 2 with exactly-once transport delivery guarantees via a 4-step handshake:
PUBLISH -> PUBREC -> PUBREL -> PUBCOMP.
Tracks protocol completion separately from subscriber receipt and application confirmation.
MQTT QoS 2 guarantees exactly-once protocol delivery between network peers,
not human or operator perception of an emergency notification.
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


class StaticQos2Policy(BaseAckPolicy):
    """Evaluates MQTT QoS 2 four-way handshake protocol strategy."""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        super().__init__(config=config)
        self.config.configured_qos = 2

    @property
    def policy_type(self) -> EvaluationPolicy:
        return EvaluationPolicy.STATIC_MQTT_QOS_2

    @property
    def corresponding_action(self) -> AckAction:
        return AckAction.MQTT_QOS_2

    def evaluate_controlled(
        self,
        message: AlertMessage,
        network: SimulatedNetworkModel,
    ) -> PolicyEvaluationResult:
        """Evaluate QoS 2 under network simulation with 4-way protocol handshake modeling."""
        attempt_id = f"{message.alert_id}_att1"
        payload_bytes = len(message.to_json().encode("utf-8"))

        # Step 1: PUBLISH -> Broker
        fwd_ok, leg1_lat = network.transmit_packet(traffic_class=message.priority, attempt=1)

        # Step 2: PUBREC <- Broker
        rec_ok, leg2_lat = network.transmit_ack(attempt=1) if fwd_ok else (False, 0.0)

        # Step 3: PUBREL -> Broker
        rel_ok, leg3_lat = network.transmit_packet(traffic_class=message.priority, attempt=1) if rec_ok else (False, 0.0)

        # Step 4: PUBCOMP <- Broker
        comp_ok, leg4_lat = network.transmit_ack(attempt=1) if rel_ok else (False, 0.0)

        protocol_confirmed = fwd_ok and rec_ok and rel_ok and comp_ok
        broker_handshake_lat = round(leg1_lat + leg2_lat + leg3_lat + leg4_lat, 2) if protocol_confirmed else None

        # Downstream subscriber receipt (broker routes exactly-once)
        subscriber_received = fwd_ok
        delivery_confirmed = protocol_confirmed and subscriber_received

        overhead = self.calculate_overhead_bytes(
            attempts=1,
            payload_bytes=payload_bytes,
            qos=2,
            has_local_ack=False,
        )

        e2e_lat = broker_handshake_lat if delivery_confirmed else None
        deadline_met = (e2e_lat <= self.config.deadline_ms) if e2e_lat is not None else False

        error_msg = None
        if not protocol_confirmed:
            error_msg = "QoS 2 4-way handshake failed due to packet loss in broker exchange"

        result = PolicyEvaluationResult(
            alert_id=message.alert_id,
            attempt_id=attempt_id,
            policy_name=self.policy_type.value,
            evaluation_mode="controlled_simulation",
            sensor_id=message.sensor_id,
            traffic_class=message.priority,
            severity=message.severity,
            configured_qos=2,
            publication_attempts=1,
            retry_count=0,
            broker_acknowledged=protocol_confirmed,
            local_acknowledged=False,
            subscriber_received=subscriber_received,
            delivery_confirmed=delivery_confirmed,
            delivery_status=DeliveryStatus.DELIVERY_CONFIRMED if delivery_confirmed else DeliveryStatus.FAILED,
            publish_latency_ms=round(leg1_lat + leg2_lat, 2) if (fwd_ok and rec_ok) else None,
            local_ack_latency_ms=None,
            subscriber_receipt_latency_ms=round(leg1_lat, 2) if fwd_ok else None,
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
        """Evaluate QoS 2 against live Mosquitto broker."""
        attempt_id = f"{message.alert_id}_att1"
        payload_bytes = len(message.to_json().encode("utf-8"))

        t_start = time.time()
        pub_ok = mqtt_manager.publish_alert(message, qos=2)
        pub_latency_ms = (time.time() - t_start) * 1000.0

        time.sleep(0.12)  # Allow 4-way handshake and subscriber dispatch
        broker_acked = False
        sub_received = False
        confirmed = False
        e2e_lat: Optional[float] = None

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

        overhead = self.calculate_overhead_bytes(
            attempts=1,
            payload_bytes=payload_bytes,
            qos=2,
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
            configured_qos=2,
            publication_attempts=1,
            retry_count=0,
            broker_acknowledged=broker_acked,
            local_acknowledged=False,
            subscriber_received=sub_received,
            delivery_confirmed=confirmed,
            delivery_status=DeliveryStatus.DELIVERY_CONFIRMED if confirmed else (DeliveryStatus.PUBLISHED if pub_ok else DeliveryStatus.FAILED),
            publish_latency_ms=round(pub_latency_ms, 2),
            local_ack_latency_ms=None,
            subscriber_receipt_latency_ms=e2e_lat,
            end_to_end_latency_ms=e2e_lat,
            message_overhead_bytes=overhead,
            deadline_met=deadline_met,
            error_message=None if pub_ok else "QoS 2 publish failed",
            simulated=False,
            simulation_timestamp=time.time(),
        )

        self._results_history.append(result)
        return result
