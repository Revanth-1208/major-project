"""Base abstractions and data models for Phase 5 ACK Policy Engine.

Defines the contract for fixed acknowledgment policies, evaluation results,
and policy configurations across controlled simulations and live MQTT execution.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any, Dict, List, Optional

from rl_ack.common.enums import (
    AckAction,
    AlertSeverity,
    DeliveryStatus,
    EvaluationPolicy,
    TrafficClass,
)
from rl_ack.common.interfaces import IAckPolicy, IMqttManager
from rl_ack.common.models import AlertMessage, NetworkStateVector


@dataclass
class PolicyConfig:
    """Configurable operational parameters for ACK policies."""

    ack_timeout_ms: float = 150.0
    max_retries: int = 3
    deadline_ms: float = 100.0
    local_ack_delay_ms: float = 2.0
    configured_qos: int = 0
    priority_selective_retransmit: bool = True


@dataclass
class PolicyEvaluationResult:
    """Structured evaluation record for an individual message processed under a policy.

    Fulfills Phase 5 Section 6 specification.
    """

    alert_id: str
    attempt_id: str
    policy_name: str
    evaluation_mode: str  # "controlled_simulation" or "live_mqtt"
    sensor_id: str
    traffic_class: TrafficClass
    severity: AlertSeverity
    configured_qos: int
    publication_attempts: int
    retry_count: int
    broker_acknowledged: bool
    local_acknowledged: bool
    subscriber_received: bool
    delivery_confirmed: bool
    delivery_status: DeliveryStatus
    publish_latency_ms: Optional[float] = None
    local_ack_latency_ms: Optional[float] = None
    subscriber_receipt_latency_ms: Optional[float] = None
    end_to_end_latency_ms: Optional[float] = None
    message_overhead_bytes: int = 0
    deadline_met: Optional[bool] = None
    error_message: Optional[str] = None
    simulated: bool = True
    simulation_timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary with serialized enums."""
        tc_val = self.traffic_class.value if hasattr(self.traffic_class, "value") else str(self.traffic_class)
        sev_val = self.severity.value if hasattr(self.severity, "value") else str(self.severity)
        stat_val = self.delivery_status.value if hasattr(self.delivery_status, "value") else str(self.delivery_status)

        return {
            "alert_id": self.alert_id,
            "attempt_id": self.attempt_id,
            "policy_name": self.policy_name,
            "evaluation_mode": self.evaluation_mode,
            "sensor_id": self.sensor_id,
            "traffic_class": tc_val,
            "severity": sev_val,
            "configured_qos": self.configured_qos,
            "publication_attempts": self.publication_attempts,
            "retry_count": self.retry_count,
            "broker_acknowledged": self.broker_acknowledged,
            "local_acknowledged": self.local_acknowledged,
            "subscriber_received": self.subscriber_received,
            "delivery_confirmed": self.delivery_confirmed,
            "delivery_status": stat_val,
            "publish_latency_ms": self.publish_latency_ms,
            "local_ack_latency_ms": self.local_ack_latency_ms,
            "subscriber_receipt_latency_ms": self.subscriber_receipt_latency_ms,
            "end_to_end_latency_ms": self.end_to_end_latency_ms,
            "message_overhead_bytes": self.message_overhead_bytes,
            "deadline_met": self.deadline_met,
            "error_message": self.error_message,
            "simulated": self.simulated,
            "simulation_timestamp": self.simulation_timestamp,
        }

    def to_json(self) -> str:
        """Serialize result to JSON string."""
        return json.dumps(self.to_dict())


class BaseAckPolicy(IAckPolicy, ABC):
    """Abstract base class implementing the Phase 5 ACK policy contract."""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()
        self._results_history: List[PolicyEvaluationResult] = []

    @property
    @abstractmethod
    def policy_type(self) -> EvaluationPolicy:
        """Return the evaluation policy identifier."""
        pass

    @property
    @abstractmethod
    def corresponding_action(self) -> AckAction:
        """Return corresponding fixed AckAction for this policy."""
        pass

    def select_action(self, state: NetworkStateVector) -> AckAction:
        """Return the fixed strategy action regardless of state."""
        return self.corresponding_action

    @abstractmethod
    def evaluate_controlled(
        self,
        message: AlertMessage,
        network: Any,  # SimulatedNetworkModel
    ) -> PolicyEvaluationResult:
        """Evaluate acknowledgment strategy under controlled simulated network model."""
        pass

    @abstractmethod
    def evaluate_live(
        self,
        message: AlertMessage,
        mqtt_manager: IMqttManager,
        timeout_s: float = 2.0,
    ) -> PolicyEvaluationResult:
        """Evaluate acknowledgment strategy using live MQTT broker connection."""
        pass

    def calculate_overhead_bytes(
        self,
        attempts: int,
        payload_bytes: int,
        qos: int,
        has_local_ack: bool = False,
    ) -> int:
        """Calculate total transmitted protocol and payload byte overhead.

        MQTT packet estimation:
        - Fixed + Variable header & topic overhead: ~28 bytes per PUBLISH
        - QoS 0: PUBLISH packet (payload + 28 bytes)
        - QoS 1: PUBLISH packet + PUBACK (4 bytes) per attempt
        - QoS 2: PUBLISH + PUBREC(4) + PUBREL(4) + PUBCOMP(4) = payload + 40 bytes per attempt
        - Local ACK: +20 bytes local proxy acknowledgment packet
        - Application-level ACK: +64 bytes JSON receipt packet
        """
        publish_overhead = 28
        base_packet = payload_bytes + publish_overhead

        if qos == 0:
            attempt_bytes = base_packet
        elif qos == 1:
            attempt_bytes = base_packet + 4  # PUBACK
        else:  # qos == 2
            attempt_bytes = base_packet + 12  # PUBREC, PUBREL, PUBCOMP

        total = attempts * attempt_bytes
        if has_local_ack:
            total += 20  # Local proxy ACK overhead
        return total

    def get_results(self) -> List[PolicyEvaluationResult]:
        """Return chronological evaluation results for this policy."""
        return list(self._results_history)

    def reset(self) -> None:
        """Reset historical evaluation state."""
        self._results_history.clear()
