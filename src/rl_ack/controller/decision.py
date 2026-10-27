"""Decision Record Data Model for Module 8: DQN Controller Integration.

Encapsulates individual live ACK policy selection decisions produced by the DQN controller,
including the 8-feature input state, model Q-values, selected action, safety override status,
and fallback metadata.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional


@dataclass
class DqnDecisionRecord:
    """Structured audit record of an ACK strategy decision made for an emergency alert."""

    decision_id: str
    alert_id: str
    sensor_id: str
    traffic_class: str
    severity: str
    reading_value: float
    threshold_value: float
    unit: str
    state_vector: List[float]  # 8 normalized continuous features
    q_values: Optional[List[float]]  # 5 Q-values from model inference (None if fallback)
    action_idx: int  # 0..4
    action_label: str  # e.g. "MQTT QoS 1 (Retries)"
    policy_name: str  # e.g. "Static MQTT QoS 1"
    configured_qos: int  # 0, 1, or 2
    is_fallback: bool  # True if model was unready or fallback was triggered
    fallback_reason: Optional[str] = None
    safety_override: bool = False  # True if safety constraints adjusted the policy
    model_id: str = "UNLOADED"  # Checkpoint filename or model identifier
    timestamp_s: float = 0.0  # Monotonic simulation clock
    timestamp_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Phase 7: Delivery Outcome Tracking
    delivery_status: str = "PENDING"
    delivery_confirmed: bool = False
    broker_acknowledged: bool = False
    subscriber_received: bool = False
    end_to_end_latency_ms: Optional[float] = None
    retry_count: int = 0
    overhead_bytes: int = 0
    failure_reason: Optional[str] = None
    evaluation_mode: str = "live_mqtt"

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision record to dictionary."""
        return {
            "decision_id": self.decision_id,
            "alert_id": self.alert_id,
            "sensor_id": self.sensor_id,
            "traffic_class": self.traffic_class,
            "severity": self.severity,
            "reading_value": round(self.reading_value, 2),
            "threshold_value": round(self.threshold_value, 2),
            "unit": self.unit,
            "state_vector": [round(x, 4) for x in self.state_vector],
            "q_values": [round(q, 4) for q in self.q_values] if self.q_values is not None else None,
            "action_idx": self.action_idx,
            "action_label": self.action_label,
            "policy_name": self.policy_name,
            "configured_qos": self.configured_qos,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "safety_override": self.safety_override,
            "model_id": self.model_id,
            "timestamp_s": round(self.timestamp_s, 2),
            "timestamp_iso": self.timestamp_iso,
            "delivery_status": self.delivery_status,
            "delivery_confirmed": self.delivery_confirmed,
            "broker_acknowledged": self.broker_acknowledged,
            "subscriber_received": self.subscriber_received,
            "end_to_end_latency_ms": round(self.end_to_end_latency_ms, 2) if self.end_to_end_latency_ms is not None else None,
            "retry_count": self.retry_count,
            "overhead_bytes": self.overhead_bytes,
            "failure_reason": self.failure_reason,
            "evaluation_mode": self.evaluation_mode,
        }

    def to_json(self) -> str:
        """Serialize decision record to JSON string."""
        return json.dumps(self.to_dict())
