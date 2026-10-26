"""Data models and transfer objects for the RL-ACK project."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from rl_ack.common.enums import (
    AckAction,
    AlertSeverity,
    DeliveryStatus,
    NodeOperationalStatus,
    SensorType,
    TrafficClass,
)


@dataclass
class SensorReading:
    """Individual virtual sensor telemetry record."""
    sensor_id: str
    host_node: str
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: float
    operational_status: NodeOperationalStatus = NodeOperationalStatus.NORMAL
    is_threshold_exceeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "host_node": self.host_node,
            "sensor_type": self.sensor_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "operational_status": self.operational_status.value,
            "is_threshold_exceeded": self.is_threshold_exceeded,
        }


@dataclass
class AlertMessage:
    """Distinct incident alert payload transmitted over MQTT."""
    alert_id: str
    sensor_id: str
    host_node: str
    sensor_type: SensorType
    severity: AlertSeverity
    traffic_class: TrafficClass
    reading_value: float
    threshold_value: float
    timestamp: float
    status: DeliveryStatus = DeliveryStatus.GENERATED
    chosen_action: Optional[AckAction] = None
    retry_count: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "sensor_id": self.sensor_id,
            "host_node": self.host_node,
            "sensor_type": self.sensor_type.value,
            "severity": self.severity.value,
            "traffic_class": self.traffic_class.value,
            "reading_value": self.reading_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "chosen_action": self.chosen_action.value if self.chosen_action is not None else None,
            "retry_count": self.retry_count,
            "payload": self.payload,
        }


@dataclass
class AckEvent:
    """Tracked lifecycle event for delivery verification."""
    event_id: str
    alert_id: str
    event_type: str
    timestamp: float
    latency_ms: Optional[float] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "alert_id": self.alert_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "details": self.details,
        }


@dataclass
class NetworkStateVector:
    """Eight-feature state representation for the DQN agent.

    Features:
    1. Network utilization [0.0, 1.0]
    2. Rolling ACK rate [0.0, 1.0]
    3. Transmit queue occupancy [0.0, 1.0]
    4. Critical Control one-hot feature (1.0 or 0.0)
    5. High Bandwidth Streaming one-hot feature (1.0 or 0.0)
    6. Best Effort one-hot feature (1.0 or 0.0)
    7. Remaining node energy [0.0, 1.0]
    8. Rolling latency (normalized, e.g. [0.0, 1.0])
    """
    network_utilization: float
    rolling_ack_rate: float
    transmit_queue_occupancy: float
    traffic_critical_control: float
    traffic_high_bandwidth_streaming: float
    traffic_best_effort: float
    remaining_node_energy: float
    rolling_latency: float

    FEATURE_COUNT: int = 8

    def to_list(self) -> List[float]:
        return [
            float(self.network_utilization),
            float(self.rolling_ack_rate),
            float(self.transmit_queue_occupancy),
            float(self.traffic_critical_control),
            float(self.traffic_high_bandwidth_streaming),
            float(self.traffic_best_effort),
            float(self.remaining_node_energy),
            float(self.rolling_latency),
        ]

    def to_numpy(self) -> np.ndarray:
        arr = np.array(self.to_list(), dtype=np.float32)
        assert arr.shape == (8,), f"Expected shape (8,), got {arr.shape}"
        return arr

    @classmethod
    def from_features(
        cls,
        network_utilization: float,
        rolling_ack_rate: float,
        transmit_queue_occupancy: float,
        traffic_class: TrafficClass,
        remaining_node_energy: float,
        rolling_latency: float,
    ) -> "NetworkStateVector":
        """Construct state vector with one-hot encoded traffic class."""
        c_ctrl, c_hb, c_be = traffic_class.to_one_hot()
        return cls(
            network_utilization=max(0.0, min(1.0, float(network_utilization))),
            rolling_ack_rate=max(0.0, min(1.0, float(rolling_ack_rate))),
            transmit_queue_occupancy=max(0.0, min(1.0, float(transmit_queue_occupancy))),
            traffic_critical_control=c_ctrl,
            traffic_high_bandwidth_streaming=c_hb,
            traffic_best_effort=c_be,
            remaining_node_energy=max(0.0, min(1.0, float(remaining_node_energy))),
            rolling_latency=max(0.0, float(rolling_latency)),
        )


@dataclass
class ExperimentRecord:
    """Summary record for policy comparative evaluations."""
    experiment_id: str
    simulation_id: str
    policy_name: str
    timestamp: float
    total_alerts: int
    delivered_alerts: int
    failed_alerts: int
    delivery_rate: float
    avg_latency_ms: float
    avg_energy_consumed: float
    total_retransmissions: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "simulation_id": self.simulation_id,
            "policy_name": self.policy_name,
            "timestamp": self.timestamp,
            "total_alerts": self.total_alerts,
            "delivered_alerts": self.delivered_alerts,
            "failed_alerts": self.failed_alerts,
            "delivery_rate": self.delivery_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_energy_consumed": self.avg_energy_consumed,
            "total_retransmissions": self.total_retransmissions,
        }
