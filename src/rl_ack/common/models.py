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
    """Individual virtual sensor telemetry record.

    Attributes:
        sensor_id: Stable identifier for sensor instance.
        sensor_type: Categorical sensor type.
        value: Valid finite numeric reading value.
        unit: Physical engineering or optical measurement unit.
        simulation_time_s: Elapsed monotonic simulation time in seconds.
        timestamp_utc: ISO 8601 formatted UTC wall-clock timestamp string.
        sampling_interval_s: Configured sampling interval for this sensor in seconds.
        host_node_id: Identifier of the hosting IoT node.
        status: Operational status of the sensor.
        sequence_number: Monotonically increasing sequence number per sensor instance.
        is_threshold_exceeded: Flag indicating if configured threshold was exceeded.
    """
    sensor_id: str
    sensor_type: SensorType
    value: float
    unit: str
    simulation_time_s: float = 0.0
    timestamp_utc: str = ""
    sampling_interval_s: float = 1.0
    host_node_id: str = ""
    status: NodeOperationalStatus = NodeOperationalStatus.NORMAL
    sequence_number: int = 0
    is_threshold_exceeded: bool = False

    # Backwards-compatibility aliases for Phase 0 calls
    host_node: Optional[str] = None
    timestamp: Optional[float] = None
    operational_status: Optional[NodeOperationalStatus] = None

    def __post_init__(self) -> None:
        from datetime import datetime, timezone

        # Synchronize host_node / host_node_id
        if self.host_node is not None and not self.host_node_id:
            self.host_node_id = self.host_node
        elif self.host_node_id and self.host_node is None:
            self.host_node = self.host_node_id

        # Synchronize timestamp / simulation_time_s
        if self.timestamp is not None and self.simulation_time_s == 0.0:
            self.simulation_time_s = float(self.timestamp)
        elif self.simulation_time_s and self.timestamp is None:
            self.timestamp = self.simulation_time_s

        # Synchronize status / operational_status
        if self.operational_status is not None and self.status == NodeOperationalStatus.NORMAL:
            self.status = self.operational_status
        elif self.status and self.operational_status is None:
            self.operational_status = self.status

        # Provide ISO 8601 UTC timestamp if not supplied
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        s_type = self.sensor_type.value if hasattr(self.sensor_type, "value") else str(self.sensor_type)
        stat = self.status.value if hasattr(self.status, "value") else str(self.status)
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": s_type,
            "value": self.value,
            "unit": self.unit,
            "simulation_time_s": self.simulation_time_s,
            "timestamp_utc": self.timestamp_utc,
            "sampling_interval_s": self.sampling_interval_s,
            "host_node_id": self.host_node_id,
            "status": stat,
            "sequence_number": self.sequence_number,
            "is_threshold_exceeded": self.is_threshold_exceeded,
            # Backward-compatibility keys
            "host_node": self.host_node_id,
            "timestamp": self.simulation_time_s,
            "operational_status": stat,
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
