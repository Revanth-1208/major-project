"""Data models and transfer objects for the RL-ACK project."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from rl_ack.common.enums import (
    AckAction,
    AlertSeverity,
    DeliveryStatus,
    IncidentState,
    NodeOperationalStatus,
    SensorType,
    TrafficClass,
)
from rl_ack.common.exceptions import MqttPayloadError


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
    """Distinct incident alert payload transmitted over MQTT.

    Supports complete emergency-alert architecture with end-to-end delivery tracking.
    """
    alert_id: str
    sensor_id: str
    severity: AlertSeverity = AlertSeverity.WARNING
    reading_value: float = 0.0
    threshold_value: float = 0.0
    unit: str = ""
    incident_type: str = "threshold_violation"
    message: str = ""
    simulation_time_s: float = 0.0
    created_at_utc: str = ""
    priority: TrafficClass = TrafficClass.CRITICAL_CONTROL
    destination_topic: str = "iot/sensors/alerts"
    delivery_status: DeliveryStatus = DeliveryStatus.GENERATED
    published_at_utc: Optional[str] = None
    broker_ack_at_utc: Optional[str] = None
    received_at_utc: Optional[str] = None
    app_ack_at_utc: Optional[str] = None
    recovered_at_utc: Optional[str] = None
    retry_count: int = 0
    failure_reason: Optional[str] = None
    chosen_action: Optional[AckAction] = None
    incident_id: Optional[str] = None

    # Backward-compatibility aliases
    host_node: str = ""
    sensor_type: Optional[SensorType] = None
    timestamp: Optional[float] = None
    traffic_class: Optional[TrafficClass] = None
    status: Optional[DeliveryStatus] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from datetime import datetime, timezone

        if self.timestamp is not None and self.simulation_time_s == 0.0:
            self.simulation_time_s = float(self.timestamp)
        elif self.simulation_time_s and self.timestamp is None:
            self.timestamp = self.simulation_time_s

        if self.traffic_class is not None and self.priority == TrafficClass.CRITICAL_CONTROL:
            self.priority = self.traffic_class
        elif self.priority and self.traffic_class is None:
            self.traffic_class = self.priority

        if self.status is not None and self.delivery_status == DeliveryStatus.GENERATED:
            self.delivery_status = self.status
        elif self.delivery_status and self.status is None:
            self.status = self.delivery_status

        if not self.created_at_utc:
            self.created_at_utc = datetime.now(timezone.utc).isoformat()

        if not self.message:
            self.message = (
                f"{self.sensor_id} {self.incident_type} condition: "
                f"{self.reading_value:.2f} {self.unit} (threshold: {self.threshold_value:.2f} {self.unit})"
            )

    @property
    def value(self) -> float:
        """Alias for reading_value."""
        return self.reading_value

    @property
    def threshold(self) -> float:
        """Alias for threshold_value."""
        return self.threshold_value

    @property
    def created_at(self) -> str:
        """Alias for created_at_utc."""
        return self.created_at_utc

    @property
    def acknowledged_at(self) -> Optional[str]:
        """Alias for app_ack_at_utc."""
        return self.app_ack_at_utc

    @property
    def recovered_at(self) -> Optional[str]:
        """Alias for recovered_at_utc."""
        return self.recovered_at_utc

    def to_dict(self) -> Dict[str, Any]:
        s_type = self.sensor_type.value if hasattr(self.sensor_type, "value") else str(self.sensor_type or "")
        return {
            "alert_id": self.alert_id,
            "sensor_id": self.sensor_id,
            "incident_type": self.incident_type,
            "severity": self.severity.value,
            "message": self.message,
            "reading_value": self.reading_value,
            "threshold_value": self.threshold_value,
            "value": self.reading_value,
            "threshold": self.threshold_value,
            "unit": self.unit,
            "simulation_time_s": self.simulation_time_s,
            "created_at_utc": self.created_at_utc,
            "created_at": self.created_at_utc,
            "priority": self.priority.value,
            "destination_topic": self.destination_topic,
            "delivery_status": self.delivery_status.value,
            "status": self.delivery_status.value,
            "published_at_utc": self.published_at_utc,
            "broker_ack_at_utc": self.broker_ack_at_utc,
            "received_at_utc": self.received_at_utc,
            "app_ack_at_utc": self.app_ack_at_utc,
            "acknowledged_at": self.app_ack_at_utc,
            "recovered_at_utc": self.recovered_at_utc,
            "recovered_at": self.recovered_at_utc,
            "retry_count": self.retry_count,
            "failure_reason": self.failure_reason,
            "chosen_action": self.chosen_action.value if self.chosen_action is not None else None,
            "incident_id": self.incident_id,
            # Backward-compatibility keys
            "host_node": self.host_node,
            "sensor_type": s_type,
            "traffic_class": self.priority.value,
            "timestamp": self.simulation_time_s,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        """Serialize alert message to JSON string."""
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertMessage":
        """Reconstruct AlertMessage from dictionary, validating required fields."""
        if not isinstance(data, dict):
            raise MqttPayloadError(f"Alert payload must be a JSON object, got {type(data).__name__}")

        if "alert_id" not in data or not data["alert_id"]:
            raise MqttPayloadError("Missing required field: alert_id")
        if "sensor_id" not in data or not data["sensor_id"]:
            raise MqttPayloadError("Missing required field: sensor_id")

        # Severity
        sev_raw = data.get("severity", "warning")
        try:
            sev = AlertSeverity(sev_raw)
        except (ValueError, KeyError):
            sev = AlertSeverity.WARNING

        # Priority
        pri_raw = data.get("priority", data.get("traffic_class", TrafficClass.CRITICAL_CONTROL.value))
        try:
            priority = TrafficClass(pri_raw)
        except (ValueError, KeyError):
            priority = TrafficClass.CRITICAL_CONTROL

        # Delivery status
        ds_raw = data.get("delivery_status", data.get("status", DeliveryStatus.GENERATED.value))
        try:
            del_stat = DeliveryStatus(ds_raw)
        except (ValueError, KeyError):
            del_stat = DeliveryStatus.GENERATED

        # Numeric values
        val = float(data.get("reading_value", data.get("value", 0.0)))
        thresh = float(data.get("threshold_value", data.get("threshold", 0.0)))
        sim_time = float(data.get("simulation_time_s", data.get("timestamp", 0.0)))

        return cls(
            alert_id=str(data["alert_id"]),
            sensor_id=str(data["sensor_id"]),
            severity=sev,
            reading_value=val,
            threshold_value=thresh,
            unit=str(data.get("unit", "")),
            incident_type=str(data.get("incident_type", "threshold_violation")),
            message=str(data.get("message", "")),
            simulation_time_s=sim_time,
            created_at_utc=str(data.get("created_at_utc", data.get("created_at", ""))),
            priority=priority,
            destination_topic=str(data.get("destination_topic", "rlack/kitchen/alerts")),
            delivery_status=del_stat,
            published_at_utc=data.get("published_at_utc"),
            broker_ack_at_utc=data.get("broker_ack_at_utc"),
            received_at_utc=data.get("received_at_utc"),
            app_ack_at_utc=data.get("app_ack_at_utc", data.get("acknowledged_at")),
            recovered_at_utc=data.get("recovered_at_utc", data.get("recovered_at")),
            retry_count=int(data.get("retry_count", 0)),
            failure_reason=data.get("failure_reason"),
            incident_id=data.get("incident_id"),
            host_node=str(data.get("host_node", "")),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AlertMessage":
        """Reconstruct AlertMessage from JSON string, handling decoding errors."""
        import json
        try:
            data = json.loads(json_str)
        except Exception as e:
            raise MqttPayloadError(f"Malformed JSON payload: {e}") from e
        return cls.from_dict(data)


@dataclass
class IncidentRecord:
    """Tracks the lifecycle of an ongoing or cleared incident.

    Prevents generating redundant alert instances on every consecutive sample
    while the same underlying incident condition persists.
    """
    incident_id: str
    sensor_id: str
    incident_type: str
    state: IncidentState = IncidentState.ACTIVE
    severity: AlertSeverity = AlertSeverity.WARNING
    trigger_reading: float = 0.0
    trigger_threshold: float = 0.0
    unit: str = ""
    trigger_time_s: float = 0.0
    created_at_utc: str = ""
    last_reading: float = 0.0
    last_updated_time_s: float = 0.0
    acknowledged_time_s: Optional[float] = None
    acknowledged_at_utc: Optional[str] = None
    recovered_time_s: Optional[float] = None
    recovered_at_utc: Optional[str] = None
    closed_time_s: Optional[float] = None
    closed_at_utc: Optional[str] = None
    associated_alert_ids: List[str] = field(default_factory=list)

    # Backward-compatibility alias
    cleared_time_s: Optional[float] = None

    def __post_init__(self) -> None:
        from datetime import datetime, timezone
        if not self.created_at_utc:
            self.created_at_utc = datetime.now(timezone.utc).isoformat()
        if self.cleared_time_s is not None and self.recovered_time_s is None:
            self.recovered_time_s = self.cleared_time_s
        elif self.recovered_time_s is not None and self.cleared_time_s is None:
            self.cleared_time_s = self.recovered_time_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "sensor_id": self.sensor_id,
            "incident_type": self.incident_type,
            "state": self.state.value,
            "severity": self.severity.value,
            "trigger_reading": self.trigger_reading,
            "trigger_threshold": self.trigger_threshold,
            "unit": self.unit,
            "trigger_time_s": self.trigger_time_s,
            "created_at_utc": self.created_at_utc,
            "last_reading": self.last_reading,
            "last_updated_time_s": self.last_updated_time_s,
            "acknowledged_time_s": self.acknowledged_time_s,
            "acknowledged_at_utc": self.acknowledged_at_utc,
            "recovered_time_s": self.recovered_time_s if self.recovered_time_s is not None else self.cleared_time_s,
            "recovered_at_utc": self.recovered_at_utc,
            "closed_time_s": self.closed_time_s,
            "closed_at_utc": self.closed_at_utc,
            "cleared_time_s": self.cleared_time_s if self.cleared_time_s is not None else self.recovered_time_s,
            "associated_alert_ids": self.associated_alert_ids,
        }

    def to_json(self) -> str:
        """Serialize incident record to JSON string."""
        import json
        return json.dumps(self.to_dict())


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
