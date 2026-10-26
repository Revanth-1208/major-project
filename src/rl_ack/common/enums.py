"""Domain enumerations for the RL-ACK IoT network simulation."""

from enum import Enum, IntEnum


class SensorType(str, Enum):
    """The four required virtual sensor types."""
    GAS = "gas"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    SMOKE = "smoke"

    @property
    def default_unit(self) -> str:
        units = {
            SensorType.GAS: "ppm",
            SensorType.TEMPERATURE: "°C",
            SensorType.HUMIDITY: "% RH",
            SensorType.SMOKE: "% obs/m",
        }
        return units[self]


class TrafficClass(str, Enum):
    """Traffic priority classes for network messages."""
    CRITICAL_CONTROL = "Critical Control"
    HIGH_BANDWIDTH_STREAMING = "High Bandwidth Streaming"
    BEST_EFFORT = "Best Effort"

    def to_one_hot(self) -> tuple[float, float, float]:
        """Convert traffic class to 3-element one-hot tuple for state vector."""
        if self == TrafficClass.CRITICAL_CONTROL:
            return (1.0, 0.0, 0.0)
        elif self == TrafficClass.HIGH_BANDWIDTH_STREAMING:
            return (0.0, 1.0, 0.0)
        elif self == TrafficClass.BEST_EFFORT:
            return (0.0, 0.0, 1.0)
        return (0.0, 0.0, 0.0)


class EvaluationPolicy(str, Enum):
    """The six required comparative evaluation policies."""
    STATIC_MQTT_QOS_0 = "Static MQTT QoS 0"
    STATIC_LOCAL_ACK = "Static Local ACK"
    STATIC_MQTT_QOS_1 = "Static MQTT QoS 1"
    LOCAL_ACK_SELECTIVE_RETRANSMIT = "Local ACK plus Selective Retransmission"
    STATIC_MQTT_QOS_2 = "Static MQTT QoS 2"
    ADAPTIVE_RL_ACK_DQN = "Adaptive RL-ACK DQN"


class AckAction(IntEnum):
    """The 5 DQN action space outputs corresponding to the 5 underlying ACK strategies.

    Architecture constraint: 5 Q-value outputs.
    """
    MQTT_QOS_0 = 0
    LOCAL_ACK = 1
    MQTT_QOS_1 = 2
    LOCAL_ACK_SELECTIVE_RETRANSMIT = 3
    MQTT_QOS_2 = 4

    @property
    def label(self) -> str:
        labels = {
            AckAction.MQTT_QOS_0: "MQTT QoS 0",
            AckAction.LOCAL_ACK: "Local ACK",
            AckAction.MQTT_QOS_1: "MQTT QoS 1",
            AckAction.LOCAL_ACK_SELECTIVE_RETRANSMIT: "Local ACK plus Selective Retransmission",
            AckAction.MQTT_QOS_2: "MQTT QoS 2",
        }
        return labels[self]


class DeliveryStatus(str, Enum):
    """Distinct delivery verification lifecycle stages.

    Never mark an alert as delivered merely because a publication was attempted.
    """
    GENERATED = "generated"
    CREATED = "created"
    PUBLISH_PENDING = "publish_pending"
    PUBLISHED = "published"
    BROKER_ACKNOWLEDGED = "broker_acknowledged"
    RECEIVED = "received"
    ACKNOWLEDGED = "acknowledged"
    DELIVERY_CONFIRMED = "delivery_confirmed"
    DELIVERED = "delivered"
    FAILED = "failed"
    TIMEOUT = "timeout"
    TIMED_OUT = "timed_out"


class AlertSeverity(str, Enum):
    """Incident alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class NodeOperationalStatus(str, Enum):
    """IoT sensor node operational and threshold status."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"
    # Backward compatibility aliases
    ALERTING = "critical"
    OFFLINE = "unavailable"


class IncidentState(str, Enum):
    """Incident tracking lifecycle states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RECOVERED = "recovered"
    CLOSED = "closed"
    CLEARED = "cleared"
