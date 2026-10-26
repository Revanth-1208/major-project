"""Custom exception hierarchy for the RL-ACK project."""


class RLAckError(Exception):
    """Base exception for all domain-specific RL-ACK errors."""
    pass


class ConfigurationError(RLAckError):
    """Raised when configuration parameters are invalid or missing."""
    pass


class BrokerConnectionError(RLAckError):
    """Raised when MQTT broker connection or communication fails."""
    pass


class PolicyExecutionError(RLAckError):
    """Raised when policy evaluation or action execution fails."""
    pass


class DQNModelError(RLAckError):
    """Raised when DQN dimension mismatch, invalid tensor, or checkpoint load fails."""
    pass


class StorageError(RLAckError):
    """Raised when database or file export operations fail."""
    pass


class SensorSimulationError(RLAckError):
    """Raised when sensor sampling or incident generation fails."""
    pass


class InvalidStateTransitionError(RLAckError):
    """Raised when an illegal incident lifecycle state transition is attempted."""
    pass


class MqttPayloadError(RLAckError):
    """Raised when an MQTT payload is malformed, missing required fields, or fails schema validation."""
    pass


class DeliveryTimeoutError(RLAckError):
    """Raised when message delivery verification exceeds configured timeout threshold."""
    pass
