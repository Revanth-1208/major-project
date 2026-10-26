"""Abstract base interfaces establishing strict module boundaries."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from rl_ack.common.enums import AckAction, EvaluationPolicy, SensorType
from rl_ack.common.models import (
    AckEvent,
    AlertMessage,
    ExperimentRecord,
    NetworkStateVector,
    SensorReading,
)


class ISensor(ABC):
    """Module 2: Contract for individual virtual sensors."""

    @property
    @abstractmethod
    def sensor_id(self) -> str:
        """Unique sensor identifier."""
        pass

    @property
    @abstractmethod
    def sensor_type(self) -> SensorType:
        """Type of the virtual sensor."""
        pass

    @property
    @abstractmethod
    def host_node(self) -> str:
        """Identifier of the hosting IoT node."""
        pass

    @abstractmethod
    def sample(self, current_time: float) -> SensorReading:
        """Generate a dynamic reading for the current simulation timestamp."""
        pass

    @abstractmethod
    def set_incident(self, active: bool, intensity: float = 1.0) -> None:
        """Inject or clear an incident scenario."""
        pass


class ISensorEngine(ABC):
    """Module 2: Contract for coordinating multiple virtual sensors."""

    @abstractmethod
    def register_sensor(self, sensor: ISensor) -> None:
        """Register a sensor in the engine."""
        pass

    @abstractmethod
    def step(self, current_time: float) -> List[SensorReading]:
        """Perform a single simulation step and collect all sensor readings."""
        pass

    @abstractmethod
    def get_sensors(self) -> List[ISensor]:
        """Return all registered sensors."""
        pass


class IIncidentDetector(ABC):
    """Module 4: Contract for incident detection and alert generation."""

    @abstractmethod
    def evaluate(self, reading: SensorReading) -> Optional[AlertMessage]:
        """Evaluate a sensor reading against configured thresholds."""
        pass

    @abstractmethod
    def get_active_incidents(self) -> List[AlertMessage]:
        """Retrieve currently active unacknowledged incidents."""
        pass


class IMqttManager(ABC):
    """Module 5: Contract for MQTT broker connection and delivery verification."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the Mosquitto MQTT broker."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully disconnect from the MQTT broker."""
        pass

    @abstractmethod
    def publish_telemetry(self, reading: SensorReading) -> bool:
        """Publish sensor telemetry to the telemetry topic."""
        pass

    @abstractmethod
    def publish_alert(self, alert: AlertMessage, qos: int) -> bool:
        """Publish alert message to the alert topic with specified QoS."""
        pass

    @abstractmethod
    def track_event(self, event: AckEvent) -> None:
        """Track distinct lifecycle acknowledgment event."""
        pass


class IAckPolicy(ABC):
    """Module 6: Contract for acknowledgment evaluation policies."""

    @property
    @abstractmethod
    def policy_type(self) -> EvaluationPolicy:
        """Return the evaluation policy identifier."""
        pass

    @abstractmethod
    def select_action(self, state: NetworkStateVector) -> AckAction:
        """Select an ACK strategy action given current network state."""
        pass


class IDQNAgent(ABC):
    """Module 7 & 8: Contract for Deep Q-Network Agent."""

    @abstractmethod
    def select_action(self, state: NetworkStateVector, explore: bool = True) -> AckAction:
        """Select action using epsilon-greedy or greedy policy."""
        pass

    @abstractmethod
    def store_transition(
        self,
        state: NetworkStateVector,
        action: AckAction,
        reward: float,
        next_state: NetworkStateVector,
        done: bool,
    ) -> None:
        """Store step transition in experience replay buffer."""
        pass

    @abstractmethod
    def train_step(self) -> Optional[float]:
        """Perform one Bellman loss optimization step; return loss if trained."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model weights and training state."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> bool:
        """Load model weights and training state."""
        pass


class ISimulationRunner(ABC):
    """Module 9: Contract for interactive simulation runner."""

    @abstractmethod
    def initialize(self) -> None:
        """Reset and prepare the simulation environment."""
        pass

    @abstractmethod
    def run_step(self) -> Dict[str, Any]:
        """Execute one simulation step across all modules."""
        pass

    @abstractmethod
    def run_batch(self, steps: int, policy: EvaluationPolicy) -> ExperimentRecord:
        """Run batch simulation for a specific policy and produce summary."""
        pass


class IStorageRepository(ABC):
    """Module 10 & 11: Contract for SQLite persistence and CSV export."""

    @abstractmethod
    def initialize_schema(self) -> None:
        """Initialize SQLite database tables and indexes."""
        pass

    @abstractmethod
    def record_reading(self, reading: SensorReading) -> None:
        """Persist a single sensor reading."""
        pass

    @abstractmethod
    def record_alert(self, alert: AlertMessage) -> None:
        """Persist an incident alert."""
        pass

    @abstractmethod
    def record_ack_event(self, event: AckEvent) -> None:
        """Persist a delivery lifecycle acknowledgment event."""
        pass

    @abstractmethod
    def record_experiment(self, experiment: ExperimentRecord) -> None:
        """Persist policy comparison experimental result."""
        pass

    @abstractmethod
    def export_to_csv(self, table_name: str, output_path: str) -> str:
        """Export specified table data to CSV file."""
        pass
