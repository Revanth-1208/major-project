"""Configuration settings and defaults for the RL-ACK project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class BrokerConfig:
    """MQTT Broker connection parameters."""
    host: str = "127.0.0.1"
    port: int = 1883
    keepalive: int = 60
    base_telemetry_topic: str = "iot/sensors/telemetry"
    base_alert_topic: str = "iot/sensors/alerts"
    base_ack_topic: str = "iot/sensors/acks"
    client_id_prefix: str = "rlack_node_"


@dataclass(frozen=True)
class SensorDefaults:
    """Default parameters for virtual sensors."""
    sampling_interval_sec: float = 1.0
    gas_range_ppm: tuple[float, float] = (0.0, 1000.0)
    gas_threshold_ppm: float = 400.0
    temp_range_celsius: tuple[float, float] = (-20.0, 100.0)
    temp_threshold_celsius: float = 55.0
    humidity_range_pct: tuple[float, float] = (0.0, 100.0)
    humidity_threshold_pct: float = 85.0
    smoke_range_obscuration: tuple[float, float] = (0.0, 20.0)
    smoke_threshold_obscuration: float = 3.5


@dataclass(frozen=True)
class DQNConfig:
    """Deep Q-Network hyperparameter and architectural configuration.

    Default architecture constraint:
    8 input features -> 64 hidden layer -> 32 hidden layer -> 5 Q-value outputs.
    """
    state_dim: int = 8
    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    action_dim: int = 5
    learning_rate: float = 1e-3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 64
    memory_capacity: int = 10000
    target_update_interval: int = 20
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation engine runtime settings."""
    tick_rate_hz: float = 1.0
    total_steps: int = 500
    random_seed: int = 42
    network_loss_probability: float = 0.05
    network_latency_mean_ms: float = 25.0
    network_latency_std_ms: float = 5.0


@dataclass(frozen=True)
class StorageConfig:
    """Data persistence and export parameters."""
    db_path: Path = field(default_factory=lambda: Path("data/rl_ack.db"))
    export_dir: Path = field(default_factory=lambda: Path("exports"))


@dataclass
class AppConfig:
    """Master application configuration aggregate."""
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    sensors: SensorDefaults = field(default_factory=SensorDefaults)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation."""
        return {
            "broker": self.broker.__dict__,
            "sensors": self.sensors.__dict__,
            "dqn": {k: str(v) if isinstance(v, Path) else v for k, v in self.dqn.__dict__.items()},
            "simulation": self.simulation.__dict__,
            "storage": {k: str(v) if isinstance(v, Path) else v for k, v in self.storage.__dict__.items()},
            "project_root": str(self.project_root),
        }
