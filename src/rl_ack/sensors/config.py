"""Sensor configuration specifications and validation for virtual sensors.

NOTE: The default operating ranges and incident thresholds defined herein are
simulation modeling parameters, not certified life-safety or regulatory limits.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import math

from rl_ack.common.enums import NodeOperationalStatus, SensorType
from rl_ack.common.exceptions import ConfigurationError


@dataclass
class SensorConfig:
    """Configuration parameters for a single virtual sensor instance.

    All ranges, intervals, and thresholds are configurable.
    """
    sensor_id: str
    sensor_type: SensorType
    unit: str
    baseline: float
    min_value: float
    max_value: float
    sampling_interval_s: float
    host_node_id: str
    threshold: float
    noise_std: float = 0.5
    mean_reversion_k: float = 0.1
    random_seed: int = 42
    initial_status: NodeOperationalStatus = NodeOperationalStatus.NORMAL
    precision_digits: int = 2

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters, raising ConfigurationError on violations."""
        if not self.sensor_id or not isinstance(self.sensor_id, str) or not self.sensor_id.strip():
            raise ConfigurationError("sensor_id must be a non-empty string.")

        if not isinstance(self.sensor_type, SensorType):
            try:
                self.sensor_type = SensorType(self.sensor_type)
            except ValueError:
                raise ConfigurationError(f"Invalid sensor_type: {self.sensor_type}")

        if not self.unit or not isinstance(self.unit, str) or not self.unit.strip():
            raise ConfigurationError("unit must be a non-empty string.")

        if not self.host_node_id or not isinstance(self.host_node_id, str) or not self.host_node_id.strip():
            raise ConfigurationError("host_node_id must be a non-empty string.")

        if not math.isfinite(self.min_value) or not math.isfinite(self.max_value):
            raise ConfigurationError("min_value and max_value must be finite numbers.")

        if self.min_value >= self.max_value:
            raise ConfigurationError(
                f"min_value ({self.min_value}) must be strictly less than max_value ({self.max_value})."
            )

        if not math.isfinite(self.baseline):
            raise ConfigurationError("baseline must be a finite number.")

        if self.baseline < self.min_value or self.baseline > self.max_value:
            raise ConfigurationError(
                f"baseline ({self.baseline}) must be within [{self.min_value}, {self.max_value}]."
            )

        if not math.isfinite(self.sampling_interval_s) or self.sampling_interval_s <= 0.0:
            raise ConfigurationError(
                f"sampling_interval_s must be a strictly positive finite number, got {self.sampling_interval_s}."
            )

        if not math.isfinite(self.noise_std) or self.noise_std < 0.0:
            raise ConfigurationError(
                f"noise_std must be a non-negative finite number, got {self.noise_std}."
            )

        if not math.isfinite(self.mean_reversion_k) or not (0.0 <= self.mean_reversion_k <= 1.0):
            raise ConfigurationError(
                f"mean_reversion_k must be between 0.0 and 1.0 inclusive, got {self.mean_reversion_k}."
            )

        if not math.isfinite(self.threshold):
            raise ConfigurationError("threshold must be a finite number.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "unit": self.unit,
            "baseline": self.baseline,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "sampling_interval_s": self.sampling_interval_s,
            "host_node_id": self.host_node_id,
            "threshold": self.threshold,
            "noise_std": self.noise_std,
            "mean_reversion_k": self.mean_reversion_k,
            "random_seed": self.random_seed,
            "initial_status": self.initial_status.value,
            "precision_digits": self.precision_digits,
        }


def get_default_gas_config(
    sensor_id: str = "GAS-001",
    host_node_id: str = "NODE-001",
    random_seed: int = 42,
    sampling_interval_s: float = 1.0,
    baseline: float = 50.0,
    threshold: float = 400.0,
) -> SensorConfig:
    """Default configuration for virtual gas sensor."""
    return SensorConfig(
        sensor_id=sensor_id,
        sensor_type=SensorType.GAS,
        unit="ppm",
        baseline=baseline,
        min_value=0.0,
        max_value=1000.0,
        sampling_interval_s=sampling_interval_s,
        host_node_id=host_node_id,
        threshold=threshold,
        noise_std=2.0,
        mean_reversion_k=0.1,
        random_seed=random_seed,
        precision_digits=2,
    )


def get_default_temperature_config(
    sensor_id: str = "TEMP-001",
    host_node_id: str = "NODE-002",
    random_seed: int = 43,
    sampling_interval_s: float = 1.0,
    baseline: float = 24.0,
    threshold: float = 55.0,
) -> SensorConfig:
    """Default configuration for virtual temperature sensor."""
    return SensorConfig(
        sensor_id=sensor_id,
        sensor_type=SensorType.TEMPERATURE,
        unit="°C",
        baseline=baseline,
        min_value=-20.0,
        max_value=100.0,
        sampling_interval_s=sampling_interval_s,
        host_node_id=host_node_id,
        threshold=threshold,
        noise_std=0.3,
        mean_reversion_k=0.1,
        random_seed=random_seed,
        precision_digits=2,
    )


def get_default_humidity_config(
    sensor_id: str = "HUM-001",
    host_node_id: str = "NODE-003",
    random_seed: int = 44,
    sampling_interval_s: float = 2.0,
    baseline: float = 45.0,
    threshold: float = 85.0,
) -> SensorConfig:
    """Default configuration for virtual relative humidity sensor."""
    return SensorConfig(
        sensor_id=sensor_id,
        sensor_type=SensorType.HUMIDITY,
        unit="% RH",
        baseline=baseline,
        min_value=0.0,
        max_value=100.0,
        sampling_interval_s=sampling_interval_s,
        host_node_id=host_node_id,
        threshold=threshold,
        noise_std=0.5,
        mean_reversion_k=0.08,
        random_seed=random_seed,
        precision_digits=2,
    )


def get_default_smoke_config(
    sensor_id: str = "SMOKE-001",
    host_node_id: str = "NODE-004",
    random_seed: int = 45,
    sampling_interval_s: float = 1.0,
    baseline: float = 0.2,
    threshold: float = 3.5,
) -> SensorConfig:
    """Default configuration for virtual optical smoke sensor."""
    return SensorConfig(
        sensor_id=sensor_id,
        sensor_type=SensorType.SMOKE,
        unit="% obs/m",
        baseline=baseline,
        min_value=0.0,
        max_value=20.0,
        sampling_interval_s=sampling_interval_s,
        host_node_id=host_node_id,
        threshold=threshold,
        noise_std=0.02,
        mean_reversion_k=0.15,
        random_seed=random_seed,
        precision_digits=2,
    )
