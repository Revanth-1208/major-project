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
    threshold: float = 0.0
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    hysteresis: float = 0.0
    warning_low: Optional[float] = None
    critical_low: Optional[float] = None
    noise_std: float = 0.5
    mean_reversion_k: float = 0.1
    random_seed: int = 42
    initial_status: NodeOperationalStatus = NodeOperationalStatus.NORMAL
    precision_digits: int = 2

    def __post_init__(self) -> None:
        # Reconcile threshold / critical_threshold
        if self.critical_threshold is None:
            self.critical_threshold = float(self.threshold)
        elif self.threshold == 0.0 and self.critical_threshold is not None:
            self.threshold = float(self.critical_threshold)

        # Set default warning thresholds based on sensor type if not explicitly supplied
        if self.warning_threshold is None:
            if self.sensor_type == SensorType.GAS:
                self.warning_threshold = 100.0
                if self.hysteresis == 0.0:
                    self.hysteresis = 5.0
            elif self.sensor_type == SensorType.TEMPERATURE:
                self.warning_threshold = 38.0
                if self.hysteresis == 0.0:
                    self.hysteresis = 1.0
            elif self.sensor_type == SensorType.SMOKE:
                self.warning_threshold = 1.0
                if self.hysteresis == 0.0:
                    self.hysteresis = 0.1
            elif self.sensor_type == SensorType.HUMIDITY:
                self.warning_threshold = 70.0
                if self.warning_low is None:
                    self.warning_low = 30.0
                if self.critical_low is None:
                    self.critical_low = 15.0
                if self.hysteresis == 0.0:
                    self.hysteresis = 2.0
            else:
                self.warning_threshold = self.baseline + (self.critical_threshold - self.baseline) * 0.5

        if self.sensor_type == SensorType.HUMIDITY:
            if self.warning_low is None:
                self.warning_low = 30.0
            if self.critical_low is None:
                self.critical_low = 15.0
            if self.hysteresis == 0.0:
                self.hysteresis = 2.0

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

        crit = self.critical_threshold if self.critical_threshold is not None else self.threshold
        if not math.isfinite(crit):
            raise ConfigurationError("threshold/critical_threshold must be a finite number.")

        if self.warning_threshold is not None and not math.isfinite(self.warning_threshold):
            raise ConfigurationError("warning_threshold must be a finite number.")

        if self.hysteresis < 0.0:
            raise ConfigurationError("hysteresis cannot be negative.")

        if self.sensor_type != SensorType.HUMIDITY and self.warning_threshold is not None:
            if self.warning_threshold >= crit:
                raise ConfigurationError(
                    f"warning_threshold ({self.warning_threshold}) must be strictly less than "
                    f"critical_threshold ({crit})."
                )

        if self.sensor_type == SensorType.HUMIDITY:
            if self.warning_low is not None and self.critical_low is not None:
                if not (self.critical_low < self.warning_low <= (self.warning_threshold or 70.0) < crit):
                    raise ConfigurationError(
                        f"Humidity thresholds must satisfy: critical_low ({self.critical_low}) < "
                        f"warning_low ({self.warning_low}) <= warning_high ({self.warning_threshold}) < "
                        f"critical_high ({crit})."
                    )

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
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "hysteresis": self.hysteresis,
            "warning_low": self.warning_low,
            "critical_low": self.critical_low,
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
    warning_threshold: float = 100.0,
    critical_threshold: float = 400.0,
    hysteresis: float = 5.0,
    threshold: Optional[float] = None,
) -> SensorConfig:
    """Default configuration for virtual gas sensor."""
    crit = critical_threshold if threshold is None else threshold
    return SensorConfig(
        sensor_id=sensor_id,
        sensor_type=SensorType.GAS,
        unit="ppm",
        baseline=baseline,
        min_value=0.0,
        max_value=1000.0,
        sampling_interval_s=sampling_interval_s,
        host_node_id=host_node_id,
        threshold=crit,
        warning_threshold=warning_threshold,
        critical_threshold=crit,
        hysteresis=hysteresis,
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
    warning_threshold: float = 38.0,
    critical_threshold: float = 55.0,
    hysteresis: float = 1.0,
    threshold: Optional[float] = None,
) -> SensorConfig:
    """Default configuration for virtual temperature sensor."""
    crit = critical_threshold if threshold is None else threshold
    return SensorConfig(
        sensor_id=sensor_id,
        sensor_type=SensorType.TEMPERATURE,
        unit="°C",
        baseline=baseline,
        min_value=-20.0,
        max_value=100.0,
        sampling_interval_s=sampling_interval_s,
        host_node_id=host_node_id,
        threshold=crit,
        warning_threshold=warning_threshold,
        critical_threshold=crit,
        hysteresis=hysteresis,
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
    warning_low: float = 30.0,
    warning_high: float = 70.0,
    critical_low: float = 15.0,
    critical_high: float = 85.0,
    hysteresis: float = 2.0,
    threshold: Optional[float] = None,
) -> SensorConfig:
    """Default configuration for virtual relative humidity sensor."""
    crit = critical_high if threshold is None else threshold
    return SensorConfig(
        sensor_id=sensor_id,
        sensor_type=SensorType.HUMIDITY,
        unit="% RH",
        baseline=baseline,
        min_value=0.0,
        max_value=100.0,
        sampling_interval_s=sampling_interval_s,
        host_node_id=host_node_id,
        threshold=crit,
        warning_threshold=warning_high,
        critical_threshold=crit,
        warning_low=warning_low,
        critical_low=critical_low,
        hysteresis=hysteresis,
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
    warning_threshold: float = 1.0,
    critical_threshold: float = 3.5,
    hysteresis: float = 0.1,
    threshold: Optional[float] = None,
) -> SensorConfig:
    """Default configuration for virtual optical smoke sensor."""
    crit = critical_threshold if threshold is None else threshold
    return SensorConfig(
        sensor_id=sensor_id,
        sensor_type=SensorType.SMOKE,
        unit="% obs/m",
        baseline=baseline,
        min_value=0.0,
        max_value=20.0,
        sampling_interval_s=sampling_interval_s,
        host_node_id=host_node_id,
        threshold=crit,
        warning_threshold=warning_threshold,
        critical_threshold=crit,
        hysteresis=hysteresis,
        noise_std=0.02,
        mean_reversion_k=0.15,
        random_seed=random_seed,
        precision_digits=2,
    )
