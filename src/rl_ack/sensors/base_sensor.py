"""Base virtual sensor abstraction implementing stochastic dynamics and lifecycle management."""

from datetime import datetime, timezone
import math
import random
from typing import Any, Dict, Optional

from rl_ack.common.enums import NodeOperationalStatus, SensorType
from rl_ack.common.interfaces import ISensor
from rl_ack.common.models import SensorReading
from rl_ack.sensors.config import SensorConfig


class BaseSensor(ISensor):
    """Abstract base class for all virtual sensors.

    Implements a bounded mean-reverting Ornstein-Uhlenbeck stochastic process:
        x_t = x_{t-1} + k * (mu - x_{t-1}) + sigma * epsilon_t
    where:
        mu = target baseline (or dynamic event target under incident conditions)
        k = mean reversion strength (0 <= k <= 1)
        sigma = noise standard deviation
        epsilon_t ~ N(0, 1) sampled from a dedicated per-sensor RNG.

    The state is strictly clamped to [min_value, max_value] and rounded to precision_digits.
    """

    def __init__(self, config: SensorConfig) -> None:
        self.config = config
        self._rng = random.Random(self.config.random_seed)
        self._current_value: float = float(self.config.baseline)
        self._sequence_number: int = 0
        self._last_reading: Optional[SensorReading] = None
        self._next_sampling_time: float = float(self.config.sampling_interval_s)
        self._status: NodeOperationalStatus = self.config.initial_status

        # Environmental event state
        self._event_active: bool = False
        self._event_target: Optional[float] = None
        self._event_start_time: Optional[float] = None
        self._event_duration_s: Optional[float] = None

    @property
    def sensor_id(self) -> str:
        return self.config.sensor_id

    @property
    def sensor_type(self) -> SensorType:
        return self.config.sensor_type

    @property
    def unit(self) -> str:
        return self.config.unit

    @property
    def host_node(self) -> str:
        """Host node identifier (alias for host_node_id)."""
        return self.config.host_node_id

    @property
    def host_node_id(self) -> str:
        return self.config.host_node_id

    @property
    def sampling_interval_s(self) -> float:
        return self.config.sampling_interval_s

    @property
    def threshold(self) -> float:
        return self.config.threshold

    @property
    def status(self) -> NodeOperationalStatus:
        return self._status

    @property
    def sequence_number(self) -> int:
        return self._sequence_number

    @property
    def last_reading(self) -> Optional[SensorReading]:
        return self._last_reading

    @property
    def next_sampling_time(self) -> float:
        return self._next_sampling_time

    @property
    def current_value(self) -> float:
        return self._current_value

    @property
    def is_event_active(self) -> bool:
        return self._event_active

    def is_due(self, current_time: float, tolerance: float = 1e-9) -> bool:
        """Check if sensor is scheduled for sampling at current_time."""
        return (current_time + tolerance) >= self._next_sampling_time

    def set_operational_status(self, status: NodeOperationalStatus) -> None:
        """Set sensor operational status (e.g. NORMAL vs OFFLINE)."""
        self._status = status

    def trigger_event(
        self,
        target_value: float,
        duration_s: Optional[float] = None,
        start_time: Optional[float] = None,
    ) -> None:
        """Trigger an environmental event that alters the sensor's mean target.

        The value will smoothly transition toward target_value via mean reversion.
        """
        clamped_target = max(self.config.min_value, min(self.config.max_value, float(target_value)))
        self._event_active = True
        self._event_target = clamped_target
        self._event_duration_s = duration_s
        self._event_start_time = start_time

    def clear_event(self) -> None:
        """Clear active environmental event; sensor returns gradually to baseline."""
        self._event_active = False
        self._event_target = None
        self._event_start_time = None
        self._event_duration_s = None

    def set_incident(self, active: bool, intensity: float = 1.0) -> None:
        """ISensor contract: inject or clear incident scenario.

        When active is True, triggers an event targeting above the configured threshold.
        When active is False, clears the event to return toward baseline.
        """
        if active:
            # Scale target based on threshold and intensity
            delta = (self.config.threshold - self.config.baseline)
            target = self.config.baseline + delta * max(1.1, float(intensity))
            self.trigger_event(target_value=target)
        else:
            self.clear_event()

    def _check_event_expiry(self, current_time: float) -> None:
        """Clear event if its duration has elapsed."""
        if self._event_active and self._event_duration_s is not None and self._event_start_time is not None:
            if current_time >= (self._event_start_time + self._event_duration_s):
                self.clear_event()

    def _step_value(self, current_time: float) -> float:
        """Perform one stochastic step according to the mean-reverting model."""
        self._check_event_expiry(current_time)

        # Determine target mean mu
        if self._event_active and self._event_target is not None:
            mu = self._event_target
        else:
            mu = self.config.baseline

        # Mean reversion pull
        reversion = self.config.mean_reversion_k * (mu - self._current_value)

        # Stochastic noise
        noise = self._rng.gauss(0.0, self.config.noise_std)

        # Update state
        new_val = self._current_value + reversion + noise

        # Clamp strictly to configured physical simulation range
        new_val = max(self.config.min_value, min(self.config.max_value, new_val))

        # Round to configured precision
        rounded_val = round(new_val, self.config.precision_digits)
        self._current_value = rounded_val
        return rounded_val

    def _is_threshold_exceeded(self, value: float) -> bool:
        """Evaluate whether value exceeds threshold based on sensor characteristics."""
        return value >= self.config.threshold

    def sample(self, current_time: float) -> Optional[SensorReading]:
        """Generate a dynamic reading for the current simulation timestamp.

        Returns None if sensor is unavailable (not NORMAL).
        """
        if self._status != NodeOperationalStatus.NORMAL:
            return None

        self._sequence_number += 1
        new_value = self._step_value(current_time)
        threshold_exceeded = self._is_threshold_exceeded(new_value)

        reading = SensorReading(
            sensor_id=self.config.sensor_id,
            sensor_type=self.config.sensor_type,
            value=new_value,
            unit=self.config.unit,
            simulation_time_s=float(current_time),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            sampling_interval_s=self.config.sampling_interval_s,
            host_node_id=self.config.host_node_id,
            status=self._status,
            sequence_number=self._sequence_number,
            is_threshold_exceeded=threshold_exceeded,
        )

        self._last_reading = reading
        self._next_sampling_time += self.config.sampling_interval_s
        return reading

    def reset(self) -> None:
        """Reset sensor state and random generator to initial conditions."""
        self._rng = random.Random(self.config.random_seed)
        self._current_value = float(self.config.baseline)
        self._sequence_number = 0
        self._last_reading = None
        self._next_sampling_time = float(self.config.sampling_interval_s)
        self._status = self.config.initial_status
        self.clear_event()

    def get_metadata(self) -> Dict[str, Any]:
        """Return full sensor configuration and runtime status metadata."""
        return {
            "config": self.config.to_dict(),
            "current_value": self._current_value,
            "sequence_number": self._sequence_number,
            "next_sampling_time": self._next_sampling_time,
            "status": self._status.value,
            "event_active": self._event_active,
            "event_target": self._event_target,
        }
