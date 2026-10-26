"""Simulation clock and discrete-event sampling scheduler for virtual sensors."""

from typing import Any, Dict, List, Optional
import math

from rl_ack.common.enums import NodeOperationalStatus, SensorType
from rl_ack.common.interfaces import ISensor, ISensorEngine
from rl_ack.common.models import SensorReading
from rl_ack.sensors.gas_sensor import GasSensor
from rl_ack.sensors.temperature_sensor import TemperatureSensor
from rl_ack.sensors.humidity_sensor import HumiditySensor
from rl_ack.sensors.smoke_sensor import SmokeSensor
from rl_ack.sensors.config import (
    get_default_gas_config,
    get_default_temperature_config,
    get_default_humidity_config,
    get_default_smoke_config,
)


class SensorEngine(ISensorEngine):
    """Coordinates virtual sensors, simulation clock progression, and discrete sampling events.

    Provides a clean, modular abstraction representing the IoT sensor plane.
    """

    def __init__(self) -> None:
        self._current_simulation_time: float = 0.0
        self._sensors: Dict[str, ISensor] = {}
        self._latest_readings: Dict[str, SensorReading] = {}
        self._time_tolerance: float = 1e-9

    @property
    def current_simulation_time(self) -> float:
        """Current monotonic simulation time in seconds."""
        return self._current_simulation_time

    def register_sensor(self, sensor: ISensor) -> None:
        """Register a virtual sensor with the engine."""
        if not isinstance(sensor, ISensor):
            raise TypeError("Sensor must implement ISensor interface.")
        if sensor.sensor_id in self._sensors:
            raise ValueError(f"Sensor with ID '{sensor.sensor_id}' is already registered.")
        self._sensors[sensor.sensor_id] = sensor

    def get_sensor(self, sensor_id: str) -> ISensor:
        """Retrieve sensor by its ID."""
        if sensor_id not in self._sensors:
            raise KeyError(f"Sensor '{sensor_id}' not found in engine registry.")
        return self._sensors[sensor_id]

    def get_sensors(self) -> List[ISensor]:
        """Return list of all registered sensors."""
        return list(self._sensors.values())

    def get_latest_readings(self) -> Dict[str, SensorReading]:
        """Return most recent valid reading for each sensor ID."""
        return dict(self._latest_readings)

    def get_sensor_statuses(self) -> Dict[str, NodeOperationalStatus]:
        """Return operational status of all registered sensors."""
        return {s_id: s.status for s_id, s in self._sensors.items()}

    def get_sensor_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata for all registered sensors."""
        result: Dict[str, Dict[str, Any]] = {}
        for s_id, sensor in self._sensors.items():
            if hasattr(sensor, "get_metadata"):
                result[s_id] = sensor.get_metadata()
            else:
                result[s_id] = {
                    "sensor_id": sensor.sensor_id,
                    "sensor_type": sensor.sensor_type.value,
                    "host_node_id": sensor.host_node_id,
                    "sampling_interval_s": sensor.sampling_interval_s,
                    "status": sensor.status.value,
                }
        return result

    def trigger_sensor_event(
        self,
        sensor_id: str,
        target_value: float,
        duration_s: Optional[float] = None,
    ) -> None:
        """Trigger an environmental event on a specific sensor."""
        sensor = self.get_sensor(sensor_id)
        if hasattr(sensor, "trigger_event"):
            sensor.trigger_event(
                target_value=target_value,
                duration_s=duration_s,
                start_time=self._current_simulation_time,
            )
        else:
            sensor.set_incident(True)

    def clear_sensor_event(self, sensor_id: str) -> None:
        """Clear active environmental event on a specific sensor."""
        sensor = self.get_sensor(sensor_id)
        if hasattr(sensor, "clear_event"):
            sensor.clear_event()
        else:
            sensor.set_incident(False)

    def set_sensor_status(self, sensor_id: str, status: NodeOperationalStatus) -> None:
        """Update operational status for a specific sensor."""
        sensor = self.get_sensor(sensor_id)
        sensor.set_operational_status(status)

    def advance(self, dt: float) -> List[SensorReading]:
        """Advance simulation clock by dt seconds and generate readings due on scheduled boundaries.

        If dt crosses multiple scheduled boundaries for a sensor, each boundary sample is
        generated at its exact scheduled simulation time.
        """
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError(f"dt must be a non-negative finite number, got {dt}.")

        if dt == 0.0:
            return []

        target_time = round(self._current_simulation_time + dt, 9)
        generated_readings: List[SensorReading] = []

        # Collect due samples across all registered sensors
        for sensor in self._sensors.values():
            # Process all scheduled sampling boundaries up to target_time
            while (sensor.next_sampling_time - self._time_tolerance) <= target_time:
                scheduled_boundary = round(sensor.next_sampling_time, 9)
                reading = sensor.sample(current_time=scheduled_boundary)
                if reading is not None:
                    generated_readings.append(reading)
                    self._latest_readings[sensor.sensor_id] = reading
                else:
                    # Sensor is unavailable: advance its next sampling time to prevent infinite loop
                    if hasattr(sensor, "_next_sampling_time"):
                        sensor._next_sampling_time += sensor.sampling_interval_s
                    else:
                        break

        self._current_simulation_time = target_time

        # Order readings deterministically by scheduled simulation time, then sensor ID
        generated_readings.sort(key=lambda r: (r.simulation_time_s, r.sensor_id))
        return generated_readings

    def step(self, current_time: float) -> List[SensorReading]:
        """Advance simulation clock to current_time (ISensorEngine contract)."""
        if current_time < self._current_simulation_time:
            raise ValueError(
                f"Cannot step backward in time: current={self._current_simulation_time}, requested={current_time}."
            )
        dt = current_time - self._current_simulation_time
        return self.advance(dt)

    def reset(self) -> None:
        """Reset simulation clock and all registered sensors to initial conditions."""
        self._current_simulation_time = 0.0
        self._latest_readings.clear()
        for sensor in self._sensors.values():
            sensor.reset()

    @classmethod
    def create_default_engine(cls, seed_offset: int = 0) -> "SensorEngine":
        """Factory method: create engine pre-configured with the four default virtual sensors."""
        engine = cls()
        gas_cfg = get_default_gas_config(random_seed=42 + seed_offset)
        temp_cfg = get_default_temperature_config(random_seed=43 + seed_offset)
        hum_cfg = get_default_humidity_config(random_seed=44 + seed_offset)
        smoke_cfg = get_default_smoke_config(random_seed=45 + seed_offset)

        engine.register_sensor(GasSensor(gas_cfg))
        engine.register_sensor(TemperatureSensor(temp_cfg))
        engine.register_sensor(HumiditySensor(hum_cfg))
        engine.register_sensor(SmokeSensor(smoke_cfg))
        return engine
