"""Virtual temperature sensor implementation."""

from typing import Optional

from rl_ack.common.enums import SensorType
from rl_ack.sensors.base_sensor import BaseSensor
from rl_ack.sensors.config import SensorConfig, get_default_temperature_config


class TemperatureSensor(BaseSensor):
    """Virtual temperature sensor measuring ambient temperature in degrees Celsius (°C).

    Default specification:
    - Sensor ID: TEMP-001
    - Sensor Type: temperature
    - Unit: °C
    - Default Sampling Interval: 1.0s
    - Operating Range: [-20.0, 100.0] °C
    - Baseline: 24.0 °C
    - Incident Threshold: 55.0 °C
    """

    def __init__(self, config: Optional[SensorConfig] = None) -> None:
        if config is None:
            config = get_default_temperature_config()
        if config.sensor_type != SensorType.TEMPERATURE:
            raise ValueError(f"TemperatureSensor requires SensorType.TEMPERATURE, got {config.sensor_type}")
        super().__init__(config)

    def simulate_heat_spike(
        self,
        target_temp_c: float = 75.0,
        duration_s: Optional[float] = None,
        start_time: Optional[float] = None,
    ) -> None:
        """Simulate an environmental heating event driving temperature above threshold."""
        self.trigger_event(target_value=target_temp_c, duration_s=duration_s, start_time=start_time)
