"""Virtual relative humidity sensor implementation."""

from typing import Optional

from rl_ack.common.enums import SensorType
from rl_ack.sensors.base_sensor import BaseSensor
from rl_ack.sensors.config import SensorConfig, get_default_humidity_config


class HumiditySensor(BaseSensor):
    """Virtual relative humidity sensor measuring atmospheric humidity percentage (% RH).

    Default specification:
    - Sensor ID: HUM-001
    - Sensor Type: humidity
    - Unit: % RH
    - Default Sampling Interval: 2.0s
    - Operating Range: [0.0, 100.0] % RH
    - Baseline: 45.0 % RH
    - Incident Threshold: 85.0 % RH
    """

    def __init__(self, config: Optional[SensorConfig] = None) -> None:
        if config is None:
            config = get_default_humidity_config()
        if config.sensor_type != SensorType.HUMIDITY:
            raise ValueError(f"HumiditySensor requires SensorType.HUMIDITY, got {config.sensor_type}")
        super().__init__(config)

    def simulate_humidity_change(
        self,
        target_pct: float,
        duration_s: Optional[float] = None,
        start_time: Optional[float] = None,
    ) -> None:
        """Simulate an environmental humidity change (e.g. steam surge or dry condition)."""
        self.trigger_event(target_value=target_pct, duration_s=duration_s, start_time=start_time)
