"""Virtual gas concentration sensor implementation."""

from typing import Optional

from rl_ack.common.enums import SensorType
from rl_ack.sensors.base_sensor import BaseSensor
from rl_ack.sensors.config import SensorConfig, get_default_gas_config


class GasSensor(BaseSensor):
    """Virtual gas sensor measuring atmospheric gas concentration in parts per million (ppm).

    Default specification:
    - Sensor ID: GAS-001
    - Sensor Type: gas
    - Unit: ppm
    - Default Sampling Interval: 1.0s
    - Operating Range: [0.0, 1000.0] ppm
    - Baseline: 50.0 ppm
    - Incident Threshold: 400.0 ppm
    """

    def __init__(self, config: Optional[SensorConfig] = None) -> None:
        if config is None:
            config = get_default_gas_config()
        if config.sensor_type != SensorType.GAS:
            raise ValueError(f"GasSensor requires SensorType.GAS, got {config.sensor_type}")
        super().__init__(config)

    def simulate_gas_leak(
        self,
        target_ppm: float = 650.0,
        duration_s: Optional[float] = None,
        start_time: Optional[float] = None,
    ) -> None:
        """Simulate an environmental gas leak driving concentration above threshold."""
        self.trigger_event(target_value=target_ppm, duration_s=duration_s, start_time=start_time)
