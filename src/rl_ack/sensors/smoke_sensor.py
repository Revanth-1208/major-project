"""Virtual optical smoke obscuration sensor implementation."""

from typing import Optional

from rl_ack.common.enums import SensorType
from rl_ack.sensors.base_sensor import BaseSensor
from rl_ack.sensors.config import SensorConfig, get_default_smoke_config


class SmokeSensor(BaseSensor):
    """Virtual smoke sensor measuring optical smoke obscuration in percentage obscuration per meter (% obs/m).

    Simulates photoelectric smoke obscuration dynamics per standard detector profiles (UL 217 / EN 54).

    Default specification:
    - Sensor ID: SMOKE-001
    - Sensor Type: smoke
    - Unit: % obs/m
    - Default Sampling Interval: 1.0s
    - Operating Range: [0.0, 20.0] % obs/m
    - Baseline: 0.2 % obs/m (clean indoor air)
    - Incident Threshold: 3.5 % obs/m
    """

    def __init__(self, config: Optional[SensorConfig] = None) -> None:
        if config is None:
            config = get_default_smoke_config()
        if config.sensor_type != SensorType.SMOKE:
            raise ValueError(f"SmokeSensor requires SensorType.SMOKE, got {config.sensor_type}")
        super().__init__(config)

    def simulate_smoke_event(
        self,
        target_obs: float = 6.0,
        duration_s: Optional[float] = None,
        start_time: Optional[float] = None,
    ) -> None:
        """Simulate an environmental combustion event driving smoke obscuration above threshold."""
        self.trigger_event(target_value=target_obs, duration_s=duration_s, start_time=start_time)
