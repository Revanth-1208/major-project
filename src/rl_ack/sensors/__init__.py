"""Module 2: Virtual Sensor Engine.

Implements realistic dynamic sensor simulation for:
- Gas sensor (ppm)
- Temperature sensor (°C)
- Relative humidity sensor (% RH)
- Smoke sensor (% obs/m)
Coordinated by SensorEngine with discrete-event sampling and environmental event injection.

NOTE: Operating ranges and incident thresholds are simulation parameters,
not certified life-safety limits.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path even when running from inside src/rl_ack
_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from rl_ack.sensors.config import (
    SensorConfig,
    get_default_gas_config,
    get_default_temperature_config,
    get_default_humidity_config,
    get_default_smoke_config,
)
from rl_ack.sensors.base_sensor import BaseSensor
from rl_ack.sensors.gas_sensor import GasSensor
from rl_ack.sensors.temperature_sensor import TemperatureSensor
from rl_ack.sensors.humidity_sensor import HumiditySensor
from rl_ack.sensors.smoke_sensor import SmokeSensor
from rl_ack.sensors.sensor_engine import SensorEngine
from rl_ack.sensors.thresholds import (
    StatusClassifier,
    ThresholdDirection,
    ThresholdRule,
    create_classifier_for_sensor,
)
from rl_ack.sensors.scenarios import (
    ScenarioEvent,
    ScenarioRunner,
    SensorScenario,
    scenario_abnormal_temperature_rise,
    scenario_gradual_gas_leak,
    scenario_humidity_change,
    scenario_increasing_smoke,
    scenario_kitchen_fire_multi_event,
    scenario_normal_operation,
)

MODULE_NAME = "sensors"
MODULE_ID = 2
IS_IMPLEMENTED = True

__all__ = [
    "MODULE_NAME",
    "MODULE_ID",
    "IS_IMPLEMENTED",
    "SensorConfig",
    "get_default_gas_config",
    "get_default_temperature_config",
    "get_default_humidity_config",
    "get_default_smoke_config",
    "BaseSensor",
    "GasSensor",
    "TemperatureSensor",
    "HumiditySensor",
    "SmokeSensor",
    "SensorEngine",
    "ThresholdRule",
    "ThresholdDirection",
    "StatusClassifier",
    "create_classifier_for_sensor",
    "ScenarioEvent",
    "SensorScenario",
    "ScenarioRunner",
    "scenario_normal_operation",
    "scenario_gradual_gas_leak",
    "scenario_abnormal_temperature_rise",
    "scenario_increasing_smoke",
    "scenario_humidity_change",
    "scenario_kitchen_fire_multi_event",
]
