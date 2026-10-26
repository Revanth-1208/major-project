"""Incident detector module for virtual sensor networks.

Evaluates sensor readings against configured warning and critical thresholds
independently of the RL agent / DQN controller.
"""

from typing import Any, Dict, List, Optional, Tuple

from rl_ack.common.enums import (
    AlertSeverity,
    NodeOperationalStatus,
    SensorType,
    TrafficClass,
)
from rl_ack.common.interfaces import IIncidentDetector
from rl_ack.common.models import AlertMessage, SensorReading
from rl_ack.sensors.config import (
    SensorConfig,
    get_default_gas_config,
    get_default_humidity_config,
    get_default_smoke_config,
    get_default_temperature_config,
)


class IncidentDetector(IIncidentDetector):
    """Rule-based incident detector evaluating readings against configured thresholds.

    Architecture guarantee:
    Incident detection is completely decoupled from DQN action selection and
    MQTT transport mechanics.
    """

    def __init__(
        self,
        sensor_configs: Optional[Dict[str, SensorConfig]] = None,
        manager: Optional[Any] = None,
    ) -> None:
        """Initialize detector with authoritative sensor configurations."""
        if sensor_configs is None:
            self.sensor_configs: Dict[str, SensorConfig] = {
                "GAS-001": get_default_gas_config(),
                "TEMP-001": get_default_temperature_config(),
                "HUM-001": get_default_humidity_config(),
                "SMOKE-001": get_default_smoke_config(),
            }
        else:
            self.sensor_configs = dict(sensor_configs)

        self.manager = manager

    def set_sensor_config(self, sensor_id: str, config: SensorConfig) -> None:
        """Update or register configuration for a sensor instance."""
        self.sensor_configs[sensor_id] = config

    def detect_condition(
        self, reading: SensorReading
    ) -> Tuple[Optional[AlertSeverity], float, str, str]:
        """Evaluate a single reading against configured thresholds.

        Returns:
            Tuple of (severity, trigger_threshold, incident_type, message).
            If condition is NORMAL, severity is None and trigger_threshold is 0.0.
        """
        cfg = self.sensor_configs.get(reading.sensor_id)
        status = reading.status

        # If sensor status is NORMAL or UNAVAILABLE, no active threshold incident
        if status == NodeOperationalStatus.NORMAL:
            return None, 0.0, self._get_incident_type(reading, cfg), ""

        if status == NodeOperationalStatus.UNAVAILABLE:
            return None, 0.0, "sensor_offline", f"{reading.sensor_id} is unavailable/offline"

        # Determine threshold direction & values from config
        is_two_sided = False
        warn_high = 0.0
        crit_high = 0.0
        warn_low: Optional[float] = None
        crit_low: Optional[float] = None

        if cfg is not None:
            crit_high = cfg.critical_threshold or cfg.threshold
            warn_high = cfg.warning_threshold or (crit_high * 0.5)
            warn_low = cfg.warning_low
            crit_low = cfg.critical_low
            is_two_sided = cfg.sensor_type == SensorType.HUMIDITY
        else:
            # Fallback if config is missing
            crit_high = 400.0 if reading.sensor_id == "GAS-001" else 55.0
            warn_high = 100.0 if reading.sensor_id == "GAS-001" else 38.0

        val = reading.value
        unit = reading.unit

        if status == NodeOperationalStatus.CRITICAL:
            severity = AlertSeverity.CRITICAL
            if is_two_sided and crit_low is not None and val <= crit_low:
                incident_type = "low_humidity_anomaly"
                threshold = crit_low
                msg = f"CRITICAL Low humidity alert: {val:.1f} {unit} is below critical limit ({threshold:.1f} {unit})"
            elif is_two_sided and val >= crit_high:
                incident_type = "high_humidity_anomaly"
                threshold = crit_high
                msg = f"CRITICAL High humidity alert: {val:.1f} {unit} exceeds critical limit ({threshold:.1f} {unit})"
            else:
                incident_type = self._get_incident_type(reading, cfg)
                threshold = crit_high
                msg = f"CRITICAL {incident_type.replace('_', ' ').title()}: {val:.2f} {unit} exceeds limit ({threshold:.2f} {unit})"
            return severity, threshold, incident_type, msg

        elif status == NodeOperationalStatus.WARNING:
            severity = AlertSeverity.WARNING
            if is_two_sided and warn_low is not None and val < warn_low:
                incident_type = "low_humidity_anomaly"
                threshold = warn_low
                msg = f"Low humidity warning: {val:.1f} {unit} is below warning limit ({threshold:.1f} {unit})"
            elif is_two_sided and val > warn_high:
                incident_type = "high_humidity_anomaly"
                threshold = warn_high
                msg = f"High humidity warning: {val:.1f} {unit} exceeds warning limit ({threshold:.1f} {unit})"
            else:
                incident_type = self._get_incident_type(reading, cfg)
                threshold = warn_high
                msg = f"Warning: {incident_type.replace('_', ' ').title()}: {val:.2f} {unit} exceeds limit ({threshold:.2f} {unit})"
            return severity, threshold, incident_type, msg

        return None, 0.0, self._get_incident_type(reading, cfg), ""

    def _get_incident_type(
        self, reading: SensorReading, cfg: Optional[SensorConfig]
    ) -> str:
        """Map sensor identifier or type to categorical incident classification."""
        s_type = cfg.sensor_type if cfg else reading.sensor_type
        if s_type == SensorType.GAS or reading.sensor_id.startswith("GAS"):
            return "gas_leak"
        elif s_type == SensorType.TEMPERATURE or reading.sensor_id.startswith("TEMP"):
            return "high_temperature"
        elif s_type == SensorType.SMOKE or reading.sensor_id.startswith("SMOKE"):
            return "smoke_combustion"
        elif s_type == SensorType.HUMIDITY or reading.sensor_id.startswith("HUM"):
            return "humidity_anomaly"
        return "threshold_violation"

    def evaluate(self, reading: SensorReading) -> Optional[AlertMessage]:
        """Fulfill IIncidentDetector interface.

        If a manager is bound, delegates reading evaluation and alert generation
        to the manager. Otherwise returns an ad-hoc alert when condition is active.
        """
        if self.manager is not None:
            return self.manager.process_reading(reading)

        severity, threshold, inc_type, msg = self.detect_condition(reading)
        if severity is None:
            return None

        priority = (
            TrafficClass.CRITICAL_CONTROL
            if severity == AlertSeverity.CRITICAL
            else TrafficClass.HIGH_BANDWIDTH_STREAMING
        )

        return AlertMessage(
            alert_id=f"ALT-ADHOC-{reading.sensor_id}",
            sensor_id=reading.sensor_id,
            severity=severity,
            reading_value=reading.value,
            threshold_value=threshold,
            unit=reading.unit,
            incident_type=inc_type,
            message=msg,
            simulation_time_s=reading.simulation_time_s,
            priority=priority,
            host_node=reading.host_node_id,
            sensor_type=reading.sensor_type,
        )

    def get_active_incidents(self) -> List[AlertMessage]:
        """Fulfill IIncidentDetector interface."""
        if self.manager is not None:
            return self.manager.get_active_incident_alerts()
        return []
