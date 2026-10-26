"""Incident and Alert Manager for RL-ACK smart kitchen IoT network.

Coordinates incident lifecycle (ACTIVE -> ACKNOWLEDGED -> RECOVERED -> CLOSED),
suppresses duplicate alerts on consecutive sensor samples, escalates severity on
worsening conditions, records timestamps, and generates emergency alerts.
"""

from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from rl_ack.common.enums import (
    AlertSeverity,
    IncidentState,
    TrafficClass,
)
from rl_ack.common.exceptions import InvalidStateTransitionError
from rl_ack.common.models import AlertMessage, IncidentRecord, SensorReading
from rl_ack.monitoring.detector import IncidentDetector


class IncidentManager:
    """Manages incident lifecycle, alert generation, and duplicate prevention.

    Enforces strict decoupling: operates independently of the DQN controller
    and MQTT transport broker.
    """

    def __init__(
        self,
        detector: Optional[IncidentDetector] = None,
        max_alerts: int = 500,
    ) -> None:
        """Initialize IncidentManager with an optional detector and alert retention limit."""
        self.detector = detector or IncidentDetector(manager=self)
        if self.detector.manager is None:
            self.detector.manager = self

        self.max_alerts = max_alerts
        self._active_incidents: Dict[str, IncidentRecord] = {}
        self._all_incidents: Dict[str, IncidentRecord] = {}
        self._alerts: deque[AlertMessage] = deque(maxlen=self.max_alerts)

        self._incident_counter: int = 0
        self._alert_counter: int = 0
        self._processed_reading_keys: Set[Tuple[str, int]] = set()

    def _next_incident_id(self) -> str:
        """Generate monotonically increasing unique incident ID."""
        self._incident_counter += 1
        return f"INC-{self._incident_counter:04d}"

    def _next_alert_id(self) -> str:
        """Generate monotonically increasing unique alert ID."""
        self._alert_counter += 1
        return f"ALT-{self._alert_counter:04d}"

    def process_reading(self, reading: SensorReading) -> Optional[AlertMessage]:
        """Process a single sensor reading and manage incident lifecycle.

        Duplicate Prevention & Escalation Rules:
        1. Repeated evaluations of the exact same reading sequence are ignored (idempotent).
        2. Normal reading while an active incident exists transitions it to RECOVERED,
           and emits a recovery notification.
        3. Abnormal reading when no incident exists creates a new ACTIVE incident and emits an alert.
        4. Abnormal reading when an active incident exists:
           - If severity escalates (WARNING -> CRITICAL): updates incident severity to CRITICAL
             and emits an escalation alert.
           - If severity is unchanged or lower: updates latest reading on the incident,
             without generating a duplicate alert.
        """
        # Idempotency check to avoid re-processing on Streamlit reruns
        reading_key = (reading.sensor_id, reading.sequence_number)
        if reading_key in self._processed_reading_keys:
            return None
        self._processed_reading_keys.add(reading_key)

        severity, threshold, inc_type, msg = self.detector.detect_condition(reading)

        # ---------------------------------------------------------------------
        # Condition A: Reading is Abnormal (WARNING or CRITICAL)
        # ---------------------------------------------------------------------
        if severity is not None:
            existing = self._active_incidents.get(reading.sensor_id)

            if existing is not None and existing.state in (
                IncidentState.ACTIVE,
                IncidentState.ACKNOWLEDGED,
            ):
                # Check for severity escalation (WARNING -> CRITICAL)
                if (
                    existing.severity == AlertSeverity.WARNING
                    and severity == AlertSeverity.CRITICAL
                ):
                    existing.severity = AlertSeverity.CRITICAL
                    existing.last_reading = reading.value
                    existing.last_updated_time_s = reading.simulation_time_s

                    alt_id = self._next_alert_id()
                    existing.associated_alert_ids.append(alt_id)

                    esc_alert = AlertMessage(
                        alert_id=alt_id,
                        incident_id=existing.incident_id,
                        sensor_id=reading.sensor_id,
                        severity=AlertSeverity.CRITICAL,
                        reading_value=reading.value,
                        threshold_value=threshold,
                        unit=reading.unit,
                        incident_type=inc_type,
                        message=f"ESCALATION: {msg}",
                        simulation_time_s=reading.simulation_time_s,
                        priority=TrafficClass.CRITICAL_CONTROL,
                        host_node=reading.host_node_id,
                        sensor_type=reading.sensor_type,
                    )
                    self._alerts.append(esc_alert)
                    return esc_alert

                # Otherwise condition persists at same or lower severity:
                # Update reading without generating duplicate alert
                existing.last_reading = reading.value
                existing.last_updated_time_s = reading.simulation_time_s
                return None

            # New Incident: No active or acknowledged incident for this sensor
            inc_id = self._next_incident_id()
            alt_id = self._next_alert_id()

            priority = (
                TrafficClass.CRITICAL_CONTROL
                if severity == AlertSeverity.CRITICAL
                else TrafficClass.HIGH_BANDWIDTH_STREAMING
            )

            alert = AlertMessage(
                alert_id=alt_id,
                incident_id=inc_id,
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

            incident = IncidentRecord(
                incident_id=inc_id,
                sensor_id=reading.sensor_id,
                incident_type=inc_type,
                state=IncidentState.ACTIVE,
                severity=severity,
                trigger_reading=reading.value,
                trigger_threshold=threshold,
                unit=reading.unit,
                trigger_time_s=reading.simulation_time_s,
                last_reading=reading.value,
                last_updated_time_s=reading.simulation_time_s,
                associated_alert_ids=[alt_id],
            )

            self._active_incidents[reading.sensor_id] = incident
            self._all_incidents[inc_id] = incident
            self._alerts.append(alert)
            return alert

        # ---------------------------------------------------------------------
        # Condition B: Reading is NORMAL (severity is None)
        # ---------------------------------------------------------------------
        existing = self._active_incidents.get(reading.sensor_id)
        if existing is not None and existing.state in (
            IncidentState.ACTIVE,
            IncidentState.ACKNOWLEDGED,
        ):
            now_utc = datetime.now(timezone.utc).isoformat()
            existing.state = IncidentState.RECOVERED
            existing.recovered_time_s = reading.simulation_time_s
            existing.cleared_time_s = reading.simulation_time_s
            existing.recovered_at_utc = now_utc
            existing.last_reading = reading.value
            existing.last_updated_time_s = reading.simulation_time_s

            # Remove from active tracking so future flare-up gets a brand new incident ID
            del self._active_incidents[reading.sensor_id]

            # Emit recovery notification
            alt_id = self._next_alert_id()
            existing.associated_alert_ids.append(alt_id)

            rec_msg = f"{reading.sensor_id} recovered to normal: {reading.value:.2f} {reading.unit}"
            rec_alert = AlertMessage(
                alert_id=alt_id,
                incident_id=existing.incident_id,
                sensor_id=reading.sensor_id,
                severity=AlertSeverity.INFO,
                reading_value=reading.value,
                threshold_value=threshold,
                unit=reading.unit,
                incident_type=existing.incident_type,
                message=rec_msg,
                simulation_time_s=reading.simulation_time_s,
                priority=TrafficClass.BEST_EFFORT,
                recovered_at_utc=now_utc,
                host_node=reading.host_node_id,
                sensor_type=reading.sensor_type,
            )
            self._alerts.append(rec_alert)
            return rec_alert

        return None

    def process_readings(
        self, readings: List[SensorReading]
    ) -> List[AlertMessage]:
        """Process a batch of readings, returning all alerts generated."""
        alerts: List[AlertMessage] = []
        for r in readings:
            alert = self.process_reading(r)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def acknowledge_incident(
        self, incident_id: str, simulation_time_s: Optional[float] = None
    ) -> IncidentRecord:
        """Operator acknowledges an active incident.

        Valid transition: ACTIVE -> ACKNOWLEDGED
        Raises:
            KeyError: if incident_id not found.
            InvalidStateTransitionError: if incident is not ACTIVE.
        """
        incident = self._all_incidents.get(incident_id)
        if incident is None:
            raise KeyError(f"Incident {incident_id} not found.")

        if incident.state != IncidentState.ACTIVE:
            raise InvalidStateTransitionError(
                f"Cannot acknowledge incident {incident_id} in state '{incident.state.value}'. "
                f"Only ACTIVE incidents can be acknowledged."
            )

        now_utc = datetime.now(timezone.utc).isoformat()
        incident.state = IncidentState.ACKNOWLEDGED
        incident.acknowledged_at_utc = now_utc
        incident.acknowledged_time_s = simulation_time_s

        # Annotate associated alerts with application-level ack timestamp
        for a in self._alerts:
            if a.incident_id == incident_id and a.app_ack_at_utc is None:
                a.app_ack_at_utc = now_utc

        return incident

    def close_incident(
        self, incident_id: str, simulation_time_s: Optional[float] = None
    ) -> IncidentRecord:
        """Operator closes an acknowledged or recovered incident.

        Valid transitions:
        - ACKNOWLEDGED -> CLOSED
        - RECOVERED -> CLOSED
        - CLEARED -> CLOSED

        Raises:
            KeyError: if incident_id not found.
            InvalidStateTransitionError: if transition is invalid.
        """
        incident = self._all_incidents.get(incident_id)
        if incident is None:
            raise KeyError(f"Incident {incident_id} not found.")

        if incident.state not in (
            IncidentState.ACKNOWLEDGED,
            IncidentState.RECOVERED,
            IncidentState.CLEARED,
        ):
            raise InvalidStateTransitionError(
                f"Cannot close incident {incident_id} in state '{incident.state.value}'. "
                f"Only ACKNOWLEDGED or RECOVERED incidents can be closed."
            )

        now_utc = datetime.now(timezone.utc).isoformat()
        incident.state = IncidentState.CLOSED
        incident.closed_at_utc = now_utc
        incident.closed_time_s = simulation_time_s

        if incident.sensor_id in self._active_incidents:
            del self._active_incidents[incident.sensor_id]

        return incident

    def get_active_incidents(self) -> List[IncidentRecord]:
        """Return all active or acknowledged incidents."""
        return [
            inc
            for inc in self._all_incidents.values()
            if inc.state in (IncidentState.ACTIVE, IncidentState.ACKNOWLEDGED)
        ]

    def get_all_incidents(self) -> List[IncidentRecord]:
        """Return all recorded incidents in chronological order."""
        return list(self._all_incidents.values())

    def get_active_incident_alerts(self) -> List[AlertMessage]:
        """Return alerts belonging to currently active or acknowledged incidents."""
        active_inc_ids = {
            inc.incident_id for inc in self.get_active_incidents()
        }
        return [a for a in self._alerts if a.incident_id in active_inc_ids]

    def get_alerts(self, limit: Optional[int] = None) -> List[AlertMessage]:
        """Return recorded alerts, latest first if limit is applied."""
        alerts = list(self._alerts)
        if limit is not None and limit > 0:
            return alerts[-limit:]
        return alerts

    def get_kpis(self) -> Dict[str, int]:
        """Derive live incident metrics directly from managed records."""
        active_cnt = sum(
            1 for inc in self._all_incidents.values() if inc.state == IncidentState.ACTIVE
        )
        crit_active = sum(
            1
            for inc in self._all_incidents.values()
            if inc.state == IncidentState.ACTIVE and inc.severity == AlertSeverity.CRITICAL
        )
        warn_active = sum(
            1
            for inc in self._all_incidents.values()
            if inc.state == IncidentState.ACTIVE and inc.severity == AlertSeverity.WARNING
        )
        ack_cnt = sum(
            1
            for inc in self._all_incidents.values()
            if inc.state == IncidentState.ACKNOWLEDGED
        )
        rec_closed = sum(
            1
            for inc in self._all_incidents.values()
            if inc.state
            in (IncidentState.RECOVERED, IncidentState.CLOSED, IncidentState.CLEARED)
        )

        return {
            "active": active_cnt,
            "critical_active": crit_active,
            "warning_active": warn_active,
            "acknowledged": ack_cnt,
            "recovered_closed": rec_closed,
            "total_incidents": len(self._all_incidents),
        }

    def reset(self) -> None:
        """Reset all incident tracking state to initial clean baseline."""
        self._active_incidents.clear()
        self._all_incidents.clear()
        self._alerts.clear()
        self._incident_counter = 0
        self._alert_counter = 0
        self._processed_reading_keys.clear()
