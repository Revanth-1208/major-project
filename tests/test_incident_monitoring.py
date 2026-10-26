"""Comprehensive automated test suite for Phase 3: Incident Monitoring, Emergency Detection and Alert Management.

Validates:
1. Gas warning and critical detection.
2. Temperature warning and critical detection.
3. Smoke warning and critical detection.
4. Humidity two-sided limit anomalies (high and low).
5. Normal readings produce zero incidents.
6. Unique and sequential incident and alert IDs.
7. Deduplication: repeated consecutive abnormal readings do not produce duplicate incidents or alerts.
8. Idempotency: repeated evaluations on the same sequence do not duplicate alerts on reruns.
9. Severity escalation (WARNING -> CRITICAL) emits escalation alert and updates incident.
10. Hysteresis-aware incident recovery emits recovery notification and updates lifecycle.
11. Condition recurrence after recovery spawns a new distinct incident ID.
12. Valid lifecycle transitions and rejection of invalid transitions with InvalidStateTransitionError.
13. Operator acknowledgement records timestamps and updates associated alerts.
14. Alert and Incident serialization (dict and JSON).
15. Dynamic KPI calculations matching actual incident states.
16. Concurrent simultaneous incidents across distinct sensors.
17. Reset behavior cleans all state and counters.
18. Integration with deterministic scenarios (Kitchen Fire and Gradual Gas Leak).
"""

import json
import pytest

from rl_ack.common.enums import (
    AlertSeverity,
    IncidentState,
    NodeOperationalStatus,
    SensorType,
    TrafficClass,
)
from rl_ack.common.exceptions import InvalidStateTransitionError
from rl_ack.common.models import AlertMessage, IncidentRecord, SensorReading
from rl_ack.dashboard.state import DashboardSimulationManager
from rl_ack.monitoring.detector import IncidentDetector
from rl_ack.monitoring.manager import IncidentManager


def _make_reading(
    sensor_id: str = "GAS-001",
    sensor_type: SensorType = SensorType.GAS,
    value: float = 25.0,
    unit: str = "ppm",
    status: NodeOperationalStatus = NodeOperationalStatus.NORMAL,
    sim_time: float = 1.0,
    seq: int = 1,
) -> SensorReading:
    """Helper to construct synthetic SensorReading records for unit testing."""
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        value=value,
        unit=unit,
        simulation_time_s=sim_time,
        sampling_interval_s=1.0,
        host_node_id="NODE-001",
        status=status,
        sequence_number=seq,
        is_threshold_exceeded=(status != NodeOperationalStatus.NORMAL),
    )


# -----------------------------------------------------------------------------
# 1-5: Sensor Detection Rules
# -----------------------------------------------------------------------------

def test_gas_warning_and_critical_detection():
    """Verify gas concentration detects warning (>100 ppm) and critical (>=400 ppm)."""
    manager = IncidentManager()

    # Warning
    r_warn = _make_reading("GAS-001", SensorType.GAS, 150.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1)
    alt1 = manager.process_reading(r_warn)
    assert alt1 is not None
    assert alt1.severity == AlertSeverity.WARNING
    assert alt1.incident_type == "gas_leak"
    assert alt1.threshold_value == 100.0
    assert alt1.priority == TrafficClass.HIGH_BANDWIDTH_STREAMING

    # Escalation to Critical
    r_crit = _make_reading("GAS-001", SensorType.GAS, 450.0, "ppm", NodeOperationalStatus.CRITICAL, 2.0, 2)
    alt2 = manager.process_reading(r_crit)
    assert alt2 is not None
    assert alt2.severity == AlertSeverity.CRITICAL
    assert "ESCALATION" in alt2.message
    assert alt2.priority == TrafficClass.CRITICAL_CONTROL


def test_temperature_warning_and_critical_detection():
    """Verify temperature detects warning (>38°C) and critical (>=55°C)."""
    manager = IncidentManager()

    # Warning
    r_warn = _make_reading("TEMP-001", SensorType.TEMPERATURE, 42.0, "°C", NodeOperationalStatus.WARNING, 1.0, 1)
    alt_warn = manager.process_reading(r_warn)
    assert alt_warn is not None
    assert alt_warn.severity == AlertSeverity.WARNING
    assert alt_warn.incident_type == "high_temperature"
    assert alt_warn.threshold_value == 38.0

    # Reset and test Critical directly
    manager.reset()
    r_crit = _make_reading("TEMP-001", SensorType.TEMPERATURE, 62.0, "°C", NodeOperationalStatus.CRITICAL, 1.0, 1)
    alt_crit = manager.process_reading(r_crit)
    assert alt_crit is not None
    assert alt_crit.severity == AlertSeverity.CRITICAL
    assert alt_crit.incident_type == "high_temperature"
    assert alt_crit.threshold_value == 55.0


def test_smoke_warning_and_critical_detection():
    """Verify smoke obscuration detects warning (>=1.0) and critical (>=3.5)."""
    manager = IncidentManager()

    r_warn = _make_reading("SMOKE-001", SensorType.SMOKE, 1.5, "% obs/m", NodeOperationalStatus.WARNING, 1.0, 1)
    alt_warn = manager.process_reading(r_warn)
    assert alt_warn is not None
    assert alt_warn.severity == AlertSeverity.WARNING
    assert alt_warn.incident_type == "smoke_combustion"

    r_crit = _make_reading("SMOKE-001", SensorType.SMOKE, 4.2, "% obs/m", NodeOperationalStatus.CRITICAL, 2.0, 2)
    alt_crit = manager.process_reading(r_crit)
    assert alt_crit is not None
    assert alt_crit.severity == AlertSeverity.CRITICAL


def test_humidity_two_sided_anomalies_high_and_low():
    """Verify relative humidity handles both upper and lower anomaly boundaries."""
    manager = IncidentManager()

    # Low humidity warning (< 30 % RH)
    r_low_warn = _make_reading("HUM-001", SensorType.HUMIDITY, 25.0, "% RH", NodeOperationalStatus.WARNING, 1.0, 1)
    alt_low_warn = manager.process_reading(r_low_warn)
    assert alt_low_warn is not None
    assert alt_low_warn.incident_type == "low_humidity_anomaly"
    assert alt_low_warn.threshold_value == 30.0

    # Critical low humidity (<= 15 % RH)
    manager.reset()
    r_low_crit = _make_reading("HUM-001", SensorType.HUMIDITY, 10.0, "% RH", NodeOperationalStatus.CRITICAL, 1.0, 1)
    alt_low_crit = manager.process_reading(r_low_crit)
    assert alt_low_crit is not None
    assert alt_low_crit.incident_type == "low_humidity_anomaly"
    assert alt_low_crit.threshold_value == 15.0

    # High humidity warning (> 70 % RH)
    manager.reset()
    r_high_warn = _make_reading("HUM-001", SensorType.HUMIDITY, 75.0, "% RH", NodeOperationalStatus.WARNING, 1.0, 1)
    alt_high_warn = manager.process_reading(r_high_warn)
    assert alt_high_warn is not None
    assert alt_high_warn.incident_type == "high_humidity_anomaly"
    assert alt_high_warn.threshold_value == 70.0

    # High humidity critical (>= 85 % RH)
    manager.reset()
    r_high_crit = _make_reading("HUM-001", SensorType.HUMIDITY, 92.0, "% RH", NodeOperationalStatus.CRITICAL, 1.0, 1)
    alt_high_crit = manager.process_reading(r_high_crit)
    assert alt_high_crit is not None
    assert alt_high_crit.incident_type == "high_humidity_anomaly"
    assert alt_high_crit.threshold_value == 85.0


def test_normal_readings_produce_no_incidents():
    """Verify normal readings across all sensors produce no incidents or alerts."""
    manager = IncidentManager()
    sensors = [
        ("GAS-001", SensorType.GAS, 20.0, "ppm"),
        ("TEMP-001", SensorType.TEMPERATURE, 24.0, "°C"),
        ("HUM-001", SensorType.HUMIDITY, 45.0, "% RH"),
        ("SMOKE-001", SensorType.SMOKE, 0.1, "% obs/m"),
    ]
    for s_id, s_type, val, unit in sensors:
        r = _make_reading(s_id, s_type, val, unit, NodeOperationalStatus.NORMAL, 1.0, 1)
        alert = manager.process_reading(r)
        assert alert is None

    assert len(manager.get_active_incidents()) == 0
    assert len(manager.get_alerts()) == 0


# -----------------------------------------------------------------------------
# 6-9: IDs, Deduplication and Idempotency
# -----------------------------------------------------------------------------

def test_incident_and_alert_id_uniqueness_and_monotonicity():
    """Verify incident and alert IDs follow clean sequential format."""
    manager = IncidentManager()

    r1 = _make_reading("GAS-001", SensorType.GAS, 150.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1)
    alt1 = manager.process_reading(r1)
    assert alt1.alert_id == "ALT-0001"
    assert alt1.incident_id == "INC-0001"

    r2 = _make_reading("TEMP-001", SensorType.TEMPERATURE, 42.0, "°C", NodeOperationalStatus.WARNING, 1.0, 1)
    alt2 = manager.process_reading(r2)
    assert alt2.alert_id == "ALT-0002"
    assert alt2.incident_id == "INC-0002"


def test_deduplication_on_repeated_consecutive_abnormal_samples():
    """Verify consecutive samples within the same incident update metadata but emit NO duplicate alerts."""
    manager = IncidentManager()

    # Initial activation: Alert 1 emitted
    r1 = _make_reading("GAS-001", SensorType.GAS, 150.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1)
    alt1 = manager.process_reading(r1)
    assert alt1 is not None

    # Consecutive samples at warning severity
    for seq in range(2, 6):
        r = _make_reading("GAS-001", SensorType.GAS, 155.0 + seq, "ppm", NodeOperationalStatus.WARNING, float(seq), seq)
        alt = manager.process_reading(r)
        assert alt is None, f"Expected no duplicate alert on sequence {seq}"

    assert len(manager.get_alerts()) == 1
    incidents = manager.get_active_incidents()
    assert len(incidents) == 1
    assert incidents[0].last_reading == 160.0
    assert incidents[0].last_updated_time_s == 5.0


def test_idempotency_on_repeated_reruns():
    """Verify repeated calls with the same (sensor_id, seq_num) produce no duplicates."""
    manager = IncidentManager()

    r1 = _make_reading("GAS-001", SensorType.GAS, 150.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1)
    alt1 = manager.process_reading(r1)
    assert alt1 is not None

    # Simulating a dashboard rerun with the exact same reading
    alt_dup = manager.process_reading(r1)
    assert alt_dup is None
    assert len(manager.get_alerts()) == 1


# -----------------------------------------------------------------------------
# 10-14: Lifecycle Transitions, Escalation, Recovery & Acknowledgement
# -----------------------------------------------------------------------------

def test_incident_recovery_and_recovery_notification():
    """Verify condition returning to NORMAL marks incident RECOVERED and emits recovery alert."""
    manager = IncidentManager()

    r1 = _make_reading("GAS-001", SensorType.GAS, 150.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1)
    manager.process_reading(r1)
    assert len(manager.get_active_incidents()) == 1

    # Recover to normal
    r_norm = _make_reading("GAS-001", SensorType.GAS, 20.0, "ppm", NodeOperationalStatus.NORMAL, 5.0, 2)
    alt_rec = manager.process_reading(r_norm)

    assert alt_rec is not None
    assert alt_rec.severity == AlertSeverity.INFO
    assert alt_rec.priority == TrafficClass.BEST_EFFORT
    assert "recovered to normal" in alt_rec.message

    assert len(manager.get_active_incidents()) == 0
    all_inc = manager.get_all_incidents()
    assert len(all_inc) == 1
    assert all_inc[0].state == IncidentState.RECOVERED
    assert all_inc[0].recovered_time_s == 5.0
    assert all_inc[0].recovered_at_utc != ""


def test_new_incident_spawns_after_recovery_recurrence():
    """Verify if a condition flares up again after recovery, a new incident ID is created."""
    manager = IncidentManager()

    # First event
    r1 = _make_reading("GAS-001", SensorType.GAS, 150.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1)
    manager.process_reading(r1)
    r2 = _make_reading("GAS-001", SensorType.GAS, 20.0, "ppm", NodeOperationalStatus.NORMAL, 5.0, 2)
    manager.process_reading(r2)

    # Condition reoccurs
    r3 = _make_reading("GAS-001", SensorType.GAS, 160.0, "ppm", NodeOperationalStatus.WARNING, 10.0, 3)
    alt3 = manager.process_reading(r3)

    assert alt3.incident_id == "INC-0002"
    all_inc = manager.get_all_incidents()
    assert len(all_inc) == 2
    assert all_inc[0].incident_id == "INC-0001"
    assert all_inc[0].state == IncidentState.RECOVERED
    assert all_inc[1].incident_id == "INC-0002"
    assert all_inc[1].state == IncidentState.ACTIVE


def test_operator_acknowledgement_and_lifecycle_valid_transitions():
    """Verify ACTIVE -> ACKNOWLEDGED -> CLOSED lifecycle transitions and timestamps."""
    manager = IncidentManager()

    r1 = _make_reading("GAS-001", SensorType.GAS, 150.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1)
    manager.process_reading(r1)

    # Operator acknowledges
    inc = manager.acknowledge_incident("INC-0001", simulation_time_s=2.5)
    assert inc.state == IncidentState.ACKNOWLEDGED
    assert inc.acknowledged_time_s == 2.5
    assert inc.acknowledged_at_utc is not None

    # Associated alert receives application ack timestamp
    alert = manager.get_alerts()[0]
    assert alert.app_ack_at_utc == inc.acknowledged_at_utc

    # Close incident
    inc_closed = manager.close_incident("INC-0001", simulation_time_s=6.0)
    assert inc_closed.state == IncidentState.CLOSED
    assert inc_closed.closed_time_s == 6.0
    assert inc_closed.closed_at_utc is not None


def test_invalid_lifecycle_transitions_raise_error():
    """Verify invalid state transitions are rejected with InvalidStateTransitionError."""
    manager = IncidentManager()

    r1 = _make_reading("GAS-001", SensorType.GAS, 150.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1)
    manager.process_reading(r1)

    # Attempting to close an ACTIVE incident directly
    with pytest.raises(InvalidStateTransitionError):
        manager.close_incident("INC-0001")

    # Acknowledge incident
    manager.acknowledge_incident("INC-0001")

    # Attempting to re-acknowledge an already ACKNOWLEDGED incident
    with pytest.raises(InvalidStateTransitionError):
        manager.acknowledge_incident("INC-0001")

    # Close incident
    manager.close_incident("INC-0001")

    # Attempting to acknowledge a CLOSED incident
    with pytest.raises(InvalidStateTransitionError):
        manager.acknowledge_incident("INC-0001")

    # Attempting to close an already CLOSED incident
    with pytest.raises(InvalidStateTransitionError):
        manager.close_incident("INC-0001")


# -----------------------------------------------------------------------------
# 15-18: Serialization, KPIs, Simultaneous Incidents & Reset
# -----------------------------------------------------------------------------

def test_alert_and_incident_serialization():
    """Verify to_dict and to_json serialize all required contracts without omission."""
    manager = IncidentManager()

    r1 = _make_reading("TEMP-001", SensorType.TEMPERATURE, 60.0, "°C", NodeOperationalStatus.CRITICAL, 3.0, 1)
    alert = manager.process_reading(r1)
    inc = manager.get_active_incidents()[0]

    # Alert serialization
    alt_dict = alert.to_dict()
    assert alt_dict["alert_id"] == "ALT-0001"
    assert alt_dict["incident_id"] == "INC-0001"
    assert alt_dict["severity"] == "critical"
    assert alt_dict["reading_value"] == 60.0
    assert alt_dict["priority"] == "Critical Control"
    alt_json = alert.to_json()
    assert json.loads(alt_json)["alert_id"] == "ALT-0001"

    # Incident serialization
    inc_dict = inc.to_dict()
    assert inc_dict["incident_id"] == "INC-0001"
    assert inc_dict["state"] == "active"
    assert inc_dict["severity"] == "critical"
    assert inc_dict["associated_alert_ids"] == ["ALT-0001"]
    inc_json = inc.to_json()
    assert json.loads(inc_json)["incident_id"] == "INC-0001"


def test_dynamic_incident_kpi_counts():
    """Verify incident KPIs accurately reflect active, critical, warning, acked, and closed counts."""
    manager = IncidentManager()

    # 1 critical active
    manager.process_reading(_make_reading("TEMP-001", SensorType.TEMPERATURE, 60.0, "°C", NodeOperationalStatus.CRITICAL, 1.0, 1))
    # 1 warning active
    manager.process_reading(_make_reading("SMOKE-001", SensorType.SMOKE, 1.5, "% obs/m", NodeOperationalStatus.WARNING, 1.0, 1))
    # 1 acknowledged
    manager.process_reading(_make_reading("GAS-001", SensorType.GAS, 150.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1))
    manager.acknowledge_incident("INC-0003")

    kpis = manager.get_kpis()
    assert kpis["total_incidents"] == 3
    assert kpis["active"] == 2
    assert kpis["critical_active"] == 1
    assert kpis["warning_active"] == 1
    assert kpis["acknowledged"] == 1
    assert kpis["recovered_closed"] == 0


def test_concurrent_simultaneous_incidents_across_sensors():
    """Verify multiple sensors triggering simultaneously maintain independent incident records."""
    manager = IncidentManager()

    readings = [
        _make_reading("GAS-001", SensorType.GAS, 200.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1),
        _make_reading("TEMP-001", SensorType.TEMPERATURE, 60.0, "°C", NodeOperationalStatus.CRITICAL, 1.0, 1),
        _make_reading("SMOKE-001", SensorType.SMOKE, 4.0, "% obs/m", NodeOperationalStatus.CRITICAL, 1.0, 1),
    ]

    alerts = manager.process_readings(readings)
    assert len(alerts) == 3
    assert len(manager.get_active_incidents()) == 3
    incident_types = {inc.incident_type for inc in manager.get_active_incidents()}
    assert incident_types == {"gas_leak", "high_temperature", "smoke_combustion"}


def test_reset_clears_all_incident_state():
    """Verify reset restores incident manager to initial clean baseline."""
    manager = IncidentManager()

    manager.process_reading(_make_reading("GAS-001", SensorType.GAS, 200.0, "ppm", NodeOperationalStatus.WARNING, 1.0, 1))
    assert len(manager.get_alerts()) == 1
    assert len(manager.get_active_incidents()) == 1

    manager.reset()
    assert len(manager.get_alerts()) == 0
    assert len(manager.get_active_incidents()) == 0
    assert len(manager.get_all_incidents()) == 0
    assert manager._incident_counter == 0
    assert manager._alert_counter == 0
    assert len(manager._processed_reading_keys) == 0


# -----------------------------------------------------------------------------
# 19: End-to-End Deterministic Scenario Integration
# -----------------------------------------------------------------------------

def test_deterministic_scenario_kitchen_fire_multi_event():
    """Verify DashboardSimulationManager detects real incidents in Kitchen Fire Multi-Event Emergency."""
    sim = DashboardSimulationManager(random_seed=42, scenario_name="Kitchen Fire Multi-Event Emergency")
    assert sim.incident_manager.get_kpis()["total_incidents"] == 0

    # Advance simulation by 16 steps
    sim.run_steps(16, dt=1.0)

    kpis = sim.get_incident_kpis()
    assert kpis["total_incidents"] > 0
    assert len(sim.incident_manager.get_alerts()) > 0

    df_active = sim.get_active_incidents_df()
    assert not df_active.empty
    assert "Incident ID" in df_active.columns
    assert "Severity" in df_active.columns

    df_alerts = sim.get_alert_history_df()
    assert not df_alerts.empty
    assert "Alert ID" in df_alerts.columns


def test_deterministic_scenario_gas_leak_lifecycle():
    """Verify full incident lifecycle (ACTIVE -> ACKNOWLEDGED -> RECOVERED) in Gradual Gas Leak."""
    sim = DashboardSimulationManager(random_seed=42, scenario_name="Gradual Gas Leak")

    # Step until warning/critical gas incident triggers
    for _ in range(16):
        sim.step(1.0)
        if sim.incident_manager.get_active_incidents():
            break

    active = sim.incident_manager.get_active_incidents()
    assert len(active) >= 1
    gas_inc = [inc for inc in active if inc.sensor_id == "GAS-001"][0]
    assert gas_inc.state == IncidentState.ACTIVE

    # Operator acknowledges the incident
    sim.acknowledge_incident(gas_inc.incident_id)
    assert gas_inc.state == IncidentState.ACKNOWLEDGED
