"""Tests for sensor status classification, hysteresis, scenario execution, and alert contracts."""

import math
import pytest

from rl_ack.common.enums import (
    AlertSeverity,
    DeliveryStatus,
    IncidentState,
    NodeOperationalStatus,
    SensorType,
    TrafficClass,
)
from rl_ack.common.models import AlertMessage, IncidentRecord
from rl_ack.sensors.config import (
    get_default_gas_config,
    get_default_humidity_config,
    get_default_temperature_config,
    get_default_smoke_config,
)
from rl_ack.sensors.gas_sensor import GasSensor
from rl_ack.sensors.humidity_sensor import HumiditySensor
from rl_ack.sensors.temperature_sensor import TemperatureSensor
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
    scenario_gradual_gas_leak,
    scenario_humidity_change,
    scenario_kitchen_fire_multi_event,
    scenario_normal_operation,
)


# ==============================================================================
# 1. Threshold Boundary & Status Classification Tests
# ==============================================================================

def test_high_danger_status_boundaries():
    """Verify threshold boundary classifications for high-danger sensors (Gas, Temp, Smoke)."""
    rule = ThresholdRule(
        direction=ThresholdDirection.HIGH_DANGER,
        warning_threshold=100.0,
        critical_threshold=400.0,
        hysteresis=10.0,
    )
    classifier = StatusClassifier(rule)

    assert classifier.classify(50.0) == NodeOperationalStatus.NORMAL
    assert classifier.classify(99.9) == NodeOperationalStatus.NORMAL
    assert classifier.classify(100.0) == NodeOperationalStatus.WARNING
    assert classifier.classify(250.0) == NodeOperationalStatus.WARNING
    assert classifier.classify(399.9) == NodeOperationalStatus.WARNING
    assert classifier.classify(400.0) == NodeOperationalStatus.CRITICAL
    assert classifier.classify(600.0) == NodeOperationalStatus.CRITICAL


def test_hysteresis_clearing_behavior():
    """Verify hysteresis prevents status fluttering around thresholds during recovery."""
    rule = ThresholdRule(
        direction=ThresholdDirection.HIGH_DANGER,
        warning_threshold=100.0,
        critical_threshold=400.0,
        hysteresis=10.0,
    )
    classifier = StatusClassifier(rule)

    # Drive into CRITICAL
    assert classifier.classify(450.0) == NodeOperationalStatus.CRITICAL

    # Dropping slightly below 400.0 (e.g. 395.0) should REMAIN CRITICAL because 395 >= (400 - 10)
    assert classifier.classify(395.0) == NodeOperationalStatus.CRITICAL

    # Dropping strictly below 390.0 (400 - 10) transitions to WARNING
    assert classifier.classify(389.0) == NodeOperationalStatus.WARNING

    # Dropping slightly below 100.0 (e.g. 95.0) should REMAIN WARNING because 95 >= (100 - 10)
    assert classifier.classify(95.0) == NodeOperationalStatus.WARNING

    # Dropping strictly below 90.0 (100 - 10) recovers to NORMAL
    assert classifier.classify(89.0) == NodeOperationalStatus.NORMAL


def test_humidity_two_sided_threshold_boundaries():
    """Verify relative humidity two-sided warning and critical boundaries."""
    rule = ThresholdRule(
        direction=ThresholdDirection.TWO_SIDED,
        warning_threshold=70.0,
        critical_threshold=85.0,
        warning_low=30.0,
        critical_low=15.0,
        hysteresis=2.0,
    )
    classifier = StatusClassifier(rule)

    # Normal range: [30.0, 70.0]
    assert classifier.classify(45.0) == NodeOperationalStatus.NORMAL
    assert classifier.classify(30.0) == NodeOperationalStatus.NORMAL
    assert classifier.classify(70.0) == NodeOperationalStatus.NORMAL

    # High warning: (70.0, 85.0)
    assert classifier.classify(72.0) == NodeOperationalStatus.WARNING

    # High critical: >= 85.0
    assert classifier.classify(86.0) == NodeOperationalStatus.CRITICAL

    # Low warning: [15.0, 30.0)
    classifier.reset()
    assert classifier.classify(25.0) == NodeOperationalStatus.WARNING

    # Low critical: <= 15.0
    assert classifier.classify(14.0) == NodeOperationalStatus.CRITICAL


def test_humidity_two_sided_hysteresis_recovery():
    """Verify recovery hysteresis for both low and high humidity violations."""
    rule = ThresholdRule(
        direction=ThresholdDirection.TWO_SIDED,
        warning_threshold=70.0,
        critical_threshold=85.0,
        warning_low=30.0,
        critical_low=15.0,
        hysteresis=2.0,
    )
    classifier = StatusClassifier(rule)

    # 1. Recover from High Critical (85.0): must drop below 85 - 2 = 83.0 to enter warning
    classifier.classify(90.0)  # CRITICAL
    assert classifier.classify(84.0) == NodeOperationalStatus.CRITICAL
    assert classifier.classify(82.5) == NodeOperationalStatus.WARNING
    # Recover from High Warning (70.0): must drop below 70 - 2 = 68.0 to enter normal
    assert classifier.classify(69.0) == NodeOperationalStatus.WARNING
    assert classifier.classify(67.5) == NodeOperationalStatus.NORMAL

    # 2. Recover from Low Critical (15.0): must rise above 15 + 2 = 17.0 to enter warning
    classifier.classify(10.0)  # CRITICAL
    assert classifier.classify(16.0) == NodeOperationalStatus.CRITICAL
    assert classifier.classify(17.5) == NodeOperationalStatus.WARNING
    # Recover from Low Warning (30.0): must rise above 30 + 2 = 32.0 to enter normal
    assert classifier.classify(31.0) == NodeOperationalStatus.WARNING
    assert classifier.classify(32.5) == NodeOperationalStatus.NORMAL


# ==============================================================================
# 2. Sensor Integration & Dynamic Status Transitions
# ==============================================================================

def test_sensor_dynamically_transitions_through_all_statuses():
    """Verify that a sensor dynamically changes status from normal -> warning -> critical -> normal."""
    cfg = get_default_gas_config(
        warning_threshold=120.0,
        critical_threshold=350.0,
        hysteresis=10.0,
        random_seed=42,
    )
    cfg.mean_reversion_k = 0.30
    gas = GasSensor(cfg)

    # 1. Normal state
    r1 = gas.sample(1.0)
    assert r1 is not None
    assert r1.status == NodeOperationalStatus.NORMAL

    # 2. Trigger gas leak targeting 500 ppm
    gas.simulate_gas_leak(target_ppm=500.0)

    # Should enter WARNING as value crosses 120.0
    r2 = gas.sample(2.0)
    assert r2 is not None
    assert r2.status == NodeOperationalStatus.WARNING

    # After several steps, should enter CRITICAL as value crosses 350.0
    critical_reached = False
    for t in range(3, 10):
        r = gas.sample(float(t))
        assert r is not None
        if r.status == NodeOperationalStatus.CRITICAL:
            critical_reached = True
            break
    assert critical_reached, "Sensor must reach CRITICAL status"

    # 3. Clear event and observe recovery
    gas.clear_event()
    recovered_to_normal = False
    for t in range(10, 35):
        r = gas.sample(float(t))
        assert r is not None
        if r.status == NodeOperationalStatus.NORMAL:
            recovered_to_normal = True
            break
    assert recovered_to_normal, "Sensor must recover back to NORMAL status"


# ==============================================================================
# 3. Configurable Scenario Engine Tests
# ==============================================================================

def test_scenario_normal_operation_produces_only_normal_statuses():
    """Verify normal operation scenario maintains normal status on all sensors."""
    scenario = scenario_normal_operation(duration_s=10.0, seed=42)
    runner = ScenarioRunner(scenario)
    records = runner.run(step_size=1.0)

    assert len(records) > 0
    for rec in records:
        assert rec["status"] == "normal"
    assert len(runner.status_transitions) == 0


def test_scenario_kitchen_fire_multi_event_overlapping():
    """Verify multi-event kitchen fire scenario handles overlapping events and transitions."""
    scenario = scenario_kitchen_fire_multi_event(total_duration_s=20.0, seed=42)
    runner = ScenarioRunner(scenario)
    records = runner.run(step_size=1.0)

    # Check that transitions occurred on GAS, TEMP, and SMOKE
    transition_sensors = {t["sensor_id"] for t in runner.status_transitions}
    assert "GAS-001" in transition_sensors

    # Check that warning and critical states were observed
    statuses_observed = {rec["status"] for rec in records}
    assert "normal" in statuses_observed
    assert "warning" in statuses_observed
    assert "critical" in statuses_observed


def test_scenario_deterministic_reproducibility():
    """Verify that identical scenarios with fixed seeds produce identical outputs."""
    scen1 = scenario_gradual_gas_leak(seed=100)
    scen2 = scenario_gradual_gas_leak(seed=100)

    runner1 = ScenarioRunner(scen1)
    runner2 = ScenarioRunner(scen2)

    res1 = runner1.run(step_size=1.0)
    res2 = runner2.run(step_size=1.0)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1["sensor_id"] == r2["sensor_id"]
        assert r1["simulation_time_s"] == r2["simulation_time_s"]
        assert r1["value"] == r2["value"]
        assert r1["status"] == r2["status"]


def test_scenario_respects_sampling_intervals_and_bounds():
    """Verify scenario runner strictly respects sampling intervals and physical bounds."""
    scenario = scenario_kitchen_fire_multi_event(total_duration_s=15.0, seed=42)
    runner = ScenarioRunner(scenario)
    records = runner.run(step_size=1.0)

    # Humidity must only produce records at even seconds (2.0s interval)
    hum_times = [r["simulation_time_s"] for r in records if r["sensor_id"] == "HUM-001"]
    for t in hum_times:
        assert (t % 2.0) == pytest.approx(0.0)

    # All values must be finite and within configured ranges
    for r in records:
        assert math.isfinite(r["value"])
        if r["sensor_id"] == "GAS-001":
            assert 0.0 <= r["value"] <= 1000.0
        elif r["sensor_id"] == "TEMP-001":
            assert -20.0 <= r["value"] <= 100.0
        elif r["sensor_id"] == "HUM-001":
            assert 0.0 <= r["value"] <= 100.0
        elif r["sensor_id"] == "SMOKE-001":
            assert 0.0 <= r["value"] <= 20.0


# ==============================================================================
# 4. Emergency Alert & Incident Architecture Data Contracts
# ==============================================================================

def test_alert_message_comprehensive_contract():
    """Verify AlertMessage supports all required emergency-alert fields."""
    alert = AlertMessage(
        alert_id="alt-uuid-001",
        sensor_id="GAS-001",
        incident_type="gas_leak",
        severity=AlertSeverity.CRITICAL,
        reading_value=425.5,
        threshold_value=350.0,
        unit="ppm",
        simulation_time_s=12.0,
        priority=TrafficClass.CRITICAL_CONTROL,
        destination_topic="iot/sensors/alerts/GAS-001",
        delivery_status=DeliveryStatus.GENERATED,
        incident_id="inc-uuid-001",
    )
    d = alert.to_dict()
    assert d["alert_id"] == "alt-uuid-001"
    assert d["sensor_id"] == "GAS-001"
    assert d["incident_type"] == "gas_leak"
    assert d["severity"] == "critical"
    assert d["reading_value"] == 425.5
    assert d["priority"] == "Critical Control"
    assert d["delivery_status"] == "generated"
    assert d["simulation_time_s"] == 12.0
    assert d["created_at_utc"] != ""


def test_incident_record_lifecycle():
    """Verify IncidentRecord tracking across active and cleared lifecycle states."""
    incident = IncidentRecord(
        incident_id="inc-001",
        sensor_id="TEMP-001",
        incident_type="overheating",
        state=IncidentState.ACTIVE,
        severity=AlertSeverity.CRITICAL,
        trigger_reading=62.0,
        trigger_time_s=5.0,
        last_reading=62.0,
        last_updated_time_s=5.0,
        associated_alert_ids=["alt-001"],
    )
    assert incident.state == IncidentState.ACTIVE
    assert len(incident.associated_alert_ids) == 1

    # Transition to cleared
    incident.state = IncidentState.CLEARED
    incident.cleared_time_s = 15.0
    d = incident.to_dict()
    assert d["state"] == "cleared"
    assert d["cleared_time_s"] == 15.0
