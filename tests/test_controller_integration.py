"""Tests for Module 8: DQN Controller Integration and Live Classification Pipeline.

Validates:
1. Dynamic sensor classification for all 4 kitchen sensors (GAS-001, TEMP-001, HUM-001, SMOKE-001)
   across NORMAL, WARNING, and CRITICAL statuses according to configured thresholds.
2. Automatic incident and alert generation with proper deduplication.
3. Deterministic DQN inference with loaded PyTorch checkpoints (strictly greedy, epsilon=0).
4. Fallback behavior when checkpoint is unready or unloaded (truthful reporting, no fake Q-values).
5. Non-suppression invariant: danger detection is fully decoupled from DQN and never suppressed.
6. Safety override policy: Critical Control emergency alerts cannot be assigned fire-and-forget QoS 0.
7. End-to-end simulation manager step loop routing alerts through DQN controller to MQTT.
"""

from pathlib import Path
import numpy as np
import pytest

from rl_ack.common.enums import (
    AckAction,
    AlertSeverity,
    EvaluationPolicy,
    NodeOperationalStatus,
    SensorType,
    TrafficClass,
)
from rl_ack.common.models import AlertMessage, SensorReading
from rl_ack.controller.controller import DqnAckController
from rl_ack.controller.decision import DqnDecisionRecord
from rl_ack.dashboard.state import DashboardSimulationManager
from rl_ack.dqn.agent import DQNAgent
from rl_ack.monitoring.detector import IncidentDetector
from rl_ack.monitoring.manager import IncidentManager
from rl_ack.sensors.config import (
    get_default_gas_config,
    get_default_humidity_config,
    get_default_smoke_config,
    get_default_temperature_config,
)
from rl_ack.sensors.gas_sensor import GasSensor
from rl_ack.sensors.humidity_sensor import HumiditySensor
from rl_ack.sensors.smoke_sensor import SmokeSensor
from rl_ack.sensors.temperature_sensor import TemperatureSensor


# ------------------------------------------------------------------------------
# 1. Dynamic Sensor Classification Tests
# ------------------------------------------------------------------------------


def test_gas_sensor_status_classification():
    """Verify GAS-001 classifies into NORMAL, WARNING, CRITICAL at exact thresholds."""
    cfg = get_default_gas_config(
        warning_threshold=30.0,
        critical_threshold=50.0,
        hysteresis=2.0,
    )
    sensor = GasSensor(config=cfg)

    # Normal range
    assert sensor.classifier.classify(15.0) == NodeOperationalStatus.NORMAL
    assert sensor.classifier.classify(29.9) == NodeOperationalStatus.NORMAL

    # Warning range (30.0 to < 50.0)
    assert sensor.classifier.classify(30.0) == NodeOperationalStatus.WARNING
    assert sensor.classifier.classify(45.0) == NodeOperationalStatus.WARNING

    # Critical range (>= 50.0)
    assert sensor.classifier.classify(50.0) == NodeOperationalStatus.CRITICAL
    assert sensor.classifier.classify(120.0) == NodeOperationalStatus.CRITICAL


def test_temperature_sensor_status_classification():
    """Verify TEMP-001 classifies into NORMAL, WARNING, CRITICAL at exact thresholds."""
    cfg = get_default_temperature_config(
        warning_threshold=45.0,
        critical_threshold=60.0,
        hysteresis=1.5,
    )
    sensor = TemperatureSensor(config=cfg)

    assert sensor.classifier.classify(22.0) == NodeOperationalStatus.NORMAL
    assert sensor.classifier.classify(44.9) == NodeOperationalStatus.NORMAL
    assert sensor.classifier.classify(45.0) == NodeOperationalStatus.WARNING
    assert sensor.classifier.classify(59.9) == NodeOperationalStatus.WARNING
    assert sensor.classifier.classify(60.0) == NodeOperationalStatus.CRITICAL
    assert sensor.classifier.classify(85.0) == NodeOperationalStatus.CRITICAL


def test_smoke_sensor_status_classification():
    """Verify SMOKE-001 classifies into NORMAL, WARNING, CRITICAL at exact thresholds."""
    cfg = get_default_smoke_config(
        warning_threshold=1.5,
        critical_threshold=3.0,
        hysteresis=0.2,
    )
    sensor = SmokeSensor(config=cfg)

    assert sensor.classifier.classify(0.2) == NodeOperationalStatus.NORMAL
    assert sensor.classifier.classify(1.49) == NodeOperationalStatus.NORMAL
    assert sensor.classifier.classify(1.5) == NodeOperationalStatus.WARNING
    assert sensor.classifier.classify(2.9) == NodeOperationalStatus.WARNING
    assert sensor.classifier.classify(3.0) == NodeOperationalStatus.CRITICAL
    assert sensor.classifier.classify(7.5) == NodeOperationalStatus.CRITICAL


def test_humidity_sensor_two_sided_status_classification():
    """Verify HUM-001 handles two-sided (low & high) thresholds correctly."""
    cfg = get_default_humidity_config(
        warning_low=30.0,
        warning_high=75.0,
        critical_low=20.0,
        critical_high=85.0,
        hysteresis=3.0,
    )
    sensor = HumiditySensor(config=cfg)

    # Normal range (30.0 to 75.0)
    assert sensor.classifier.classify(50.0) == NodeOperationalStatus.NORMAL
    # High side
    assert sensor.classifier.classify(76.0) == NodeOperationalStatus.WARNING
    assert sensor.classifier.classify(86.0) == NodeOperationalStatus.CRITICAL
    # Low side
    assert sensor.classifier.classify(25.0) == NodeOperationalStatus.WARNING
    assert sensor.classifier.classify(15.0) == NodeOperationalStatus.CRITICAL


# ------------------------------------------------------------------------------
# 2. Incident Detection & Deduplication Tests
# ------------------------------------------------------------------------------


def test_automatic_alert_generation_and_deduplication():
    """Verify that repeated elevated readings in the same incident are deduplicated."""
    configs = {
        "GAS-001": get_default_gas_config(
            warning_threshold=30.0,
            critical_threshold=50.0,
        )
    }
    detector = IncidentDetector(sensor_configs=configs)
    mgr = IncidentManager(detector=detector)

    def _mk_reading(val: float, status: NodeOperationalStatus, seq: int, t: float):
        return SensorReading(
            sensor_id="GAS-001",
            sensor_type=SensorType.GAS,
            value=val,
            unit="ppm",
            simulation_time_s=t,
            sampling_interval_s=1.0,
            host_node_id="NODE-001",
            status=status,
            sequence_number=seq,
            is_threshold_exceeded=(status != NodeOperationalStatus.NORMAL),
        )

    # 1. Normal reading -> No alert
    r1 = _mk_reading(18.0, NodeOperationalStatus.NORMAL, 1, 1.0)
    alerts = mgr.process_readings([r1])
    assert len(alerts) == 0

    # 2. Reading crosses Warning threshold -> Alert generated
    r2 = _mk_reading(35.0, NodeOperationalStatus.WARNING, 2, 2.0)
    alerts = mgr.process_readings([r2])
    assert len(alerts) == 1
    assert alerts[0].severity == AlertSeverity.WARNING
    assert alerts[0].sensor_id == "GAS-001"

    # 3. Subsequent reading still in Warning -> Deduplicated (no new alert)
    r3 = _mk_reading(38.0, NodeOperationalStatus.WARNING, 3, 3.0)
    alerts = mgr.process_readings([r3])
    assert len(alerts) == 0, "Repeated warning reading should be deduplicated."

    # 4. Reading escalates to Critical -> New escalated alert generated
    r4 = _mk_reading(65.0, NodeOperationalStatus.CRITICAL, 4, 4.0)
    alerts = mgr.process_readings([r4])
    assert len(alerts) == 1
    assert alerts[0].severity == AlertSeverity.CRITICAL


# ------------------------------------------------------------------------------
# 3. DQN Controller Deterministic Inference & Non-Suppression Tests
# ------------------------------------------------------------------------------


def test_deterministic_dqn_inference_with_loaded_checkpoint():
    """Verify loaded checkpoint produces deterministic greedy Q-values and valid action."""
    ckpt_path = Path("checkpoints/dqn_best.pt")
    if not ckpt_path.is_file():
        ckpt_path = Path("checkpoints/dqn_latest.pt")
    assert ckpt_path.is_file(), "A saved checkpoint must exist in checkpoints/."

    controller = DqnAckController()
    loaded = controller.load_checkpoint(ckpt_path)
    assert loaded is True
    assert controller.model_ready is True
    assert controller.last_error is None

    # Construct test emergency alert
    alert = AlertMessage(
        alert_id="ALT-TEST-001",
        sensor_id="GAS-001",
        priority=TrafficClass.CRITICAL_CONTROL,
        severity=AlertSeverity.CRITICAL,
        reading_value=72.0,
        threshold_value=50.0,
        unit="ppm",
        simulation_time_s=10.0,
    )

    # Execute inference
    decision1 = controller.select_action(alert)
    assert isinstance(decision1, DqnDecisionRecord)
    assert decision1.is_fallback is False
    assert decision1.q_values is not None
    assert len(decision1.q_values) == 5
    assert all(np.isfinite(decision1.q_values))
    assert 0 <= decision1.action_idx <= 4
    assert decision1.configured_qos in (0, 1, 2)

    # Determinism: repeating inference on identical input produces identical Q-values and action
    decision2 = controller.select_action(alert)
    assert decision1.action_idx == decision2.action_idx
    assert np.allclose(decision1.q_values, decision2.q_values)


def test_safe_fallback_when_checkpoint_unloaded():
    """Verify controller gracefully and transparently falls back when model is unloaded."""
    controller = DqnAckController(fallback_policy=EvaluationPolicy.STATIC_MQTT_QOS_1)
    controller.unload_checkpoint()
    assert controller.model_ready is False

    alert = AlertMessage(
        alert_id="ALT-FALLBACK-001",
        sensor_id="TEMP-001",
        priority=TrafficClass.HIGH_BANDWIDTH_STREAMING,
        severity=AlertSeverity.WARNING,
        reading_value=48.0,
        threshold_value=45.0,
        unit="°C",
        simulation_time_s=15.0,
    )

    decision = controller.select_action(alert)
    assert decision.is_fallback is True
    assert decision.fallback_reason is not None
    assert decision.q_values is None, "No fake Q-values should be fabricated during fallback."
    assert decision.action_idx == int(AckAction.MQTT_QOS_1)
    assert decision.configured_qos == 1
    assert decision.model_id == "UNLOADED (Fallback Mode)"


def test_critical_alert_safety_override():
    """Verify safety override upgrades QoS 0 to QoS 1 for CRITICAL emergency alerts."""
    controller = DqnAckController(enable_safety_override=True)

    # Mock agent to return action 0 (QoS 0)
    agent = DQNAgent()
    agent.get_q_values = lambda state: np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    controller.agent = agent
    controller.model_ready = True

    # 1. Critical alert: must trigger safety override
    critical_alert = AlertMessage(
        alert_id="ALT-CRIT-001",
        sensor_id="GAS-001",
        priority=TrafficClass.CRITICAL_CONTROL,
        severity=AlertSeverity.CRITICAL,
        reading_value=85.0,
        threshold_value=50.0,
        unit="ppm",
    )
    dec_crit = controller.select_action(critical_alert)
    assert dec_crit.safety_override is True
    assert dec_crit.action_idx == int(AckAction.MQTT_QOS_1)
    assert dec_crit.configured_qos == 1

    # 2. Warning alert: safety override is not applied even if action 0 is chosen
    warning_alert = AlertMessage(
        alert_id="ALT-WARN-001",
        sensor_id="TEMP-001",
        priority=TrafficClass.HIGH_BANDWIDTH_STREAMING,
        severity=AlertSeverity.WARNING,
        reading_value=48.0,
        threshold_value=45.0,
        unit="°C",
    )
    dec_warn = controller.select_action(warning_alert)
    assert dec_warn.safety_override is False
    assert dec_warn.action_idx == int(AckAction.MQTT_QOS_0)


def test_non_suppression_invariant():
    """Verify danger condition always triggers alert regardless of DQN state."""
    # Test with model unready
    mgr = DashboardSimulationManager(scenario_name="Gradual Gas Leak")
    mgr.dqn_controller.unload_checkpoint()

    # Step into leak
    mgr.run_steps(10)
    history = mgr.incident_manager.get_active_incidents()
    alerts = mgr.incident_manager.get_alerts()

    assert len(history) > 0 or len(alerts) > 0, "DQN unreadiness must never suppress incident detection."
    # Decisions were still logged with fallback
    decisions = mgr.get_recent_dqn_decisions()
    assert len(decisions) > 0
    assert decisions[0].is_fallback is True


# ------------------------------------------------------------------------------
# 4. Simulation Manager End-to-End Pipeline Tests
# ------------------------------------------------------------------------------


def test_simulation_manager_step_routes_through_dqn_and_mqtt():
    """Verify step() advances sensors, detects incidents, queries DQN, and queues alert."""
    mgr = DashboardSimulationManager(scenario_name="Kitchen Fire Multi-Event Emergency")
    assert mgr.dqn_controller.model_ready is True

    # Advance until kitchen fire triggers (starts at t=4.0s)
    mgr.run_steps(8)  # 8 * 1.0s = 8.0s > 4.0s

    # Alerts should have been generated
    alerts = mgr.incident_manager.get_alerts()
    assert len(alerts) > 0

    # Controller should have recorded decisions
    decisions = mgr.get_recent_dqn_decisions()
    assert len(decisions) > 0

    latest_dec = mgr.get_latest_dqn_decision()
    assert latest_dec is not None
    assert latest_dec.alert_id == alerts[-1].alert_id

    # Decisions DataFrame
    df = mgr.get_dqn_decisions_df()
    assert not df.empty
    assert "decision_id" in df.columns
    assert "state_vector" in df.columns
    assert "configured_qos" in df.columns

    # CSV export
    csv_str = mgr.export_dqn_decisions_csv()
    assert "DEC-0001" in csv_str
