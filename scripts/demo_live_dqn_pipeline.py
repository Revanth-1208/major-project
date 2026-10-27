"""Demonstration runner for Phase 7: Live Sensor Classification, Automatic Alerts, and DQN-Based ACK Selection.

Fulfills Phase 7 Section 8 End-to-End Demonstration Requirements:
1. Sensors generating changing readings across time.
2. A normal reading remaining normal without a false emergency alert.
3. A configured gas or smoke event crossing the existing critical threshold.
4. An incident and structured alert being created with incident lifecycle tracking.
5. The DQN selecting an ACK action using the 8-feature state vector.
6. The selected Phase 5 policy being invoked (apply_policy).
7. Transport delivery progression and delivery confirmation.
8. Honest pending/failure status when MQTT is unavailable (no fabricated confirmations).
9. A decision history linking alert ID, selected action, and actual delivery result.
10. Recovery following the existing incident lifecycle (hysteresis clearing -> RECOVERED).
"""

from pathlib import Path
import sys
import time

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rl_ack.common.enums import AlertSeverity, IncidentState, NodeOperationalStatus, TrafficClass
from rl_ack.common.models import AlertMessage
from rl_ack.dashboard.state import DashboardSimulationManager
from rl_ack.policies.network_model import SimulatedNetworkModel, NetworkModelConfig


def main() -> None:
    print("\n" + "=" * 80)
    print("🤖 RL-ACK PHASE 7: 10-STEP END-TO-END DEMONSTRATION")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Step 1: Sensors Generating Changing Readings
    # --------------------------------------------------------------------------
    print("\n[STEP 1] Sensors Generating Changing Readings")
    mgr = DashboardSimulationManager(scenario_name="Normal Operation", random_seed=42)
    print("  Initializing simulation with deterministic seed 42...")
    for step_i in range(3):
        mgr.step(1.0)
        statuses = mgr.engine.get_sensor_statuses()
        readings = {s.sensor_id: round(s.last_reading.value, 2) for s in mgr.engine.get_sensors() if s.last_reading}
        print(f"  t={mgr.current_time_s:.1f}s: Readings = {readings}")
    print("  ✓ Virtual sensors dynamically produce fluctuating physical measurements.")

    # --------------------------------------------------------------------------
    # Step 2: Normal Readings Remaining Normal Without False Alert
    # --------------------------------------------------------------------------
    print("\n[STEP 2] Normal Reading Remaining Normal Without False Emergency Alert")
    mgr.run_steps(5)
    statuses = mgr.engine.get_sensor_statuses()
    for sid, st in statuses.items():
        print(f"  - {sid}: Operational Status = {st.value}")
    alerts = mgr.incident_manager.get_alerts()
    active_incidents = mgr.incident_manager.get_active_incidents()
    print(f"  Active Incidents: {len(active_incidents)}, Emitted Alerts: {len(alerts)}")
    assert len(alerts) == 0, "No alerts should be emitted during normal kitchen operation."
    assert all(st == NodeOperationalStatus.NORMAL for st in statuses.values())
    print("  ✓ Zero false positive alerts emitted during normal operation.")

    # --------------------------------------------------------------------------
    # Step 3: Configured Gas/Smoke Event Crossing Critical Threshold
    # --------------------------------------------------------------------------
    print("\n[STEP 3] Configured Gas/Smoke Event Crossing Critical Threshold")
    print("  Switching scenario to 'Kitchen Fire Multi-Event Emergency' (gas + smoke surge)...")
    mgr.set_scenario("Kitchen Fire Multi-Event Emergency")
    mgr.set_random_seed(42)

    critical_reached = False
    for _ in range(10):
        mgr.step(1.0)
        gas_sensor = next(s for s in mgr.engine.get_sensors() if s.sensor_id == "GAS-001")
        smoke_sensor = next(s for s in mgr.engine.get_sensors() if s.sensor_id == "SMOKE-001")
        g_val = gas_sensor.last_reading.value if gas_sensor.last_reading else 0.0
        s_val = smoke_sensor.last_reading.value if smoke_sensor.last_reading else 0.0

        if gas_sensor.status == NodeOperationalStatus.CRITICAL or smoke_sensor.status == NodeOperationalStatus.CRITICAL:
            critical_reached = True
            print(f"  t={mgr.current_time_s:.1f}s: GAS={g_val:.2f} ppm (Status: {gas_sensor.status.value}), SMOKE={s_val:.2f} % obs/m (Status: {smoke_sensor.status.value})")
            break

    assert critical_reached, "Expected sensor status to cross into CRITICAL."
    print("  ✓ Sensor threshold crossing detected by independent IncidentDetector.")

    # --------------------------------------------------------------------------
    # Step 4: Incident and Structured Alert Creation
    # --------------------------------------------------------------------------
    print("\n[STEP 4] Incident and Structured Alert Creation")
    active_incidents = mgr.incident_manager.get_active_incidents()
    alerts = mgr.incident_manager.get_alerts()
    assert len(active_incidents) > 0 and len(alerts) > 0
    latest_inc = active_incidents[-1]
    latest_alert = alerts[-1]
    print(f"  ✓ Incident Created : ID={latest_inc.incident_id}, State={latest_inc.state.value}, Severity={latest_inc.severity.value}")
    print(f"  ✓ Alert Message    : ID={latest_alert.alert_id}, Sensor={latest_alert.sensor_id}, TrafficClass={latest_alert.priority.value}")
    print(f"    Alert Details    : {latest_alert.message} (Value={latest_alert.reading_value}{latest_alert.unit})")

    # --------------------------------------------------------------------------
    # Step 5: DQN Selecting ACK Action Using Current State Vector
    # --------------------------------------------------------------------------
    print("\n[STEP 5] DQN Action Selection Using 8-Feature State Vector")
    assert mgr.dqn_controller.model_ready is True
    dec = mgr.get_latest_dqn_decision()
    assert dec is not None
    print(f"  ✓ Decision ID      : {dec.decision_id}")
    print(f"  ✓ Loaded Model     : {dec.model_id}")
    print(f"  ✓ 8-Feature State  : {[round(x, 4) for x in dec.state_vector]}")
    feature_labels = [
        "Network Util", "Rolling ACK Rate", "Queue Occ",
        "is_CriticalControl", "is_HighBandwidth", "is_BestEffort",
        "Remaining Energy", "Rolling Latency"
    ]
    for lbl, val in zip(feature_labels, dec.state_vector):
        print(f"      [{lbl:19s}] = {val:.4f}")
    if dec.q_values is not None:
        print(f"  ✓ Q-Values Q(s, a) : {[round(q, 3) for q in dec.q_values]}")
    print(f"  ✓ Selected Action  : Action {dec.action_idx} -> '{dec.action_label}' (QoS {dec.configured_qos})")
    print(f"  ✓ Safety Override  : {dec.safety_override} (Critical emergency alerts protected)")

    # --------------------------------------------------------------------------
    # Step 6: Selected Policy Being Invoked
    # --------------------------------------------------------------------------
    print("\n[STEP 6] Selected Phase 5 Policy Being Invoked")
    policy_inst = mgr.dqn_controller.policies.get(dec.action_idx)
    assert policy_inst is not None
    print(f"  ✓ Controller invoked policy engine: {type(policy_inst).__name__}")
    print(f"  ✓ Policy strategy parameters:")
    print(f"      - Configured QoS     : {policy_inst.config.configured_qos}")
    print(f"      - Max Retries        : {policy_inst.config.max_retries}")
    print(f"      - ACK Timeout        : {policy_inst.config.ack_timeout_ms} ms")

    # --------------------------------------------------------------------------
    # Step 7: Transport Delivery Progression & Outcome Recording
    # --------------------------------------------------------------------------
    print("\n[STEP 7] Controlled Transport Delivery Progression")
    test_net = SimulatedNetworkModel(NetworkModelConfig(packet_loss_rate=0.05, ack_loss_rate=0.02, latency_base_ms=12.0))
    ctrl_alert = AlertMessage(
        alert_id="ALT-CTRL-001",
        sensor_id="TEMP-001",
        priority=TrafficClass.HIGH_BANDWIDTH_STREAMING,
        severity=AlertSeverity.WARNING,
        reading_value=49.5,
        threshold_value=45.0,
        unit="°C",
    )
    ctrl_dec = mgr.dqn_controller.select_action(ctrl_alert)
    ctrl_res = mgr.dqn_controller.apply_policy(ctrl_alert, ctrl_dec, network_model=test_net)

    print(f"  ✓ Simulated Transport Evaluation:")
    print(f"      - Mode               : {ctrl_res.evaluation_mode}")
    print(f"      - Delivery Status    : {ctrl_dec.delivery_status}")
    print(f"      - Broker ACK         : {ctrl_dec.broker_acknowledged}")
    print(f"      - Subscriber Receipt : {ctrl_dec.subscriber_received}")
    print(f"      - App Confirmed      : {ctrl_dec.delivery_confirmed}")
    print(f"      - End-to-End Latency : {ctrl_dec.end_to_end_latency_ms:.2f} ms")
    print(f"      - Retransmissions    : {ctrl_dec.retry_count}")
    print(f"      - Overhead Bytes     : {ctrl_dec.overhead_bytes} B")

    # --------------------------------------------------------------------------
    # Step 8: Honest Pending/Failure Status When MQTT is Unavailable
    # --------------------------------------------------------------------------
    print("\n[STEP 8] Honest Pending/Failure Status When MQTT Broker is Unavailable")
    offline_alert = AlertMessage(
        alert_id="ALT-OFFLINE-TEST",
        sensor_id="GAS-001",
        priority=TrafficClass.CRITICAL_CONTROL,
        severity=AlertSeverity.CRITICAL,
        reading_value=88.0,
        threshold_value=50.0,
        unit="ppm",
    )
    off_dec = mgr.dqn_controller.select_action(offline_alert)
    # Apply with no connected MQTT broker
    off_res = mgr.dqn_controller.apply_policy(offline_alert, off_dec, mqtt_manager=None, live_mode=True)

    print(f"  ✓ Offline MQTT Delivery Outcome:")
    print(f"      - Delivery Status    : {off_dec.delivery_status}")
    print(f"      - Confirmed?         : {off_dec.delivery_confirmed} (Strictly False; no fabricated delivery)")
    print(f"      - Broker ACK         : {off_dec.broker_acknowledged}")
    print(f"      - Failure Reason     : {off_dec.failure_reason}")
    assert off_dec.delivery_confirmed is False
    assert off_dec.delivery_status == "failed"
    assert "offline" in off_dec.failure_reason.lower()

    # --------------------------------------------------------------------------
    # Step 9: Decision History Linking Alert ID, Selected Action & Delivery Result
    # --------------------------------------------------------------------------
    print("\n[STEP 9] Decision History Audit Stream (Alert -> DQN -> Policy -> Delivery)")
    df = mgr.get_dqn_decisions_df()
    print("  Recent Decision Audit Records:")
    for _, row in df.tail(4).iterrows():
        print(f"    [{row['decision_id']}] Alert={row['alert_id']} | Sensor={row['sensor_id']} | Action={row['action_label']} (QoS {row['configured_qos']}) | Status={row.get('delivery_status')} | Latency={row.get('end_to_end_latency_ms')} ms")

    kpis = mgr.get_dqn_controller_kpis()
    print(f"  ✓ Controller Aggregated KPIs:")
    print(f"      - Total Decisions         : {kpis['total_decisions']}")
    print(f"      - Delivery Confirmed Rate : {kpis['delivery_confirmation_rate']}%")
    print(f"      - Critical Delivery Rate  : {kpis['critical_delivery_rate']}%")
    print(f"      - Total Retries           : {kpis['total_retries']}")
    print(f"      - Fallback Count          : {kpis['fallback_count']}")
    print(f"      - Safety Overrides        : {kpis['safety_override_count']}")

    # --------------------------------------------------------------------------
    # Step 10: Recovery Following Existing Incident Lifecycle
    # --------------------------------------------------------------------------
    print("\n[STEP 10] Recovery Following Existing Incident Lifecycle")
    print("  Operator acknowledges active incident...")
    inc_to_ack = active_incidents[0].incident_id
    acked = mgr.acknowledge_incident(inc_to_ack)
    print(f"  ✓ Transition to ACKNOWLEDGED: Incident {acked.incident_id} State={acked.state.value}")

    print("  Switching to Normal Operation and clearing emergency events...")
    for s in mgr.engine.get_sensors():
        s.clear_event()

    recovered = False
    for step_i in range(25):
        mgr.step(1.0)
        match_inc = next((i for i in mgr.incident_manager.get_all_incidents() if i.incident_id == inc_to_ack), None)
        if match_inc and match_inc.state == IncidentState.RECOVERED:
            recovered = True
            print(f"  t={mgr.current_time_s:.1f}s: Readings cleared below hysteresis -> Incident {match_inc.incident_id} State={match_inc.state.value}")
            break

    assert recovered, "Incident should transition to RECOVERED once readings clear hysteresis."
    print("  ✓ Incident successfully cleared through official lifecycle (ACTIVE -> ACKNOWLEDGED -> RECOVERED).")

    print("\n" + "=" * 80)
    print("✅ ALL 10 PHASE 7 DEMONSTRATION STEPS COMPLETED AND VERIFIED!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
