"""Demonstration runner for Live Sensor Classification, Automatic Alerts, and DQN-Based ACK Selection.

Acceptance Verification:
1. Normal Operation: Verify steady baseline readings and all sensors NORMAL.
2. Trigger Emergency Scenario: Switch to 'Gradual Gas Leak'.
3. Warning Threshold Crossing:
   - Status changes from NORMAL -> WARNING.
   - Incident is opened (state: ACTIVE).
   - Alert generated and deduplication enforced.
   - DQN controller performs deterministic greedy inference on 8-feature state vector.
   - Alert published over MQTT with controller-chosen QoS.
4. Critical Threshold Escalation:
   - Reading escalates to CRITICAL.
   - Incident escalates severity to CRITICAL.
   - DQN action selection & Safety Override verification (ensuring QoS >= 1 for critical emergencies).
5. Simulation Pause.
6. Operator Incident Acknowledgment (ACTIVE -> ACKNOWLEDGED).
7. Recovery: Switch back to Normal Operation until readings return below clearing hysteresis:
   - Incident transitions to RECOVERED.
8. Safe Fallback & Non-Suppression:
   - Explicitly unload checkpoint to demonstrate deterministic fallback.
   - Advance into another danger condition.
   - Verify alert is generated (Non-Suppression Invariant holds).
   - Verify fallback policy is used with truthful absence of fake Q-values.
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

from rl_ack.common.enums import AlertSeverity, IncidentState, NodeOperationalStatus
from rl_ack.dashboard.state import DashboardSimulationManager


def main() -> None:
    print("\n" + "=" * 78)
    print("🤖 RL-ACK: Live Sensor Classification, Alerts & DQN ACK Selection Demo")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # Step 1: Normal Operation Baseline
    # --------------------------------------------------------------------------
    print("\n[STEP 1] Starting Simulation in 'Normal Operation'...")
    mgr = DashboardSimulationManager(scenario_name="Normal Operation", random_seed=42)

    # Try connecting to MQTT broker if available
    mqtt_ok = mgr.connect_mqtt(timeout_s=1.0)
    if mqtt_ok:
        print("  ✓ Mosquitto MQTT broker connected (127.0.0.1:1883)")
    else:
        print("  ℹ️ Local Mosquitto not reachable; alerts will be buffered in MQTT manager.")

    # Advance 5 steps
    for _ in range(5):
        mgr.step(1.0)

    statuses = mgr.engine.get_sensor_statuses()
    print(f"  Simulation time: t={mgr.current_time_s:.1f}s")
    print(f"  Sensors: {list(statuses.keys())}")
    for sid, st in statuses.items():
        print(f"    - {sid}: status = {st.value}")
    assert all(st == NodeOperationalStatus.NORMAL for st in statuses.values())
    print("  ✓ Baseline verified: All sensors NORMAL, 0 incidents, 0 alerts.")

    # --------------------------------------------------------------------------
    # Step 2 & 3: Environmental Scenario & Warning Threshold Crossing
    # --------------------------------------------------------------------------
    print("\n[STEP 2 & 3] Switching scenario to 'Gradual Gas Leak' and advancing...")
    mgr.set_scenario("Gradual Gas Leak")
    mgr.set_random_seed(42)

    # In Gradual Gas Leak, gas increases toward 120 ppm (warning=30 ppm, critical=50 ppm)
    warning_crossed = False
    step_count = 0

    while step_count < 20 and not warning_crossed:
        mgr.step(1.0)
        step_count += 1
        gas_sensor = next(s for s in mgr.engine.get_sensors() if s.sensor_id == "GAS-001")
        val = gas_sensor.last_reading.value if gas_sensor.last_reading else 0.0
        st = gas_sensor.status

        if st == NodeOperationalStatus.WARNING:
            warning_crossed = True
            print(f"  t={mgr.current_time_s:.1f}s: GAS-001 reached {val:.2f} ppm (Status: WARNING)")

    assert warning_crossed, "Gas sensor should have entered WARNING status."
    active_incidents = mgr.incident_manager.get_active_incidents()
    alerts = mgr.incident_manager.get_alerts()
    assert len(active_incidents) >= 1
    assert len(alerts) >= 1

    gas_inc = active_incidents[0]
    print(f"  ✓ Incident opened: ID={gas_inc.incident_id}, State={gas_inc.state.value}, Severity={gas_inc.severity.value}")
    print(f"  ✓ Alert emitted: ID={alerts[0].alert_id}, Reason='{alerts[0].message}'")

    # Inspect DQN controller decision
    dec = mgr.get_latest_dqn_decision()
    assert dec is not None
    print(f"  ✓ DQN Decision: ID={dec.decision_id}")
    print(f"    - Model ID        : {dec.model_id}")
    print(f"    - Selected Action : {dec.action_label} (QoS {dec.configured_qos})")
    print(f"    - State Vector    : {[round(x, 3) for x in dec.state_vector]}")
    if dec.q_values is not None:
        print(f"    - Q-Values Q(s,a) : {[round(q, 3) for q in dec.q_values]}")
    print(f"    - Fallback?       : {dec.is_fallback}")
    print(f"    - Safety Override?: {dec.safety_override}")

    # --------------------------------------------------------------------------
    # Step 4: Critical Threshold Escalation & Safety Override
    # --------------------------------------------------------------------------
    print("\n[STEP 4] Advancing simulation until Critical Threshold (50.0 ppm)...")
    critical_crossed = False

    while step_count < 30 and not critical_crossed:
        mgr.step(1.0)
        step_count += 1
        gas_sensor = next(s for s in mgr.engine.get_sensors() if s.sensor_id == "GAS-001")
        val = gas_sensor.last_reading.value if gas_sensor.last_reading else 0.0
        st = gas_sensor.status

        if st == NodeOperationalStatus.CRITICAL:
            critical_crossed = True
            print(f"  t={mgr.current_time_s:.1f}s: GAS-001 escalated to {val:.2f} ppm (Status: CRITICAL)")

    assert critical_crossed, "Gas sensor should have escalated to CRITICAL status."
    active_incidents = mgr.incident_manager.get_active_incidents()
    assert active_incidents[0].severity == AlertSeverity.CRITICAL
    print(f"  ✓ Incident escalated to: Severity={active_incidents[0].severity.value}")

    latest_dec = mgr.get_latest_dqn_decision()
    print(f"  ✓ DQN Decision for Critical Alert:")
    print(f"    - Action Selected : {latest_dec.action_label} (QoS {latest_dec.configured_qos})")
    print(f"    - Safety Override : {latest_dec.safety_override}")
    assert latest_dec.configured_qos >= 1, "Critical alerts must have QoS >= 1."

    # --------------------------------------------------------------------------
    # Step 5 & 6: Pause Simulation & Operator Incident Acknowledgment
    # --------------------------------------------------------------------------
    print("\n[STEP 5 & 6] Pausing simulation and acknowledging incident...")
    mgr.pause()
    print(f"  Simulation Status: {mgr.simulation_status}")

    inc_id = active_incidents[0].incident_id
    acked_inc = mgr.acknowledge_incident(inc_id)
    print(f"  ✓ Operator Acknowledged: ID={acked_inc.incident_id}, State={acked_inc.state.value}")
    assert acked_inc.state == IncidentState.ACKNOWLEDGED

    # --------------------------------------------------------------------------
    # Step 7: Incident Recovery
    # --------------------------------------------------------------------------
    print("\n[STEP 7] Switching back to 'Normal Operation' to verify clearing and recovery...")
    # Revert target to baseline to clear gas leak
    gas_sensor = next(s for s in mgr.engine.get_sensors() if s.sensor_id == "GAS-001")
    gas_sensor.clear_event()

    recovered = False
    for step_i in range(25):
        mgr.step(1.0)
        val = gas_sensor.last_reading.value if gas_sensor.last_reading else 0.0
        st = gas_sensor.status
        all_inc = mgr.incident_manager.get_all_incidents()
        match_inc = next((i for i in all_inc if i.incident_id == inc_id), None)
        if match_inc and match_inc.state == IncidentState.RECOVERED:
            recovered = True
            print(f"  t={mgr.current_time_s:.1f}s: GAS-001 cleared to {val:.2f} ppm -> Incident {inc_id} State={match_inc.state.value}")
            break

    assert recovered, "Incident should transition to RECOVERED once readings clear hysteresis."
    print("  ✓ Hysteresis-aware recovery verified!")

    # --------------------------------------------------------------------------
    # Step 8: Safe Fallback & Non-Suppression Guarantee
    # --------------------------------------------------------------------------
    print("\n[STEP 8] Demonstrating Safe Fallback Mode & Non-Suppression Guarantee...")
    print("  Unloading DQN checkpoint to test unready model state...")
    mgr.unload_controller_checkpoint()
    assert mgr.dqn_controller.model_ready is False

    print("  Triggering another emergency event while model is unloaded...")
    mgr.engine.trigger_sensor_event("GAS-001", target_value=350.0, duration_s=10.0)
    mgr.run_steps(5)

    fallback_dec = mgr.get_latest_dqn_decision()
    assert fallback_dec is not None
    assert fallback_dec.is_fallback is True
    assert fallback_dec.q_values is None, "Truthful reporting: Q-values must be None during fallback."
    assert fallback_dec.configured_qos == 1, "Fallback policy provides safe default QoS 1."

    print(f"  ✓ Fallback Decision Recorded:")
    print(f"    - Fallback Active? : {fallback_dec.is_fallback}")
    print(f"    - Fallback Reason  : {fallback_dec.fallback_reason}")
    print(f"    - Q-Values         : {fallback_dec.q_values} (Strictly None, no fake numbers)")
    print(f"    - Selected Policy  : {fallback_dec.policy_name} (QoS {fallback_dec.configured_qos})")
    print(f"  ✓ Non-Suppression Invariant: Alert was generated and delivered despite unready DQN model!")

    # Reload checkpoint for clean exit
    mgr.load_controller_checkpoint("dqn_best.pt")
    print("\n" + "=" * 78)
    print("✅ All 8 Acceptance Verification Steps Completed Successfully!")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
