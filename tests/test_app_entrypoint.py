"""Tests for Streamlit application entrypoint."""

import py_compile
from pathlib import Path


def test_app_syntax_compilation():
    """Verify app.py compiles without syntax errors."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.exists(), "app.py must exist at project root"
    compiled_path = py_compile.compile(str(app_path), doraise=True)
    assert compiled_path is not None


def test_dashboard_components_import_and_definitions():
    """Verify all dashboard presentation components import cleanly and have valid signatures."""
    from rl_ack.dashboard.components import (
        STATUS_BADGES,
        render_header,
        render_kpi_cards,
        render_sensor_cards,
        render_preview_charts,
        render_topology_section,
        render_incident_monitoring_tab,
        render_mqtt_delivery_tab,
        render_policy_comparison_tab,
        render_dqn_tab,
        render_dqn_decision_monitor_tab,
    )
    assert "NORMAL" in STATUS_BADGES
    assert callable(render_header)
    assert callable(render_kpi_cards)
    assert callable(render_sensor_cards)
    assert callable(render_preview_charts)
    assert callable(render_topology_section)
    assert callable(render_incident_monitoring_tab)
    assert callable(render_mqtt_delivery_tab)
    assert callable(render_policy_comparison_tab)
    assert callable(render_dqn_tab)
    assert callable(render_dqn_decision_monitor_tab)


def test_dashboard_components_render_without_exceptions():
    """Verify all dashboard presentation components execute with live engine state without raising exceptions."""
    from rl_ack.dashboard.state import DashboardSimulationManager
    from rl_ack.topology.models import create_smart_kitchen_topology
    from rl_ack.dashboard.components import (
        render_header,
        render_kpi_cards,
        render_sensor_cards,
        render_preview_charts,
        render_topology_section,
        render_incident_monitoring_tab,
        render_mqtt_delivery_tab,
        render_policy_comparison_tab,
        render_dqn_tab,
        render_dqn_decision_monitor_tab,
    )

    manager = DashboardSimulationManager(random_seed=42)
    topology = create_smart_kitchen_topology()

    # Initial empty state render
    render_header()
    render_kpi_cards(manager.get_kpis())
    render_sensor_cards(manager.engine.get_sensors(), manager.engine.get_latest_readings())
    render_preview_charts(manager.get_time_series_df(), sensor_configs=manager.get_sensor_configs())
    render_topology_section(topology)
    render_incident_monitoring_tab(manager)
    render_mqtt_delivery_tab(manager)
    render_policy_comparison_tab(manager)
    render_dqn_tab(manager)
    render_dqn_decision_monitor_tab(manager)

    # Populated state render after steps
    manager.step(1.0)
    render_kpi_cards(manager.get_kpis())
    render_sensor_cards(manager.engine.get_sensors(), manager.engine.get_latest_readings())
    render_preview_charts(manager.get_time_series_df(), sensor_configs=manager.get_sensor_configs())
    render_incident_monitoring_tab(manager)
    render_mqtt_delivery_tab(manager)
    render_policy_comparison_tab(manager)
    render_dqn_tab(manager)
    render_dqn_decision_monitor_tab(manager)


def test_dqn_tab_populated_rendering():
    """Verify DQN tab renders cleanly with training curves, evaluation metrics, and checkpoints."""
    from rl_ack.dashboard.state import DashboardSimulationManager
    from rl_ack.dashboard.components import render_dqn_tab

    manager = DashboardSimulationManager(random_seed=42)

    # 1. Train 1 quick episode to populate training history
    manager.train_dqn(num_episodes=1, steps_per_episode=3)
    assert not manager.get_dqn_training_history_df().empty

    # 2. Run quick evaluation to populate comparative benchmark metrics
    manager.run_dqn_evaluation(workload_count=3, seed=42)
    assert not manager.get_dqn_evaluation_summary_df().empty

    # 3. Render DQN tab with populated state across all sub-tabs
    render_dqn_tab(manager)


def test_no_duplicate_element_ids_in_components_source():
    """Regression test ensuring all interactive widgets in components.py have distinct keys."""
    import re
    from pathlib import Path

    components_path = Path(__file__).resolve().parent.parent / "src" / "rl_ack" / "dashboard" / "components.py"
    content = components_path.read_text(encoding="utf-8")

    # Find all explicit keys defined in components.py
    keys = re.findall(r'key=["\']([^"\']+)["\']', content)
    duplicate_keys = [k for k in keys if keys.count(k) > 1]
    assert len(duplicate_keys) == 0, f"Found duplicate widget keys in components.py: {set(duplicate_keys)}"


def test_dqn_decision_monitor_chart_renders_with_actual_telemetry_schema():
    """Regression test ensuring render_dqn_decision_monitor_tab renders its telemetry chart without column mismatches."""
    from rl_ack.dashboard.state import DashboardSimulationManager
    from rl_ack.dashboard.components import render_dqn_decision_monitor_tab

    # 1. Test with empty telemetry DataFrame (graceful handling, no exception)
    mgr_empty = DashboardSimulationManager(scenario_name="Normal Operation", random_seed=42)
    assert mgr_empty.get_time_series_df().empty
    render_dqn_decision_monitor_tab(mgr_empty)

    # 2. Test with populated emergency decisions and real telemetry columns
    mgr = DashboardSimulationManager(scenario_name="Kitchen Fire Multi-Event Emergency", random_seed=42)
    mgr.run_steps(8)

    # Verify decisions and telemetry exist
    kpis = mgr.get_dqn_controller_kpis()
    assert kpis["total_decisions"] > 0, "Decisions must be generated to trigger the performance metrics section."

    ts_df = mgr.get_time_series_df()
    assert not ts_df.empty
    expected_cols = ["simulation_time_s", "sensor_id", "value", "unit", "status"]
    for col in expected_cols:
        assert col in ts_df.columns, f"Expected column {col} in telemetry DataFrame"

    # Must render without ValueError: Value of 'x' is not the name of a column
    render_dqn_decision_monitor_tab(mgr)


