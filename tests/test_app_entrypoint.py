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
    )
    assert "NORMAL" in STATUS_BADGES
    assert callable(render_header)
    assert callable(render_kpi_cards)
    assert callable(render_sensor_cards)
    assert callable(render_preview_charts)
    assert callable(render_topology_section)
    assert callable(render_incident_monitoring_tab)


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

    # Populated state render after steps
    manager.step(1.0)
    render_kpi_cards(manager.get_kpis())
    render_sensor_cards(manager.engine.get_sensors(), manager.engine.get_latest_readings())
    render_preview_charts(manager.get_time_series_df(), sensor_configs=manager.get_sensor_configs())
    render_incident_monitoring_tab(manager)
