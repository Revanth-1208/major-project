"""Tests for Phase 2: Dashboard Simulation Manager and UI State Integration."""

import pytest
import pandas as pd

from rl_ack.common.enums import NodeOperationalStatus, SensorType
from rl_ack.dashboard.state import DashboardSimulationManager, SCENARIO_BUILDERS


def test_dashboard_simulation_manager_init():
    """Verify default initialization of simulation manager."""
    mgr = DashboardSimulationManager(random_seed=42, scenario_name="Normal Operation")
    assert mgr.random_seed == 42
    assert mgr.scenario_name == "Normal Operation"
    assert mgr.step_size == 1.0
    assert mgr.simulation_status == "Stopped"
    assert mgr.current_time_s == 0.0
    assert len(mgr.readings_history) == 0

    sensors = mgr.engine.get_sensors()
    assert len(sensors) == 4
    sensor_ids = {s.sensor_id for s in sensors}
    assert sensor_ids == {"GAS-001", "TEMP-001", "HUM-001", "SMOKE-001"}


def test_dashboard_step_advances_simulation_clock():
    """Verify single step execution increments clock and records readings."""
    mgr = DashboardSimulationManager(random_seed=42)
    new_readings = mgr.step(dt=1.0)

    assert mgr.current_time_s == 1.0
    assert mgr.simulation_status == "Running"
    # At t=1.0s, sensors with interval=1.0s (GAS, TEMP, SMOKE) emit readings
    assert len(new_readings) >= 3
    assert len(mgr.readings_history) == len(new_readings)


def test_dashboard_run_steps_batch_advance():
    """Verify batch advancement over multiple seconds."""
    mgr = DashboardSimulationManager(random_seed=42)
    readings = mgr.run_steps(count=5, dt=1.0)

    assert mgr.current_time_s == 5.0
    assert len(readings) > 0
    # At t=5.0s, GAS (5 samples), TEMP (5 samples), SMOKE (5 samples), HUM (2 samples: t=2,4)
    # Total = 17 readings
    assert len(mgr.readings_history) == 17


def test_dashboard_reset_restores_clean_state():
    """Verify reset clears history, resets clock, and restores stopped status."""
    mgr = DashboardSimulationManager(random_seed=42)
    mgr.run_steps(5)
    assert mgr.current_time_s == 5.0
    assert len(mgr.readings_history) > 0

    mgr.reset()
    assert mgr.current_time_s == 0.0
    assert mgr.simulation_status == "Stopped"
    assert len(mgr.readings_history) == 0


def test_dashboard_kpis_derive_dynamically():
    """Verify KPI counts derive dynamically from actual sensor engine states."""
    mgr = DashboardSimulationManager(random_seed=42, scenario_name="Normal Operation")
    kpis = mgr.get_kpis()

    assert kpis["total_sensors"] == 4
    assert kpis["normal_count"] == 4
    assert kpis["warning_count"] == 0
    assert kpis["critical_count"] == 0
    assert kpis["unavailable_count"] == 0
    assert kpis["simulation_time_s"] == 0.0
    assert kpis["simulation_status"] == "Stopped"

    # Run emergency scenario to force dynamic status changes
    mgr.set_scenario("Kitchen Fire Multi-Event Emergency")
    mgr.run_steps(10)  # t=10.0s, high heat & smoke event active
    kpis_after = mgr.get_kpis()

    assert kpis_after["total_sensors"] == 4
    assert kpis_after["simulation_time_s"] == 10.0
    assert kpis_after["simulation_status"] == "Running"
    # Total status count must strictly sum to 4
    total_statuses = (
        kpis_after["normal_count"]
        + kpis_after["warning_count"]
        + kpis_after["critical_count"]
        + kpis_after["unavailable_count"]
    )
    assert total_statuses == 4
    # During fire event, not all sensors remain normal
    assert kpis_after["warning_count"] + kpis_after["critical_count"] > 0


def test_dashboard_sensor_inventory_format():
    """Verify sensor inventory row generation derives correctly from engine configurations."""
    mgr = DashboardSimulationManager(random_seed=42)
    rows = mgr.get_sensor_inventory()

    assert len(rows) == 4
    row_by_id = {r["Sensor ID"]: r for r in rows}

    assert "GAS-001" in row_by_id
    assert "TEMP-001" in row_by_id
    assert "HUM-001" in row_by_id
    assert "SMOKE-001" in row_by_id

    gas_row = row_by_id["GAS-001"]
    assert gas_row["Sensor Type"] == "gas"
    assert gas_row["Unit"] == "ppm"
    assert gas_row["Sampling Interval"] == "1.0 s"
    assert ">= 100.0" in gas_row["Warning Limit"]
    assert ">= 400.0" in gas_row["Critical Limit"]

    hum_row = row_by_id["HUM-001"]
    assert hum_row["Sensor Type"] == "humidity"
    assert hum_row["Unit"] == "% RH"
    assert hum_row["Sampling Interval"] == "2.0 s"
    # Humidity has two-sided intervals [low, high]
    assert "[" in hum_row["Warning Limit"]
    assert "[" in hum_row["Critical Limit"]


def test_dashboard_telemetry_dataframes():
    """Verify DataFrame generation for preview charts and stream table."""
    mgr = DashboardSimulationManager(random_seed=42)

    # Empty initially
    df_empty_stream = mgr.get_recent_readings_df()
    assert isinstance(df_empty_stream, pd.DataFrame)
    assert len(df_empty_stream) == 0

    df_empty_ts = mgr.get_time_series_df()
    assert isinstance(df_empty_ts, pd.DataFrame)
    assert len(df_empty_ts) == 0

    # Advance simulation
    mgr.run_steps(4)
    df_stream = mgr.get_recent_readings_df(limit=10)
    assert isinstance(df_stream, pd.DataFrame)
    assert len(df_stream) > 0
    assert "Sensor ID" in df_stream.columns
    assert "Value" in df_stream.columns
    assert "Status" in df_stream.columns

    df_ts = mgr.get_time_series_df()
    assert isinstance(df_ts, pd.DataFrame)
    assert "simulation_time_s" in df_ts.columns
    assert "value" in df_ts.columns
    assert "sensor_id" in df_ts.columns
