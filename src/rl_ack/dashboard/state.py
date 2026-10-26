"""Simulation state manager for the RL-ACK Streamlit dashboard."""

from collections import deque
from typing import Any, Dict, List, Optional
import pandas as pd

from rl_ack.common.enums import NodeOperationalStatus, SensorType
from rl_ack.common.models import SensorReading
from rl_ack.sensors.sensor_engine import SensorEngine
from rl_ack.sensors.scenarios import (
    SensorScenario,
    scenario_abnormal_temperature_rise,
    scenario_gradual_gas_leak,
    scenario_humidity_change,
    scenario_increasing_smoke,
    scenario_kitchen_fire_multi_event,
    scenario_normal_operation,
)


SCENARIO_BUILDERS = {
    "Normal Operation": scenario_normal_operation,
    "Gradual Gas Leak": scenario_gradual_gas_leak,
    "Abnormal Temperature Rise": scenario_abnormal_temperature_rise,
    "Increasing Smoke Obscuration": scenario_increasing_smoke,
    "Humidity Anomaly (High Steam)": lambda seed=44: scenario_humidity_change(mode="high", seed=seed),
    "Humidity Anomaly (Dry Air)": lambda seed=44: scenario_humidity_change(mode="low", seed=seed),
    "Kitchen Fire Multi-Event Emergency": scenario_kitchen_fire_multi_event,
}


class DashboardSimulationManager:
    """Encapsulates simulation state, scenario execution, and telemetry history for the UI."""

    def __init__(
        self,
        random_seed: int = 42,
        scenario_name: str = "Normal Operation",
        step_size: float = 1.0,
        max_history: int = 200,
    ) -> None:
        self.random_seed = random_seed
        self.scenario_name = scenario_name
        self.step_size = step_size
        self.max_history = max_history
        self.simulation_status = "Stopped"

        self.engine: SensorEngine = SensorEngine.create_default_engine(seed_offset=self.random_seed)
        self.active_scenario: Optional[SensorScenario] = None
        self._history: deque[SensorReading] = deque(maxlen=self.max_history)
        self._applied_event_indices: set[int] = set()

        self._configure_scenario(self.scenario_name)

    def _configure_scenario(self, scenario_name: str) -> None:
        """Bind selected environmental scenario to the simulation manager."""
        self.scenario_name = scenario_name
        builder = SCENARIO_BUILDERS.get(scenario_name, scenario_normal_operation)
        self.active_scenario = builder(seed=self.random_seed)
        self._applied_event_indices.clear()

    def set_random_seed(self, seed: int) -> None:
        """Update random seed and reinitialize engine."""
        self.random_seed = int(seed)
        self.reset()

    def set_scenario(self, scenario_name: str) -> None:
        """Change environmental scenario and reset starting conditions."""
        self._configure_scenario(scenario_name)
        self.reset()

    def reset(self) -> None:
        """Clear telemetry history and reset sensor engine to baseline t=0."""
        self.engine.reset()
        self._history.clear()
        self._applied_event_indices.clear()
        self.simulation_status = "Stopped"

    @property
    def current_time_s(self) -> float:
        """Current monotonic simulation time in seconds."""
        return self.engine.current_simulation_time

    @property
    def readings_history(self) -> List[SensorReading]:
        """Chronological list of all retained readings up to max_history."""
        return list(self._history)

    def step(self, dt: Optional[float] = None) -> List[SensorReading]:
        """Advance simulation by dt seconds (default step_size) and process due events."""
        step_dt = self.step_size if dt is None else dt
        current_time = self.engine.current_simulation_time

        # Check and apply scenario events due at or before current_time
        if self.active_scenario:
            for idx, event in enumerate(self.active_scenario.events):
                if idx not in self._applied_event_indices:
                    if current_time >= event.start_time_s:
                        if event.is_active_at(current_time):
                            self.engine.trigger_sensor_event(
                                event.sensor_id,
                                target_value=event.target_value,
                                duration_s=event.duration_s,
                            )
                            self._applied_event_indices.add(idx)

        new_readings = self.engine.advance(step_dt)
        for r in new_readings:
            self._history.append(r)

        self.simulation_status = "Running"
        return new_readings

    def run_steps(self, count: int, dt: Optional[float] = None) -> List[SensorReading]:
        """Execute multiple steps consecutively."""
        all_new: List[SensorReading] = []
        for _ in range(count):
            all_new.extend(self.step(dt))
        return all_new

    def get_kpis(self) -> Dict[str, Any]:
        """Compute live system-level metrics directly from the actual sensor engine."""
        sensors = self.engine.get_sensors()
        statuses = self.engine.get_sensor_statuses()

        normal_cnt = sum(1 for s in statuses.values() if s == NodeOperationalStatus.NORMAL)
        warning_cnt = sum(1 for s in statuses.values() if s == NodeOperationalStatus.WARNING)
        critical_cnt = sum(1 for s in statuses.values() if s == NodeOperationalStatus.CRITICAL)
        unavail_cnt = sum(1 for s in statuses.values() if s == NodeOperationalStatus.UNAVAILABLE)

        return {
            "total_sensors": len(sensors),
            "normal_count": normal_cnt,
            "warning_count": warning_cnt,
            "critical_count": critical_cnt,
            "unavailable_count": unavail_cnt,
            "simulation_time_s": self.engine.current_simulation_time,
            "active_events_count": self.engine.active_event_count,
            "simulation_status": self.simulation_status,
        }

    def get_sensor_inventory(self) -> List[Dict[str, Any]]:
        """Extract sensor inventory table rows derived directly from engine configurations."""
        latest_readings = self.engine.get_latest_readings()
        rows: List[Dict[str, Any]] = []

        for sensor in self.engine.get_sensors():
            reading = latest_readings.get(sensor.sensor_id)
            curr_val = f"{reading.value:.2f}" if reading else f"{sensor.current_value:.2f}"
            seq_num = reading.sequence_number if reading else sensor.sequence_number

            cfg = getattr(sensor, "config", None)
            if cfg:
                if cfg.sensor_type == SensorType.HUMIDITY:
                    warn_str = f"[{cfg.warning_low:.1f}, {cfg.warning_threshold:.1f}]"
                    crit_str = f"[{cfg.critical_low:.1f}, {cfg.critical_threshold:.1f}]"
                else:
                    warn_str = f">= {cfg.warning_threshold:.1f}"
                    crit_str = f">= {cfg.critical_threshold:.1f}"
                host_id = cfg.host_node_id
            else:
                warn_str = "N/A"
                crit_str = f">= {sensor.threshold:.1f}"
                host_id = sensor.host_node_id

            rows.append({
                "Sensor ID": sensor.sensor_id,
                "Sensor Type": sensor.sensor_type.value,
                "Unit": sensor.unit,
                "Current Reading": curr_val,
                "Warning Limit": warn_str,
                "Critical Limit": crit_str,
                "Sampling Interval": f"{sensor.sampling_interval_s:.1f} s",
                "Sequence #": seq_num,
                "Derived Status": sensor.status.value.upper(),
                "Host Node": host_id,
            })

        return rows

    def get_recent_readings_df(self, limit: int = 50) -> pd.DataFrame:
        """Return pandas DataFrame of recent readings (most recent first)."""
        if not self._history:
            return pd.DataFrame(columns=[
                "Sim Time (s)", "Sensor ID", "Value", "Unit", "Status", "Seq #", "Threshold Exceeded"
            ])

        records = []
        for r in reversed(list(self._history)[-limit:]):
            records.append({
                "Sim Time (s)": f"{r.simulation_time_s:.1f}",
                "Sensor ID": r.sensor_id,
                "Value": f"{r.value:.2f}",
                "Unit": r.unit,
                "Status": r.status.value.upper(),
                "Seq #": r.sequence_number,
                "Threshold Exceeded": "YES" if r.is_threshold_exceeded else "NO",
            })
        return pd.DataFrame(records)

    def get_time_series_df(self) -> pd.DataFrame:
        """Return pandas DataFrame formatted for time-series charting."""
        if not self._history:
            return pd.DataFrame(columns=["simulation_time_s", "sensor_id", "value", "unit", "status"])

        data = [{
            "simulation_time_s": r.simulation_time_s,
            "sensor_id": r.sensor_id,
            "value": r.value,
            "unit": r.unit,
            "status": r.status.value,
        } for r in self._history]

        return pd.DataFrame(data)
