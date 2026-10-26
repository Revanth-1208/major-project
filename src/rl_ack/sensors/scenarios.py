"""Configurable scenario engine supporting multi-event environmental simulation.

Simulates normal conditions, individual parameter anomalies, and multi-sensor overlapping
kitchen emergency scenarios with deterministic random seed reproducibility.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rl_ack.common.enums import NodeOperationalStatus, SensorType
from rl_ack.sensors.sensor_engine import SensorEngine


@dataclass
class ScenarioEvent:
    """A scheduled environmental event within a simulation scenario."""
    sensor_id: str
    start_time_s: float
    duration_s: Optional[float]
    target_value: float
    description: str = ""

    def is_active_at(self, current_time_s: float) -> bool:
        """Check whether the event is active at the given simulation time."""
        if current_time_s < self.start_time_s:
            return False
        if self.duration_s is not None:
            return current_time_s < (self.start_time_s + self.duration_s)
        return True


@dataclass
class SensorScenario:
    """Specification of an environmental scenario comprising multiple scheduled events."""
    name: str
    description: str
    events: List[ScenarioEvent] = field(default_factory=list)
    total_duration_s: float = 20.0
    random_seed: int = 42


# ==============================================================================
# Standard Predefined Scenarios
# ==============================================================================

def scenario_normal_operation(duration_s: float = 15.0, seed: int = 42) -> SensorScenario:
    """Scenario 1: Normal steady-state operation with small stochastic fluctuations."""
    return SensorScenario(
        name="Normal Operation",
        description="All sensors operate near baseline with small Gaussian noise; all statuses remain normal.",
        events=[],
        total_duration_s=duration_s,
        random_seed=seed,
    )


def scenario_gradual_gas_leak(
    start_time_s: float = 3.0,
    duration_s: float = 7.0,
    target_ppm: float = 850.0,
    total_duration_s: float = 18.0,
    seed: int = 42,
) -> SensorScenario:
    """Scenario 2: Gas concentration climbs through warning (100 ppm) to critical (400 ppm), then clears."""
    return SensorScenario(
        name="Gradual Gas Leak",
        description="Gas concentration steadily rises above warning and critical thresholds, then recovers.",
        events=[
            ScenarioEvent(
                sensor_id="GAS-001",
                start_time_s=start_time_s,
                duration_s=duration_s,
                target_value=target_ppm,
                description="Simulated unignited LPG gas accumulation",
            )
        ],
        total_duration_s=total_duration_s,
        random_seed=seed,
    )


def scenario_abnormal_temperature_rise(
    start_time_s: float = 3.0,
    duration_s: float = 7.0,
    target_temp_c: float = 95.0,
    total_duration_s: float = 18.0,
    seed: int = 43,
) -> SensorScenario:
    """Scenario 3: Temperature rises through warning (38°C) to critical (55°C), then cools."""
    return SensorScenario(
        name="Abnormal Temperature Rise",
        description="Thermal overheating event crossing warning and critical boundaries, followed by cooling.",
        events=[
            ScenarioEvent(
                sensor_id="TEMP-001",
                start_time_s=start_time_s,
                duration_s=duration_s,
                target_value=target_temp_c,
                description="Cooktop runaway heating event",
            )
        ],
        total_duration_s=total_duration_s,
        random_seed=seed,
    )


def scenario_increasing_smoke(
    start_time_s: float = 3.0,
    duration_s: float = 6.0,
    target_obs: float = 7.5,
    total_duration_s: float = 18.0,
    seed: int = 45,
) -> SensorScenario:
    """Scenario 4: Optical smoke obscuration rises through warning (1.0) and critical (3.5 % obs/m)."""
    return SensorScenario(
        name="Increasing Smoke Obscuration",
        description="Combustion smoke release crossing warning and critical thresholds, then ventilating.",
        events=[
            ScenarioEvent(
                sensor_id="SMOKE-001",
                start_time_s=start_time_s,
                duration_s=duration_s,
                target_value=target_obs,
                description="Smoldering combustion smoke obscuration",
            )
        ],
        total_duration_s=total_duration_s,
        random_seed=seed,
    )


def scenario_humidity_change(
    mode: str = "high",
    start_time_s: float = 2.0,
    duration_s: float = 16.0,
    total_duration_s: float = 26.0,
    seed: int = 44,
) -> SensorScenario:
    """Scenario 5: Relative humidity excursion (two-sided: either high condensation or severe dryness)."""
    target = 99.0 if mode == "high" else 1.0
    desc = "High steam condensation event" if mode == "high" else "Severe HVAC dryness event"
    return SensorScenario(
        name=f"Humidity Anomaly ({mode.capitalize()})",
        description=f"Relative humidity crossing {mode} warning/critical bounds, then equilibrating.",
        events=[
            ScenarioEvent(
                sensor_id="HUM-001",
                start_time_s=start_time_s,
                duration_s=duration_s,
                target_value=target,
                description=desc,
            )
        ],
        total_duration_s=total_duration_s,
        random_seed=seed,
    )


def scenario_kitchen_fire_multi_event(
    total_duration_s: float = 20.0,
    seed: int = 42,
) -> SensorScenario:
    """Scenario 6: Multi-event overlapping kitchen emergency.

    Timeline:
    - t=2.0s: Gas leak initiates (GAS-001 climbing toward 850 ppm, crossing warning and critical).
    - t=5.0s: Ignition causes heat spike (TEMP-001 climbing toward 95°C, crossing critical).
    - t=6.0s: Heavy combustion smoke develops (SMOKE-001 climbing toward 8.0 % obs/m, crossing critical).
    - t=9.0s: Gas shutoff valve activates; gas begins recovery.
    - t=11.0s: Fire suppression and exhaust ventilation activate; temperature and smoke begin recovery.
    - t=16.0s+: All sensors recover through warning to normal.
    """
    return SensorScenario(
        name="Kitchen Fire Multi-Event Emergency",
        description="Overlapping gas accumulation, thermal ignition, and smoke combustion with progressive recovery.",
        events=[
            ScenarioEvent(
                sensor_id="GAS-001",
                start_time_s=2.0,
                duration_s=7.0,  # Active t=2.0s to t=9.0s
                target_value=850.0,
                description="Gas leak accumulation",
            ),
            ScenarioEvent(
                sensor_id="TEMP-001",
                start_time_s=5.0,
                duration_s=6.0,  # Active t=5.0s to t=11.0s
                target_value=95.0,
                description="Combustion ignition heat spike",
            ),
            ScenarioEvent(
                sensor_id="SMOKE-001",
                start_time_s=6.0,
                duration_s=5.0,  # Active t=6.0s to t=11.0s
                target_value=8.0,
                description="Combustion optical smoke obscuration",
            ),
        ],
        total_duration_s=total_duration_s,
        random_seed=seed,
    )


# ==============================================================================
# Scenario Execution Runner
# ==============================================================================

class ScenarioRunner:
    """Coordinates execution of a SensorScenario across the SensorEngine."""

    def __init__(self, scenario: SensorScenario, engine: Optional[SensorEngine] = None) -> None:
        self.scenario = scenario
        self.engine = engine if engine is not None else SensorEngine.create_default_engine(
            seed_offset=self.scenario.random_seed
        )
        self._active_events: Dict[str, ScenarioEvent] = {}
        self._status_transitions: List[Dict[str, Any]] = []
        self._prev_statuses: Dict[str, NodeOperationalStatus] = {}

    @property
    def status_transitions(self) -> List[Dict[str, Any]]:
        """Log of all observed status transitions (e.g. normal -> warning -> critical -> normal)."""
        return list(self._status_transitions)

    def run(self, step_size: float = 1.0) -> List[Dict[str, Any]]:
        """Execute the scenario from t=0 to total_duration_s, returning all step telemetry."""
        self.engine.reset()
        self._active_events.clear()
        self._status_transitions.clear()
        self._prev_statuses = {s.sensor_id: s.status for s in self.engine.get_sensors()}

        timeline_records: List[Dict[str, Any]] = []
        current_time = 0.0

        while current_time < self.scenario.total_duration_s:
            # 1. Check for newly active or expired scenario events
            for event in self.scenario.events:
                # Trigger event if due and not yet triggered
                if (current_time >= event.start_time_s) and (event.sensor_id not in self._active_events):
                    if event.is_active_at(current_time):
                        self._active_events[event.sensor_id] = event
                        self.engine.trigger_sensor_event(
                            event.sensor_id,
                            target_value=event.target_value,
                            duration_s=event.duration_s,
                        )

                # Check if an active event has expired
                if event.sensor_id in self._active_events:
                    if not event.is_active_at(current_time):
                        del self._active_events[event.sensor_id]
                        # Clearance is handled by duration_s expiry inside BaseSensor,
                        # but we ensure explicit clearance here
                        self.engine.clear_sensor_event(event.sensor_id)

            # 2. Advance simulation engine clock
            readings = self.engine.advance(step_size)
            current_time = self.engine.current_simulation_time

            # 3. Process readings and log transitions
            for r in readings:
                prev_status = self._prev_statuses.get(r.sensor_id, NodeOperationalStatus.NORMAL)
                if r.status != prev_status:
                    self._status_transitions.append({
                        "simulation_time_s": r.simulation_time_s,
                        "sensor_id": r.sensor_id,
                        "from_status": prev_status.value,
                        "to_status": r.status.value,
                        "value": r.value,
                        "unit": r.unit,
                    })
                    self._prev_statuses[r.sensor_id] = r.status

                timeline_records.append({
                    "simulation_time_s": r.simulation_time_s,
                    "sensor_id": r.sensor_id,
                    "sensor_type": r.sensor_type.value,
                    "value": r.value,
                    "unit": r.unit,
                    "sequence_number": r.sequence_number,
                    "status": r.status.value,
                    "is_threshold_exceeded": r.is_threshold_exceeded,
                })

        return timeline_records
