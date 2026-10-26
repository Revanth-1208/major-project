"""Workload Generator for Phase 5 ACK Policy Comparative Evaluation.

Generates standardized, reproducible workloads spanning all three traffic classes:
1. Critical Control (CC) - High priority emergency alerts (Gas, Smoke, Fire)
2. High Bandwidth Streaming (HBS) - High-frequency virtual sensor telemetry
3. Best Effort Traffic (BET) - Non-critical routine status and telemetry
"""

from typing import List, Optional
import random

from rl_ack.common.enums import AlertSeverity, TrafficClass
from rl_ack.common.models import AlertMessage


class WorkloadGenerator:
    """Produces reproducible batches of heterogeneous smart kitchen messages."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self._rng = random.Random(self.random_seed)

    def set_seed(self, seed: int) -> None:
        """Reset pseudo-random generator with fixed seed."""
        self.random_seed = seed
        self._rng = random.Random(self.random_seed)

    def generate_workload(
        self,
        count: int = 20,
        cc_ratio: float = 0.30,
        hbs_ratio: float = 0.40,
        bet_ratio: float = 0.30,
    ) -> List[AlertMessage]:
        """Generate a deterministic synthetic workload with specified traffic class proportions.

        Args:
            count: Total number of messages to generate.
            cc_ratio: Proportion of Critical Control emergency alerts.
            hbs_ratio: Proportion of High Bandwidth Streaming telemetry.
            bet_ratio: Proportion of Best Effort Traffic readings.

        Returns:
            List of AlertMessage objects ready for policy evaluation.
        """
        # Normalize ratios
        total_ratio = cc_ratio + hbs_ratio + bet_ratio
        cc_p = cc_ratio / total_ratio
        hbs_p = hbs_ratio / total_ratio

        workload: List[AlertMessage] = []
        sim_time = 0.0

        for i in range(1, count + 1):
            sim_time += round(self._rng.uniform(0.5, 2.0), 1)
            draw = self._rng.random()

            if draw < cc_p:
                # Critical Control (Emergency)
                sensor_id = self._rng.choice(["GAS-001", "SMOKE-001", "TEMP-001"])
                if sensor_id == "GAS-001":
                    val = round(self._rng.uniform(410.0, 580.0), 1)
                    unit = "ppm"
                    threshold = 400.0
                    msg = f"CRITICAL Gas Leak Detected: {val:.1f} ppm"
                elif sensor_id == "SMOKE-001":
                    val = round(self._rng.uniform(3.1, 5.5), 2)
                    unit = "% obs/m"
                    threshold = 3.0
                    msg = f"CRITICAL Dense Smoke Detected: {val:.2f} % obs/m"
                else:
                    val = round(self._rng.uniform(61.0, 78.0), 1)
                    unit = "°C"
                    threshold = 60.0
                    msg = f"CRITICAL Thermal Anomaly: {val:.1f} °C"

                alert = AlertMessage(
                    alert_id=f"WKLD-CC-{i:04d}",
                    sensor_id=sensor_id,
                    severity=AlertSeverity.CRITICAL,
                    reading_value=val,
                    threshold_value=threshold,
                    unit=unit,
                    incident_type="emergency_alarm",
                    message=msg,
                    simulation_time_s=sim_time,
                    priority=TrafficClass.CRITICAL_CONTROL,
                )

            elif draw < (cc_p + hbs_p):
                # High Bandwidth Streaming
                sensor_id = self._rng.choice(["TEMP-001", "HUM-001", "GAS-001"])
                val = round(self._rng.uniform(22.0, 35.0), 1) if "TEMP" in sensor_id else round(self._rng.uniform(40.0, 60.0), 1)
                unit = "°C" if "TEMP" in sensor_id else ("% RH" if "HUM" in sensor_id else "ppm")
                alert = AlertMessage(
                    alert_id=f"WKLD-HBS-{i:04d}",
                    sensor_id=sensor_id,
                    severity=AlertSeverity.INFO,
                    reading_value=val,
                    threshold_value=60.0,
                    unit=unit,
                    incident_type="continuous_telemetry",
                    message=f"Streaming Telemetry Sample: {val:.1f} {unit}",
                    simulation_time_s=sim_time,
                    priority=TrafficClass.HIGH_BANDWIDTH_STREAMING,
                )

            else:
                # Best Effort Traffic
                sensor_id = "HUM-001"
                val = round(self._rng.uniform(45.0, 55.0), 1)
                alert = AlertMessage(
                    alert_id=f"WKLD-BET-{i:04d}",
                    sensor_id=sensor_id,
                    severity=AlertSeverity.INFO,
                    reading_value=val,
                    threshold_value=75.0,
                    unit="% RH",
                    incident_type="routine_report",
                    message=f"Periodic Status Update: {val:.1f} % RH",
                    simulation_time_s=sim_time,
                    priority=TrafficClass.BEST_EFFORT,
                )

            workload.append(alert)

        return workload
