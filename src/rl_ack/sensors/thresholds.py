"""Threshold rules and status classification logic with hysteresis for virtual sensors.

NOTE: All thresholds and operating limits defined herein are simulation modeling parameters,
not certified life-safety or regulatory limits.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rl_ack.common.enums import NodeOperationalStatus, SensorType


class ThresholdDirection(str, Enum):
    """Directionality of sensor threshold violations."""
    HIGH_DANGER = "high_danger"  # Elevated readings indicate abnormal/danger (Gas, Temp, Smoke)
    TWO_SIDED = "two_sided"      # Both high and low deviations are abnormal (Humidity)


@dataclass
class ThresholdRule:
    """Configurable threshold specifications and clearing rules with hysteresis."""
    direction: ThresholdDirection
    warning_threshold: float
    critical_threshold: float
    hysteresis: float = 0.0

    # Two-sided limits (used when direction == TWO_SIDED)
    warning_low: Optional[float] = None
    critical_low: Optional[float] = None

    def __post_init__(self) -> None:
        if self.direction == ThresholdDirection.HIGH_DANGER:
            if self.warning_threshold >= self.critical_threshold:
                raise ValueError(
                    f"warning_threshold ({self.warning_threshold}) must be strictly less "
                    f"than critical_threshold ({self.critical_threshold}) for HIGH_DANGER sensors."
                )
        elif self.direction == ThresholdDirection.TWO_SIDED:
            if self.warning_low is None or self.critical_low is None:
                raise ValueError("TWO_SIDED threshold rules require warning_low and critical_low.")
            if not (self.critical_low < self.warning_low <= self.warning_threshold < self.critical_threshold):
                raise ValueError(
                    f"TWO_SIDED limits must satisfy: critical_low ({self.critical_low}) < "
                    f"warning_low ({self.warning_low}) <= warning_high ({self.warning_threshold}) < "
                    f"critical_high ({self.critical_threshold})."
                )
        if self.hysteresis < 0.0:
            raise ValueError(f"hysteresis must be non-negative, got {self.hysteresis}")


class StatusClassifier:
    """Reusable stateful status classifier evaluating readings against thresholds with hysteresis.

    Applies hysteresis during recovery/clearing transitions to prevent status chattering
    around threshold boundaries.
    """

    def __init__(
        self,
        rule: ThresholdRule,
        initial_status: NodeOperationalStatus = NodeOperationalStatus.NORMAL,
    ) -> None:
        self.rule = rule
        self._current_status = initial_status

    @property
    def current_status(self) -> NodeOperationalStatus:
        """Currently active classified status."""
        return self._current_status

    def reset(self, initial_status: NodeOperationalStatus = NodeOperationalStatus.NORMAL) -> None:
        """Reset classifier state."""
        self._current_status = initial_status

    def classify(
        self,
        value: float,
        current_status: Optional[NodeOperationalStatus] = None,
    ) -> NodeOperationalStatus:
        """Evaluate a reading value and return the derived status, honoring hysteresis."""
        state = current_status if current_status is not None else self._current_status

        # If sensor is explicitly unavailable, maintain unavailable status
        if state == NodeOperationalStatus.UNAVAILABLE:
            return NodeOperationalStatus.UNAVAILABLE

        if self.rule.direction == ThresholdDirection.HIGH_DANGER:
            new_status = self._classify_high_danger(value, state)
        else:
            new_status = self._classify_two_sided(value, state)

        self._current_status = new_status
        return new_status

    def _classify_high_danger(
        self,
        value: float,
        current_state: NodeOperationalStatus,
    ) -> NodeOperationalStatus:
        warn = self.rule.warning_threshold
        crit = self.rule.critical_threshold
        h = self.rule.hysteresis

        # 1. Critical condition
        if value >= crit:
            return NodeOperationalStatus.CRITICAL

        # 2. Transition from Critical -> lower state (hysteresis clearing check)
        if current_state == NodeOperationalStatus.CRITICAL:
            if value >= (crit - h):
                # Remains critical until it drops below crit - hysteresis
                return NodeOperationalStatus.CRITICAL
            elif value >= warn:
                return NodeOperationalStatus.WARNING
            elif value < (warn - h):
                return NodeOperationalStatus.NORMAL
            else:
                return NodeOperationalStatus.WARNING

        # 3. Warning condition
        if value >= warn:
            return NodeOperationalStatus.WARNING

        # 4. Transition from Warning -> Normal (hysteresis clearing check)
        if current_state == NodeOperationalStatus.WARNING:
            if value >= (warn - h):
                # Remains warning until it drops below warn - hysteresis
                return NodeOperationalStatus.WARNING
            else:
                return NodeOperationalStatus.NORMAL

        # 5. Normal condition
        return NodeOperationalStatus.NORMAL

    def _classify_two_sided(
        self,
        value: float,
        current_state: NodeOperationalStatus,
    ) -> NodeOperationalStatus:
        crit_low = self.rule.critical_low  # type: ignore[assignment]
        warn_low = self.rule.warning_low  # type: ignore[assignment]
        warn_high = self.rule.warning_threshold
        crit_high = self.rule.critical_threshold
        h = self.rule.hysteresis

        # 1. Critical condition (either high or low)
        if value <= crit_low or value >= crit_high:
            return NodeOperationalStatus.CRITICAL

        # 2. Recovery from Critical
        if current_state == NodeOperationalStatus.CRITICAL:
            if value <= (crit_low + h) or value >= (crit_high - h):
                return NodeOperationalStatus.CRITICAL
            elif value < warn_low or value > warn_high:
                return NodeOperationalStatus.WARNING
            else:
                return NodeOperationalStatus.NORMAL

        # 3. Warning condition (either high or low)
        if value < warn_low or value > warn_high:
            return NodeOperationalStatus.WARNING

        # 4. Recovery from Warning
        if current_state == NodeOperationalStatus.WARNING:
            if (value < warn_low + h) or (value > warn_high - h):
                return NodeOperationalStatus.WARNING
            else:
                return NodeOperationalStatus.NORMAL

        # 5. Normal condition
        return NodeOperationalStatus.NORMAL


def create_classifier_for_sensor(
    sensor_type: SensorType,
    warning_threshold: float,
    critical_threshold: float,
    hysteresis: float,
    warning_low: Optional[float] = None,
    critical_low: Optional[float] = None,
) -> StatusClassifier:
    """Factory helper to build appropriate classifier for a given sensor type."""
    direction = ThresholdDirection.TWO_SIDED if sensor_type == SensorType.HUMIDITY else ThresholdDirection.HIGH_DANGER
    rule = ThresholdRule(
        direction=direction,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        hysteresis=hysteresis,
        warning_low=warning_low,
        critical_low=critical_low,
    )
    return StatusClassifier(rule)
