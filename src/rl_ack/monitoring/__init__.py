"""Module 4: Sensor Monitoring and Incident Detection.

Responsible for:
- Live sensor telemetry processing
- Threshold violation evaluation (gas, temperature, humidity, smoke)
- Alert generation with severity and traffic priority tagging
- Complete incident lifecycle (ACTIVE -> ACKNOWLEDGED -> RECOVERED -> CLOSED)
- Deduplication and severity escalation
- Independence guarantee: incident generation is independent of DQN action selection.
"""

from rl_ack.common.enums import AlertSeverity, IncidentState
from rl_ack.common.models import AlertMessage, IncidentRecord
from rl_ack.monitoring.detector import IncidentDetector
from rl_ack.monitoring.manager import IncidentManager

MODULE_NAME = "monitoring"
MODULE_ID = 4
IS_IMPLEMENTED = True

__all__ = [
    "MODULE_NAME",
    "MODULE_ID",
    "IS_IMPLEMENTED",
    "IncidentDetector",
    "IncidentManager",
    "IncidentState",
    "AlertMessage",
    "IncidentRecord",
    "AlertSeverity",
]
