"""Module 6 / Phase 5: ACK Policy Engine and Comparative Evaluation.

Implements and evaluates the five fixed acknowledgment policies:
1. Policy A — Static MQTT QoS 0
2. Policy B — Static Local ACK
3. Policy C — Static MQTT QoS 1 with Retries
4. Policy D — Local ACK plus Selective Retransmission
5. Policy E — Static MQTT QoS 2

Provides controlled channel simulation, live MQTT integration,
statistical metric aggregation, and benchmark workload generation.
"""

from rl_ack.policies.base import (
    BaseAckPolicy,
    PolicyConfig,
    PolicyEvaluationResult,
)
from rl_ack.policies.local_ack_retry import LocalAckSelectiveRetryPolicy
from rl_ack.policies.network_model import (
    NetworkModelConfig,
    SimulatedNetworkModel,
)
from rl_ack.policies.runner import (
    PolicyExperimentRunner,
    PolicySummaryMetrics,
)
from rl_ack.policies.static_local_ack import StaticLocalAckPolicy
from rl_ack.policies.static_qos0 import StaticQos0Policy
from rl_ack.policies.static_qos1 import StaticQos1RetryPolicy
from rl_ack.policies.static_qos2 import StaticQos2Policy
from rl_ack.policies.workload import WorkloadGenerator

MODULE_NAME = "policies"
MODULE_ID = 6
IS_IMPLEMENTED = True

__all__ = [
    "BaseAckPolicy",
    "PolicyConfig",
    "PolicyEvaluationResult",
    "StaticQos0Policy",
    "StaticLocalAckPolicy",
    "StaticQos1RetryPolicy",
    "LocalAckSelectiveRetryPolicy",
    "StaticQos2Policy",
    "SimulatedNetworkModel",
    "NetworkModelConfig",
    "WorkloadGenerator",
    "PolicyExperimentRunner",
    "PolicySummaryMetrics",
    "MODULE_NAME",
    "MODULE_ID",
    "IS_IMPLEMENTED",
]
