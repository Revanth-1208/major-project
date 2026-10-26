"""Action Space Definition and Policy Mapping for the RL-ACK DQN Agent.

Defines the 5-action output space and canonical mapping to Phase 5 fixed policies.
Ensures identical action semantics across training, evaluation, and future controller integration.

Action Mapping:
0 -> Static MQTT QoS 0 (Fire-and-forget, zero retries, minimal overhead)
1 -> Static Local ACK (Immediate proximate edge proxy response)
2 -> Static MQTT QoS 1 (Broker PUBACK tracking with bounded timeout retries)
3 -> Local ACK with Selective Retransmission (Edge proxy ACK + prioritized CC retries)
4 -> Static MQTT QoS 2 (4-way exactly-once protocol handshake)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from rl_ack.common.enums import AckAction, EvaluationPolicy
from rl_ack.policies.base import BaseAckPolicy, PolicyConfig
from rl_ack.policies.local_ack_retry import LocalAckSelectiveRetryPolicy
from rl_ack.policies.static_local_ack import StaticLocalAckPolicy
from rl_ack.policies.static_qos0 import StaticQos0Policy
from rl_ack.policies.static_qos1 import StaticQos1RetryPolicy
from rl_ack.policies.static_qos2 import StaticQos2Policy


ACTION_DIM: int = 5


@dataclass(frozen=True)
class ActionMetadata:
    """Canonical descriptor for each discrete DQN action."""

    action_idx: int
    action_enum: AckAction
    policy_enum: EvaluationPolicy
    policy_class: Type[BaseAckPolicy]
    label: str
    qos_level: int
    description: str
    base_overhead_bytes: int


ACTION_REGISTRY: Dict[int, ActionMetadata] = {
    0: ActionMetadata(
        action_idx=0,
        action_enum=AckAction.MQTT_QOS_0,
        policy_enum=EvaluationPolicy.STATIC_MQTT_QOS_0,
        policy_class=StaticQos0Policy,
        label="Static MQTT QoS 0",
        qos_level=0,
        description="QoS 0 fire-and-forget; no broker or subscriber acknowledgments, zero retries, lowest overhead.",
        base_overhead_bytes=32,
    ),
    1: ActionMetadata(
        action_idx=1,
        action_enum=AckAction.LOCAL_ACK,
        policy_enum=EvaluationPolicy.STATIC_LOCAL_ACK,
        policy_class=StaticLocalAckPolicy,
        label="Static Local ACK",
        qos_level=0,
        description="Immediate proximate edge proxy acknowledgment, fast local roundtrip, no end-to-end guarantee.",
        base_overhead_bytes=64,
    ),
    2: ActionMetadata(
        action_idx=2,
        action_enum=AckAction.MQTT_QOS_1,
        policy_enum=EvaluationPolicy.STATIC_MQTT_QOS_1,
        policy_class=StaticQos1RetryPolicy,
        label="Static MQTT QoS 1 (Retries)",
        qos_level=1,
        description="MQTT QoS 1 with broker PUBACK tracking and bounded application-level retries on timeout.",
        base_overhead_bytes=96,
    ),
    3: ActionMetadata(
        action_idx=3,
        action_enum=AckAction.LOCAL_ACK_SELECTIVE_RETRANSMIT,
        policy_enum=EvaluationPolicy.LOCAL_ACK_SELECTIVE_RETRANSMIT,
        policy_class=LocalAckSelectiveRetryPolicy,
        label="Local ACK + Selective Retry",
        qos_level=1,
        description="Immediate edge proxy ACK coupled with downstream selective retransmissions prioritizing Critical Control.",
        base_overhead_bytes=80,
    ),
    4: ActionMetadata(
        action_idx=4,
        action_enum=AckAction.MQTT_QOS_2,
        policy_enum=EvaluationPolicy.STATIC_MQTT_QOS_2,
        policy_class=StaticQos2Policy,
        label="Static MQTT QoS 2",
        qos_level=2,
        description="MQTT QoS 2 four-way handshake (PUBLISH -> PUBREC -> PUBREL -> PUBCOMP); exactly-once transport delivery.",
        base_overhead_bytes=160,
    ),
}


def get_action_metadata(action: Union[int, AckAction]) -> ActionMetadata:
    """Retrieve metadata descriptor for a given action index or enum."""
    idx = int(action)
    if idx not in ACTION_REGISTRY:
        raise ValueError(f"Invalid DQN action index: {idx}. Must be an integer in 0..{ACTION_DIM - 1}")
    return ACTION_REGISTRY[idx]


def action_to_policy_enum(action: Union[int, AckAction]) -> EvaluationPolicy:
    """Map action integer or enum to EvaluationPolicy enum."""
    return get_action_metadata(action).policy_enum


def policy_enum_to_action(policy: EvaluationPolicy) -> AckAction:
    """Reverse map EvaluationPolicy enum to AckAction enum."""
    for meta in ACTION_REGISTRY.values():
        if meta.policy_enum == policy:
            return meta.action_enum
    raise ValueError(f"No matching DQN action for policy: {policy}")


def instantiate_policy_for_action(
    action: Union[int, AckAction],
    config: Optional[PolicyConfig] = None,
) -> BaseAckPolicy:
    """Instantiate the corresponding BaseAckPolicy for a selected DQN action."""
    meta = get_action_metadata(action)
    return meta.policy_class(config=config)
