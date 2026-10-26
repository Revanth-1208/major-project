"""Module 7: Deep Q-Network (DQN) Architecture, Training, and Evaluation.

Responsible for:
- 8 input features -> 64-neuron hidden -> 32-neuron hidden -> 5 Q-value outputs
- Online network and Target network (QNetwork)
- Experience replay buffer with uniform sampling (ReplayBuffer)
- Epsilon-greedy exploration schedule with decay
- Bellman loss optimization and gradient clipping
- Multi-objective reward calculation balancing delivery, deadline, latency, overhead, and energy
- Model checkpointing and training history persistence
- Greedy evaluation against Phase 5 fixed policy baselines.

Implementation completed in Phase 6.
"""

from rl_ack.dqn.actions import (
    ACTION_DIM,
    ACTION_REGISTRY,
    ActionMetadata,
    action_to_policy_enum,
    get_action_metadata,
    instantiate_policy_for_action,
    policy_enum_to_action,
)
from rl_ack.dqn.agent import DQNAgent
from rl_ack.dqn.checkpoint import (
    CheckpointError,
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from rl_ack.dqn.environment import EnvConfig, SmartKitchenIoTEnv
from rl_ack.dqn.evaluator import DQNEvaluationReport, DQNEvaluator
from rl_ack.dqn.model import QNetwork
from rl_ack.dqn.replay_buffer import ReplayBuffer
from rl_ack.dqn.reward import RewardCalculator, RewardConfig
from rl_ack.dqn.state import (
    DEFAULT_BASELINE_LATENCY_MS,
    DEFAULT_WINDOW_SIZE,
    InvalidStateError,
    MAX_LATENCY_NORMALIZATION_MS,
    NetworkStateTracker,
    STATE_DIM,
    create_state_vector,
    validate_and_normalize_state,
)
from rl_ack.dqn.trainer import DQNTrainer, EpisodeMetrics

MODULE_NAME = "dqn"
MODULE_ID = 7
IS_IMPLEMENTED = True

__all__ = [
    "MODULE_NAME",
    "MODULE_ID",
    "IS_IMPLEMENTED",
    "STATE_DIM",
    "ACTION_DIM",
    "MAX_LATENCY_NORMALIZATION_MS",
    "DEFAULT_BASELINE_LATENCY_MS",
    "DEFAULT_WINDOW_SIZE",
    "InvalidStateError",
    "validate_and_normalize_state",
    "create_state_vector",
    "NetworkStateTracker",
    "ActionMetadata",
    "ACTION_REGISTRY",
    "get_action_metadata",
    "action_to_policy_enum",
    "policy_enum_to_action",
    "instantiate_policy_for_action",
    "QNetwork",
    "ReplayBuffer",
    "RewardConfig",
    "RewardCalculator",
    "EnvConfig",
    "SmartKitchenIoTEnv",
    "DQNAgent",
    "DQNTrainer",
    "EpisodeMetrics",
    "DQNEvaluator",
    "DQNEvaluationReport",
    "CheckpointMetadata",
    "CheckpointError",
    "save_checkpoint",
    "load_checkpoint",
]
