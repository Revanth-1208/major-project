"""Controlled Smart Kitchen IoT Environment for DQN Training and Evaluation.

Implements a gym-style interface (reset, step) integrating Phase 5's:
- SimulatedNetworkModel (deterministic packet & ACK loss, latency, jitter)
- WorkloadGenerator (balanced CC emergency alarms, HBS telemetry, BET readings)
- Five fixed ACK policies (QoS 0, Local ACK, QoS 1 Retry, Selective Retry, QoS 2)
- Eight-feature state tracker and multi-objective reward calculator.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from rl_ack.common.enums import AckAction, EvaluationPolicy, TrafficClass
from rl_ack.common.models import AlertMessage, NetworkStateVector
from rl_ack.dqn.actions import ACTION_DIM, get_action_metadata, instantiate_policy_for_action
from rl_ack.dqn.reward import RewardCalculator, RewardConfig
from rl_ack.dqn.state import (
    NetworkStateTracker,
    STATE_DIM,
    validate_and_normalize_state,
)
from rl_ack.policies.base import BaseAckPolicy, PolicyConfig, PolicyEvaluationResult
from rl_ack.policies.network_model import NetworkModelConfig, SimulatedNetworkModel
from rl_ack.policies.workload import WorkloadGenerator


@dataclass
class EnvConfig:
    """Configuration for controlled DQN IoT training environment."""

    steps_per_episode: int = 50
    cc_ratio: float = 0.35
    hbs_ratio: float = 0.35
    bet_ratio: float = 0.30
    random_seed: int = 42
    network_loss_rate: float = 0.12
    network_latency_ms: float = 20.0
    initial_energy: float = 1.0


class SmartKitchenIoTEnv:
    """Controlled simulated IoT environment for training and evaluating ACK decision agents."""

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        policy_config: Optional[PolicyConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        self.config = config or EnvConfig()
        self.policy_config = policy_config or PolicyConfig()
        self.reward_config = reward_config or RewardConfig()

        self.random_seed = self.config.random_seed
        self.workload_generator = WorkloadGenerator(random_seed=self.random_seed)

        self.network_config = NetworkModelConfig(
            latency_base_ms=self.config.network_latency_ms,
            jitter_ms=5.0,
            packet_loss_rate=self.config.network_loss_rate,
            ack_loss_rate=0.05,
            random_seed=self.random_seed,
        )
        self.network_model = SimulatedNetworkModel(config=self.network_config)

        # Pre-instantiate the 5 ACK policies
        self.policies: Dict[int, BaseAckPolicy] = {
            a_idx: instantiate_policy_for_action(a_idx, config=self.policy_config)
            for a_idx in range(ACTION_DIM)
        }

        self.reward_calculator = RewardCalculator(config=self.reward_config)
        self.state_tracker = NetworkStateTracker(initial_energy=self.config.initial_energy)

        self._current_step: int = 0
        self._current_workload: List[AlertMessage] = []
        self._current_message: Optional[AlertMessage] = None

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment to initial conditions for a new episode.

        Args:
            seed: Optional integer to re-seed deterministic generation.

        Returns:
            Initial 8-dimensional state vector as np.ndarray of shape (8,).
        """
        if seed is not None:
            self.random_seed = seed
            self.network_config.random_seed = seed

        self.network_model.set_seed(self.random_seed)
        self.workload_generator.set_seed(self.random_seed)
        self.state_tracker.reset(initial_energy=self.config.initial_energy)

        # Reset all 5 policies
        for pol in self.policies.values():
            pol.reset()

        # Generate fresh workload for this episode
        self._current_workload = self.workload_generator.generate_workload(
            count=self.config.steps_per_episode,
            cc_ratio=self.config.cc_ratio,
            hbs_ratio=self.config.hbs_ratio,
            bet_ratio=self.config.bet_ratio,
        )

        self._current_step = 0
        self._current_message = self._current_workload[0]

        # Initial state corresponds to the first message's traffic class
        initial_state = self.state_tracker.build_state_array(self._current_message.priority)
        return initial_state

    def step(self, action: Union[int, AckAction]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advance simulation by executing chosen ACK action for current message.

        Args:
            action: Discrete action index in 0..4 (or AckAction enum).

        Returns:
            Tuple of (next_state, reward, done, info_dict).
        """
        act_idx = int(action)
        if not (0 <= act_idx < ACTION_DIM):
            raise ValueError(f"Action {act_idx} is out of bounds (must be 0..{ACTION_DIM - 1})")

        if self._current_step >= len(self._current_workload):
            raise RuntimeError("Cannot step an environment that is already done. Call reset() first.")

        message = self._current_workload[self._current_step]
        prior_state = self.state_tracker.build_state(message.priority)

        # Register outgoing message in state queue
        self.state_tracker.enqueue_message(message.priority)

        # Execute the chosen policy
        policy = self.policies[act_idx]
        eval_result = policy.evaluate_controlled(message=message, network=self.network_model)

        # Compute multi-objective reward
        reward, reward_terms = self.reward_calculator.compute_reward(eval_result, prior_state)

        # Update rolling state tracker with observed outcome
        self.state_tracker.record_outcome(
            confirmed=eval_result.delivery_confirmed,
            latency_ms=eval_result.end_to_end_latency_ms,
            overhead_bytes=eval_result.message_overhead_bytes,
        )

        self._current_step += 1
        done = self._current_step >= len(self._current_workload)

        # Build next state
        if not done:
            self._current_message = self._current_workload[self._current_step]
            next_traffic = self._current_message.priority
        else:
            self._current_message = None
            next_traffic = TrafficClass.BEST_EFFORT

        next_state = self.state_tracker.build_state_array(next_traffic)

        info: Dict[str, Any] = {
            "step": self._current_step,
            "action": act_idx,
            "action_label": get_action_metadata(act_idx).label,
            "traffic_class": message.priority.value,
            "sensor_id": message.sensor_id,
            "delivery_confirmed": eval_result.delivery_confirmed,
            "broker_acknowledged": eval_result.broker_acknowledged,
            "local_acknowledged": eval_result.local_acknowledged,
            "latency_ms": eval_result.end_to_end_latency_ms,
            "overhead_bytes": eval_result.message_overhead_bytes,
            "retry_count": eval_result.retry_count,
            "deadline_met": eval_result.deadline_met,
            "reward_terms": reward_terms,
            "simulated": True,
        }

        return next_state, reward, done, info
