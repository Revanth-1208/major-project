"""Deep Q-Network Agent Implementation for the RL-ACK Framework.

Implements IDQNAgent with:
- Dual Q-networks (Online & Target)
- Experience replay buffer
- Epsilon-greedy action selection
- Bellman loss optimization with gradient clipping
- Terminal-state masking
- Atomic checkpoint save and load capabilities.
"""

from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_ack.common.enums import AckAction
from rl_ack.common.interfaces import IDQNAgent
from rl_ack.common.models import NetworkStateVector
from rl_ack.config import DQNConfig
from rl_ack.dqn.actions import ACTION_DIM, get_action_metadata
from rl_ack.dqn.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from rl_ack.dqn.model import QNetwork
from rl_ack.dqn.replay_buffer import ReplayBuffer
from rl_ack.dqn.state import STATE_DIM, validate_and_normalize_state


class DQNAgent(IDQNAgent):
    """Deep Q-Network learning agent for adaptive ACK policy selection."""

    def __init__(
        self,
        config: Optional[DQNConfig] = None,
        device: Optional[torch.device] = None,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.config = config or DQNConfig()
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

        # Device selection: CPU default, CUDA only if available
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # Initialize Online and Target Networks
        self.online_net = QNetwork(
            state_dim=self.config.state_dim,
            hidden_dim_1=self.config.hidden_dim_1,
            hidden_dim_2=self.config.hidden_dim_2,
            action_dim=self.config.action_dim,
            device=self.device,
        )
        self.target_net = QNetwork(
            state_dim=self.config.state_dim,
            hidden_dim_1=self.config.hidden_dim_1,
            hidden_dim_2=self.config.hidden_dim_2,
            action_dim=self.config.action_dim,
            device=self.device,
        )
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.config.learning_rate,
        )

        # Experience Replay Buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.memory_capacity,
            random_seed=random_seed,
            device=self.device,
        )

        # Exploration & tracking state
        self.epsilon: float = self.config.epsilon_start
        self.total_steps: int = 0
        self.episodes_trained: int = 0

    def select_action(
        self,
        state: Union[NetworkStateVector, np.ndarray, List[float]],
        explore: bool = True,
    ) -> AckAction:
        """Select action using epsilon-greedy (explore=True) or greedy (explore=False).

        Args:
            state: 8-dimensional state vector.
            explore: Whether to apply epsilon-greedy exploration.

        Returns:
            AckAction enum corresponding to the chosen action.
        """
        s_arr = validate_and_normalize_state(state)

        if explore and self._rng.random() < self.epsilon:
            action_idx = self._rng.randint(0, self.config.action_dim - 1)
        else:
            with torch.no_grad():
                q_vals = self.online_net(s_arr)
                action_idx = int(torch.argmax(q_vals).item())

        return AckAction(action_idx)

    def get_q_values(
        self,
        state: Union[NetworkStateVector, np.ndarray, List[float]],
    ) -> np.ndarray:
        """Retrieve the 5 predicted Q-values for a given state without exploration."""
        s_arr = validate_and_normalize_state(state)
        with torch.no_grad():
            q_vals = self.online_net(s_arr)
            return q_vals.cpu().numpy().astype(np.float32)

    def store_transition(
        self,
        state: Union[NetworkStateVector, np.ndarray, List[float]],
        action: Union[AckAction, int],
        reward: float,
        next_state: Union[NetworkStateVector, np.ndarray, List[float]],
        done: bool,
    ) -> None:
        """Store step transition in experience replay buffer."""
        act_idx = int(action)
        self.replay_buffer.push(state, act_idx, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Perform one Bellman loss optimization step; return loss if trained."""
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        # Current Q-values: Q_online(s, a)
        q_eval = self.online_net(states).gather(1, actions)

        # Target Q-values: y = r + (1 - done) * gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + (1.0 - dones) * self.config.gamma * q_next

        # Compute Huber loss (Smooth L1 Loss)
        loss = F.smooth_l1_loss(q_eval, q_target)

        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1

        # Periodically update target network
        if self.total_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Decay exploration rate epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

        return float(loss.item())

    def update_target_network(self) -> None:
        """Immediately synchronize target network weights with online network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_checkpoint(
        self,
        path: Union[str, Path],
        reward_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Save model weights and training state to checkpoint file."""
        save_checkpoint(
            filepath=path,
            online_network=self.online_net,
            target_network=self.target_net,
            optimizer=self.optimizer,
            epsilon=self.epsilon,
            episodes_trained=self.episodes_trained,
            total_steps=self.total_steps,
            config=self.config,
            reward_weights=reward_weights,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> bool:
        """Load model weights and training state from checkpoint file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            meta = load_checkpoint(
                filepath=path,
                online_network=self.online_net,
                target_network=self.target_net,
                optimizer=self.optimizer,
                device=self.device,
            )
            self.epsilon = float(meta.get("epsilon", self.config.epsilon_min))
            self.total_steps = int(meta.get("total_steps", 0))
            self.episodes_trained = int(meta.get("episodes_trained", 0))
            return True
        except CheckpointError:
            return False
