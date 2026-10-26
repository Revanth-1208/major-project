"""Experience Replay Buffer for the RL-ACK DQN Agent.

Stores transition tuples (state, action, reward, next_state, done) with configurable
capacity, circular FIFO overwrite, reproducible batch sampling, and strict validation.
"""

from collections import deque
import random
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from rl_ack.dqn.actions import ACTION_DIM
from rl_ack.dqn.state import STATE_DIM, validate_and_normalize_state


class ReplayBuffer:
    """Bounded experience replay memory with uniform random sampling."""

    def __init__(
        self,
        capacity: int = 10000,
        random_seed: Optional[int] = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)
        self.device = device or torch.device("cpu")

        self._buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def set_seed(self, seed: int) -> None:
        """Reset internal random generator with new seed for deterministic sampling."""
        self.random_seed = seed
        self._rng = random.Random(seed)

    def push(
        self,
        state: Union[np.ndarray, List[float]],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, List[float]],
        done: bool,
    ) -> None:
        """Insert a validated transition into the replay memory.

        Args:
            state: Pre-transition 8-feature state vector.
            action: Selected discrete action index in 0..4.
            reward: Finite scalar reward value.
            next_state: Post-transition 8-feature state vector.
            done: Terminal state indicator.

        Raises:
            ValueError: If inputs are invalid, non-finite, or out of range.
        """
        # Validate states
        s = validate_and_normalize_state(state)
        ns = validate_and_normalize_state(next_state)

        # Validate action
        act_idx = int(action)
        if not (0 <= act_idx < ACTION_DIM):
            raise ValueError(f"Action {act_idx} is out of bounds (must be 0..{ACTION_DIM - 1})")

        # Validate reward
        rew = float(reward)
        if not np.isfinite(rew):
            raise ValueError(f"Reward must be finite, got {rew}")

        is_done = bool(done)

        self._buffer.append((s, act_idx, rew, ns, is_done))

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a mini-batch of transitions uniformly at random.

        Args:
            batch_size: Number of transitions to draw.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as PyTorch tensors.

        Raises:
            ValueError: If requested batch_size exceeds current buffer length.
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with {len(self._buffer)} items"
            )

        indices = self._rng.sample(range(len(self._buffer)), batch_size)
        batch = [self._buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        s_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        a_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        d_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        return (s_tensor, a_tensor, r_tensor, ns_tensor, d_tensor)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has accumulated sufficient transitions to sample."""
        return len(self._buffer) >= batch_size

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        """Empty buffer contents."""
        self._buffer.clear()
