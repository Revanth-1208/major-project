"""Deep Q-Network (DQN) Neural Network Architecture.

Implements the 8 -> 64 -> 32 -> 5 fully-connected feedforward Q-network in PyTorch.
Ensures CPU execution by default and CUDA only when explicitly available.
"""

from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn

from rl_ack.dqn.actions import ACTION_DIM
from rl_ack.dqn.state import STATE_DIM


class QNetwork(nn.Module):
    """Feedforward Deep Q-Network for approximating Q-values in smart kitchen IoT.

    Architecture:
        Input: 8 normalized network state features
        Hidden 1: Linear(8, 64) + ReLU
        Hidden 2: Linear(64, 32) + ReLU
        Output: Linear(32, 5) -> 5 Q-values
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        action_dim: int = ACTION_DIM,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.action_dim = action_dim

        # Auto device detection: CPU by default, CUDA only if available
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim_1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_2, self.action_dim),
        )

        self._init_weights()
        self.to(self.device)

    def _init_weights(self) -> None:
        """Initialize layer weights using Kaiming normal for ReLU and small Xavier for output."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Scale final output layer lightly to prevent initial Q-value divergence
        last_linear = self.network[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.xavier_uniform_(last_linear.weight, gain=0.5)

    def forward(self, state: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Compute 5 Q-values for the input state vector(s).

        Args:
            state: Tensor or ndarray of shape (8,) or (N, 8).

        Returns:
            torch.Tensor of shape (5,) or (N, 5) with finite Q-values.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(device=self.device, dtype=torch.float32)

        # Handle single state vector shape (8,) -> (1, 8)
        single_input = state.dim() == 1
        if single_input:
            state = state.unsqueeze(0)

        if state.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected state input feature dimension {self.state_dim}, got {state.shape[1]}"
            )

        q_values = self.network(state)

        # Squeeze back if single input
        if single_input:
            q_values = q_values.squeeze(0)

        return q_values
