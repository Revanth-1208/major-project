"""Model Checkpoint Persistence and Validation for the RL-ACK DQN Agent.

Handles atomic saving, schema validation, architecture verification, and loading of
DQN checkpoints to ensure robust model persistence without file corruption.
"""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Union
import torch

from rl_ack.config import DQNConfig
from rl_ack.dqn.actions import ACTION_DIM
from rl_ack.dqn.state import STATE_DIM


CHECKPOINT_FORMAT_VERSION: str = "1.0"


class CheckpointError(Exception):
    """Raised when a checkpoint file is missing, corrupted, or incompatible."""
    pass


@dataclass
class CheckpointMetadata:
    """Metadata describing the training run and architecture of a saved checkpoint."""

    format_version: str
    timestamp_utc: str
    state_dim: int
    action_dim: int
    hidden_dim_1: int
    hidden_dim_2: int
    episodes_trained: int
    total_steps: int
    epsilon: float
    hyperparameters: Dict[str, Any]
    reward_weights: Dict[str, float]


def save_checkpoint(
    filepath: Union[str, Path],
    online_network: torch.nn.Module,
    target_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epsilon: float,
    episodes_trained: int,
    total_steps: int,
    config: Optional[DQNConfig] = None,
    reward_weights: Optional[Dict[str, float]] = None,
) -> Path:
    """Atomically save model weights, optimizer state, and training metadata to disk.

    Args:
        filepath: Target destination .pt or .pth file path.
        online_network: Online Q-network module.
        target_network: Target Q-network module.
        optimizer: PyTorch optimizer instance.
        epsilon: Current exploration rate.
        episodes_trained: Number of completed training episodes.
        total_steps: Number of total environment/training steps.
        config: DQN configuration object.
        reward_weights: Dictionary of reward formulation weights.

    Returns:
        Path to the saved checkpoint file.
    """
    from datetime import datetime, timezone

    dest_path = Path(filepath)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = config or DQNConfig()
    metadata = CheckpointMetadata(
        format_version=CHECKPOINT_FORMAT_VERSION,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        state_dim=getattr(online_network, "state_dim", STATE_DIM),
        action_dim=getattr(online_network, "action_dim", ACTION_DIM),
        hidden_dim_1=getattr(online_network, "hidden_dim_1", 64),
        hidden_dim_2=getattr(online_network, "hidden_dim_2", 32),
        episodes_trained=episodes_trained,
        total_steps=total_steps,
        epsilon=float(epsilon),
        hyperparameters={
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "batch_size": cfg.batch_size,
            "target_update_interval": cfg.target_update_interval,
            "memory_capacity": cfg.memory_capacity,
        },
        reward_weights=reward_weights or {},
    )

    payload: Dict[str, Any] = {
        "metadata": asdict(metadata),
        "online_state_dict": online_network.state_dict(),
        "target_state_dict": target_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # Atomic write pattern: write to temporary file in same directory, then rename
    temp_fd, temp_path_str = tempfile.mkstemp(dir=str(dest_path.parent), suffix=".tmp")
    os.close(temp_fd)
    temp_path = Path(temp_path_str)

    try:
        torch.save(payload, str(temp_path))
        # Atomic rename on POSIX; on Windows replace if exists
        temp_path.replace(dest_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise CheckpointError(f"Failed to atomically write checkpoint to {dest_path}: {e}") from e

    return dest_path


def load_checkpoint(
    filepath: Union[str, Path],
    online_network: torch.nn.Module,
    target_network: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load and validate a saved DQN checkpoint.

    Args:
        filepath: Source checkpoint path.
        online_network: Target online network to populate with weights.
        target_network: Optional target network to populate with weights.
        optimizer: Optional optimizer to restore state.
        device: Device to map tensors to.

    Returns:
        Dictionary containing checkpoint metadata and restored training parameters.

    Raises:
        CheckpointError: If file is missing, invalid, or architecturally incompatible.
    """
    path = Path(filepath)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint file not found: {path}")

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    try:
        checkpoint = torch.load(str(path), map_location=dev)
    except Exception as e:
        raise CheckpointError(f"Corrupted or unreadable checkpoint file {path}: {e}") from e

    if not isinstance(checkpoint, dict) or "metadata" not in checkpoint:
        raise CheckpointError(f"Invalid checkpoint format in {path}: missing 'metadata' root dictionary")

    meta = checkpoint["metadata"]

    # Architecture validation: ensure state_dim and action_dim match
    exp_state_dim = getattr(online_network, "state_dim", STATE_DIM)
    exp_action_dim = getattr(online_network, "action_dim", ACTION_DIM)

    if meta.get("state_dim") != exp_state_dim:
        raise CheckpointError(
            f"Checkpoint architecture mismatch: model expects state_dim={exp_state_dim}, "
            f"but checkpoint has state_dim={meta.get('state_dim')}"
        )

    if meta.get("action_dim") != exp_action_dim:
        raise CheckpointError(
            f"Checkpoint architecture mismatch: model expects action_dim={exp_action_dim}, "
            f"but checkpoint has action_dim={meta.get('action_dim')}"
        )

    # Load weights
    try:
        online_network.load_state_dict(checkpoint["online_state_dict"])
        online_network.to(dev)

        if target_network is not None and "target_state_dict" in checkpoint:
            target_network.load_state_dict(checkpoint["target_state_dict"])
            target_network.to(dev)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except Exception as e:
        raise CheckpointError(f"Failed to restore network state from checkpoint: {e}") from e

    return meta
