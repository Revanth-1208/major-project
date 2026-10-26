"""Training Loop and Experiment Orchestrator for the RL-ACK DQN Agent.

Coordinates multi-episode training, epsilon exploration decay, replay buffer accumulation,
loss optimization, live metric tracking, and safe checkpointing.
"""

from dataclasses import asdict, dataclass
import io
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from rl_ack.common.enums import AckAction, TrafficClass
from rl_ack.config import DQNConfig
from rl_ack.dqn.actions import ACTION_DIM
from rl_ack.dqn.agent import DQNAgent
from rl_ack.dqn.environment import EnvConfig, SmartKitchenIoTEnv


@dataclass
class EpisodeMetrics:
    """Detailed statistical performance metrics recorded per training episode."""

    episode: int
    total_reward: float
    avg_loss: Optional[float]
    epsilon: float
    delivery_rate: float
    cc_delivery_rate: float
    avg_latency_ms: Optional[float]
    deadline_compliance_rate: float
    total_overhead_bytes: int
    action_0_qos0: int
    action_1_local_ack: int
    action_2_qos1: int
    action_3_selective: int
    action_4_qos2: int
    total_steps: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


class DQNTrainer:
    """Manages the lifecycle of DQN training runs with metric logging and checkpointing."""

    def __init__(
        self,
        agent: Optional[DQNAgent] = None,
        env: Optional[SmartKitchenIoTEnv] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.agent = agent or DQNAgent()
        self.env = env or SmartKitchenIoTEnv()
        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[EpisodeMetrics] = []
        self.best_reward: float = -float("inf")

    def train(
        self,
        num_episodes: int = 30,
        progress_callback: Optional[Callable[[int, int, EpisodeMetrics], None]] = None,
        save_checkpoints: bool = True,
    ) -> List[EpisodeMetrics]:
        """Execute full training loop across multiple episodes.

        Args:
            num_episodes: Number of episodes to run.
            progress_callback: Optional callback(current_ep, total_eps, metrics) for UI updates.
            save_checkpoints: Whether to automatically save checkpoints during training.

        Returns:
            List of recorded EpisodeMetrics objects.
        """
        for ep in range(1, num_episodes + 1):
            ep_seed = (self.env.random_seed + ep * 17) if self.env.random_seed is not None else None
            state = self.env.reset(seed=ep_seed)

            ep_reward: float = 0.0
            losses: List[float] = []
            confirmed_count: int = 0
            cc_total: int = 0
            cc_confirmed: int = 0
            latencies: List[float] = []
            deadline_met_count: int = 0
            total_overhead: int = 0
            action_counts: Dict[int, int] = {i: 0 for i in range(ACTION_DIM)}

            done = False
            while not done:
                # Epsilon-greedy action selection
                action = self.agent.select_action(state, explore=True)
                act_idx = int(action)
                action_counts[act_idx] += 1

                # Step environment
                next_state, reward, done, info = self.env.step(action)
                ep_reward += reward

                # Record transition in replay memory
                self.agent.store_transition(state, action, reward, next_state, done)

                # Perform gradient optimization step
                loss = self.agent.train_step()
                if loss is not None:
                    losses.append(loss)

                # Tally episode diagnostics
                if info.get("delivery_confirmed"):
                    confirmed_count += 1
                if info.get("traffic_class") == TrafficClass.CRITICAL_CONTROL.value:
                    cc_total += 1
                    if info.get("delivery_confirmed"):
                        cc_confirmed += 1
                lat = info.get("latency_ms")
                if lat is not None and info.get("delivery_confirmed"):
                    latencies.append(lat)
                if info.get("deadline_met"):
                    deadline_met_count += 1
                total_overhead += int(info.get("overhead_bytes", 0))

                state = next_state

            # Tally metrics for this episode
            total_steps_in_ep = self.env.config.steps_per_episode
            del_rate = confirmed_count / max(1, total_steps_in_ep)
            cc_del_rate = (cc_confirmed / cc_total) if cc_total > 0 else 1.0
            avg_lat = (sum(latencies) / len(latencies)) if latencies else None
            ddl_rate = deadline_met_count / max(1, total_steps_in_ep)
            avg_loss = (sum(losses) / len(losses)) if losses else None

            self.agent.episodes_trained += 1

            metrics = EpisodeMetrics(
                episode=self.agent.episodes_trained,
                total_reward=round(ep_reward, 2),
                avg_loss=round(avg_loss, 4) if avg_loss is not None else None,
                epsilon=round(self.agent.epsilon, 4),
                delivery_rate=round(del_rate, 4),
                cc_delivery_rate=round(cc_del_rate, 4),
                avg_latency_ms=round(avg_lat, 2) if avg_lat is not None else None,
                deadline_compliance_rate=round(ddl_rate, 4),
                total_overhead_bytes=total_overhead,
                action_0_qos0=action_counts[0],
                action_1_local_ack=action_counts[1],
                action_2_qos1=action_counts[2],
                action_3_selective=action_counts[3],
                action_4_qos2=action_counts[4],
                total_steps=self.agent.total_steps,
            )
            self.history.append(metrics)

            # Checkpoint persistence
            if save_checkpoints:
                latest_ckpt = self.checkpoint_dir / "dqn_latest.pt"
                self.agent.save_checkpoint(latest_ckpt, reward_weights=self.env.reward_config.to_dict())

                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward
                    best_ckpt = self.checkpoint_dir / "dqn_best.pt"
                    self.agent.save_checkpoint(best_ckpt, reward_weights=self.env.reward_config.to_dict())

            if progress_callback:
                progress_callback(ep, num_episodes, metrics)

        return self.history

    def get_history_df(self) -> pd.DataFrame:
        """Convert training history to a pandas DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in self.history])

    def export_metrics_csv(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Export training history metrics to a CSV file."""
        dest = Path(filepath or (self.checkpoint_dir / "dqn_training_metrics.csv"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = self.get_history_df()
        df.to_csv(str(dest), index=False)
        return dest
