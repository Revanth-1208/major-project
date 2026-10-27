"""Deep Q-Network Adaptive ACK Controller for Smart Kitchen IoT Networks.

Module 8: DQN Controller Integration
Coordinates:
- Real-time 8-feature state extraction from live sensor telemetry and network conditions.
- Strictly deterministic greedy Q-value inference from trained PyTorch checkpoints.
- Safe, transparent fallback behavior when checkpoints are unloaded or incompatible.
- Non-suppression guarantee: alerts are generated independently; the controller optimizes
  acknowledgment and transmission QoS without suppressing danger alarms.
- Safety override policy ensuring Critical Control alerts maintain delivery reliability.
"""

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from rl_ack.common.enums import AckAction, AlertSeverity, EvaluationPolicy, TrafficClass
from rl_ack.common.models import AlertMessage
from rl_ack.controller.decision import DqnDecisionRecord
from rl_ack.dqn.actions import (
    ACTION_DIM,
    action_to_policy_enum,
    get_action_metadata,
    policy_enum_to_action,
)
from rl_ack.dqn.agent import DQNAgent
from rl_ack.dqn.checkpoint import CheckpointError, load_checkpoint
from rl_ack.dqn.state import (
    NetworkStateTracker,
    STATE_DIM,
    create_state_vector,
    validate_and_normalize_state,
)


class DqnAckController:
    """Online DQN controller for adaptive ACK policy selection on emergency IoT alerts."""

    def __init__(
        self,
        agent: Optional[DQNAgent] = None,
        state_tracker: Optional[NetworkStateTracker] = None,
        fallback_policy: EvaluationPolicy = EvaluationPolicy.STATIC_MQTT_QOS_1,
        enable_safety_override: bool = True,
        max_history: int = 200,
    ) -> None:
        self.agent = agent or DQNAgent()
        self.state_tracker = state_tracker or NetworkStateTracker()
        self.fallback_policy = fallback_policy
        self.enable_safety_override = enable_safety_override
        self.max_history = max_history

        self.model_ready: bool = False
        self.current_model_path: Optional[Path] = None
        self.checkpoint_metadata: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None

        self._decision_counter: int = 0
        self._decision_history: deque[DqnDecisionRecord] = deque(maxlen=self.max_history)

    def load_checkpoint(self, filepath: Union[str, Path]) -> bool:
        """Load and validate a trained DQN model checkpoint.

        Args:
            filepath: Path to .pt checkpoint file.

        Returns:
            True if checkpoint loaded and validated successfully, False otherwise.
        """
        path = Path(filepath)
        if not path.is_file():
            self.model_ready = False
            self.last_error = f"Checkpoint file not found: {path}"
            return False

        try:
            meta = load_checkpoint(
                filepath=path,
                online_network=self.agent.online_net,
                target_network=self.agent.target_net,
                optimizer=self.agent.optimizer,
                device=self.agent.device,
            )
            self.agent.epsilon = float(meta.get("epsilon", 0.05))
            self.agent.total_steps = int(meta.get("total_steps", 0))
            self.agent.episodes_trained = int(meta.get("episodes_trained", 0))

            self.model_ready = True
            self.current_model_path = path
            self.checkpoint_metadata = meta
            self.last_error = None
            return True
        except CheckpointError as e:
            self.model_ready = False
            self.last_error = str(e)
            return False
        except Exception as e:
            self.model_ready = False
            self.last_error = f"Unexpected error loading checkpoint {path.name}: {e}"
            return False

    def unload_checkpoint(self) -> None:
        """Unload active model to test deterministic fallback behavior."""
        self.model_ready = False
        self.current_model_path = None
        self.checkpoint_metadata = None
        self.last_error = "Model explicitly unloaded by operator."

    def extract_state_vector(
        self,
        alert: AlertMessage,
        network_metrics: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Construct the 8-feature normalized state vector from alert and network metrics.

        Features:
        1. network_utilization [0.0, 1.0]
        2. rolling_ack_rate [0.0, 1.0]
        3. queue_occupancy [0.0, 1.0]
        4. is_CC {0.0, 1.0}
        5. is_HBS {0.0, 1.0}
        6. is_BET {0.0, 1.0}
        7. remaining_energy [0.0, 1.0]
        8. rolling_latency [0.0, 1.0]
        """
        metrics = network_metrics or {}

        net_util = float(metrics.get("network_utilization", self.state_tracker._channel_load_factor))
        rolling_ack = float(metrics.get("rolling_ack_rate", self.state_tracker.rolling_ack_rate))
        queue_occ = float(metrics.get("queue_occupancy", self.state_tracker.queue_occupancy))
        energy = float(metrics.get("remaining_energy", self.state_tracker.remaining_energy))
        lat_ms = float(metrics.get("rolling_latency_ms", self.state_tracker.rolling_latency_ms))

        state_vec = create_state_vector(
            network_utilization=net_util,
            rolling_ack_rate=rolling_ack,
            queue_occupancy=queue_occ,
            traffic_class=alert.priority,
            remaining_energy=energy,
            rolling_latency_ms=lat_ms,
        )

        return validate_and_normalize_state(state_vec)

    def select_action(
        self,
        alert: AlertMessage,
        network_metrics: Optional[Dict[str, float]] = None,
        simulation_time_s: float = 0.0,
    ) -> DqnDecisionRecord:
        """Select optimal ACK strategy for an outgoing alert.

        Non-Suppression Guarantee:
            Alert generation and severity are decided strictly by the IncidentDetector.
            This controller decides transmission ACK semantics (QoS, retries, headers);
            it NEVER drops, suppresses, or modifies alert severity.

        Args:
            alert: Outgoing AlertMessage to transmit.
            network_metrics: Current dynamic channel telemetry.
            simulation_time_s: Current simulation timestamp.

        Returns:
            DqnDecisionRecord detailing Q-values, chosen policy, and fallback/override flags.
        """
        # 1. Feature Extraction
        state_arr = self.extract_state_vector(alert, network_metrics)
        state_list = [float(x) for x in state_arr]

        # 2. Inference vs Fallback
        if self.model_ready:
            # Deterministic inference (strictly greedy, explore=False)
            q_vals = self.agent.get_q_values(state_arr)
            q_values_list: Optional[List[float]] = [float(q) for q in q_vals]
            action_idx = int(np.argmax(q_vals))
            is_fallback = False
            fallback_reason = None
            model_id = self.current_model_path.name if self.current_model_path else "dqn_active"

            # 3. Safety Override Check for Critical Emergency Alerts
            # If an emergency alert has CRITICAL severity (e.g. gas leak / dense smoke)
            # and DQN chose QoS 0 (fire-and-forget), safety override upgrades to QoS 1
            # with bounded retries to safeguard emergency delivery.
            safety_override = False
            if self.enable_safety_override and alert.severity == AlertSeverity.CRITICAL:
                if action_idx == int(AckAction.MQTT_QOS_0):
                    action_idx = int(AckAction.MQTT_QOS_1)
                    safety_override = True

        else:
            # Deterministic fallback mode when model is unready
            action_idx = int(policy_enum_to_action(self.fallback_policy))
            q_values_list = None
            is_fallback = True
            fallback_reason = (
                self.last_error
                or f"No trained DQN model loaded. Operating on deterministic fallback ({self.fallback_policy.value})."
            )
            safety_override = False
            model_id = "UNLOADED (Fallback Mode)"

        # 4. Map Action to Policy Metadata
        meta = get_action_metadata(action_idx)

        # 5. Record Decision
        self._decision_counter += 1
        decision = DqnDecisionRecord(
            decision_id=f"DEC-{self._decision_counter:04d}",
            alert_id=alert.alert_id,
            sensor_id=alert.sensor_id,
            traffic_class=alert.priority.value,
            severity=alert.severity.value,
            reading_value=alert.reading_value,
            threshold_value=alert.threshold_value,
            unit=alert.unit,
            state_vector=state_list,
            q_values=q_values_list,
            action_idx=action_idx,
            action_label=meta.label,
            policy_name=meta.policy_enum.value,
            configured_qos=meta.qos_level,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
            safety_override=safety_override,
            model_id=model_id,
            timestamp_s=simulation_time_s or alert.simulation_time_s,
        )

        self._decision_history.append(decision)
        return decision

    def record_delivery_outcome(
        self,
        confirmed: bool,
        latency_ms: Optional[float],
        overhead_bytes: int = 32,
    ) -> None:
        """Update internal rolling state tracker from observed delivery events."""
        self.state_tracker.record_outcome(
            confirmed=confirmed,
            latency_ms=latency_ms,
            overhead_bytes=overhead_bytes,
        )

    def get_recent_decisions(self, limit: int = 50) -> List[DqnDecisionRecord]:
        """Return chronological list of recent decisions up to limit."""
        return list(self._decision_history)[-limit:]

    def get_decisions_df(self, limit: int = 50) -> pd.DataFrame:
        """Convert recent decisions to pandas DataFrame for dashboard visualization."""
        decisions = self.get_recent_decisions(limit=limit)
        if not decisions:
            return pd.DataFrame()
        return pd.DataFrame([d.to_dict() for d in decisions])

    def get_latest_decision(self) -> Optional[DqnDecisionRecord]:
        """Return the most recent decision record, if any exists."""
        if not self._decision_history:
            return None
        return self._decision_history[-1]

    def reset(self) -> None:
        """Reset decision history and rolling state tracker."""
        self._decision_history.clear()
        self._decision_counter = 0
        self.state_tracker.reset()
