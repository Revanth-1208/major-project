"""Eight-Dimensional State Space Representation for the RL-ACK DQN Agent.

Defines the normalized 8-feature network state vector, rolling statistics tracking,
validation rules, and normalization transformations.

Features:
1. network_utilization: [0.0, 1.0] - Channel occupancy / utilization fraction.
2. rolling_ack_rate: [0.0, 1.0] - Ratio of successful ACKs over rolling window.
3. queue_occupancy: [0.0, 1.0] - Outgoing transmission queue saturation fraction.
4. is_CC: {0.0, 1.0} - Critical Control traffic indicator.
5. is_HBS: {0.0, 1.0} - High Bandwidth Streaming traffic indicator.
6. is_BET: {0.0, 1.0} - Best Effort Traffic indicator.
7. remaining_energy: [0.0, 1.0] - Normalized node battery / power remaining.
8. rolling_latency: [0.0, 1.0] - Normalized rolling delivery/ACK latency.

Traffic class indicators (is_CC, is_HBS, is_BET) are mutually exclusive.
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import numpy as np

from rl_ack.common.enums import TrafficClass
from rl_ack.common.models import NetworkStateVector


STATE_DIM: int = 8
MAX_LATENCY_NORMALIZATION_MS: float = 200.0
DEFAULT_BASELINE_LATENCY_MS: float = 15.0
DEFAULT_WINDOW_SIZE: int = 20


class InvalidStateError(ValueError):
    """Raised when an 8-dimensional state vector contains invalid or non-finite values."""
    pass


def validate_and_normalize_state(
    raw_state: Union[List[float], np.ndarray, NetworkStateVector],
) -> np.ndarray:
    """Validate and enforce normalization bounds on an 8-dimensional state vector.

    Args:
        raw_state: State vector as list, numpy array, or NetworkStateVector.

    Returns:
        np.ndarray of shape (8,) with np.float32 bounded values.

    Raises:
        InvalidStateError: If dimensions differ from 8, or if non-finite values exist,
                           or if traffic class indicators are not mutually exclusive.
    """
    if isinstance(raw_state, NetworkStateVector):
        vec = raw_state.to_numpy()
    elif isinstance(raw_state, (list, tuple)):
        vec = np.array(raw_state, dtype=np.float32)
    elif isinstance(raw_state, np.ndarray):
        vec = raw_state.astype(np.float32)
    else:
        raise InvalidStateError(f"Unsupported state vector type: {type(raw_state).__name__}")

    if vec.shape != (STATE_DIM,):
        raise InvalidStateError(f"Expected state vector shape ({STATE_DIM},), got {vec.shape}")

    # Check for NaN or Inf
    if not np.all(np.isfinite(vec)):
        raise InvalidStateError(f"State vector contains non-finite values (NaN/Inf): {vec}")

    net_util, ack_rate, queue_occ, is_cc, is_hbs, is_bet, energy, latency = vec

    # Validate traffic class mutual exclusivity (must sum to ~1.0, each must be 0.0 or 1.0)
    tc_sum = is_cc + is_hbs + is_bet
    if not math.isclose(tc_sum, 1.0, abs_tol=1e-3):
        raise InvalidStateError(
            f"Traffic class indicators must be mutually exclusive and sum to 1.0, got {is_cc}, {is_hbs}, {is_bet} (sum={tc_sum})"
        )

    # Normalize/clamp continuous features to [0.0, 1.0]
    normalized = np.array([
        np.clip(net_util, 0.0, 1.0),
        np.clip(ack_rate, 0.0, 1.0),
        np.clip(queue_occ, 0.0, 1.0),
        1.0 if is_cc >= 0.5 else 0.0,
        1.0 if is_hbs >= 0.5 else 0.0,
        1.0 if is_bet >= 0.5 else 0.0,
        np.clip(energy, 0.0, 1.0),
        np.clip(latency, 0.0, 1.0),
    ], dtype=np.float32)

    return normalized


def create_state_vector(
    network_utilization: float,
    rolling_ack_rate: float,
    queue_occupancy: float,
    traffic_class: TrafficClass,
    remaining_energy: float,
    rolling_latency_ms: float,
) -> NetworkStateVector:
    """Create a validated NetworkStateVector from domain parameters.

    Args:
        network_utilization: Channel utilization fraction [0.0, 1.0].
        rolling_ack_rate: Rolling ACK success rate [0.0, 1.0].
        queue_occupancy: Transmit queue depth fraction [0.0, 1.0].
        traffic_class: Domain TrafficClass (Critical Control, Streaming, Best Effort).
        remaining_energy: Remaining node power budget [0.0, 1.0].
        rolling_latency_ms: Rolling latency in milliseconds (normalized by MAX_LATENCY_NORMALIZATION_MS).

    Returns:
        Validated NetworkStateVector instance.
    """
    norm_latency = min(1.0, max(0.0, rolling_latency_ms / MAX_LATENCY_NORMALIZATION_MS))
    return NetworkStateVector.from_features(
        network_utilization=network_utilization,
        rolling_ack_rate=rolling_ack_rate,
        transmit_queue_occupancy=queue_occupancy,
        traffic_class=traffic_class,
        remaining_node_energy=remaining_energy,
        rolling_latency=norm_latency,
    )


class NetworkStateTracker:
    """Maintains empirical rolling statistics and dynamic telemetry to construct states.

    Tracks:
    - Recent delivery confirmations over a sliding window (rolling_ack_rate).
    - Recent acknowledgment and delivery latencies over a sliding window (rolling_latency).
    - Current node transmission queue depth (queue_occupancy).
    - Node energy depletion model (remaining_energy).
    - Channel utilization estimator (network_utilization).
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_queue_capacity: int = 50,
        initial_energy: float = 1.0,
    ) -> None:
        self.window_size = max(5, window_size)
        self.max_queue_capacity = max(1, max_queue_capacity)
        self.initial_energy = float(initial_energy)

        self._ack_history: Deque[bool] = deque(maxlen=self.window_size)
        self._latency_history: Deque[float] = deque(maxlen=self.window_size)
        self._current_queue_count: int = 0
        self._remaining_energy: float = self.initial_energy
        self._channel_load_factor: float = 0.10

    def reset(self, initial_energy: Optional[float] = None) -> None:
        """Reset internal history and reinitialize rolling buffers."""
        self._ack_history.clear()
        self._latency_history.clear()
        self._current_queue_count = 0
        self._remaining_energy = self.initial_energy if initial_energy is None else float(initial_energy)
        self._channel_load_factor = 0.10

    def record_outcome(
        self,
        confirmed: bool,
        latency_ms: Optional[float],
        overhead_bytes: int = 32,
    ) -> None:
        """Record transmission outcome into rolling statistic buffers.

        Args:
            confirmed: Whether the message achieved verified end-to-end delivery.
            latency_ms: Observed latency in milliseconds (None if unconfirmed/lost).
            overhead_bytes: Byte overhead expended during transmission.
        """
        self._ack_history.append(bool(confirmed))

        if latency_ms is not None and latency_ms > 0:
            self._latency_history.append(float(latency_ms))

        # Energy consumption model: ~0.0001 per 100 bytes transmitted
        energy_cost = (overhead_bytes / 100.0) * 0.0002
        self._remaining_energy = max(0.0, self._remaining_energy - energy_cost)

        # Queue depth update: dequeue successful or dropped packet
        self._current_queue_count = max(0, self._current_queue_count - 1)

    def enqueue_message(self, traffic_class: TrafficClass) -> None:
        """Increment queue depth for an incoming transmission request."""
        self._current_queue_count = min(self.max_queue_capacity, self._current_queue_count + 1)
        if traffic_class == TrafficClass.HIGH_BANDWIDTH_STREAMING:
            self._channel_load_factor = min(1.0, self._channel_load_factor + 0.05)
        else:
            self._channel_load_factor = max(0.05, self._channel_load_factor - 0.01)

    def set_network_utilization(self, utilization: float) -> None:
        """Directly update the channel utilization estimator."""
        self._channel_load_factor = max(0.0, min(1.0, float(utilization)))

    @property
    def rolling_ack_rate(self) -> float:
        """Calculate recent ACK success rate.

        Defaults to 1.0 (optimistic cold start) when history is empty.
        """
        if not self._ack_history:
            return 1.0
        return sum(1.0 for s in self._ack_history if s) / len(self._ack_history)

    @property
    def rolling_latency_ms(self) -> float:
        """Calculate recent mean delivery latency.

        Defaults to baseline 15.0 ms when history is empty.
        """
        if not self._latency_history:
            return DEFAULT_BASELINE_LATENCY_MS
        return sum(self._latency_history) / len(self._latency_history)

    @property
    def queue_occupancy(self) -> float:
        """Calculate current transmit queue occupancy fraction in [0.0, 1.0]."""
        return min(1.0, self._current_queue_count / self.max_queue_capacity)

    @property
    def remaining_energy(self) -> float:
        """Return remaining normalized energy budget in [0.0, 1.0]."""
        return max(0.0, min(1.0, self._remaining_energy))

    def build_state(self, traffic_class: TrafficClass) -> NetworkStateVector:
        """Construct a validated 8-dimensional state vector for an incoming message."""
        return create_state_vector(
            network_utilization=self._channel_load_factor,
            rolling_ack_rate=self.rolling_ack_rate,
            queue_occupancy=self.queue_occupancy,
            traffic_class=traffic_class,
            remaining_energy=self.remaining_energy,
            rolling_latency_ms=self.rolling_latency_ms,
        )

    def build_state_array(self, traffic_class: TrafficClass) -> np.ndarray:
        """Construct an 8-dimensional float32 numpy array ready for neural network input."""
        state_vec = self.build_state(traffic_class)
        return validate_and_normalize_state(state_vec)
