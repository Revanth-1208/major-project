"""Controlled Simulated Network Model for Phase 5 Comparative Evaluation.

Simulates deterministic and reproducible transport conditions including:
- Base propagation latency and stochastic jitter
- Packet loss in transit (forward channel)
- Acknowledgment loss (reverse channel)
- Traffic-class congestion delays
- Local edge proxy responses
"""

from dataclasses import dataclass
import random
from typing import Optional, Tuple

from rl_ack.common.enums import TrafficClass


@dataclass
class NetworkModelConfig:
    """Parameters defining controlled network environment conditions."""

    latency_base_ms: float = 15.0
    jitter_ms: float = 5.0
    packet_loss_rate: float = 0.10
    ack_loss_rate: float = 0.05
    congestion_factor: float = 0.0
    local_ack_latency_ms: float = 2.0
    random_seed: int = 42


class SimulatedNetworkModel:
    """Deterministic channel simulator modeling IoT kitchen mesh network behavior."""

    def __init__(self, config: Optional[NetworkModelConfig] = None) -> None:
        self.config = config or NetworkModelConfig()
        self.random_seed = self.config.random_seed
        self._rng = random.Random(self.random_seed)

    def set_seed(self, seed: int) -> None:
        """Reset pseudo-random generator with fixed seed for reproducible workloads."""
        self.random_seed = seed
        self._rng = random.Random(self.random_seed)

    def reset(self) -> None:
        """Reset internal RNG to the configured starting seed."""
        self._rng = random.Random(self.random_seed)

    def transmit_packet(
        self,
        traffic_class: TrafficClass = TrafficClass.BEST_EFFORT,
        attempt: int = 1,
    ) -> Tuple[bool, float]:
        """Simulate forward-channel packet transmission to downstream subscriber.

        Returns:
            Tuple of (is_received, one_way_latency_ms).
        """
        # Extra congestion penalty for high bandwidth traffic
        congestion_penalty = 0.0
        if traffic_class == TrafficClass.HIGH_BANDWIDTH_STREAMING:
            congestion_penalty = self.config.congestion_factor * 10.0

        # Draw forward loss
        loss_prob = self.config.packet_loss_rate
        is_lost = self._rng.random() < loss_prob

        # Calculate sample latency with gaussian jitter
        jitter = self._rng.gauss(0.0, max(0.1, self.config.jitter_ms))
        latency = max(
            1.0,
            self.config.latency_base_ms + jitter + congestion_penalty,
        )

        return (not is_lost, round(latency, 2))

    def transmit_ack(self, attempt: int = 1) -> Tuple[bool, float]:
        """Simulate reverse-channel acknowledgment return to publisher.

        Returns:
            Tuple of (is_ack_received, return_latency_ms).
        """
        loss_prob = self.config.ack_loss_rate
        is_lost = self._rng.random() < loss_prob

        jitter = self._rng.gauss(0.0, max(0.1, self.config.jitter_ms * 0.5))
        latency = max(1.0, (self.config.latency_base_ms * 0.6) + jitter)

        return (not is_lost, round(latency, 2))

    def query_local_proxy(self) -> Tuple[bool, float]:
        """Simulate immediate local edge proxy acknowledgment.

        Local proxy is on the same local broadcast domain (e.g. kitchen router),
        with minimal delay and negligible loss probability (< 0.1%).

        Returns:
            Tuple of (is_local_acked, local_ack_latency_ms).
        """
        is_lost = self._rng.random() < 0.001
        jitter = self._rng.gauss(0.0, 0.3)
        latency = max(0.5, self.config.local_ack_latency_ms + jitter)

        return (not is_lost, round(latency, 2))
