"""Reward Formulation and Multi-Objective Evaluation for the RL-ACK DQN Agent.

Calculates multi-objective rewards balancing:
1. End-to-end verified delivery confirmation (with enhanced weights for Critical Control).
2. Latency compliance against traffic-class deadlines.
3. Delivery failure penalties (harsh penalty for lost emergency alerts).
4. Communication byte overhead penalties (curbing unnecessary packet bloat).
5. Retransmission churn penalties (penalizing repetitive channel contention).
6. Node energy consumption costs.
7. Local ACK distinction (proxy ACK is never credited as end-to-end delivery).

Invariants:
- Broker PUBACK is distinct from subscriber receipt.
- Local proxy ACK does not satisfy E2E delivery.
- No outcome is double-counted.
- All terms and weights are strictly documented and configurable.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from rl_ack.common.enums import AlertSeverity, TrafficClass
from rl_ack.common.models import NetworkStateVector
from rl_ack.policies.base import PolicyEvaluationResult


@dataclass
class RewardConfig:
    """Configurable weights for the multi-objective RL-ACK reward function."""

    # Delivery rewards
    w_delivery_success: float = 10.0
    w_cc_delivery_bonus: float = 15.0
    w_broker_ingress_signal: float = 1.0
    w_local_proxy_signal: float = 0.5

    # Failure penalties
    w_delivery_failure: float = 15.0
    w_cc_failure_penalty: float = 25.0

    # Deadline & latency terms
    w_deadline_met: float = 5.0
    w_deadline_violation: float = 10.0
    w_cc_deadline_violation: float = 15.0
    w_latency_scale: float = 3.0

    # Resource & channel overhead penalties
    w_overhead_per_100b: float = 1.0
    w_retry_penalty: float = 2.0
    w_energy_penalty: float = 1.0
    w_congestion_penalty: float = 1.5

    # Target deadline reference in milliseconds
    cc_deadline_ms: float = 60.0
    hbs_deadline_ms: float = 120.0
    bet_deadline_ms: float = 250.0

    def to_dict(self) -> Dict[str, float]:
        """Export reward configuration as dictionary."""
        return asdict(self)


class RewardCalculator:
    """Computes bounded, documented scalar rewards from policy evaluation results."""

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()

    def get_deadline_for_traffic(self, traffic_class: TrafficClass) -> float:
        """Determine latency deadline requirement by traffic class."""
        if traffic_class == TrafficClass.CRITICAL_CONTROL:
            return self.config.cc_deadline_ms
        elif traffic_class == TrafficClass.HIGH_BANDWIDTH_STREAMING:
            return self.config.hbs_deadline_ms
        else:
            return self.config.bet_deadline_ms

    def compute_reward(
        self,
        result: PolicyEvaluationResult,
        state: Optional[NetworkStateVector] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute multi-objective reward and explainable component breakdown.

        Args:
            result: Detailed evaluation record of the action outcome.
            state: Pre-transition network state vector (for congestion & energy info).

        Returns:
            Tuple of (scalar_reward, terms_breakdown_dict).
        """
        is_cc = result.traffic_class == TrafficClass.CRITICAL_CONTROL
        deadline_ms = self.get_deadline_for_traffic(result.traffic_class)

        terms: Dict[str, float] = {}

        # ----------------------------------------------------------------------
        # 1. Delivery Outcome Term
        # ----------------------------------------------------------------------
        if result.delivery_confirmed:
            # End-to-end subscriber delivery confirmed
            r_del = self.config.w_delivery_success
            if is_cc:
                r_del += self.config.w_cc_delivery_bonus
            terms["delivery_reward"] = r_del
        else:
            # Delivery failed or dropped
            p_fail = -self.config.w_delivery_failure
            if is_cc:
                p_fail -= self.config.w_cc_failure_penalty
            terms["delivery_penalty"] = p_fail

            # Partial progress credit (broker ingress or local proxy)
            # NEVER treated as delivery, but provides mild gradient signal
            if result.broker_acknowledged:
                terms["broker_ingress_signal"] = self.config.w_broker_ingress_signal
            elif result.local_acknowledged:
                terms["local_proxy_signal"] = self.config.w_local_proxy_signal

        # ----------------------------------------------------------------------
        # 2. Deadline and Latency Term
        # ----------------------------------------------------------------------
        if result.delivery_confirmed and result.end_to_end_latency_ms is not None:
            e2e_ms = result.end_to_end_latency_ms

            # Deadline compliance
            if e2e_ms <= deadline_ms:
                terms["deadline_bonus"] = self.config.w_deadline_met
            else:
                p_ddl = -self.config.w_deadline_violation
                if is_cc:
                    p_ddl -= self.config.w_cc_deadline_violation
                terms["deadline_violation"] = p_ddl

            # Normalized latency penalty
            norm_lat = min(2.0, e2e_ms / max(1.0, deadline_ms))
            terms["latency_penalty"] = -round(self.config.w_latency_scale * norm_lat, 3)
        else:
            # Undelivered messages violate deadline by default
            p_ddl = -self.config.w_deadline_violation
            if is_cc:
                p_ddl -= self.config.w_cc_deadline_violation
            terms["deadline_violation"] = p_ddl
            terms["latency_penalty"] = -round(self.config.w_latency_scale * 1.5, 3)

        # ----------------------------------------------------------------------
        # 3. Communication Overhead Term
        # ----------------------------------------------------------------------
        overhead_units = result.message_overhead_bytes / 100.0
        p_overhead = -round(self.config.w_overhead_per_100b * overhead_units, 3)
        terms["overhead_penalty"] = p_overhead

        # ----------------------------------------------------------------------
        # 4. Retransmission Churn Penalty
        # ----------------------------------------------------------------------
        if result.retry_count > 0:
            terms["retry_penalty"] = -round(self.config.w_retry_penalty * result.retry_count, 3)

        # ----------------------------------------------------------------------
        # 5. Energy and Congestion Penalties (from state if available)
        # ----------------------------------------------------------------------
        if state is not None:
            # Energy depletion penalty: harsher if battery is running low
            energy_factor = max(0.0, 1.0 - state.remaining_node_energy)
            terms["energy_penalty"] = -round(self.config.w_energy_penalty * energy_factor * overhead_units, 3)

            # Congestion penalty: penalize excessive transmissions when channel/queue saturated
            if state.transmit_queue_occupancy > 0.7 or state.network_utilization > 0.7:
                cong_severity = max(state.transmit_queue_occupancy, state.network_utilization)
                terms["congestion_penalty"] = -round(self.config.w_congestion_penalty * cong_severity, 3)

        total_reward = round(sum(terms.values()), 3)
        return total_reward, terms
