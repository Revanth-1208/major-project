"""Module 8: DQN Controller Integration.

Responsible for:
- Live network state feature extraction (8 features)
- Strictly deterministic greedy Q-value inference from PyTorch checkpoints
- Safe, transparent fallback behavior when checkpoints are unready or missing
- Non-suppression guarantee (DQN optimizes transmission/ACK, never suppresses critical alerts)
- Safety override policy ensuring Critical Control alerts maintain reliable QoS
- Online decision dispatch to transmission pipeline.

Implementation completed in Phase 7.
"""

from rl_ack.controller.controller import DqnAckController
from rl_ack.controller.decision import DqnDecisionRecord

MODULE_NAME = "controller"
MODULE_ID = 8
IS_IMPLEMENTED = True

__all__ = [
    "MODULE_NAME",
    "MODULE_ID",
    "IS_IMPLEMENTED",
    "DqnAckController",
    "DqnDecisionRecord",
]
