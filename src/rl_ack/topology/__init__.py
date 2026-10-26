"""Module 3: System Overview and IoT Topology.

Defines the heterogeneous smart kitchen network topology, node definitions,
data flow representations, and component readiness mapping (Implemented vs Planned).
"""

from rl_ack.topology.models import (
    ComponentPhaseStatus,
    IoTNode,
    KitchenIoTTopology,
    NodeType,
    TopologyEdge,
    create_smart_kitchen_topology,
)

MODULE_NAME = "topology"
MODULE_ID = 3
IS_IMPLEMENTED = True

__all__ = [
    "MODULE_NAME",
    "MODULE_ID",
    "IS_IMPLEMENTED",
    "NodeType",
    "ComponentPhaseStatus",
    "IoTNode",
    "TopologyEdge",
    "KitchenIoTTopology",
    "create_smart_kitchen_topology",
]
