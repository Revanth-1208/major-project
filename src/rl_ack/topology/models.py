"""Data models and topology representation for the Smart Kitchen IoT Network."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import plotly.graph_objects as go


class NodeType(str, Enum):
    """Categorical classification of nodes within the IoT network topology."""
    SENSOR_NODE = "sensor_node"
    EDGE_GATEWAY = "edge_gateway"
    MQTT_BROKER = "mqtt_broker"
    MONITORING_DASHBOARD = "monitoring_dashboard"
    DQN_CONTROLLER = "dqn_controller"
    CLOUD_BACKEND = "cloud_backend"


class ComponentPhaseStatus(str, Enum):
    """Implementation readiness stage of topology components."""
    IMPLEMENTED_PHASE_1_2 = "Implemented (Phase 1 & 2)"
    PLANNED_LATER_PHASE = "Planned (Subsequent Phases)"


@dataclass
class IoTNode:
    """Represents a physical or logical entity within the IoT kitchen network."""
    node_id: str
    name: str
    node_type: NodeType
    phase_status: ComponentPhaseStatus
    phase_label: str
    associated_sensors: List[str] = field(default_factory=list)
    network_address: str = "127.0.0.1"
    description: str = ""
    # 2D coordinates for visual topology mapping
    x: float = 0.0
    y: float = 0.0

    @property
    def is_implemented(self) -> bool:
        """True if component is operational in the current project phase."""
        return self.phase_status == ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "phase_status": self.phase_status.value,
            "phase_label": self.phase_label,
            "associated_sensors": self.associated_sensors,
            "network_address": self.network_address,
            "description": self.description,
            "x": self.x,
            "y": self.y,
            "is_implemented": self.is_implemented,
        }


@dataclass
class TopologyEdge:
    """Represents a data transmission link between two IoT topology nodes."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    label: str
    protocol: str
    phase_status: ComponentPhaseStatus

    @property
    def is_implemented(self) -> bool:
        return self.phase_status == ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "label": self.label,
            "protocol": self.protocol,
            "phase_status": self.phase_status.value,
            "is_implemented": self.is_implemented,
        }


class KitchenIoTTopology:
    """Manages the network graph, nodes, edges, and rendering for the Kitchen IoT Network."""

    def __init__(self) -> None:
        self._nodes: Dict[str, IoTNode] = {}
        self._edges: List[TopologyEdge] = []

    def add_node(self, node: IoTNode) -> None:
        """Add an IoT node to the network topology."""
        self._nodes[node.node_id] = node

    def add_edge(self, edge: TopologyEdge) -> None:
        """Add a communication edge between two nodes."""
        self._edges.append(edge)

    def get_node(self, node_id: str) -> IoTNode:
        return self._nodes[node_id]

    def get_nodes(self) -> List[IoTNode]:
        return list(self._nodes.values())

    def get_edges(self) -> List[TopologyEdge]:
        return list(self._edges)

    def get_implemented_nodes(self) -> List[IoTNode]:
        return [n for n in self._nodes.values() if n.is_implemented]

    def get_planned_nodes(self) -> List[IoTNode]:
        return [n for n in self._nodes.values() if not n.is_implemented]

    def to_plotly_figure(self) -> go.Figure:
        """Generate an interactive Plotly graph visualizing the IoT topology and component readiness."""
        fig = go.Figure()

        # 1. Draw connecting edge lines
        edge_x = []
        edge_y = []
        edge_colors = []

        for edge in self._edges:
            src = self._nodes[edge.source_node_id]
            tgt = self._nodes[edge.target_node_id]

            # Midpoint for arrow / directional hint
            mid_x = (src.x + tgt.x) / 2
            mid_y = (src.y + tgt.y) / 2

            line_color = "#10B981" if edge.is_implemented else "#94A3B8"
            line_dash = "solid" if edge.is_implemented else "dot"

            fig.add_trace(go.Scatter(
                x=[src.x, tgt.x],
                y=[src.y, tgt.y],
                mode="lines",
                line=dict(width=2.5, color=line_color, dash=line_dash),
                hoverinfo="text",
                text=f"{edge.label} ({edge.protocol}) - {edge.phase_status.value}",
                showlegend=False,
            ))

        # 2. Draw Implemented Nodes (Green / Active)
        impl_nodes = self.get_implemented_nodes()
        if impl_nodes:
            fig.add_trace(go.Scatter(
                x=[n.x for n in impl_nodes],
                y=[n.y for n in impl_nodes],
                mode="markers+text",
                name="Implemented (Phase 1 & 2)",
                marker=dict(
                    symbol="circle",
                    size=38,
                    color="#10B981",  # Emerald Green
                    line=dict(width=3, color="#065F46"),
                ),
                text=[n.name for n in impl_nodes],
                textposition="bottom center",
                textfont=dict(size=11, color="#1F2937", family="sans-serif"),
                hoverinfo="text",
                hovertext=[
                    f"<b>{n.name}</b> ({n.node_id})<br>"
                    f"Type: {n.node_type.value}<br>"
                    f"Status: {n.phase_label}<br>"
                    f"Address: {n.network_address}<br>"
                    f"Sensors: {', '.join(n.associated_sensors) if n.associated_sensors else 'None'}<br>"
                    f"Description: {n.description}"
                    for n in impl_nodes
                ],
            ))

        # 3. Draw Planned Nodes (Slate / Dashed Border)
        plan_nodes = self.get_planned_nodes()
        if plan_nodes:
            fig.add_trace(go.Scatter(
                x=[n.x for n in plan_nodes],
                y=[n.y for n in plan_nodes],
                mode="markers+text",
                name="Planned (Subsequent Phases)",
                marker=dict(
                    symbol="hexagon",
                    size=42,
                    color="#F1F5F9",  # Soft Slate
                    line=dict(width=3, color="#64748B"),
                ),
                text=[n.name for n in plan_nodes],
                textposition="bottom center",
                textfont=dict(size=11, color="#475569", family="sans-serif"),
                hoverinfo="text",
                hovertext=[
                    f"<b>{n.name}</b> ({n.node_id})<br>"
                    f"Type: {n.node_type.value}<br>"
                    f"Status: {n.phase_label}<br>"
                    f"Address: {n.network_address}<br>"
                    f"Role: {n.description}"
                    for n in plan_nodes
                ],
            ))

        fig.update_layout(
            title=dict(
                text="<b>Smart Kitchen IoT Network Topology & Data Flow</b><br>"
                     "<sup>Solid green = Implemented in Phase 1 & 2 | Dotted gray = Planned for subsequent phases</sup>",
                font=dict(size=15),
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#FAFAFA",
            margin=dict(l=40, r=40, t=80, b=40),
            height=460,
        )
        return fig


def create_smart_kitchen_topology() -> KitchenIoTTopology:
    """Factory creating the reference Smart Kitchen IoT Network topology."""
    topo = KitchenIoTTopology()

    # 1. Virtual Sensor Nodes (Implemented in Phase 1)
    topo.add_node(IoTNode(
        node_id="NODE-001",
        name="Gas Sensor Node\n(GAS-001)",
        node_type=NodeType.SENSOR_NODE,
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
        phase_label="Implemented in Phase 1",
        associated_sensors=["GAS-001"],
        network_address="192.168.1.101",
        description="Measures atmospheric LPG/natural gas concentration in ppm (1.0s interval).",
        x=1.0,
        y=4.0,
    ))

    topo.add_node(IoTNode(
        node_id="NODE-002",
        name="Temp Sensor Node\n(TEMP-001)",
        node_type=NodeType.SENSOR_NODE,
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
        phase_label="Implemented in Phase 1",
        associated_sensors=["TEMP-001"],
        network_address="192.168.1.102",
        description="Monitors ambient kitchen cooktop temperature in °C (1.0s interval).",
        x=1.0,
        y=2.8,
    ))

    topo.add_node(IoTNode(
        node_id="NODE-003",
        name="Humidity Node\n(HUM-001)",
        node_type=NodeType.SENSOR_NODE,
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
        phase_label="Implemented in Phase 1",
        associated_sensors=["HUM-001"],
        network_address="192.168.1.103",
        description="Measures relative humidity % RH (2.0s interval).",
        x=1.0,
        y=1.6,
    ))

    topo.add_node(IoTNode(
        node_id="NODE-004",
        name="Smoke Sensor Node\n(SMOKE-001)",
        node_type=NodeType.SENSOR_NODE,
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
        phase_label="Implemented in Phase 1",
        associated_sensors=["SMOKE-001"],
        network_address="192.168.1.104",
        description="Measures photoelectric smoke obscuration in % obs/m (1.0s interval).",
        x=1.0,
        y=0.4,
    ))

    # 2. Kitchen IoT Edge Gateway (Implemented Phase 2 Coordination / In-Memory SensorEngine)
    topo.add_node(IoTNode(
        node_id="EDGE-GW-01",
        name="Kitchen IoT Edge Gateway\n(SensorEngine & Scheduler)",
        node_type=NodeType.EDGE_GATEWAY,
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
        phase_label="Implemented in Phase 1 & 2",
        network_address="192.168.1.1",
        description="Coordinates discrete-event simulation clock, executes sampling schedules, classifies status.",
        x=3.0,
        y=2.2,
    ))

    # 3. Central Monitoring Dashboard (Implemented in Phase 2)
    topo.add_node(IoTNode(
        node_id="DASH-01",
        name="Monitoring UI\n(Streamlit Dashboard)",
        node_type=NodeType.MONITORING_DASHBOARD,
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
        phase_label="Implemented in Phase 2",
        network_address="http://localhost:8501",
        description="Faculty demonstration interface displaying live KPIs, sensor inventory, and status.",
        x=5.2,
        y=3.5,
    ))

    # 4. Planned Components (Planned for Phases 3 through 8)
    topo.add_node(IoTNode(
        node_id="BROKER-01",
        name="Mosquitto Broker\n(Planned Phase 5)",
        node_type=NodeType.MQTT_BROKER,
        phase_status=ComponentPhaseStatus.PLANNED_LATER_PHASE,
        phase_label="Planned for Phase 5 (MQTT Transport)",
        network_address="127.0.0.1:1883",
        description="Eclipse Mosquitto MQTT broker handling isolated telemetry, alert, and ACK topics.",
        x=5.2,
        y=1.0,
    ))

    topo.add_node(IoTNode(
        node_id="DQN-CTRL-01",
        name="Adaptive ACK Controller\n(Planned Phase 7/8)",
        node_type=NodeType.DQN_CONTROLLER,
        phase_status=ComponentPhaseStatus.PLANNED_LATER_PHASE,
        phase_label="Planned for Phase 7/8 (DQN & Controller)",
        network_address="In-Process PyTorch Model",
        description="8-feature DQN selecting between 5 ACK strategies based on channel conditions.",
        x=7.2,
        y=1.0,
    ))

    # 5. Edges
    # Sensor Nodes -> Edge Gateway (Implemented Phase 1 & 2 via SensorEngine in-memory bus)
    topo.add_edge(TopologyEdge(
        edge_id="EDGE-001",
        source_node_id="NODE-001",
        target_node_id="EDGE-GW-01",
        label="Gas Telemetry",
        protocol="Sensor Engine Bus (1s)",
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
    ))
    topo.add_edge(TopologyEdge(
        edge_id="EDGE-002",
        source_node_id="NODE-002",
        target_node_id="EDGE-GW-01",
        label="Temp Telemetry",
        protocol="Sensor Engine Bus (1s)",
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
    ))
    topo.add_edge(TopologyEdge(
        edge_id="EDGE-003",
        source_node_id="NODE-003",
        target_node_id="EDGE-GW-01",
        label="Humidity Telemetry",
        protocol="Sensor Engine Bus (2s)",
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
    ))
    topo.add_edge(TopologyEdge(
        edge_id="EDGE-004",
        source_node_id="NODE-004",
        target_node_id="EDGE-GW-01",
        label="Smoke Telemetry",
        protocol="Sensor Engine Bus (1s)",
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
    ))

    # Edge Gateway -> Streamlit Dashboard (Implemented in Phase 2)
    topo.add_edge(TopologyEdge(
        edge_id="EDGE-005",
        source_node_id="EDGE-GW-01",
        target_node_id="DASH-01",
        label="Live Dashboard Feed",
        protocol="Python Session State",
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
    ))

    # Edge Gateway -> Mosquitto Broker (Planned for Phase 5)
    topo.add_edge(TopologyEdge(
        edge_id="EDGE-006",
        source_node_id="EDGE-GW-01",
        target_node_id="BROKER-01",
        label="MQTT Telemetry & Alerts",
        protocol="Paho MQTT / TCP",
        phase_status=ComponentPhaseStatus.PLANNED_LATER_PHASE,
    ))

    # Mosquitto Broker <-> Adaptive DQN Controller (Planned for Phase 7/8)
    topo.add_edge(TopologyEdge(
        edge_id="EDGE-007",
        source_node_id="BROKER-01",
        target_node_id="DQN-CTRL-01",
        label="Channel State & ACK Control",
        protocol="PyTorch Inference Loop",
        phase_status=ComponentPhaseStatus.PLANNED_LATER_PHASE,
    ))

    return topo
