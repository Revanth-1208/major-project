"""Tests for Module 3: System Overview and IoT Network Topology."""

import pytest
import plotly.graph_objects as go

from rl_ack.topology.models import (
    ComponentPhaseStatus,
    IoTNode,
    KitchenIoTTopology,
    NodeType,
    TopologyEdge,
    create_smart_kitchen_topology,
)


def test_iot_node_attributes_and_implementation_flag():
    """Verify IoTNode properties and is_implemented status deduction."""
    node_impl = IoTNode(
        node_id="TEST-001",
        name="Test Sensor",
        node_type=NodeType.SENSOR_NODE,
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
        phase_label="Phase 1",
        associated_sensors=["GAS-001"],
        network_address="192.168.1.100",
        description="Testing node",
        x=1.0,
        y=2.0,
    )
    assert node_impl.is_implemented is True
    d = node_impl.to_dict()
    assert d["node_id"] == "TEST-001"
    assert d["node_type"] == "sensor_node"
    assert d["is_implemented"] is True

    node_plan = IoTNode(
        node_id="TEST-002",
        name="Planned Broker",
        node_type=NodeType.MQTT_BROKER,
        phase_status=ComponentPhaseStatus.PLANNED_LATER_PHASE,
        phase_label="Phase 5",
    )
    assert node_plan.is_implemented is False
    assert node_plan.to_dict()["is_implemented"] is False


def test_topology_edge_attributes_and_dict():
    """Verify TopologyEdge creation and serialization."""
    edge = TopologyEdge(
        edge_id="EDGE-TEST",
        source_node_id="SRC-01",
        target_node_id="TGT-01",
        label="Test Link",
        protocol="MQTT/TCP",
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
    )
    assert edge.is_implemented is True
    d = edge.to_dict()
    assert d["edge_id"] == "EDGE-TEST"
    assert d["protocol"] == "MQTT/TCP"


def test_kitchen_iot_topology_graph_operations():
    """Verify adding nodes and querying implemented vs planned partitions."""
    topo = KitchenIoTTopology()
    node1 = IoTNode(
        node_id="N1",
        name="Node 1",
        node_type=NodeType.SENSOR_NODE,
        phase_status=ComponentPhaseStatus.IMPLEMENTED_PHASE_1_2,
        phase_label="Phase 1",
    )
    node2 = IoTNode(
        node_id="N2",
        name="Node 2",
        node_type=NodeType.MQTT_BROKER,
        phase_status=ComponentPhaseStatus.PLANNED_LATER_PHASE,
        phase_label="Phase 5",
    )
    edge = TopologyEdge(
        edge_id="E1",
        source_node_id="N1",
        target_node_id="N2",
        label="Link",
        protocol="TCP",
        phase_status=ComponentPhaseStatus.PLANNED_LATER_PHASE,
    )

    topo.add_node(node1)
    topo.add_node(node2)
    topo.add_edge(edge)

    assert len(topo.get_nodes()) == 2
    assert len(topo.get_edges()) == 1
    assert topo.get_node("N1") == node1
    assert len(topo.get_implemented_nodes()) == 1
    assert topo.get_implemented_nodes()[0].node_id == "N1"
    assert len(topo.get_planned_nodes()) == 1
    assert topo.get_planned_nodes()[0].node_id == "N2"


def test_create_smart_kitchen_topology_reference_graph():
    """Verify reference topology construction matches smart kitchen IoT specifications."""
    topo = create_smart_kitchen_topology()

    # Verify 8 total nodes: 4 sensors, 1 edge gateway, 1 dashboard, 1 broker, 1 DQN
    nodes = topo.get_nodes()
    assert len(nodes) == 8

    node_ids = {n.node_id for n in nodes}
    expected_ids = {
        "NODE-001", "NODE-002", "NODE-003", "NODE-004",
        "EDGE-GW-01", "DASH-01", "BROKER-01", "DQN-CTRL-01"
    }
    assert node_ids == expected_ids

    # 6 implemented (Phase 1 & 2), 2 planned (Phase 5 & 7/8)
    impl_nodes = topo.get_implemented_nodes()
    plan_nodes = topo.get_planned_nodes()
    assert len(impl_nodes) == 6
    assert len(plan_nodes) == 2

    # Verify edge connectivity
    edges = topo.get_edges()
    assert len(edges) == 7

    # 4 sensor telemetry links + 1 dashboard feed = 5 implemented edges
    impl_edges = [e for e in edges if e.is_implemented]
    plan_edges = [e for e in edges if not e.is_implemented]
    assert len(impl_edges) == 5
    assert len(plan_edges) == 2


def test_to_plotly_figure_generation():
    """Verify Plotly figure generation produces valid visual network structure."""
    topo = create_smart_kitchen_topology()
    fig = topo.to_plotly_figure()

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3  # edges + implemented nodes + planned nodes
    assert fig.layout.title.text is not None
    assert "Smart Kitchen IoT Network Topology" in fig.layout.title.text

    # Trace names
    trace_names = [t.name for t in fig.data if t.name is not None]
    assert "Implemented (Phase 1 & 2)" in trace_names
    assert "Planned (Subsequent Phases)" in trace_names


def test_topology_node_lookup_by_id():
    """Verify get_node properly retrieves node and raises KeyError for invalid id."""
    topo = create_smart_kitchen_topology()
    node = topo.get_node("EDGE-GW-01")
    assert node.name.startswith("Kitchen IoT Edge Gateway")
    assert node.node_type == NodeType.EDGE_GATEWAY

    with pytest.raises(KeyError):
        topo.get_node("NONEXISTENT_NODE")
