"""Tests for module boundaries and architectural package structure."""

import importlib
import pytest
from rl_ack.common.interfaces import (
    ISensor,
    ISensorEngine,
    IIncidentDetector,
    IMqttManager,
    IAckPolicy,
    IDQNAgent,
    ISimulationRunner,
    IStorageRepository,
)


MODULE_SPECS = [
    ("rl_ack.sensors", 2, "sensors"),
    ("rl_ack.topology", 3, "topology"),
    ("rl_ack.monitoring", 4, "monitoring"),
    ("rl_ack.mqtt", 5, "mqtt"),
    ("rl_ack.policies", 6, "policies"),
    ("rl_ack.dqn", 7, "dqn"),
    ("rl_ack.controller", 8, "controller"),
    ("rl_ack.simulation", 9, "simulation"),
    ("rl_ack.storage", 10, "storage"),
    ("rl_ack.analysis", 11, "analysis"),
]


@pytest.mark.parametrize("mod_name,expected_id,expected_slug", MODULE_SPECS)
def test_module_boundary_exists(mod_name: str, expected_id: int, expected_slug: str):
    """Verify each module boundary package is importable and has identity metadata."""
    mod = importlib.import_module(mod_name)
    assert getattr(mod, "MODULE_ID") == expected_id
    assert getattr(mod, "MODULE_NAME") == expected_slug
    assert getattr(mod, "IS_IMPLEMENTED") is False, (
        f"Module {mod_name} should not be prematurely implemented in Phase 0."
    )


def test_abstract_interfaces_cannot_be_instantiated():
    """Verify abstract base interfaces enforce contractual separation."""
    interfaces = [
        ISensor,
        ISensorEngine,
        IIncidentDetector,
        IMqttManager,
        IAckPolicy,
        IDQNAgent,
        ISimulationRunner,
        IStorageRepository,
    ]
    for iface in interfaces:
        with pytest.raises(TypeError):
            iface()
