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
    ("rl_ack.sensors", 2, "sensors", True),  # Module 2 implemented in Phase 1
    ("rl_ack.topology", 3, "topology", True),  # Module 3 implemented in Phase 2
    ("rl_ack.monitoring", 4, "monitoring", True),  # Module 4 implemented in Phase 3
    ("rl_ack.mqtt", 5, "mqtt", False),
    ("rl_ack.policies", 6, "policies", False),
    ("rl_ack.dqn", 7, "dqn", False),
    ("rl_ack.controller", 8, "controller", False),
    ("rl_ack.simulation", 9, "simulation", False),
    ("rl_ack.storage", 10, "storage", False),
    ("rl_ack.analysis", 11, "analysis", False),
]


@pytest.mark.parametrize("mod_name,expected_id,expected_slug,is_implemented", MODULE_SPECS)
def test_module_boundary_exists(mod_name: str, expected_id: int, expected_slug: str, is_implemented: bool):
    """Verify each module boundary package is importable and has identity metadata."""
    mod = importlib.import_module(mod_name)
    assert getattr(mod, "MODULE_ID") == expected_id
    assert getattr(mod, "MODULE_NAME") == expected_slug
    assert getattr(mod, "IS_IMPLEMENTED") is is_implemented, (
        f"Module {mod_name} implementation status mismatch: expected {is_implemented}"
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
