"""Tests for configuration dataclasses and validation."""

import pytest
from pathlib import Path
from rl_ack.config import (
    AppConfig,
    BrokerConfig,
    SensorDefaults,
    DQNConfig,
    SimulationConfig,
    StorageConfig,
)


def test_app_config_initialization():
    """Verify default AppConfig builds all child configurations."""
    config = AppConfig()
    assert isinstance(config.broker, BrokerConfig)
    assert isinstance(config.sensors, SensorDefaults)
    assert isinstance(config.dqn, DQNConfig)
    assert isinstance(config.simulation, SimulationConfig)
    assert isinstance(config.storage, StorageConfig)
    assert isinstance(config.project_root, Path)


def test_dqn_architecture_constraints():
    """Verify DQN configuration strictly enforces 8 inputs and 5 action outputs."""
    dqn = DQNConfig()
    assert dqn.state_dim == 8, "DQN input dimension must be 8"
    assert dqn.hidden_dim_1 == 64, "DQN first hidden layer must be 64"
    assert dqn.hidden_dim_2 == 32, "DQN second hidden layer must be 32"
    assert dqn.action_dim == 5, "DQN output dimension must be 5"
    assert dqn.learning_rate > 0
    assert 0 < dqn.gamma <= 1.0


def test_broker_config_defaults():
    """Verify default broker connection properties."""
    broker = BrokerConfig()
    assert broker.host == "127.0.0.1"
    assert broker.port == 1883
    assert broker.base_telemetry_topic == "iot/sensors/telemetry"
    assert broker.base_alert_topic == "iot/sensors/alerts"
    assert broker.base_ack_topic == "iot/sensors/acks"


def test_sensor_defaults():
    """Verify default operating ranges for the 4 virtual sensors."""
    sensors = SensorDefaults()
    assert sensors.gas_range_ppm == (0.0, 1000.0)
    assert sensors.gas_threshold_ppm == 400.0
    assert sensors.temp_range_celsius == (-20.0, 100.0)
    assert sensors.temp_threshold_celsius == 55.0
    assert sensors.humidity_range_pct == (0.0, 100.0)
    assert sensors.humidity_threshold_pct == 85.0
    assert sensors.smoke_range_obscuration == (0.0, 20.0)
    assert sensors.smoke_threshold_obscuration == 3.5


def test_config_to_dict():
    """Verify AppConfig serialization to dictionary."""
    config = AppConfig()
    data = config.to_dict()
    assert "broker" in data
    assert "sensors" in data
    assert "dqn" in data
    assert "simulation" in data
    assert "storage" in data
    assert data["dqn"]["state_dim"] == 8
    assert data["dqn"]["action_dim"] == 5
