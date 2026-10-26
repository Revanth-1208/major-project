"""Comprehensive unit and behavioral tests for Phase 1 Virtual Sensor Engine."""

from datetime import datetime
import math
import pytest

from rl_ack.common.enums import NodeOperationalStatus, SensorType
from rl_ack.common.exceptions import ConfigurationError
from rl_ack.sensors.config import (
    SensorConfig,
    get_default_gas_config,
    get_default_temperature_config,
    get_default_humidity_config,
    get_default_smoke_config,
)
from rl_ack.sensors.base_sensor import BaseSensor
from rl_ack.sensors.gas_sensor import GasSensor
from rl_ack.sensors.temperature_sensor import TemperatureSensor
from rl_ack.sensors.humidity_sensor import HumiditySensor
from rl_ack.sensors.smoke_sensor import SmokeSensor
from rl_ack.sensors.sensor_engine import SensorEngine


# ==============================================================================
# TEST GROUP A: Sensor Initialization & Configuration Validation
# ==============================================================================

def test_four_sensors_initialize_with_default_specs():
    """Verify all four sensor types initialize with their required default specifications."""
    gas = GasSensor()
    assert gas.sensor_id == "GAS-001"
    assert gas.sensor_type == SensorType.GAS
    assert gas.unit == "ppm"
    assert gas.sampling_interval_s == 1.0
    assert gas.host_node_id == "NODE-001"
    assert gas.status == NodeOperationalStatus.NORMAL

    temp = TemperatureSensor()
    assert temp.sensor_id == "TEMP-001"
    assert temp.sensor_type == SensorType.TEMPERATURE
    assert temp.unit == "°C"
    assert temp.sampling_interval_s == 1.0
    assert temp.host_node_id == "NODE-002"

    hum = HumiditySensor()
    assert hum.sensor_id == "HUM-001"
    assert hum.sensor_type == SensorType.HUMIDITY
    assert hum.unit == "% RH"
    assert hum.sampling_interval_s == 2.0  # 2.0s required
    assert hum.host_node_id == "NODE-003"

    smoke = SmokeSensor()
    assert smoke.sensor_id == "SMOKE-001"
    assert smoke.sensor_type == SensorType.SMOKE
    assert smoke.unit == "% obs/m"  # optical obscuration unit
    assert smoke.sampling_interval_s == 1.0
    assert smoke.host_node_id == "NODE-004"


def test_invalid_config_min_greater_than_max():
    """Verify ConfigurationError raised when min_value >= max_value."""
    with pytest.raises(ConfigurationError, match="min_value .* must be strictly less than max_value"):
        SensorConfig(
            sensor_id="TEST-01",
            sensor_type=SensorType.GAS,
            unit="ppm",
            baseline=50.0,
            min_value=100.0,
            max_value=50.0,
            sampling_interval_s=1.0,
            host_node_id="NODE-01",
            threshold=200.0,
        )


def test_invalid_config_non_positive_interval():
    """Verify ConfigurationError raised when sampling interval <= 0."""
    with pytest.raises(ConfigurationError, match="sampling_interval_s must be a strictly positive"):
        SensorConfig(
            sensor_id="TEST-01",
            sensor_type=SensorType.GAS,
            unit="ppm",
            baseline=50.0,
            min_value=0.0,
            max_value=100.0,
            sampling_interval_s=0.0,
            host_node_id="NODE-01",
            threshold=80.0,
        )


def test_invalid_config_baseline_out_of_range():
    """Verify ConfigurationError raised when baseline is outside [min, max]."""
    with pytest.raises(ConfigurationError, match="baseline .* must be within"):
        SensorConfig(
            sensor_id="TEST-01",
            sensor_type=SensorType.GAS,
            unit="ppm",
            baseline=150.0,
            min_value=0.0,
            max_value=100.0,
            sampling_interval_s=1.0,
            host_node_id="NODE-01",
            threshold=80.0,
        )


def test_invalid_config_empty_ids_and_negative_noise():
    """Verify ConfigurationError raised on missing IDs or negative noise."""
    with pytest.raises(ConfigurationError, match="sensor_id must be a non-empty string"):
        get_default_gas_config(sensor_id="")

    with pytest.raises(ConfigurationError, match="host_node_id must be a non-empty string"):
        get_default_gas_config(host_node_id="   ")

    with pytest.raises(ConfigurationError, match="noise_std must be a non-negative"):
        cfg = get_default_gas_config()
        cfg.noise_std = -1.0
        cfg.validate()


# ==============================================================================
# TEST GROUP B: Reading Validity & Data Contracts
# ==============================================================================

def test_reading_fields_completeness_and_finite_numeric_values():
    """Verify all 10 required fields in SensorReading are present, finite, and well-formed."""
    sensor = GasSensor()
    reading = sensor.sample(current_time=1.0)
    assert reading is not None

    assert reading.sensor_id == "GAS-001"
    assert reading.sensor_type == SensorType.GAS
    assert math.isfinite(reading.value)
    assert 0.0 <= reading.value <= 1000.0
    assert reading.unit == "ppm"
    assert reading.simulation_time_s == 1.0
    assert reading.sampling_interval_s == 1.0
    assert reading.host_node_id == "NODE-001"
    assert reading.status == NodeOperationalStatus.NORMAL
    assert reading.sequence_number == 1

    # Validate ISO 8601 UTC timestamp format
    dt = datetime.fromisoformat(reading.timestamp_utc.replace("Z", "+00:00"))
    assert dt.tzinfo is not None


def test_sequence_numbers_increment_strictly():
    """Verify sequence numbers strictly increment without gaps."""
    sensor = TemperatureSensor()
    for seq in range(1, 10):
        reading = sensor.sample(current_time=float(seq))
        assert reading is not None
        assert reading.sequence_number == seq


def test_values_stay_within_configured_ranges():
    """Verify sensor output stays within bounds over numerous stochastic steps."""
    hum = HumiditySensor()
    for t in range(1, 100):
        reading = hum.sample(current_time=float(t))
        assert reading is not None
        assert 0.0 <= reading.value <= 100.0


# ==============================================================================
# TEST GROUP C: Simulation Clock & Sampling Scheduler
# ==============================================================================

def test_sensor_sampling_due_boundaries():
    """Verify is_due correctly tracks sampling schedule."""
    gas = GasSensor()  # interval 1.0s
    assert not gas.is_due(0.5)
    assert gas.is_due(1.0)

    hum = HumiditySensor()  # interval 2.0s
    assert not hum.is_due(1.0)
    assert hum.is_due(2.0)


def test_engine_respects_different_sampling_intervals():
    """Verify Gas (1s) samples at t=1, 2, 3; Humidity (2s) samples only at t=2, 4."""
    engine = SensorEngine.create_default_engine(seed_offset=0)

    # Advance to t = 1.0s
    readings_t1 = engine.advance(1.0)
    sensor_ids_t1 = [r.sensor_id for r in readings_t1]
    assert "GAS-001" in sensor_ids_t1
    assert "TEMP-001" in sensor_ids_t1
    assert "SMOKE-001" in sensor_ids_t1
    assert "HUM-001" not in sensor_ids_t1  # Humidity interval is 2.0s

    # Advance to t = 2.0s
    readings_t2 = engine.advance(1.0)
    sensor_ids_t2 = [r.sensor_id for r in readings_t2]
    assert "HUM-001" in sensor_ids_t2  # Humidity is now due at t=2.0s
    assert "GAS-001" in sensor_ids_t2


def test_multi_boundary_advance_does_not_skip_samples():
    """Verify advancing by dt=3.0 generates all intermediate samples for due sensors."""
    engine = SensorEngine.create_default_engine(seed_offset=0)
    readings = engine.advance(3.0)

    # In 3.0s:
    # 1s sensors (GAS, TEMP, SMOKE) should each emit 3 readings (t=1, 2, 3)
    # 2s sensor (HUM) should emit 1 reading (t=2)
    gas_readings = [r for r in readings if r.sensor_id == "GAS-001"]
    hum_readings = [r for r in readings if r.sensor_id == "HUM-001"]

    assert len(gas_readings) == 3
    assert [r.simulation_time_s for r in gas_readings] == [1.0, 2.0, 3.0]
    assert len(hum_readings) == 1
    assert hum_readings[0].simulation_time_s == 2.0


def test_engine_reset_restores_initial_state():
    """Verify reset restores simulation clock and sensor states."""
    engine = SensorEngine.create_default_engine(seed_offset=0)
    engine.advance(5.0)
    assert engine.current_simulation_time == 5.0

    engine.reset()
    assert engine.current_simulation_time == 0.0
    gas = engine.get_sensor("GAS-001")
    assert gas.sequence_number == 0
    assert gas.last_reading is None
    assert gas.next_sampling_time == 1.0


# ==============================================================================
# TEST GROUP D: Randomness, Independence & Reproducibility
# ==============================================================================

def test_fixed_seed_reproducibility():
    """Verify that two runs with identical seeds produce identical sequences."""
    engine1 = SensorEngine.create_default_engine(seed_offset=10)
    engine2 = SensorEngine.create_default_engine(seed_offset=10)

    readings1 = engine1.advance(5.0)
    readings2 = engine2.advance(5.0)

    assert len(readings1) == len(readings2)
    for r1, r2 in zip(readings1, readings2):
        assert r1.sensor_id == r2.sensor_id
        assert r1.simulation_time_s == r2.simulation_time_s
        assert r1.value == r2.value


def test_different_seeds_produce_different_sequences():
    """Verify that different seeds produce different stochastic sequences."""
    cfg1 = get_default_gas_config(random_seed=100)
    cfg2 = get_default_gas_config(random_seed=200)

    s1 = GasSensor(cfg1)
    s2 = GasSensor(cfg2)

    vals1 = [s1.sample(float(t)).value for t in range(1, 10)]
    vals2 = [s2.sample(float(t)).value for t in range(1, 10)]

    assert vals1 != vals2


def test_isolated_sensor_rng():
    """Verify sampling one sensor does not affect another sensor's random sequence."""
    # Run sensor A alongside sensor B
    gas_cfg_a = get_default_gas_config(random_seed=777)
    gas_cfg_b = get_default_gas_config(random_seed=777)
    temp_cfg = get_default_temperature_config(random_seed=888)

    gas_a = GasSensor(gas_cfg_a)
    gas_b = GasSensor(gas_cfg_b)
    temp = TemperatureSensor(temp_cfg)

    # In path 1: only sample gas_a
    vals_a = [gas_a.sample(float(t)).value for t in range(1, 6)]

    # In path 2: sample temp between samples of gas_b
    vals_b = []
    for t in range(1, 6):
        temp.sample(float(t))
        temp.sample(float(t) + 0.1)  # extra draws on temp
        vals_b.append(gas_b.sample(float(t)).value)

    assert vals_a == vals_b


# ==============================================================================
# TEST GROUP E: Environmental Event Injection
# ==============================================================================

def test_gas_leak_event_drives_concentration_up_and_reverts():
    """Verify gas leak event causes concentration rise and eventual return toward baseline."""
    gas = GasSensor(get_default_gas_config(random_seed=42))
    baseline = gas.config.baseline

    # Sample normally for 3 steps
    for t in range(1, 4):
        gas.sample(float(t))
    initial_val = gas.current_value

    # Trigger gas leak targeting 700 ppm
    gas.simulate_gas_leak(target_ppm=700.0)
    leak_vals = []
    for t in range(4, 15):
        reading = gas.sample(float(t))
        leak_vals.append(reading.value)

    # Values should climb significantly higher than initial baseline
    assert max(leak_vals) > (initial_val + 100.0)
    assert leak_vals[-1] > baseline

    # Clear event: sensor should gradually pull back down toward baseline
    gas.clear_event()
    recovery_vals = []
    for t in range(15, 30):
        reading = gas.sample(float(t))
        recovery_vals.append(reading.value)

    # Trend must be decreasing toward baseline
    assert recovery_vals[-1] < leak_vals[-1]


def test_temperature_heat_event():
    """Verify simulated heating event increases temperature."""
    temp = TemperatureSensor(get_default_temperature_config(random_seed=43))
    temp.simulate_heat_spike(target_temp_c=80.0)
    vals = [temp.sample(float(t)).value for t in range(1, 10)]
    assert vals[-1] > temp.config.baseline + 10.0


def test_smoke_combustion_event():
    """Verify simulated smoke event increases optical obscuration."""
    smoke = SmokeSensor(get_default_smoke_config(random_seed=45))
    smoke.simulate_smoke_event(target_obs=8.0)
    vals = [smoke.sample(float(t)).value for t in range(1, 8)]
    assert vals[-1] > smoke.config.threshold


def test_event_clamping_to_physical_limits():
    """Verify event targeting beyond max physical range remains strictly clamped."""
    gas = GasSensor()
    gas.trigger_event(target_value=999999.0)  # extreme target
    for t in range(1, 20):
        reading = gas.sample(float(t))
        assert reading.value <= gas.config.max_value


# ==============================================================================
# TEST GROUP F: Sensor Availability & Recovery
# ==============================================================================

def test_sensor_unavailability_and_recovery():
    """Verify unavailable sensor stops generating readings and resumes upon recovery."""
    engine = SensorEngine.create_default_engine(seed_offset=0)

    # Advance to t=1.0s: all 1s sensors active
    r1 = engine.advance(1.0)
    assert any(r.sensor_id == "GAS-001" for r in r1)

    # Take GAS-001 offline
    engine.set_sensor_status("GAS-001", NodeOperationalStatus.OFFLINE)
    assert engine.get_sensor("GAS-001").status == NodeOperationalStatus.OFFLINE

    # Advance to t=2.0s: GAS-001 should not produce reading
    r2 = engine.advance(1.0)
    gas_in_r2 = [r for r in r2 if r.sensor_id == "GAS-001"]
    assert len(gas_in_r2) == 0

    # Other sensors continued functioning normally
    assert any(r.sensor_id == "TEMP-001" for r in r2)
    assert any(r.sensor_id == "HUM-001" for r in r2)

    # Restore GAS-001
    engine.set_sensor_status("GAS-001", NodeOperationalStatus.NORMAL)
    r3 = engine.advance(1.0)
    gas_in_r3 = [r for r in r3 if r.sensor_id == "GAS-001"]
    assert len(gas_in_r3) == 1
    assert gas_in_r3[0].status == NodeOperationalStatus.NORMAL


# ==============================================================================
# TEST GROUP G: Regression & Non-Regression Scope
# ==============================================================================

def test_no_external_services_required():
    """Verify Virtual Sensor Engine operates purely in-memory with zero network or broker dependencies."""
    engine = SensorEngine.create_default_engine()
    readings = engine.advance(10.0)
    assert len(readings) > 0
    assert all(math.isfinite(r.value) for r in readings)
