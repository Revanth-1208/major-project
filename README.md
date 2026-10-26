# RL-ACK: Adaptive Acknowledgment Mechanism for MQTT-based IoT Sensor Networks

## Overview

RL-ACK is a modular, interactive research and simulation platform evaluating acknowledgment (ACK) mechanisms over MQTT-enabled IoT sensor networks. The system features:
- **Virtual Sensors**: Gas, Temperature, Relative Humidity, and Smoke sensors producing realistic dynamic telemetry.
- **Incident Monitoring**: Threshold-based incident detection with independent alert generation.
- **MQTT Transport**: Real Eclipse Mosquitto broker integration with multi-stage delivery verification.
- **Policy Comparison**: Six comparative evaluation policies.
- **Adaptive Control**: Deep Q-Network (DQN) controller selecting between five underlying ACK strategies based on an 8-feature network state.
- **Storage & Analytics**: SQLite persistence and CSV export for experimental verification.

> [!IMPORTANT]
> **Safety Notice**: All virtual sensor readings, operating ranges, and incident thresholds in this framework are computer-simulated modeling parameters. They are not certified life-safety limits and must never be used as a substitute for calibrated physical safety sensors.

---

## Technology Stack

| Component | Library / System | Version / Role |
| :--- | :--- | :--- |
| Programming Language | Python | 3.13.2 |
| Interactive Dashboard | Streamlit | 1.64.0 |
| Deep Reinforcement Learning | PyTorch | 2.14.0 |
| MQTT Client Library | Paho MQTT | 2.1.0 |
| MQTT Message Broker | Eclipse Mosquitto | 2.0+ (Local broker binary) |
| Data Processing | Pandas & NumPy | 3.0.5 & 2.5.3 |
| Visualization | Plotly | 7.1.0 |
| Relational Storage | SQLite3 | Python standard library |
| Automated Testing | Pytest | 9.1.1 |

---

## Architectural Modules

The system is partitioned into twelve distinct modules:

1. **Module 1: Project Setup & Architecture** (`rl_ack/config.py`, `rl_ack/common/`, `pyproject.toml`) — *Verified*
2. **Module 2: Virtual Sensor Engine** (`rl_ack/sensors/`) — *Implemented (Phase 1)*
3. **Module 3: System Overview and IoT Topology** (`rl_ack/topology/`, `rl_ack/dashboard/`, `app.py`) — *Implemented (Phase 2)*
4. **Module 4: Sensor Monitoring and Incident Detection** (`rl_ack/monitoring/`) — *Subsequent Phase*
5. **Module 5: MQTT Alert Transmission and Delivery Verification** (`rl_ack/mqtt/`) — *Subsequent Phase*
6. **Module 6: Six-Policy Comparative Analysis** (`rl_ack/policies/`) — *Subsequent Phase*
7. **Module 7: DQN Architecture and Training** (`rl_ack/dqn/`) — *Subsequent Phase*
8. **Module 8: DQN Controller Integration** (`rl_ack/controller/`) — *Subsequent Phase*
9. **Module 9: Interactive Simulation Runner** (`rl_ack/simulation/`) — *Subsequent Phase*
10. **Module 10: Experiment Data Storage** (`rl_ack/storage/`) — *Subsequent Phase*
11. **Module 11: Performance Analysis and Export** (`rl_ack/analysis/`) — *Subsequent Phase*
12. **Module 12: End-to-End Testing Suite** (`tests/`) — *Continuous Verification*

---

## Phase 1: Virtual Sensor Engine Architecture

The Virtual Sensor Engine acts as the single source of truth for all telemetry data across the platform.

### Mathematical Dynamics
Each sensor model is governed by a discrete-time mean-reverting bounded Ornstein-Uhlenbeck stochastic process:

$$x_t = \text{clamp}\Big(x_{t-1} + k \cdot (\mu_t - x_{t-1}) + \sigma \cdot \epsilon_t, \; [\text{min\_value}, \text{max\_value}]\Big)$$

- $\mu_t$: Target baseline under normal conditions, or dynamic target under active environmental incidents.
- $k \in [0.0, 1.0]$: Mean-reversion pull rate.
- $\sigma \ge 0.0$: Gaussian noise standard deviation.
- $\epsilon_t \sim \mathcal{N}(0, 1)$: Pseudorandom variate from a dedicated per-sensor RNG.

### The Four Required Virtual Sensors

| Sensor ID | Type | Measurement | Unit | Interval | Range | Baseline | Threshold | Host Node |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `GAS-001` | `gas` | Gas concentration | `ppm` | 1.0 s | $[0.0, 1000.0]$ | 50.0 ppm | 400.0 ppm | `NODE-001` |
| `TEMP-001` | `temperature` | Ambient temperature | `°C` | 1.0 s | $[-20.0, 100.0]$ | 24.0 °C | 55.0 °C | `NODE-002` |
| `HUM-001` | `humidity` | Relative humidity | `% RH` | 2.0 s | $[0.0, 100.0]$ | 45.0 % RH | 85.0 % RH | `NODE-003` |
| `SMOKE-001` | `smoke` | Optical obscuration | `% obs/m` | 1.0 s | $[0.0, 20.0]$ | 0.2 % obs/m | 3.5 % obs/m | `NODE-004` |

### Sensor Reading Data Contract (`SensorReading`)
Each generated reading is an immutable-style typed dataclass containing all 10 required fields:
1. `sensor_id`: Stable instance identifier (e.g. `GAS-001`).
2. `sensor_type`: Categorical type (`SensorType.GAS`, etc.).
3. `value`: Finite numeric float clamped to the sensor's physical range.
4. `unit`: Physical measurement unit string (e.g. `ppm`, `°C`, `% RH`, `% obs/m`).
5. `simulation_time_s`: Elapsed monotonic simulation clock in seconds.
6. `timestamp_utc`: ISO 8601 UTC timestamp string (`YYYY-MM-DDTHH:MM:SS.mmmmmm+00:00`).
7. `sampling_interval_s`: Configured sampling period in seconds.
8. `host_node_id`: Configured host node identifier.
9. `status`: Operational status (`NodeOperationalStatus.NORMAL` or `OFFLINE`).
10. `sequence_number`: Strictly monotonic integer starting at 1 per sensor.

### Simulation Time vs UTC Wall-Clock Time
- **Simulation Time (`simulation_time_s`)**: Monotonic discrete clock advancing by explicit time increments ($\Delta t$) via the `SensorEngine`. It is unaffected by real-world CPU pauses, thread scheduling, or wall-clock time zones.
- **UTC Wall-Clock Time (`timestamp_utc`)**: Real-world timestamp recorded at sample generation for auditing and logging purposes.

### Isolated Reproducibility via Fixed Random Seeds
Every sensor owns an independent `random.Random(seed)` instance. Sampling one sensor does not advance the PRNG state of any other sensor, ensuring that experiments are 100% reproducible regardless of thread or invocation interleaving.

### Environmental Event Injection
Simulated environmental events (e.g. gas leaks, heat spikes, combustion smoke, humidity changes) are applied programmatically:
```python
# Inject gas leak targeting 650.0 ppm for 5 seconds
engine.trigger_sensor_event("GAS-001", target_value=650.0, duration_s=5.0)

# Clear event early if desired
engine.clear_sensor_event("GAS-001")
```
When an event ends, the sensor returns gradually toward its baseline via mean reversion. Events change the underlying physical model rather than fabricating sample records.

---

## Phase 2: System Overview, Sensor Dashboard & IoT Topology

Phase 2 builds an interactive user interface (via Streamlit and Plotly) directly connected to the Phase 1 Virtual Sensor Engine. It introduces the full Smart Kitchen IoT Network graph, dynamic KPI derivation, sensor cards, inventory tables, and real-time telemetry charting.

### Dashboard Architecture & Features

1. **Simulation Controls & Determinism**:
   - **Step Button (`▶️ Step`)**: Advances the simulation clock monotonically by $\Delta t$ (e.g. 1.0s).
   - **Batch Advance (`⏩ Run N Steps`)**: Executes consecutive simulation steps (5, 10, 15, 20 steps) in a single update.
   - **Reset Button (`🔄 Reset`)**: Clears history buffer, resets sensor engines to $t=0$, and restores baseline conditions.
   - **Environmental Scenario Selector**: Allows triggering preconfigured scenarios (Normal Operation, Gradual Gas Leak, Abnormal Temperature Rise, Increasing Smoke Obscuration, Steam Anomaly, Dry Air, and Kitchen Fire Multi-Event Emergency).
   - **Deterministic Random Seed**: Fixed numeric seed ensures identical pseudo-random walks across re-runs.

2. **Derived System KPIs**:
   - KPIs are computed dynamically from actual sensor state: **Total Sensors (4)**, **Normal Count**, **Warning Count**, **Critical Count**, **Unavailable Count**, **Simulation Time (s)**, and **Active Events**.
   - No hardcoded numbers or static text.

3. **Individual Sensor Status Cards**:
   - Distinct cards for `GAS-001`, `TEMP-001`, `HUM-001`, and `SMOKE-001`.
   - Clear visual status badges:
     - 🟢 **NORMAL**: Reading is within standard operating limits.
     - 🟡 **WARNING**: Reading has exceeded warning limit.
     - 🔴 **CRITICAL**: Reading has crossed critical incident threshold.
     - ⚪ **UNAVAILABLE**: Sensor is offline or uninitialized.
   - Displays current reading, unit, sampling interval, sequence number, and last sample time.

4. **Sensor Inventory Table**:
   - Tabular summary of all configured sensors: Sensor ID, Type, Unit, Current Reading, Warning Limit, Critical Limit, Sampling Interval, Sequence Number, Derived Status, and Host Node ID.

5. **Live Telemetry Preview Charts (Plotly)**:
   - Real-time line plots of sensor telemetry against simulation time.
   - Includes horizontal reference lines for Warning and Critical thresholds.
   - Preserves discrete sampling intervals (1.0s for Gas, Temp, Smoke; 2.0s for Humidity).

6. **Interactive Smart Kitchen IoT Network Topology**:
   - Graph visualization using Plotly scatter traces without external OS binaries.
   - Clearly distinguishes between:
     - 🟢 **Implemented (Phase 1 & 2)**: Sensor Nodes (`NODE-001` .. `NODE-004`), Edge Gateway (`EDGE-GW-01`), Central Dashboard (`DASH-01`), and internal telemetry communication links.
     - ⚪ **Planned (Subsequent Phases)**: Mosquitto Broker (`BROKER-01`), Adaptive DQN Controller (`DQN-CTRL-01`), and external MQTT channels.

---

## DQN Architecture & Policies (Summary)

### State Vector (8 Features)
1. `network_utilization` ($[0.0, 1.0]$)
2. `rolling_ack_rate` ($[0.0, 1.0]$)
3. `transmit_queue_occupancy` ($[0.0, 1.0]$)
4. `traffic_critical_control` (one-hot: $1.0$ or $0.0$)
5. `traffic_high_bandwidth_streaming` (one-hot: $1.0$ or $0.0$)
6. `traffic_best_effort` (one-hot: $1.0$ or $0.0$)
7. `remaining_node_energy` ($[0.0, 1.0]$)
8. `rolling_latency` (normalized)

### Neural Network
$$\text{Input (8)} \longrightarrow \text{Hidden (64, ReLU)} \longrightarrow \text{Hidden (32, ReLU)} \longrightarrow \text{Output (5 Q-values)}$$

### Five DQN Actions
- Action 0: MQTT QoS 0
- Action 1: Local ACK
- Action 2: MQTT QoS 1
- Action 3: Local ACK plus Selective Retransmission
- Action 4: MQTT QoS 2

### Six Evaluation Policies
1. Static MQTT QoS 0
2. Static Local ACK
3. Static MQTT QoS 1
4. Local ACK plus Selective Retransmission
5. Static MQTT QoS 2
6. Adaptive RL-ACK DQN (selects among the 5 actions via the trained DQN)

---

## Project Structure

```text
d:\Maj_2
├── app.py                     # Streamlit application entrypoint (Phase 2)
├── main.py                    # Top-level CLI launcher & diagnostics
├── pyproject.toml             # Packaging and build specifications
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Dependency pin specifications
├── README.md                  # System overview and usage guide
├── docs/
│   └── EMERGENCY_ALERT_ARCHITECTURE.md # Architecture specification for emergency alerts
├── scripts/
│   └── demo_sensors.py        # Phase 1 sensor verification demonstration runner
├── src/
│   └── rl_ack/
│       ├── __init__.py        # Package root and versioning
│       ├── config.py          # Centralized typed configuration
│       ├── cli.py             # CLI verification and diagnostic runner
│       ├── common/            # Shared contracts, enums, models, exceptions
│       │   ├── __init__.py
│       │   ├── enums.py
│       │   ├── exceptions.py
│       │   ├── interfaces.py
│       │   └── models.py
│       ├── sensors/           # Module 2: Virtual Sensor Engine (Implemented Phase 1)
│       │   ├── __init__.py
│       │   ├── base_sensor.py
│       │   ├── config.py
│       │   ├── demo.py
│       │   ├── gas_sensor.py
│       │   ├── humidity_sensor.py
│       │   ├── scenarios.py   # Environmental scenario engine
│       │   ├── sensor_engine.py
│       │   ├── smoke_sensor.py
│       │   ├── temperature_sensor.py
│       │   └── thresholds.py  # Status classifier with hysteresis
│       ├── topology/          # Module 3: IoT Topology & Nodes (Implemented Phase 2)
│       │   ├── __init__.py
│       │   └── models.py      # Network graph, nodes, edges, Plotly rendering
│       ├── dashboard/         # UI Presentation Subsystem (Implemented Phase 2)
│       │   ├── __init__.py
│       │   ├── components.py  # Modular UI presentation widgets
│       │   └── state.py       # Simulation manager and state encapsulation
│       ├── monitoring/        # Module 4: Incident Detection (Planned Phase 3/4)
│       ├── mqtt/              # Module 5: MQTT Communication & Verification (Planned Phase 5)
│       ├── policies/          # Module 6: Six ACK Policies (Planned Phase 6)
│       ├── dqn/               # Module 7: Neural Network & Replay Buffer (Planned Phase 7)
│       ├── controller/        # Module 8: DQN Online Controller (Planned Phase 8)
│       ├── simulation/        # Module 9: Simulation Engine (Planned Phase 9)
│       ├── storage/           # Module 10: SQLite Storage (Planned Phase 10)
│       └── analysis/          # Module 11: Export & Visualization (Planned Phase 11)
└── tests/
    ├── __init__.py
    ├── test_app_entrypoint.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_dashboard_state.py           # Phase 2 Dashboard state tests
    ├── test_module_boundaries.py         # Boundary validation for Modules 1-12
    ├── test_sensor_scenarios_and_status.py # Scenario & hysteresis tests
    ├── test_topology.py                  # Phase 2 IoT Topology unit tests
    ├── test_types_and_contracts.py       # Domain contracts and serialization tests
    └── test_virtual_sensors.py           # Phase 1 Sensor suite (74 total tests)
```

---

## Execution Commands

### 1. Launch the Streamlit Dashboard (Phase 2 UI)
```powershell
streamlit run app.py
```

### 2. Run the Full Automated Pytest Suite (74 Tests)
```powershell
python -m pytest tests/
```

### 3. Run the Virtual Sensor CLI Verification Demo
```powershell
python scripts/demo_sensors.py
```

### 4. Run Diagnostic Stack Check
```powershell
python main.py
```

---

## Phase 2 Scope Boundaries & Subsystems Status

In strict adherence to Phase 2 scope:
- **Implemented Subsystems**:
  - Central Streamlit dashboard (`app.py`).
  - Simulation state manager (`DashboardSimulationManager`).
  - IoT Network Topology model and interactive Plotly visualization (`KitchenIoTTopology`).
  - Dynamic KPI cards and status badges.
  - Sensor inventory table.
  - Live time-series telemetry preview charts.
  - Bounded recent telemetry stream buffer and active scenario events log.
- **Subsystems Intentionally Deferred to Future Phases**:
  - **No MQTT Publishing**: Mosquitto broker integration and real network transport will be implemented in Phase 5.
  - **No Incident Detection / Alert Transmission**: Formal alert dispatch over MQTT topics will be implemented in Phases 4 & 5.
  - **No DQN Training or Inference**: Neural network policies and RL-ACK controller will be implemented in Phases 7 & 8.
  - **No SQLite Persistence**: Experiment runs and telemetry database logging will be implemented in Phase 10.
