# RL-ACK: Adaptive Acknowledgment Mechanism for MQTT-based IoT Sensor Networks

## Overview

RL-ACK is a modular, interactive research and simulation platform evaluating acknowledgment (ACK) mechanisms over MQTT-enabled IoT sensor networks. The system features:
- **Virtual Sensors**: Gas, Temperature, Relative Humidity, and Smoke sensors producing realistic dynamic telemetry.
- **Incident Monitoring**: Threshold-based incident detection with independent alert generation.
- **MQTT Transport**: Real Eclipse Mosquitto broker integration with multi-stage delivery verification.
- **Policy Comparison**: Six comparative evaluation policies.
- **Adaptive Control**: Deep Q-Network (DQN) controller selecting between five underlying ACK strategies based on an 8-feature network state.
- **Storage & Analytics**: SQLite persistence and CSV export for experimental verification.

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

1. **Module 1: Project Setup & Architecture** (`rl_ack/config.py`, `rl_ack/common/`, `pyproject.toml`)
2. **Module 2: Virtual Sensor Engine** (`rl_ack/sensors/`)
3. **Module 3: System Overview and IoT Topology** (`rl_ack/topology/`)
4. **Module 4: Sensor Monitoring and Incident Detection** (`rl_ack/monitoring/`)
5. **Module 5: MQTT Alert Transmission and Delivery Verification** (`rl_ack/mqtt/`)
6. **Module 6: Six-Policy Comparative Analysis** (`rl_ack/policies/`)
7. **Module 7: DQN Architecture and Training** (`rl_ack/dqn/`)
8. **Module 8: DQN Controller Integration** (`rl_ack/controller/`)
9. **Module 9: Interactive Simulation Runner** (`rl_ack/simulation/`)
10. **Module 10: Experiment Data Storage** (`rl_ack/storage/`)
11. **Module 11: Performance Analysis and Export** (`rl_ack/analysis/`)
12. **Module 12: End-to-End Testing Suite** (`tests/`)

---

## DQN Architecture & Policies

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
├── app.py                     # Streamlit application entrypoint
├── main.py                    # Top-level CLI launcher
├── pyproject.toml             # Packaging and build specifications
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Dependency pin specifications
├── README.md                  # System overview and usage guide
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
│       ├── sensors/           # Module 2: Virtual Sensor Engine
│       ├── topology/          # Module 3: IoT Topology & Nodes
│       ├── monitoring/        # Module 4: Incident Detection
│       ├── mqtt/              # Module 5: MQTT Communication & Verification
│       ├── policies/          # Module 6: Six ACK Policies
│       ├── dqn/               # Module 7: Neural Network & Replay Buffer
│       ├── controller/        # Module 8: DQN Online Controller
│       ├── simulation/        # Module 9: Simulation Engine
│       ├── storage/           # Module 10: SQLite Storage
│       └── analysis/          # Module 11: Export & Visualization
└── tests/
    ├── __init__.py
    ├── test_app_entrypoint.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_module_boundaries.py
    └── test_types_and_contracts.py
```

---

## Quickstart & Verification Commands

### 1. Run the Diagnostic CLI Check
```powershell
python main.py
```

### 2. Run the Full Test Suite
```powershell
python -m pytest tests/
```

### 3. Launch the Streamlit Dashboard (Smoke Verification)
```powershell
streamlit run app.py
```
