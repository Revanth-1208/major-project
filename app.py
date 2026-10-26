"""Streamlit Application Entrypoint for RL-ACK Project.

Phase 0: Project Setup and Architecture Verification.
Displays system overview, environment diagnostic status, and module boundaries.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st
from rl_ack import __version__
from rl_ack.config import AppConfig
from rl_ack.cli import check_dependencies

st.set_page_config(
    page_title="RL-ACK Simulation Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📡 RL-ACK: Adaptive Acknowledgment Mechanism for MQTT IoT Networks")
st.caption(f"Framework Version: v{__version__} | Architectural Status: Phase 0 Setup Verified")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🛠️ Technology Stack Status")
    dep_results = check_dependencies()
    for dep_name, (is_ok, detail) in dep_results.items():
        if is_ok:
            st.success(f"**{dep_name}**: {detail}")
        else:
            st.error(f"**{dep_name}**: {detail}")

with col2:
    st.subheader("📐 Architecture Configuration")
    config = AppConfig()
    st.json({
        "MQTT Broker Host": config.broker.host,
        "MQTT Broker Port": config.broker.port,
        "DQN Architecture": f"{config.dqn.state_dim} inputs -> {config.dqn.hidden_dim_1} -> {config.dqn.hidden_dim_2} -> {config.dqn.action_dim} Q-values",
        "Sensor Defaults": {
            "Gas Range (ppm)": config.sensors.gas_range_ppm,
            "Temp Range (°C)": config.sensors.temp_range_celsius,
            "Humidity Range (%)": config.sensors.humidity_range_pct,
            "Smoke Range (% obs)": config.sensors.smoke_range_obscuration,
        },
        "Database Path": str(config.storage.db_path),
    })

st.markdown("---")
st.subheader("📦 Module Boundary Registration (Phase 0)")

modules = [
    ("Module 1", "Project Setup & Architecture", "Configured & Active", True),
    ("Module 2", "Virtual Sensor Engine (Gas, Temp, Humidity, Smoke)", "Interface Defined (Phase 1)", False),
    ("Module 3", "System Overview & IoT Topology", "Boundary Defined", False),
    ("Module 4", "Sensor Monitoring & Incident Detection", "Boundary Defined", False),
    ("Module 5", "MQTT Alert Transmission & Delivery Verification", "Boundary Defined", False),
    ("Module 6", "Six-Policy Comparative Analysis", "Boundary Defined", False),
    ("Module 7", "DQN Architecture & Training (8-64-32-5)", "Boundary Defined", False),
    ("Module 8", "DQN Controller Integration", "Boundary Defined", False),
    ("Module 9", "Interactive Simulation Runner", "Boundary Defined", False),
    ("Module 10", "Experiment Data Storage (SQLite)", "Boundary Defined", False),
    ("Module 11", "Performance Analysis & Export (CSV/Plotly)", "Boundary Defined", False),
    ("Module 12", "End-to-End Testing", "Pytest Suite Configured", True),
]

cols = st.columns(3)
for idx, (mod_id, name, status, active) in enumerate(modules):
    target_col = cols[idx % 3]
    with target_col:
        badge = "🟢" if active else "⚪"
        st.info(f"{badge} **{mod_id}**: {name}\n\n*Status*: {status}")

st.sidebar.title("RL-ACK Controls")
st.sidebar.info(
    "Phase 0 is currently active. Module boundaries are established and ready for Phase 1 (Virtual Sensor Engine) upon user approval."
)
