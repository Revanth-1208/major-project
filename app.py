"""Streamlit Application Entrypoint for RL-ACK Project.

Phase 2: System Overview, Virtual Sensor Dashboard and IoT Topology.
Demonstrates live virtual sensor engine telemetry, dynamic status transitions,
configurable environmental scenarios, and interactive IoT network topology.
"""

import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st
import pandas as pd

from rl_ack import __version__
from rl_ack.cli import check_dependencies
from rl_ack.dashboard.state import DashboardSimulationManager, SCENARIO_BUILDERS
from rl_ack.dashboard.components import (
    render_header,
    render_kpi_cards,
    render_sensor_cards,
    render_preview_charts,
    render_topology_section,
    render_incident_monitoring_tab,
    render_mqtt_delivery_tab,
)
from rl_ack.topology.models import create_smart_kitchen_topology

# Configure Streamlit page
st.set_page_config(
    page_title="RL-ACK Smart Kitchen IoT Simulation",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Session State Initialization (Persistent across Streamlit reruns)
# ------------------------------------------------------------------------------

if "sim_manager" not in st.session_state:
    st.session_state.sim_manager = DashboardSimulationManager(
        random_seed=42,
        scenario_name="Normal Operation",
        step_size=1.0,
        max_history=250,
    )

if "topology" not in st.session_state:
    st.session_state.topology = create_smart_kitchen_topology()

manager: DashboardSimulationManager = st.session_state.sim_manager
topology = st.session_state.topology

# ------------------------------------------------------------------------------
# Sidebar: Simulation Controls & Scenario Selection
# ------------------------------------------------------------------------------

st.sidebar.title("🎛️ Simulation Controls")

# Scenario Selection
scenario_options = list(SCENARIO_BUILDERS.keys())
current_scen_idx = scenario_options.index(manager.scenario_name) if manager.scenario_name in scenario_options else 0
selected_scenario = st.sidebar.selectbox(
    "Environmental Scenario",
    options=scenario_options,
    index=current_scen_idx,
    help="Select an environmental condition to test sensor responses and status transitions.",
)

if selected_scenario != manager.scenario_name:
    manager.set_scenario(selected_scenario)
    st.rerun()

# Random Seed Setting
seed_val = st.sidebar.number_input(
    "Deterministic Random Seed",
    min_value=1,
    max_value=999999,
    value=manager.random_seed,
    step=1,
    help="Fixed seed guarantees reproducible stochastic random walks.",
)
if seed_val != manager.random_seed:
    manager.set_random_seed(seed_val)
    st.rerun()

# Step Size Configuration
step_size_val = st.sidebar.slider(
    "Step Increment (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=manager.step_size,
    step=0.5,
    help="Simulation time advance per single step.",
)
manager.step_size = step_size_val

st.sidebar.markdown("---")
st.sidebar.subheader("🕹️ Execution Commands")

# Continuous Auto-Run toggle
is_auto_running = st.sidebar.toggle(
    "▶️ Continuous Auto-Run",
    value=st.session_state.get("auto_running", False),
    help="Continuously advance the simulation clock in real time.",
)
st.session_state.auto_running = is_auto_running

# Single Step button
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("▶️ Step (1s)", width="stretch", help="Advance simulation by 1 step"):
        st.session_state.auto_running = False
        manager.step()
        manager.pause()
        st.rerun()

with col_btn2:
    if st.button("🔄 Reset", width="stretch", help="Reset simulation clock to t=0"):
        st.session_state.auto_running = False
        manager.reset()
        st.rerun()

# Batch run button
batch_count = st.sidebar.selectbox(
    "Batch Advance Steps",
    options=[5, 10, 15, 20],
    index=1,
    help="Execute consecutive steps in a single update.",
)

if st.sidebar.button(f"⏩ Run {batch_count} Steps ({batch_count * step_size_val:.1f}s)", width="stretch"):
    st.session_state.auto_running = False
    manager.run_steps(batch_count)
    manager.pause()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size: 12px; color: #64748B;">
        <b>Demo Guide for Faculty:</b><br/>
        1. Ensure Mosquitto broker is running (or check connection in MQTT tab).<br/>
        2. Select <i>Kitchen Fire Multi-Event Emergency</i> or <i>Gradual Gas Leak</i>.<br/>
        3. Advance steps (<b>Run 10 Steps</b> or <b>Continuous Auto-Run</b>).<br/>
        4. Switch to <b>🚨 Incident Monitoring</b> to view detected incidents.<br/>
        5. Switch to <b>📡 MQTT Alert Delivery</b> to observe real-time MQTT publishing, broker PUBACK, subscriber receipt, and application-level delivery confirmation with latency measurements.
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Main Dashboard Body
# ------------------------------------------------------------------------------

render_header()

# Top-level derived KPI metrics
kpis = manager.get_kpis()
render_kpi_cards(kpis)

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

# Individual sensor status cards (GAS-001, TEMP-001, HUM-001, SMOKE-001)
render_sensor_cards(
    engine_sensors=manager.engine.get_sensors(),
    latest_readings=manager.engine.get_latest_readings(),
)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Main Tabbed Presentation
tab_telemetry, tab_incident, tab_mqtt, tab_topology, tab_stream, tab_arch = st.tabs([
    "📊 Live Telemetry & Inventory",
    "🚨 Incident Monitoring & Alerts",
    "📡 MQTT Alert Delivery",
    "🌐 IoT Network Topology",
    "📜 Telemetry Stream & Events",
    "ℹ️ System Architecture & Roadmaps",
])

# ------------------------------------------------------------------------------
# Tab 1: Live Telemetry & Inventory
# ------------------------------------------------------------------------------
with tab_telemetry:
    st.subheader("📋 Virtual Sensor Inventory")
    st.caption("Live configuration, current readings, and derived operational statuses directly from the engine.")

    inventory_rows = manager.get_sensor_inventory()
    df_inventory = pd.DataFrame(inventory_rows)
    st.dataframe(df_inventory, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("📈 Real Telemetry Preview Charts")
    st.caption("Plots real sensor data points generated at their scheduled boundaries against monotonic simulation time.")

    time_series_df = manager.get_time_series_df()
    render_preview_charts(time_series_df, sensor_configs=manager.get_sensor_configs())

# ------------------------------------------------------------------------------
# Tab 2: Incident Monitoring & Alerts (Phase 3)
# ------------------------------------------------------------------------------
with tab_incident:
    render_incident_monitoring_tab(manager)

# ------------------------------------------------------------------------------
# Tab 3: MQTT Alert Delivery & Verification (Phase 4)
# ------------------------------------------------------------------------------
with tab_mqtt:
    render_mqtt_delivery_tab(manager)

# ------------------------------------------------------------------------------
# Tab 2: IoT Network Topology
# ------------------------------------------------------------------------------
with tab_topology:
    render_topology_section(topology)

# ------------------------------------------------------------------------------
# Tab 3: Telemetry Stream & Events Log
# ------------------------------------------------------------------------------
with tab_stream:
    col_ev, col_log = st.columns([1, 2])

    with col_ev:
        st.subheader("⚡ Environmental Scenario Events")
        if manager.active_scenario and manager.active_scenario.events:
            st.markdown(f"**Active Scenario:** `{manager.active_scenario.name}`")
            st.markdown(f"*{manager.active_scenario.description}*")
            for ev in manager.active_scenario.events:
                is_active = ev.is_active_at(manager.current_time_s)
                badge = "🔴 ACTIVE" if is_active else ("⏳ UPCOMING" if manager.current_time_s < ev.start_time_s else "⚪ EXPIRED")
                dur_str = f"{ev.duration_s:.1f} s" if ev.duration_s else "Indefinite"
                st.info(
                    f"**{badge}**: `{ev.sensor_id}` &rarr; Target: **{ev.target_value:.1f}**\n\n"
                    f"*Start:* {ev.start_time_s:.1f}s | *Duration:* {dur_str}\n\n"
                    f"{ev.description}"
                )
        else:
            st.success("🟢 Normal Operation: No scheduled environmental anomalies active.")

        st.warning(
            "📌 **Phase Note:** Incident alert messages and MQTT delivery verification are scheduled for "
            "Phases 4 & 5. Telemetry shown is produced by the in-memory Virtual Sensor Engine."
        )

    with col_log:
        st.subheader("📜 Recent Telemetry Stream (Bounded Buffer)")
        st.caption(f"Showing up to the last 50 samples emitted by the engine (Buffer Capacity: {manager.max_history} samples).")
        recent_df = manager.get_recent_readings_df(limit=50)
        st.dataframe(recent_df, width="stretch", hide_index=True, height=350)

# ------------------------------------------------------------------------------
# Tab 4: System Architecture & Roadmaps
# ------------------------------------------------------------------------------
with tab_arch:
    col_diag, col_plan = st.columns(2)

    with col_diag:
        st.subheader("🛠️ Technology Stack Diagnostics")
        dep_results = check_dependencies()
        for dep_name, (is_ok, detail) in dep_results.items():
            if is_ok:
                st.success(f"**{dep_name}**: {detail}")
            else:
                st.error(f"**{dep_name}**: {detail}")

    with col_plan:
        st.subheader("📐 Implementation Roadmap")
        modules_status = [
            ("Module 1", "Project Setup & Architecture", "Complete (Phase 0)", True),
            ("Module 2", "Virtual Sensor Engine & Scenarios", "Complete (Phase 1)", True),
            ("Module 3", "System Overview & IoT Topology", "Complete (Phase 2)", True),
            ("Module 4", "Sensor Monitoring & Incident Detection", "Complete (Phase 3)", True),
            ("Module 5", "MQTT Alert Transmission & Delivery Verification", "Complete (Phase 4)", True),
            ("Module 6", "Six-Policy Comparative Analysis", "Planned (Phase 6)", False),
            ("Module 7", "DQN Architecture & Training", "Planned (Phase 7)", False),
            ("Module 8", "DQN Controller Integration", "Planned (Phase 8)", False),
            ("Module 9", "Interactive Simulation Runner", "Planned (Phase 9)", False),
            ("Module 10", "Experiment Data Storage (SQLite)", "Planned (Phase 10)", False),
            ("Module 11", "Performance Analysis & CSV Export", "Planned (Phase 11)", False),
            ("Module 12", "End-to-End Test Suite", "Active (62+ Automated Tests)", True),
        ]
        for mod_id, name, phase_str, is_done in modules_status:
            icon = "🟢" if is_done else "⚪"
            st.markdown(f"- {icon} **{mod_id}: {name}** — *{phase_str}*")

# ------------------------------------------------------------------------------
# Continuous Auto-Run Execution Loop
# ------------------------------------------------------------------------------
if is_auto_running:
    manager.resume()
    manager.step()
    time.sleep(0.3)
    st.rerun()
