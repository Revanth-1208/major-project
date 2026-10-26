"""Reusable Streamlit and Plotly UI presentation components for Phase 2."""

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rl_ack.common.enums import AlertSeverity, IncidentState, NodeOperationalStatus
from rl_ack.common.exceptions import InvalidStateTransitionError
from rl_ack.topology.models import KitchenIoTTopology


STATUS_BADGES = {
    "NORMAL": ("🟢", "#D1FAE5", "#065F46"),      # Soft green bg, dark green text
    "WARNING": ("🟡", "#FEF3C7", "#92400E"),     # Soft amber bg, dark amber text
    "CRITICAL": ("🔴", "#FEE2E2", "#991B1B"),    # Soft red bg, dark red text
    "UNAVAILABLE": ("⚪", "#F1F5F9", "#475569"), # Soft slate bg, dark slate text
}


def render_header() -> None:
    """Render application header and faculty demonstration banner."""
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%); 
                    padding: 20px 25px; border-radius: 10px; margin-bottom: 20px; color: white;">
            <h1 style="margin: 0; font-size: 26px; font-weight: 700; color: #38BDF8;">
                📡 RL-ACK: Smart Kitchen IoT Simulation Platform
            </h1>
            <p style="margin: 6px 0 0 0; font-size: 14px; color: #94A3B8;">
                Phase 4: Virtual Sensors, IoT Topology, Incident Detection & MQTT Alert Delivery Verification
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(kpis: Dict[str, Any]) -> None:
    """Render top-level KPI cards derived dynamically from engine state."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            label="⏱️ Simulation Time",
            value=f"{kpis['simulation_time_s']:.1f} s",
            delta=kpis["simulation_status"],
            delta_color="normal" if kpis["simulation_status"] == "Running" else "off",
        )

    with col2:
        st.metric(
            label="📦 Total Sensors",
            value=kpis["total_sensors"],
        )

    with col3:
        st.metric(
            label="🟢 Normal State",
            value=kpis["normal_count"],
        )

    with col4:
        st.metric(
            label="🟡 Warning State",
            value=kpis["warning_count"],
        )

    with col5:
        st.metric(
            label="🔴 Critical State",
            value=kpis["critical_count"],
        )

    with col6:
        st.metric(
            label="⚡ Active Events",
            value=kpis["active_events_count"],
        )


def render_sensor_cards(engine_sensors: List[Any], latest_readings: Dict[str, Any]) -> None:
    """Render individual cards for GAS-001, TEMP-001, HUM-001, SMOKE-001."""
    cols = st.columns(4)

    for idx, sensor in enumerate(engine_sensors):
        target_col = cols[idx % 4]
        reading = latest_readings.get(sensor.sensor_id)
        val_str = f"{reading.value:.2f}" if reading else f"{sensor.current_value:.2f}"
        seq_str = str(reading.sequence_number) if reading else str(sensor.sequence_number)
        time_str = f"{reading.simulation_time_s:.1f} s" if reading else "0.0 s"
        status_key = sensor.status.value.upper()

        icon, bg_color, text_color = STATUS_BADGES.get(status_key, STATUS_BADGES["NORMAL"])

        with target_col:
            st.markdown(
                f"""
                <div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; 
                            background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; font-size: 15px; color: #1E293B;">{sensor.sensor_id}</span>
                        <span style="background: {bg_color}; color: {text_color}; font-size: 11px; font-weight: 700; 
                                     padding: 3px 8px; border-radius: 9999px;">
                            {icon} {status_key}
                        </span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span style="font-size: 26px; font-weight: 800; color: #0F172A;">{val_str}</span>
                        <span style="font-size: 14px; font-weight: 600; color: #64748B; margin-left: 4px;">{sensor.unit}</span>
                    </div>
                    <hr style="margin: 8px 0; border: none; border-top: 1px solid #F1F5F9;" />
                    <div style="font-size: 11px; color: #64748B; line-height: 1.6;">
                        <div><b>Type:</b> {sensor.sensor_type.value.capitalize()}</div>
                        <div><b>Sampling Interval:</b> {sensor.sampling_interval_s:.1f} s</div>
                        <div><b>Last Sample Time:</b> {time_str}</div>
                        <div><b>Sequence #:</b> {seq_str}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_preview_charts(
    time_series_df: pd.DataFrame,
    sensor_configs: Optional[Dict[str, Any]] = None,
) -> None:
    """Render multi-panel time-series plots showing real sensor data against simulation time.
    
    Warning and critical threshold reference lines are dynamically derived from configured sensor parameters.
    """
    if time_series_df.empty:
        st.info("ℹ️ Telemetry charts will populate as the simulation clock advances.")
        return

    configs = sensor_configs or {}
    gas_cfg = configs.get("GAS-001")
    temp_cfg = configs.get("TEMP-001")
    hum_cfg = configs.get("HUM-001")
    smoke_cfg = configs.get("SMOKE-001")

    col1, col2 = st.columns(2)

    # 1. Gas concentration (ppm)
    gas_warn = getattr(gas_cfg, "warning_threshold", 100.0) if gas_cfg else 100.0
    gas_crit = getattr(gas_cfg, "critical_threshold", 400.0) if gas_cfg else 400.0
    gas_df = time_series_df[time_series_df["sensor_id"] == "GAS-001"]
    with col1:
        fig_gas = px.line(
            gas_df,
            x="simulation_time_s",
            y="value",
            markers=True,
            title="GAS-001 Concentration (ppm) vs Simulation Time",
            labels={"simulation_time_s": "Simulation Time (s)", "value": "ppm"},
            color_discrete_sequence=["#EF4444"],
        )
        fig_gas.add_hline(y=gas_warn, line_dash="dash", line_color="#F59E0B", annotation_text=f"Warning ({gas_warn:.0f} ppm)")
        fig_gas.add_hline(y=gas_crit, line_dash="dash", line_color="#DC2626", annotation_text=f"Critical ({gas_crit:.0f} ppm)")
        fig_gas.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gas, width="stretch")

    # 2. Temperature (°C)
    temp_warn = getattr(temp_cfg, "warning_threshold", 38.0) if temp_cfg else 38.0
    temp_crit = getattr(temp_cfg, "critical_threshold", 55.0) if temp_cfg else 55.0
    temp_df = time_series_df[time_series_df["sensor_id"] == "TEMP-001"]
    with col2:
        fig_temp = px.line(
            temp_df,
            x="simulation_time_s",
            y="value",
            markers=True,
            title="TEMP-001 Temperature (°C) vs Simulation Time",
            labels={"simulation_time_s": "Simulation Time (s)", "value": "°C"},
            color_discrete_sequence=["#F97316"],
        )
        fig_temp.add_hline(y=temp_warn, line_dash="dash", line_color="#F59E0B", annotation_text=f"Warning ({temp_warn:.0f} °C)")
        fig_temp.add_hline(y=temp_crit, line_dash="dash", line_color="#DC2626", annotation_text=f"Critical ({temp_crit:.0f} °C)")
        fig_temp.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_temp, width="stretch")

    col3, col4 = st.columns(2)

    # 3. Relative Humidity (% RH) - Note 2.0s interval
    hum_warn_low = getattr(hum_cfg, "warning_low", 30.0) if hum_cfg else 30.0
    hum_warn_high = getattr(hum_cfg, "warning_threshold", 70.0) if hum_cfg else 70.0
    hum_crit_low = getattr(hum_cfg, "critical_low", 15.0) if hum_cfg else 15.0
    hum_crit_high = getattr(hum_cfg, "critical_threshold", 85.0) if hum_cfg else 85.0
    hum_df = time_series_df[time_series_df["sensor_id"] == "HUM-001"]
    with col3:
        fig_hum = px.line(
            hum_df,
            x="simulation_time_s",
            y="value",
            markers=True,
            title="HUM-001 Relative Humidity (% RH) vs Simulation Time (2s Interval)",
            labels={"simulation_time_s": "Simulation Time (s)", "value": "% RH"},
            color_discrete_sequence=["#06B6D4"],
        )
        fig_hum.add_hrect(
            y0=hum_warn_low,
            y1=hum_warn_high,
            line_width=0,
            fillcolor="#10B981",
            opacity=0.1,
            annotation_text=f"Normal ({hum_warn_low:.0f}-{hum_warn_high:.0f}%)",
        )
        fig_hum.add_hline(y=hum_crit_low, line_dash="dash", line_color="#DC2626", annotation_text=f"Crit Low ({hum_crit_low:.0f}%)")
        fig_hum.add_hline(y=hum_crit_high, line_dash="dash", line_color="#DC2626", annotation_text=f"Crit High ({hum_crit_high:.0f}%)")
        fig_hum.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hum, width="stretch")

    # 4. Smoke Obscuration (% obs/m)
    smoke_warn = getattr(smoke_cfg, "warning_threshold", 1.0) if smoke_cfg else 1.0
    smoke_crit = getattr(smoke_cfg, "critical_threshold", 3.5) if smoke_cfg else 3.5
    smoke_df = time_series_df[time_series_df["sensor_id"] == "SMOKE-001"]
    with col4:
        fig_smoke = px.line(
            smoke_df,
            x="simulation_time_s",
            y="value",
            markers=True,
            title="SMOKE-001 Optical Obscuration (% obs/m) vs Simulation Time",
            labels={"simulation_time_s": "Simulation Time (s)", "value": "% obs/m"},
            color_discrete_sequence=["#64748B"],
        )
        fig_smoke.add_hline(y=smoke_warn, line_dash="dash", line_color="#F59E0B", annotation_text=f"Warning ({smoke_warn:.1f})")
        fig_smoke.add_hline(y=smoke_crit, line_dash="dash", line_color="#DC2626", annotation_text=f"Critical ({smoke_crit:.1f})")
        fig_smoke.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_smoke, width="stretch")


def render_topology_section(topology: KitchenIoTTopology) -> None:
    """Render interactive network topology graph and component readiness breakdown."""
    st.subheader("🌐 Heterogeneous Smart Kitchen IoT Topology")
    st.caption("Visualizing communication paths from virtual sensors to edge gateway and upstream cloud entities.")

    fig = topology.to_plotly_figure()
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Operational Components (Phase 1 & 2)")
        impl_nodes = topology.get_implemented_nodes()
        for node in impl_nodes:
            sensors_tag = f" — *Sensors: {', '.join(node.associated_sensors)}*" if node.associated_sensors else ""
            st.markdown(
                f"- **{node.name.splitlines()[0]}** (`{node.node_id}`): "
                f"{node.description}{sensors_tag}"
            )

    with col2:
        st.markdown("#### ⚪ Planned Subsystems (Subsequent Phases)")
        plan_nodes = topology.get_planned_nodes()
        for node in plan_nodes:
            st.markdown(
                f"- **{node.name.splitlines()[0]}** (`{node.node_id}`): "
                f"{node.description} *[{node.phase_label}]*"
            )


def render_incident_monitoring_tab(manager: Any) -> None:
    """Render Phase 3 Incident Monitoring, Emergency Detection, and Alert Management interface."""
    st.subheader("🚨 Incident Monitoring & Emergency Alert Center")
    st.caption(
        "Real-time rule-based threshold evaluation, dynamic incident lifecycle tracking, "
        "and duplicate-suppressed alert management."
    )

    # 1. Incident Summary KPI metrics
    kpis = manager.get_incident_kpis()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="🚨 Active Incidents", value=kpis["active"])
    with col2:
        st.metric(label="🔴 Critical Active", value=kpis["critical_active"])
    with col3:
        st.metric(label="🟡 Warning Active", value=kpis["warning_active"])
    with col4:
        st.metric(label="🔵 Acknowledged", value=kpis["acknowledged"])
    with col5:
        st.metric(label="🟢 Recovered / Closed", value=kpis["recovered_closed"])

    st.markdown("---")

    col_active, col_actions = st.columns([2, 1])

    with col_active:
        st.subheader("⚠️ Active Incidents")
        active_incidents = manager.incident_manager.get_active_incidents()
        if not active_incidents:
            st.info("ℹ️ No active incidents. Monitoring simulated sensor readings.")
        else:
            df_active = manager.get_active_incidents_df()
            st.dataframe(df_active, width="stretch", hide_index=True)

    with col_actions:
        st.subheader("👤 Operator Actions")
        if not active_incidents:
            st.caption("No incidents currently requiring operator intervention.")
        else:
            incident_map = {
                f"{inc.incident_id} | {inc.sensor_id} ({inc.severity.value.upper()})": inc.incident_id
                for inc in active_incidents
            }
            selected_label = st.selectbox(
                "Select Incident to Manage",
                options=list(incident_map.keys()),
                key="operator_incident_select",
            )
            selected_inc_id = incident_map[selected_label]
            selected_inc = manager.incident_manager._all_incidents.get(selected_inc_id)

            if selected_inc:
                st.caption(
                    f"**Current State:** `{selected_inc.state.value.upper()}` | "
                    f"**Trigger Time:** `{selected_inc.trigger_time_s:.1f}s`"
                )

                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    ack_disabled = selected_inc.state != IncidentState.ACTIVE
                    if st.button("✅ Acknowledge", disabled=ack_disabled, width="stretch"):
                        try:
                            manager.acknowledge_incident(selected_inc_id)
                            st.success(f"Incident {selected_inc_id} acknowledged.")
                            st.rerun()
                        except InvalidStateTransitionError as e:
                            st.error(str(e))
                with col_a2:
                    close_disabled = selected_inc.state not in (
                        IncidentState.ACKNOWLEDGED,
                        IncidentState.RECOVERED,
                        IncidentState.CLEARED,
                    )
                    if st.button("🔒 Close", disabled=close_disabled, width="stretch"):
                        try:
                            manager.close_incident(selected_inc_id)
                            st.success(f"Incident {selected_inc_id} closed.")
                            st.rerun()
                        except InvalidStateTransitionError as e:
                            st.error(str(e))

        st.caption(
            "📌 **Decoupling Note:** Operator incident acknowledgement is an application-level "
            "workflow action within the monitoring dashboard, distinct from MQTT transport PUBACK."
        )

    st.markdown("---")
    st.subheader("📜 Emergency Alert History (Bounded Retention)")
    st.caption(
        f"Chronological record of generated emergency alerts "
        f"(Buffer Capacity: {manager.incident_manager.max_alerts})."
    )
    df_alerts = manager.get_alert_history_df(limit=50)
    st.dataframe(df_alerts, width="stretch", hide_index=True, height=260)


def render_mqtt_delivery_tab(manager: Any) -> None:
    """Render Phase 4 MQTT Integration, Emergency Alert Transmission and Delivery Verification interface."""
    st.subheader("📡 MQTT Emergency Alert Delivery & Verification Center")
    st.caption(
        "Real-time MQTT transport monitoring separating protocol-level broker PUBACK "
        "from independent subscriber receipts and application delivery confirmations."
    )

    # Automatically attempt initial connection on first visit if broker is listening
    if "mqtt_auto_attempted" not in st.session_state:
        st.session_state.mqtt_auto_attempted = True
        if not manager.mqtt_manager.is_connected:
            manager.connect_mqtt(timeout_s=0.5)

    metrics = manager.get_mqtt_metrics()
    cfg = manager.mqtt_manager.config
    pub_connected = manager.mqtt_manager.publisher.is_connected
    sub_connected = manager.mqtt_manager.subscriber.is_connected
    sub_subscribed = manager.mqtt_manager.subscriber.is_subscribed
    is_connected = manager.mqtt_manager.is_connected

    # Connection Status Banner & Config Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])

    with col_ctrl1:
        st.markdown(f"**Target Broker:** `{cfg.host}:{cfg.port}` | **Alert Topic:** `{cfg.alert_topic}`")
        st.markdown(f"**Ack Topic:** `{cfg.ack_topic}` | **Clean Session:** `{cfg.clean_session}`")

    with col_ctrl2:
        new_qos = st.selectbox(
            "MQTT Alert QoS",
            options=[0, 1, 2],
            index=manager.mqtt_qos,
            key="mqtt_qos_select",
            help="Select transport protocol QoS level (0: At-most-once, 1: At-least-once, 2: Exactly-once).",
        )
        if new_qos != manager.mqtt_qos:
            manager.set_mqtt_qos(new_qos)
            st.rerun()

    with col_ctrl3:
        if not (pub_connected and sub_connected):
            if st.button("🔌 Connect Clients", width="stretch", help="Initiate MQTT connection for publisher and subscriber"):
                with st.spinner("Connecting to MQTT broker..."):
                    ok = manager.connect_mqtt(timeout_s=1.0)
                if ok:
                    st.toast("Connected to MQTT Broker!", icon="🟢")
                else:
                    err = manager.mqtt_manager.connection_error or "Connection timed out"
                    st.toast(f"Connection failed: {err}", icon="⚠️")
                st.rerun()
        else:
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("⚡ Test Publish", width="stretch", help="Publish a diagnostic test alert to verify delivery"):
                    test_alert = manager.trigger_test_mqtt_alert()
                    if test_alert:
                        st.toast(f"Published test alert {test_alert.alert_id}!", icon="📤")
                    else:
                        st.toast("Failed to publish test alert", icon="⚠️")
                    st.rerun()
            with col_act2:
                if st.button("🛑 Disconnect", width="stretch", help="Gracefully disconnect MQTT clients"):
                    manager.disconnect_mqtt()
                    st.toast("Disconnected from MQTT Broker.", icon="ℹ️")
                    st.rerun()

    # Client Status Indicators driven by actual MQTT callbacks
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        b_color = "🟢" if is_connected else ("🟡" if (pub_connected or sub_connected) else "🔴")
        st.markdown(f"**Broker Link:** {b_color} `{metrics['broker_status']}`")
    with col_s2:
        p_icon = "🟢" if pub_connected else "🔴"
        p_extra = " *(ACK Listener)*" if manager.mqtt_manager.publisher.is_subscribed else ""
        st.markdown(f"**Publisher Client:** {p_icon} `{cfg.publisher_client_id}`{p_extra}")
    with col_s3:
        s_icon = "🟢" if sub_connected else "🔴"
        s_extra = " *(Subscribed)*" if sub_subscribed else (" *(Subscribing...)*" if sub_connected else "")
        st.markdown(f"**Subscriber Client:** {s_icon} `{cfg.subscriber_client_id}`{s_extra}")

    if not is_connected:
        st.warning(
            "⚠️ **Broker Connection Offline:** Messages will not traverse MQTT until the broker is running.\n\n"
            "• Ensure Eclipse Mosquitto is running on `127.0.0.1:1883`.\n"
            "• To start Mosquitto locally on Windows PowerShell: `& 'C:\\Program Files\\mosquitto\\mosquitto.exe' -v`\n"
            "• Click **🔌 Connect Clients** above once Mosquitto is running."
        )

    st.markdown("---")

    # Delivery KPI Metrics
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    with col_m1:
        st.metric(label="📤 Published", value=metrics["total_published"])
    with col_m2:
        st.metric(label="⚡ Broker Acked", value=metrics["broker_acknowledged"])
    with col_m3:
        st.metric(label="📥 Sub Received", value=metrics["subscriber_received"])
    with col_m4:
        st.metric(label="✅ Confirmed", value=metrics["confirmed_deliveries"])
    with col_m5:
        st.metric(label="⏳ Pending", value=metrics["pending_receipts"])
    with col_m6:
        lat_val = (
            f"{metrics['avg_e2e_latency_ms']:.1f} ms"
            if metrics["avg_e2e_latency_ms"] is not None
            else "—"
        )
        st.metric(label="⏱️ Avg E2E Latency", value=lat_val)

    st.markdown("---")

    # Delivery Verification Audit Table
    st.subheader("📋 Multi-Stage Delivery Verification Log")
    st.caption("Verifies alert progression through the 5 discrete delivery verification stages.")
    df_delivery = manager.get_mqtt_delivery_df()
    st.dataframe(df_delivery, width="stretch", hide_index=True, height=260)

    st.info(
        "📌 **Verification Pipeline Definition:**\n\n"
        "1. **GENERATED**: Created by IncidentManager.\n"
        "2. **PUBLISH_PENDING**: Alert queued for publication over MQTT.\n"
        "3. **PUBLISHED**: Confirmed at transport layer by MQTT broker (PUBACK for QoS 1/2).\n"
        "4. **RECEIVED**: Independently received and validated by separate subscriber.\n"
        "5. **DELIVERY_CONFIRMED**: Application-level receipt ACK returned on `rlack/kitchen/alerts/ack`."
    )


def render_policy_comparison_tab(manager: Any) -> None:
    """Render Phase 5 ACK Policy Engine and Comparative Evaluation Center."""
    st.subheader("⚖️ ACK Policy Engine & Comparative Evaluation Center")
    st.caption(
        "Empirical benchmark evaluating five fixed acknowledgment strategies under identical traffic "
        "workloads and channel conditions before adaptive DQN controller implementation."
    )

    # Evaluation Mode & Parameter Controls
    col_mode, col_info = st.columns([1, 2])
    with col_mode:
        eval_mode = st.radio(
            "Evaluation Environment",
            options=["Controlled Simulation (Deterministic)", "Live MQTT Broker (Mosquitto)"],
            index=0,
            help="Controlled Simulation enforces identical packet drop & delay events; Live MQTT tests physical sockets.",
        )
    with col_info:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px 16px; border-radius: 8px; font-size: 13px;">
                <b>Five Evaluated Strategies:</b><br/>
                • <b>Policy A (Static QoS 0):</b> Fire-and-forget, zero retries, minimal overhead.<br/>
                • <b>Policy B (Static Local ACK):</b> Proximate edge proxy ACK, decouples sender without E2E proof.<br/>
                • <b>Policy C (Static QoS 1 + Retries):</b> Broker PUBACK + application-level bounded retries on timeout.<br/>
                • <b>Policy D (Local ACK + Selective Retransmit):</b> Immediate edge ACK + retries reserved for Critical Control (CC).<br/>
                • <b>Policy E (Static QoS 2):</b> 4-way protocol handshake (PUBLISH &rarr; PUBREC &rarr; PUBREL &rarr; PUBCOMP).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    is_controlled = "Controlled" in eval_mode

    if is_controlled:
        st.markdown("##### 🎛️ Controlled Simulation Parameters")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            workload_size = st.slider("Workload Size (Messages)", min_value=10, max_value=100, value=25, step=5)
        with col_c2:
            seed_val = st.number_input("Deterministic Random Seed", min_value=1, max_value=999999, value=42, step=1)
        with col_c3:
            loss_rate = st.slider("Channel Packet Loss Rate", min_value=0.0, max_value=0.35, value=0.10, step=0.05, format="%.2f")
        with col_c4:
            base_lat = st.slider("Base Latency (ms)", min_value=5.0, max_value=60.0, value=15.0, step=5.0)

        with st.expander("🛠️ Advanced Network & Protocol Thresholds", expanded=False):
            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            with col_a1:
                jitter_val = st.slider("Stochastic Jitter (ms)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
            with col_a2:
                ack_loss_rate = st.slider("Reverse ACK Loss Rate", min_value=0.0, max_value=0.25, value=0.05, step=0.05, format="%.2f")
            with col_a3:
                ack_timeout = st.slider("ACK Timeout (ms)", min_value=50.0, max_value=500.0, value=150.0, step=25.0)
            with col_a4:
                max_retries = st.slider("Max Retransmissions", min_value=1, max_value=5, value=3, step=1)
            deadline_val = st.slider("Delivery Latency Deadline (ms)", min_value=50.0, max_value=300.0, value=100.0, step=25.0)
    else:
        st.markdown("##### 🔌 Live MQTT Evaluation Configuration")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            live_msg_count = st.slider("Live Test Messages", min_value=3, max_value=15, value=5, step=1)
        with col_l2:
            is_broker_ready = manager.mqtt_manager.is_connected
            status_color = "🟢" if is_broker_ready else "🔴"
            st.markdown(f"**Live Broker Link:** {status_color} `{manager.mqtt_manager.broker_status}`")
            if not is_broker_ready:
                st.warning("⚠️ Connect clients in the '📡 MQTT Alert Delivery' tab before running live comparisons.")

    # Execution Action Button
    if st.button("🚀 Run Comparative Policy Evaluation", width="stretch"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, msg: str):
            progress_bar.progress(pct)
            status_text.text(msg)

        if is_controlled:
            df_summary = manager.run_policy_comparison_controlled(
                workload_size=workload_size,
                seed=int(seed_val),
                packet_loss_rate=loss_rate,
                base_latency_ms=base_lat,
                jitter_ms=jitter_val,
                ack_loss_rate=ack_loss_rate,
                ack_timeout_ms=ack_timeout,
                max_retries=max_retries,
                deadline_ms=deadline_val,
                progress_callback=update_progress,
            )
        else:
            if not manager.mqtt_manager.is_connected:
                st.error("Cannot run live evaluation: MQTT broker is offline.")
                return
            df_summary = manager.run_policy_comparison_live(
                message_count=live_msg_count,
                progress_callback=update_progress,
            )

        progress_bar.progress(1.0)
        status_text.text("✅ Comparative Evaluation Complete!")
        st.session_state.has_run_policy_eval = True
        st.rerun()

    st.markdown("---")

    # Present Results if available
    df_summary = manager.get_policy_summary_df()
    df_results = manager.get_policy_results_df()

    if not df_summary.empty:
        st.subheader("📊 Comparative Policy Performance Summary")
        st.caption("Side-by-side analysis under identical workload traffic and channel conditions.")
        st.dataframe(df_summary, width="stretch", hide_index=True)

        # Comparative Visualization Charts
        st.markdown("##### 📈 Visual Performance Benchmarks")
        tab_c1, tab_c2, tab_c3 = st.tabs(["🎯 Delivery Reliability", "⏱️ End-to-End Latency", "📦 Protocol Overhead & Retries"])

        with tab_c1:
            # Bar chart comparing Delivery Confirmation Rate & Broker/Local ACK rates
            fig_rel = go.Figure()
            policies = df_summary["Policy"].tolist()
            fig_rel.add_trace(go.Bar(
                name="Delivery Confirmed Rate (%)",
                x=policies,
                y=df_summary["Delivery Confirmed Rate (%)"],
                marker_color="#10B981",
            ))
            fig_rel.add_trace(go.Bar(
                name="Subscriber Received Rate (%)",
                x=policies,
                y=df_summary["Subscriber Receipt Rate (%)"],
                marker_color="#3B82F6",
            ))
            fig_rel.add_trace(go.Bar(
                name="Local Proxy ACK Rate (%)",
                x=policies,
                y=df_summary["Local ACK Rate (%)"],
                marker_color="#F59E0B",
            ))
            fig_rel.update_layout(
                barmode="group",
                title="Delivery Reliability & Verification Rates Across Strategies",
                yaxis_title="Rate (%)",
                yaxis=dict(range=[0, 105]),
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_rel, width="stretch")

        with tab_c2:
            # Latency Comparison: Mean, Median, P95
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Bar(
                name="Mean Latency (ms)",
                x=policies,
                y=df_summary["Mean E2E Latency (ms)"],
                marker_color="#6366F1",
            ))
            fig_lat.add_trace(go.Bar(
                name="Median Latency (ms)",
                x=policies,
                y=df_summary["Median E2E Latency (ms)"],
                marker_color="#8B5CF6",
            ))
            fig_lat.add_trace(go.Bar(
                name="P95 Latency (ms)",
                x=policies,
                y=df_summary["P95 E2E Latency (ms)"],
                marker_color="#EC4899",
            ))
            fig_lat.update_layout(
                barmode="group",
                title="End-to-End Delivery Latency Metrics (Confirmed Messages Only)",
                yaxis_title="Latency (ms)",
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_lat, width="stretch")

        with tab_c3:
            # Overhead (Bytes) & Retransmission counts
            fig_ovh = go.Figure()
            fig_ovh.add_trace(go.Bar(
                name="Avg Overhead per Msg (Bytes)",
                x=policies,
                y=df_summary["Avg Overhead / Msg (B)"],
                marker_color="#0EA5E9",
                yaxis="y",
            ))
            fig_ovh.add_trace(go.Scatter(
                name="Retransmission Count",
                x=policies,
                y=df_summary["Retransmissions"],
                mode="lines+markers",
                marker=dict(size=10, color="#EF4444"),
                line=dict(width=3, color="#EF4444"),
                yaxis="y2",
            ))
            fig_ovh.update_layout(
                title="Transmitted Protocol Overhead vs Application Retransmissions",
                yaxis=dict(title="Overhead (Bytes / Msg)"),
                yaxis2=dict(title="Total Retransmissions", overlaying="y", side="right"),
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_ovh, width="stretch")

        # Critical Control (CC) Emergency Performance Section
        st.markdown("---")
        st.subheader("🚨 Critical Control (CC) Emergency Performance")
        st.caption("Evaluates resilience specifically for emergency gas leaks and fire/smoke alerts.")
        cc_cols = ["Policy", "CC Total", "CC Delivery Rate (%)", "CC Mean Latency (ms)", "HBS Delivery Rate (%)", "BET Delivery Rate (%)"]
        st.dataframe(df_summary[cc_cols], width="stretch", hide_index=True)

        # Granular Per-Message Audit Trail
        st.markdown("---")
        st.subheader("📋 Granular Message Evaluation Audit Trail")
        st.caption("Inspect individual message routing, retries, and latency measurements.")
        selected_pol = st.selectbox("Filter by Policy", options=["All Policies"] + policies)
        if selected_pol != "All Policies":
            filtered_df = df_results[df_results["policy_name"] == selected_pol]
        else:
            filtered_df = df_results
        st.dataframe(filtered_df, width="stretch", hide_index=True, height=280)

        # CSV Export Buttons
        st.markdown("---")
        st.subheader("📥 Export Evaluation Data")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_summary = manager.export_policy_summary_csv()
            st.download_button(
                label="📄 Download Summary Metrics CSV",
                data=csv_summary,
                file_name="rlack_policy_comparison_summary.csv",
                mime="text/csv",
                width="stretch",
            )
        with col_exp2:
            csv_results = manager.export_policy_results_csv()
            st.download_button(
                label="📄 Download Granular Audit Trail CSV",
                data=csv_results,
                file_name="rlack_policy_message_results.csv",
                mime="text/csv",
                width="stretch",
            )
    else:
        st.info(
            "💡 **No Evaluation Run Yet:** Select evaluation mode above and click "
            "**🚀 Run Comparative Policy Evaluation** to compare all five acknowledgment strategies "
            "under identical conditions."
        )



