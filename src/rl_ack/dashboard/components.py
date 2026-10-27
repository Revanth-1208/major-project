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
            key="policy_eval_mode",
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
            workload_size = st.slider("Workload Size (Messages)", min_value=10, max_value=100, value=25, step=5, key="policy_eval_workload_size")
        with col_c2:
            seed_val = st.number_input("Deterministic Random Seed", min_value=1, max_value=999999, value=manager.random_seed, step=1, key="policy_eval_random_seed")
        with col_c3:
            loss_rate = st.slider("Channel Packet Loss Rate", min_value=0.0, max_value=0.35, value=0.10, step=0.05, format="%.2f", key="policy_eval_loss_rate")
        with col_c4:
            base_lat = st.slider("Base Latency (ms)", min_value=5.0, max_value=60.0, value=15.0, step=5.0, key="policy_eval_base_lat")

        with st.expander("🛠️ Advanced Network & Protocol Thresholds", expanded=False):
            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            with col_a1:
                jitter_val = st.slider("Stochastic Jitter (ms)", min_value=1.0, max_value=20.0, value=5.0, step=1.0, key="policy_eval_jitter")
            with col_a2:
                ack_loss_rate = st.slider("Reverse ACK Loss Rate", min_value=0.0, max_value=0.25, value=0.05, step=0.05, format="%.2f", key="policy_eval_ack_loss_rate")
            with col_a3:
                ack_timeout = st.slider("ACK Timeout (ms)", min_value=50.0, max_value=500.0, value=150.0, step=25.0, key="policy_eval_ack_timeout")
            with col_a4:
                max_retries = st.slider("Max Retransmissions", min_value=1, max_value=5, value=3, step=1, key="policy_eval_max_retries")
            deadline_val = st.slider("Delivery Latency Deadline (ms)", min_value=50.0, max_value=300.0, value=100.0, step=25.0, key="policy_eval_deadline")
    else:
        st.markdown("##### 🔌 Live MQTT Evaluation Configuration")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            live_msg_count = st.slider("Live Test Messages", min_value=3, max_value=15, value=5, step=1, key="policy_eval_live_msg_count")
        with col_l2:
            is_broker_ready = manager.mqtt_manager.is_connected
            status_color = "🟢" if is_broker_ready else "🔴"
            st.markdown(f"**Live Broker Link:** {status_color} `{manager.mqtt_manager.broker_status}`")
            if not is_broker_ready:
                st.warning("⚠️ Connect clients in the '📡 MQTT Alert Delivery' tab before running live comparisons.")

    # Execution Action Button
    if st.button("🚀 Run Comparative Policy Evaluation", width="stretch", key="policy_eval_btn_run"):
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
        selected_pol = st.selectbox("Filter by Policy", options=["All Policies"] + policies, key="policy_eval_filter_pol")
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
                key="policy_eval_btn_download_summary",
            )
        with col_exp2:
            csv_results = manager.export_policy_results_csv()
            st.download_button(
                label="📄 Download Granular Audit Trail CSV",
                data=csv_results,
                file_name="rlack_policy_message_results.csv",
                mime="text/csv",
                width="stretch",
                key="policy_eval_btn_download_results",
            )
    else:
        st.info(
            "💡 **No Evaluation Run Yet:** Select evaluation mode above and click "
            "**🚀 Run Comparative Policy Evaluation** to compare all five acknowledgment strategies "
            "under identical conditions."
        )


def render_dqn_tab(manager: Any) -> None:
    """Render Phase 6 Deep Q-Network Training, Model Checkpointing, and Comparative Evaluation."""
    st.subheader("🧠 Deep Q-Network (DQN) Training & Comparative Evaluation")
    st.caption("Train a multi-objective reinforcement learning agent and evaluate its adaptive ACK decisions against Phase 5 baselines.")

    # Architectural Highlights Banner
    st.markdown(
        """
        <div style="background-color: #0F172A; border-left: 4px solid #38BDF8; padding: 12px 18px; border-radius: 6px; margin-bottom: 20px;">
            <b style="color: #38BDF8;">📐 Phase 6 DQN Architecture Specifications:</b><br/>
            <span style="color: #94A3B8; font-size: 13px;">
            • <b>8-Feature State Space:</b> [Network Utilization, Rolling ACK Rate, Queue Occupancy, is_CC, is_HBS, is_BET, Node Energy, Rolling Latency]<br/>
            • <b>Deep Q-Network:</b> Fully Connected <code>Linear(8, 64) &rarr; ReLU &rarr; Linear(64, 32) &rarr; ReLU &rarr; Linear(32, 5)</code><br/>
            • <b>5 Discrete Actions:</b> 0: QoS 0 | 1: Local ACK | 2: QoS 1 Retry | 3: Selective Retry | 4: QoS 2<br/>
            • <b>Multi-Objective Optimization:</b> Huber loss, experience replay buffer, epsilon-greedy decay, and target network synchronization.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_train, tab_eval, tab_ckpt = st.tabs([
        "🏋️ Agent Training Loop",
        "⚖️ Comparative Evaluation vs Baselines",
        "💾 Checkpoint Manager",
    ])

    # --------------------------------------------------------------------------
    # Sub-Tab 1: Training Loop
    # --------------------------------------------------------------------------
    with tab_train:
        st.markdown("##### ⚙️ Hyperparameters & Training Controls")

        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            episodes_val = st.slider("Training Episodes", min_value=3, max_value=60, value=15, step=1, key="dqn_train_episodes")
            steps_val = st.slider("Steps per Episode", min_value=10, max_value=60, value=30, step=5, key="dqn_train_steps")
            seed_val = st.number_input("Deterministic Random Seed", min_value=1, max_value=999999, value=manager.random_seed, step=1, key="dqn_training_seed", help="Random seed for deterministic training replay.")
        with col_h2:
            lr_val = st.selectbox("Learning Rate", options=[0.0005, 0.001, 0.002, 0.005], index=1, key="dqn_train_lr")
            gamma_val = st.slider("Discount Factor (γ)", min_value=0.80, max_value=0.99, value=0.95, step=0.01, key="dqn_train_gamma")
            batch_val = st.selectbox("Mini-Batch Size", options=[16, 32, 64], index=1, key="dqn_train_batch_size")
        with col_h3:
            eps_start = st.slider("Epsilon Start", min_value=0.5, max_value=1.0, value=1.0, step=0.05, key="dqn_train_eps_start")
            eps_min = st.slider("Epsilon Floor", min_value=0.01, max_value=0.20, value=0.05, step=0.01, key="dqn_train_eps_min")
            eps_decay = st.slider("Epsilon Decay / Step", min_value=0.980, max_value=0.999, value=0.995, step=0.001, key="dqn_train_eps_decay")

        with st.expander("⚖️ Multi-Objective Reward Weights & Penalties", expanded=False):
            st.caption("Configurable terms balancing reliability, latency compliance, packet bloat, and battery depletion.")
            r_c1, r_c2, r_c3 = st.columns(3)
            with r_c1:
                w_del = st.number_input("Delivery Confirmation Reward", value=10.0, step=1.0, key="dqn_rew_w_del")
                w_cc_bonus = st.number_input("Critical Control (CC) Bonus", value=15.0, step=1.0, key="dqn_rew_w_cc_bonus")
                w_fail = st.number_input("Delivery Failure Penalty", value=15.0, step=1.0, key="dqn_rew_w_fail")
            with r_c2:
                w_cc_fail = st.number_input("CC Lost Alert Penalty", value=25.0, step=1.0, key="dqn_rew_w_cc_fail")
                w_ddl_met = st.number_input("Deadline Compliance Bonus", value=5.0, step=1.0, key="dqn_rew_w_ddl_met")
                w_ddl_viol = st.number_input("Deadline Violation Penalty", value=10.0, step=1.0, key="dqn_rew_w_ddl_viol")
            with r_c3:
                w_lat = st.number_input("Latency Scale Penalty", value=3.0, step=0.5, key="dqn_rew_w_lat")
                w_overhead = st.number_input("Overhead Penalty / 100B", value=1.0, step=0.2, key="dqn_rew_w_overhead")
                w_retry = st.number_input("Retransmission Churn Penalty", value=2.0, step=0.5, key="dqn_rew_w_retry")

        # Trigger Training Button
        btn_disabled = manager.is_dqn_training
        if st.button("🚀 Start DQN Training Loop", width="stretch", disabled=btn_disabled, key="dqn_btn_start_training"):
            from rl_ack.dqn.reward import RewardConfig
            reward_cfg = RewardConfig(
                w_delivery_success=w_del,
                w_cc_delivery_bonus=w_cc_bonus,
                w_delivery_failure=w_fail,
                w_cc_failure_penalty=w_cc_fail,
                w_deadline_met=w_ddl_met,
                w_deadline_violation=w_ddl_viol,
                w_latency_scale=w_lat,
                w_overhead_per_100b=w_overhead,
                w_retry_penalty=w_retry,
            )

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_train_progress(ep: int, total_eps: int, m: Any):
                pct = ep / total_eps
                loss_disp = f"{m.avg_loss:.4f}" if m.avg_loss is not None else "N/A"
                progress_bar.progress(pct)
                status_text.text(
                    f"Episode {ep}/{total_eps} | Reward: {m.total_reward:+.1f} | "
                    f"Delivery: {m.delivery_rate * 100:.1f}% | Loss: {loss_disp} | ε: {m.epsilon:.3f}"
                )

            manager.train_dqn(
                num_episodes=episodes_val,
                steps_per_episode=steps_val,
                learning_rate=lr_val,
                gamma=gamma_val,
                batch_size=batch_val,
                epsilon_start=eps_start,
                epsilon_min=eps_min,
                epsilon_decay=eps_decay,
                reward_config=reward_cfg,
                progress_callback=update_train_progress,
            )

            progress_bar.progress(1.0)
            status_text.text("✅ Training Complete! Checkpoints saved to checkpoints/dqn_latest.pt and dqn_best.pt")
            st.rerun()

        # Display Training Metrics & Curves if available
        df_train = manager.get_dqn_training_history_df()
        if not df_train.empty:
            st.markdown("---")
            st.subheader("📈 Training Performance Curves")

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                # Cumulative Episode Reward Curve
                fig_rew = px.line(
                    df_train,
                    x="episode",
                    y="total_reward",
                    title="Episode Reward Progression",
                    markers=True,
                    labels={"episode": "Episode", "total_reward": "Total Reward"},
                )
                fig_rew.update_traces(line_color="#10B981")
                st.plotly_chart(fig_rew, width="stretch")

            with t_col2:
                # Bellman Loss Curve
                fig_loss = px.line(
                    df_train[df_train["avg_loss"].notnull()],
                    x="episode",
                    y="avg_loss",
                    title="Average Bellman Huber Loss",
                    markers=True,
                    labels={"episode": "Episode", "avg_loss": "Huber Loss"},
                )
                fig_loss.update_traces(line_color="#F59E0B")
                st.plotly_chart(fig_loss, width="stretch")

            t_col3, t_col4 = st.columns(2)
            with t_col3:
                # Delivery & Deadline Compliance
                fig_rates = go.Figure()
                fig_rates.add_trace(go.Scatter(
                    x=df_train["episode"],
                    y=df_train["delivery_rate"] * 100.0,
                    mode="lines+markers",
                    name="Delivery Rate (%)",
                    line=dict(color="#38BDF8"),
                ))
                fig_rates.add_trace(go.Scatter(
                    x=df_train["episode"],
                    y=df_train["cc_delivery_rate"] * 100.0,
                    mode="lines+markers",
                    name="CC Emergency Delivery Rate (%)",
                    line=dict(color="#EF4444", dash="dash"),
                ))
                fig_rates.add_trace(go.Scatter(
                    x=df_train["episode"],
                    y=df_train["deadline_compliance_rate"] * 100.0,
                    mode="lines+markers",
                    name="Deadline Compliance (%)",
                    line=dict(color="#10B981"),
                ))
                fig_rates.update_layout(
                    title="Reliability & Emergency Deadline Compliance",
                    xaxis_title="Episode",
                    yaxis_title="Percentage (%)",
                    yaxis_range=[0, 105],
                )
                st.plotly_chart(fig_rates, width="stretch")

            with t_col4:
                # Action Selection Distribution per Episode
                fig_act = go.Figure()
                fig_act.add_trace(go.Bar(name="0: QoS 0", x=df_train["episode"], y=df_train["action_0_qos0"], marker_color="#94A3B8"))
                fig_act.add_trace(go.Bar(name="1: Local ACK", x=df_train["episode"], y=df_train["action_1_local_ack"], marker_color="#F59E0B"))
                fig_act.add_trace(go.Bar(name="2: QoS 1 Retry", x=df_train["episode"], y=df_train["action_2_qos1"], marker_color="#3B82F6"))
                fig_act.add_trace(go.Bar(name="3: Selective Retry", x=df_train["episode"], y=df_train["action_3_selective"], marker_color="#10B981"))
                fig_act.add_trace(go.Bar(name="4: QoS 2", x=df_train["episode"], y=df_train["action_4_qos2"], marker_color="#8B5CF6"))
                fig_act.update_layout(
                    title="Action Selection Distribution Across Episodes",
                    xaxis_title="Episode",
                    yaxis_title="Action Counts",
                    barmode="stack",
                )
                st.plotly_chart(fig_act, width="stretch")

            # Training History Table & CSV Export
            st.markdown("##### 📜 Recorded Training History Table")
            st.dataframe(df_train, width="stretch", hide_index=True)

            csv_train = manager.export_dqn_training_csv()
            st.download_button(
                label="📄 Download Training Metrics CSV",
                data=csv_train,
                file_name="rlack_dqn_training_metrics.csv",
                mime="text/csv",
                width="stretch",
                key="dqn_btn_download_training_csv",
            )
        else:
            st.info("💡 **No Training Run Yet:** Configure hyperparameters above and click **🚀 Start DQN Training Loop** to train the agent.")

    # --------------------------------------------------------------------------
    # Sub-Tab 2: Comparative Evaluation vs Baselines
    # --------------------------------------------------------------------------
    with tab_eval:
        st.markdown("##### ⚖️ Benchmark DQN Agent Against 5 Fixed Policies")
        st.caption("Evaluates the trained DQN in strictly greedy mode (no exploration) and contrasts it with all 5 Phase 5 baselines under identical channel conditions.")

        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            eval_samples = st.slider("Test Workload Size (Messages)", min_value=15, max_value=80, value=40, step=5, key="dqn_eval_samples")
        with e_col2:
            eval_loss = st.slider("Channel Packet Loss Rate", min_value=0.0, max_value=0.30, value=0.12, step=0.02, key="dqn_eval_packet_loss_rate")
        with e_col3:
            eval_seed = st.number_input("Evaluation Random Seed", min_value=1, max_value=999999, value=manager.random_seed + 950, step=1, key="dqn_eval_seed")

        if st.button("🚀 Run Comparative Benchmark", width="stretch", key="dqn_btn_run_benchmark"):
            p_bar = st.progress(0.0)
            s_text = st.empty()

            def update_eval_progress(pct: float, msg: str):
                p_bar.progress(pct)
                s_text.text(msg)

            df_eval = manager.run_dqn_evaluation(
                workload_count=eval_samples,
                seed=int(eval_seed),
                loss_rate=eval_loss,
                latency_ms=20.0,
                progress_callback=update_eval_progress,
            )

            p_bar.progress(1.0)
            s_text.text("✅ Comparative Benchmark Complete!")
            st.rerun()

        # Display Comparative Benchmark Results if available
        df_eval_summary = manager.get_dqn_evaluation_summary_df()
        if not df_eval_summary.empty:
            st.markdown("---")
            st.subheader("📊 Comparative Benchmark Summary")
            st.dataframe(df_eval_summary, width="stretch", hide_index=True)

            # Visual Benchmark Charts
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                fig_comp_rel = go.Figure()
                pols = df_eval_summary["Policy"].tolist()
                fig_comp_rel.add_trace(go.Bar(
                    name="Delivery Confirmed Rate (%)",
                    x=pols,
                    y=df_eval_summary["Delivery Confirmed Rate (%)"],
                    marker_color=["#94A3B8", "#F59E0B", "#3B82F6", "#10B981", "#8B5CF6", "#06B6D4"],
                ))
                fig_comp_rel.add_trace(go.Scatter(
                    name="CC Emergency Delivery Rate (%)",
                    x=pols,
                    y=df_eval_summary["CC Delivery Rate (%)"],
                    mode="lines+markers",
                    marker=dict(size=10, color="#EF4444"),
                    line=dict(color="#EF4444", width=3),
                ))
                fig_comp_rel.update_layout(
                    title="Delivery Reliability & Critical Emergency Performance",
                    yaxis_title="Percentage (%)",
                    yaxis_range=[0, 105],
                )
                st.plotly_chart(fig_comp_rel, width="stretch")

            with b_col2:
                fig_comp_lat = go.Figure()
                fig_comp_lat.add_trace(go.Bar(
                    name="Mean Latency (ms)",
                    x=pols,
                    y=df_eval_summary["Mean E2E Latency (ms)"],
                    marker_color="#38BDF8",
                ))
                fig_comp_lat.add_trace(go.Bar(
                    name="P95 Latency (ms)",
                    x=pols,
                    y=df_eval_summary["P95 E2E Latency (ms)"],
                    marker_color="#F97316",
                ))
                fig_comp_lat.update_layout(
                    title="Mean vs P95 Tail Latency Comparison",
                    yaxis_title="Milliseconds (ms)",
                    barmode="group",
                )
                st.plotly_chart(fig_comp_lat, width="stretch")

            # Action Distribution Breakdown by Traffic Class for DQN
            if manager.last_dqn_eval_report and manager.last_dqn_eval_report.dqn_action_distribution_by_class:
                st.markdown("##### 🎯 Adaptive DQN Action Decisions by Traffic Class")
                st.caption("Demonstrates how the trained DQN agent differentiates policies based on message priority (CC vs HBS vs BET).")

                dist_data = []
                for tc, act_counts in manager.last_dqn_eval_report.dqn_action_distribution_by_class.items():
                    row = {"Traffic Class": tc}
                    row.update(act_counts)
                    dist_data.append(row)
                df_dist = pd.DataFrame(dist_data)
                st.dataframe(df_dist, width="stretch", hide_index=True)

            csv_eval = manager.export_dqn_evaluation_csv()
            st.download_button(
                label="📄 Download Comparative Benchmark CSV",
                data=csv_eval,
                file_name="rlack_dqn_comparative_benchmark.csv",
                mime="text/csv",
                width="stretch",
                key="dqn_btn_download_eval_csv",
            )
        else:
            st.info("💡 **No Benchmark Run Yet:** Click **🚀 Run Comparative Benchmark** above to evaluate DQN alongside Phase 5 baselines.")

    # --------------------------------------------------------------------------
    # Sub-Tab 3: Checkpoint Manager
    # --------------------------------------------------------------------------
    with tab_ckpt:
        st.markdown("##### 💾 Checkpoint Repository & Model Weights")
        available_ckpts = manager.get_available_checkpoints()

        if available_ckpts:
            col_ck1, col_ck2 = st.columns([2, 1])
            with col_ck1:
                selected_ckpt = st.selectbox(
                    "Available Checkpoints (.pt)",
                    options=available_ckpts,
                    key="dqn_selected_checkpoint",
                )
            with col_ck2:
                if st.button("📥 Load Selected Checkpoint", width="stretch", key="dqn_btn_load_checkpoint"):
                    manager.load_dqn_checkpoint(selected_ckpt)
                    st.rerun()

            # Persistent Status on Checkpoint Manager Page
            status = getattr(manager, "checkpoint_load_status", None)
            if status is not None:
                if status.get("success"):
                    meta = status.get("metadata") or {}
                    s_dim = meta.get("state_dim", 8)
                    a_dim = meta.get("action_dim", 5)
                    st.success(
                        f"✅ **Model Loaded Successfully!**\n\n"
                        f"- **Checkpoint File:** `{status['filename']}`\n"
                        f"- **Target Device:** `{status['device']}`\n"
                        f"- **Architecture Verified:** State Dim = `{s_dim}`, Action Dim = `{a_dim}` (Matches QNetwork `8 -> 64 -> 32 -> 5`)\n"
                        f"- **Trained Episodes:** `{status.get('episodes_trained', 0)}` | **Total Steps:** `{status.get('total_steps', 0)}` | **Epsilon (ε):** `{status.get('epsilon', 0.0):.4f}`\n"
                        f"- **Status:** Active in both Evaluation and Online ACK Controller."
                    )
                else:
                    st.error(
                        f"❌ **Model Not Loaded:**\n\n"
                        f"- **Target File:** `{status.get('filename', 'Unknown')}`\n"
                        f"- **Error Details:** {status.get('error', 'Incompatible format or architecture.')}\n"
                        f"- **Integrity Guarantee:** Previous model weights were preserved unchanged."
                    )
            else:
                st.info("💡 Select a saved `.pt` checkpoint file above and click **📥 Load Selected Checkpoint** to restore model weights.")

            st.markdown("---")
            st.markdown("##### 📊 Active Agent Parameters")
            st.info(
                f"**Exploration Epsilon (ε):** `{manager.dqn_agent.epsilon:.4f}` | "
                f"**Trained Episodes:** `{manager.dqn_agent.episodes_trained}` | "
                f"**Total Steps:** `{manager.dqn_agent.total_steps}` | "
                f"**Device:** `{manager.dqn_agent.device}`"
            )
        else:
            st.warning("⚠️ No saved checkpoints found in `checkpoints/`. Train the agent in the '🏋️ Agent Training Loop' tab to create checkpoints.")


def render_dqn_decision_monitor_tab(manager: Any) -> None:
    """Render Module 8 DQN Adaptive ACK Decision Monitor tab."""
    st.subheader("🤖 Deep Q-Network (DQN) Adaptive ACK Decision Monitor")
    st.caption(
        "Real-time observation of DQN inference, 8-feature state vector extraction, "
        "and dynamic acknowledgment action selection for incoming emergency alerts."
    )

    controller = manager.dqn_controller

    # 1. Model Readiness & Checkpoint Configuration Banner
    col_stat1, col_stat2 = st.columns([2, 1])

    with col_stat1:
        if controller.model_ready:
            model_name = controller.current_model_path.name if controller.current_model_path else "Active"
            st.success(
                f"🟢 **DQN Model Active & Ready** (`{model_name}`)\n\n"
                f"- **Inference Mode:** Strictly Deterministic Greedy ($\\epsilon = 0.0$)\n"
                f"- **Input State Space:** 8 Continuous Features (Normalized $[0, 1]$)\n"
                f"- **Action Space:** 5 Discrete ACK Strategies (QoS 0, Local ACK, QoS 1 Retry, Selective Retry, QoS 2)\n"
                f"- **Agent Total Steps:** `{controller.agent.total_steps}` | **Episodes Trained:** `{controller.agent.episodes_trained}`"
            )
        else:
            reason = controller.last_error or "No checkpoint loaded."
            st.warning(
                f"⚠️ **Safe Fallback Active** (Policy: `{controller.fallback_policy.value}`)\n\n"
                f"- **Status:** {reason}\n"
                f"- **Non-Suppression Guarantee:** Danger alerts are never suppressed or dropped; operating on deterministic fallback QoS."
            )

    with col_stat2:
        st.markdown("##### ⚙️ Controller Controls")
        available_ckpts = manager.get_available_checkpoints()
        if available_ckpts:
            col_sel, col_load = st.columns([2, 1])
            with col_sel:
                selected_ckpt = st.selectbox(
                    "Model Checkpoint",
                    options=available_ckpts,
                    key="dqn_mon_ckpt_select",
                    label_visibility="collapsed",
                )
            with col_load:
                if st.button("📥 Load", key="dqn_mon_btn_load_ckpt", width="stretch"):
                    ok = manager.load_controller_checkpoint(selected_ckpt)
                    if ok:
                        st.toast(f"Loaded {selected_ckpt} into controller!", icon="🟢")
                    else:
                        st.toast(f"Failed to load {selected_ckpt}", icon="❌")
                    st.rerun()

        col_unload, col_override = st.columns(2)
        with col_unload:
            if st.button("🛑 Unload", key="dqn_mon_btn_unload", width="stretch", help="Unload active model to test deterministic fallback behavior"):
                manager.unload_controller_checkpoint()
                st.toast("Model unloaded. Fallback mode active.", icon="⚠️")
                st.rerun()

        with col_override:
            safety_val = st.toggle(
                "Safety Override",
                value=controller.enable_safety_override,
                key="dqn_mon_toggle_safety",
                help="When active, Critical Control alerts (gas/smoke) cannot be assigned QoS 0; automatically upgraded to QoS 1.",
            )
            if safety_val != controller.enable_safety_override:
                manager.set_controller_safety_override(safety_val)
                st.rerun()

    st.markdown("---")

    # 2. Latest Live Decision Inspection
    latest_decision = manager.get_latest_dqn_decision()

    if latest_decision is None:
        st.info(
            "💡 **No Emergency Alerts Processed Yet:**\n\n"
            "Virtual sensors are currently operating within normal parameters. To trigger emergency alerts:\n"
            "1. Switch scenario to **Kitchen Fire Multi-Event Emergency** or **Gradual Gas Leak** in the sidebar.\n"
            "2. Click **▶️ Start Simulation** or **⏩ Run 10 Steps** in the Live Telemetry tab.\n"
            "3. Return here to observe live DQN Q-values, 8-feature state extraction, and action selection."
        )
    else:
        st.subheader("🎯 Latest Emergency Decision Inspection")
        st.caption(f"Audit record for Decision `{latest_decision.decision_id}` (Generated at t={latest_decision.timestamp_s:.1f}s)")

        # KPI Badges for Latest Decision
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        with c_kpi1:
            st.metric(
                label="🚨 Outgoing Alert",
                value=f"{latest_decision.alert_id}",
                delta=f"Sensor: {latest_decision.sensor_id}",
                delta_color="off",
            )
        with c_kpi2:
            st.metric(
                label="📊 Sensor Value",
                value=f"{latest_decision.reading_value:.1f} {latest_decision.unit}",
                delta=f"Threshold: {latest_decision.threshold_value:.1f} ({latest_decision.severity})",
                delta_color="inverse",
            )
        with c_kpi3:
            st.metric(
                label="🤖 Chosen ACK Action",
                value=f"{latest_decision.action_label}",
                delta=f"Configured MQTT QoS {latest_decision.configured_qos}",
            )
        with c_kpi4:
            if latest_decision.safety_override:
                mode_str = "🛡️ Safety Override Applied"
                mode_delta = "Upgraded to QoS 1"
            elif latest_decision.is_fallback:
                mode_str = "⚠️ Safe Fallback Active"
                mode_delta = "Model Unloaded"
            else:
                mode_str = "🧠 DQN Greedy Inference"
                mode_delta = f"Model: {latest_decision.model_id}"
            st.metric(label="⚙️ Inference Mode", value=mode_str, delta=mode_delta, delta_color="off")

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        col_chart, col_state = st.columns([1, 1])

        with col_chart:
            st.markdown("##### 📈 Q-Value Distribution Across 5 Discrete Actions")
            if latest_decision.q_values is not None:
                action_names = [
                    "0: QoS 0",
                    "1: Local ACK",
                    "2: QoS 1 Retry",
                    "3: Selective Retry",
                    "4: QoS 2",
                ]
                q_vals = latest_decision.q_values
                colors = ["#38BDF8"] * 5
                colors[latest_decision.action_idx] = "#10B981"  # Emerald for chosen action

                fig_q = go.Figure()
                fig_q.add_trace(go.Bar(
                    x=action_names,
                    y=q_vals,
                    marker_color=colors,
                    text=[f"{q:.3f}" for q in q_vals],
                    textposition="auto",
                ))
                fig_q.update_layout(
                    title=f"Q(s, a) Outputs (Selected: Action {latest_decision.action_idx})",
                    xaxis_title="ACK Action Candidate",
                    yaxis_title="Estimated Expected Return Q(s, a)",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_q, width="stretch")
                st.caption("📌 **Note on Q-Values:** Outputs represent expected cumulative multi-objective returns $Q(s, a)$, not probabilities. The action with $\\arg\\max_a Q(s, a)$ is selected.")
            else:
                st.info(
                    f"⚠️ **Q-Values Unavailable:** Decision was produced via deterministic fallback policy "
                    f"(`{latest_decision.policy_name}`) because the DQN checkpoint was unready. "
                    f"No fabricated Q-values are displayed."
                )

        with col_state:
            st.markdown("##### 🧩 Extracted 8-Feature State Vector")
            vec = latest_decision.state_vector
            feature_defs = [
                ("1. Network Utilization", vec[0] if len(vec) > 0 else 0.0, "Channel bandwidth utilization [0, 1]"),
                ("2. Rolling ACK Rate", vec[1] if len(vec) > 1 else 0.0, "Recent delivery confirmation ratio [0, 1]"),
                ("3. Queue Occupancy", vec[2] if len(vec) > 2 else 0.0, "Local transmission buffer fill level [0, 1]"),
                ("4. is_Critical_Control (CC)", vec[3] if len(vec) > 3 else 0.0, "1.0 if Gas/Smoke emergency alert"),
                ("5. is_High_Bandwidth_Streaming (HBS)", vec[4] if len(vec) > 4 else 0.0, "1.0 if Periodic Temp/Humidity reading"),
                ("6. is_Best_Effort (BET)", vec[5] if len(vec) > 5 else 0.0, "1.0 if Non-critical background telemetry"),
                ("7. Remaining Energy", vec[6] if len(vec) > 6 else 0.0, "Simulated device battery reserve [0, 1]"),
                ("8. Rolling Latency", vec[7] if len(vec) > 7 else 0.0, "Normalized end-to-end channel latency [0, 1]"),
            ]
            df_state = pd.DataFrame([
                {"Feature": f[0], "Normalized Value": f"{f[1]:.4f}", "Interpretation": f[2]}
                for f in feature_defs
            ])
            st.dataframe(df_state, width="stretch", hide_index=True, height=320)

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        # Delivery Result for Latest Decision (Phase 7 Section 7)
        st.markdown("##### 📡 Actual Policy Execution & Multi-Stage Delivery Result")
        c_d1, c_d2, c_d3, c_d4, c_d5 = st.columns(5)
        with c_d1:
            st.metric("Policy Invoked", latest_decision.policy_name, f"Action {latest_decision.action_idx}")
        with c_d2:
            stat_badge = "🟢 CONFIRMED" if latest_decision.delivery_confirmed else (
                "🟡 " + latest_decision.delivery_status if latest_decision.delivery_status != "FAILED" else "🔴 FAILED"
            )
            st.metric("Delivery Status", stat_badge, latest_decision.evaluation_mode)
        with c_d3:
            b_ack = "✅ ACKED" if latest_decision.broker_acknowledged else ("N/A (QoS 0)" if latest_decision.configured_qos == 0 else "❌ No ACK")
            st.metric("Broker ACK", b_ack)
        with c_d4:
            s_rec = "✅ RECEIVED" if latest_decision.subscriber_received else "⏳ Pending / Offline"
            st.metric("Subscriber Receipt", s_rec)
        with c_d5:
            lat_str = f"{latest_decision.end_to_end_latency_ms:.1f} ms" if latest_decision.end_to_end_latency_ms is not None else "—"
            st.metric("E2E Latency", lat_str, f"Retries: {latest_decision.retry_count} | Overhead: {latest_decision.overhead_bytes}B")

        if latest_decision.failure_reason:
            st.warning(f"ℹ️ **Transport Diagnostic:** {latest_decision.failure_reason}")

    # Aggregated Live History and Performance Metrics
    kpis = manager.get_dqn_controller_kpis()
    if kpis["total_decisions"] > 0:
        st.markdown("---")
        st.subheader("📊 Live ACK Controller Performance Metrics")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Total Decisions", kpis["total_decisions"], f"Overrides: {kpis['safety_override_count']}")
        with m_col2:
            st.metric("Delivery Confirmed", f"{kpis['delivery_confirmation_rate']}%")
        with m_col3:
            st.metric("Critical Emergency Rate", f"{kpis['critical_delivery_rate']}%")
        with m_col4:
            mean_lat_disp = f"{kpis['mean_e2e_latency_ms']:.1f} ms" if kpis["mean_e2e_latency_ms"] is not None else "—"
            st.metric("Avg E2E Latency", mean_lat_disp)
        with m_col5:
            st.metric("Retransmissions", kpis["total_retries"], f"Overhead: {kpis['total_overhead_bytes']} B")

        c_act_chart, c_ts = st.columns([1, 1])
        with c_act_chart:
            st.markdown("##### 🎯 Action Selection Breakdown")
            counts = kpis["action_counts"]
            act_labels = ["0: QoS 0", "1: Local ACK", "2: QoS 1 Retry", "3: Selective Retry", "4: QoS 2"]
            fig_act = go.Figure()
            fig_act.add_trace(go.Bar(
                x=act_labels,
                y=[counts.get(i, 0) for i in range(5)],
                marker_color=["#94A3B8", "#F59E0B", "#3B82F6", "#10B981", "#8B5CF6"],
                text=[str(counts.get(i, 0)) for i in range(5)],
                textposition="auto",
            ))
            fig_act.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="ACK Action Candidate", yaxis_title="Selection Count")
            st.plotly_chart(fig_act, width="stretch")

        with c_ts:
            st.markdown("##### 📈 Live Sensor Telemetry Time-Series")
            ts_df = manager.get_time_series_df()
            if ts_df is None or ts_df.empty or "simulation_time_s" not in ts_df.columns or "value" not in ts_df.columns:
                st.info("ℹ️ No sensor telemetry recorded yet. Start or step the simulation to collect readings.")
            else:
                available_sensors = sorted(ts_df["sensor_id"].unique().tolist())
                filter_col1, filter_col2 = st.columns([1, 1])
                with filter_col1:
                    selected_sensor = st.selectbox(
                        "Filter Sensor",
                        options=["All Sensors"] + available_sensors,
                        index=0,
                        key="dqn_ts_sensor_filter",
                        label_visibility="collapsed",
                    )
                with filter_col2:
                    st.caption(f"📊 {len(ts_df)} readings recorded")

                filtered_df = ts_df if selected_sensor == "All Sensors" else ts_df[ts_df["sensor_id"] == selected_sensor]

                if filtered_df.empty:
                    st.info(f"ℹ️ No readings recorded for {selected_sensor} yet.")
                else:
                    units = filtered_df["unit"].dropna().unique().tolist()
                    y_label = f"Reading ({units[0]})" if len(units) == 1 else "Sensor Reading"

                    fig_ts = px.line(
                        filtered_df,
                        x="simulation_time_s",
                        y="value",
                        color="sensor_id",
                        markers=True,
                        hover_data=["unit", "status"],
                        labels={
                            "simulation_time_s": "Simulation Time (s)",
                            "value": y_label,
                            "sensor_id": "Sensor",
                        },
                    )
                    fig_ts.update_layout(
                        height=260,
                        margin=dict(l=20, r=20, t=10, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_ts, width="stretch")

    st.markdown("---")

    # 3. Live Decision Audit Stream Table
    st.subheader("📜 Live ACK Decision Stream (Audit Trail)")
    st.caption("Chronological record of every ACK strategy decision evaluated for generated emergency alerts.")

    df_decisions = manager.get_dqn_decisions_df(limit=50)
    if not df_decisions.empty:
        display_cols = [
            "decision_id",
            "timestamp_s",
            "alert_id",
            "sensor_id",
            "severity",
            "traffic_class",
            "action_label",
            "configured_qos",
            "delivery_status",
            "delivery_confirmed",
            "end_to_end_latency_ms",
            "retry_count",
            "overhead_bytes",
            "safety_override",
            "is_fallback",
            "model_id",
        ]
        available_cols = [c for c in display_cols if c in df_decisions.columns]
        st.dataframe(df_decisions[available_cols], width="stretch", hide_index=True, height=280)

        csv_decisions = manager.export_dqn_decisions_csv()
        st.download_button(
            label="📄 Download DQN Decisions CSV",
            data=csv_decisions,
            file_name="rlack_dqn_decisions_audit.csv",
            mime="text/csv",
            width="stretch",
            key="dqn_mon_btn_download_csv",
        )
    else:
        st.caption("No decisions logged in current session.")





