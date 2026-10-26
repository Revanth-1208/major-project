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
                Phase 3: Virtual Sensor Engine, IoT Topology, Incident Detection & Emergency Alert Management
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

