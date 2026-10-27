"""Simulation state manager for the RL-ACK Streamlit dashboard."""

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd
import time
from rl_ack.common.enums import (
    AckAction,
    AlertSeverity,
    EvaluationPolicy,
    NodeOperationalStatus,
    SensorType,
    TrafficClass,
)
from rl_ack.common.models import AlertMessage, SensorReading
from rl_ack.controller.controller import DqnAckController
from rl_ack.controller.decision import DqnDecisionRecord
from rl_ack.monitoring.detector import IncidentDetector
from rl_ack.monitoring.manager import IncidentManager
from rl_ack.mqtt.manager import MqttManager
from rl_ack.config import DQNConfig
from rl_ack.dqn.agent import DQNAgent
from rl_ack.dqn.environment import EnvConfig, SmartKitchenIoTEnv
from rl_ack.dqn.evaluator import DQNEvaluationReport, DQNEvaluator
from rl_ack.dqn.reward import RewardConfig
from rl_ack.dqn.trainer import DQNTrainer, EpisodeMetrics
from rl_ack.policies.base import PolicyConfig
from rl_ack.policies.network_model import NetworkModelConfig
from rl_ack.policies.runner import PolicyExperimentRunner
from rl_ack.policies.workload import WorkloadGenerator
from rl_ack.sensors.sensor_engine import SensorEngine
from rl_ack.sensors.scenarios import (
    SensorScenario,
    scenario_abnormal_temperature_rise,
    scenario_gradual_gas_leak,
    scenario_humidity_change,
    scenario_increasing_smoke,
    scenario_kitchen_fire_multi_event,
    scenario_normal_operation,
)


SCENARIO_BUILDERS = {
    "Normal Operation": scenario_normal_operation,
    "Gradual Gas Leak": scenario_gradual_gas_leak,
    "Abnormal Temperature Rise": scenario_abnormal_temperature_rise,
    "Increasing Smoke Obscuration": scenario_increasing_smoke,
    "Humidity Anomaly (High Steam)": lambda seed=44: scenario_humidity_change(mode="high", seed=seed),
    "Humidity Anomaly (Dry Air)": lambda seed=44: scenario_humidity_change(mode="low", seed=seed),
    "Kitchen Fire Multi-Event Emergency": scenario_kitchen_fire_multi_event,
}


class DashboardSimulationManager:
    """Encapsulates simulation state, scenario execution, and telemetry history for the UI."""

    def __init__(
        self,
        random_seed: int = 42,
        scenario_name: str = "Normal Operation",
        step_size: float = 1.0,
        max_history: int = 200,
    ) -> None:
        self.random_seed = random_seed
        self.scenario_name = scenario_name
        self.step_size = step_size
        self.max_history = max_history
        self.simulation_status = "Stopped"

        self.engine: SensorEngine = SensorEngine.create_default_engine(seed_offset=self.random_seed)
        self.incident_manager = IncidentManager(
            detector=IncidentDetector(sensor_configs=self.get_sensor_configs())
        )
        self.mqtt_manager = MqttManager()
        self.auto_publish_mqtt: bool = True
        self.mqtt_qos: int = 1
        self.policy_runner = PolicyExperimentRunner()
        self.workload_generator = WorkloadGenerator(random_seed=self.random_seed)

        # Module 7: DQN Components
        self.dqn_agent = DQNAgent(random_seed=self.random_seed)
        self.dqn_env = SmartKitchenIoTEnv(
            config=EnvConfig(random_seed=self.random_seed),
            policy_config=self.policy_runner.default_policy_config,
        )
        self.dqn_trainer = DQNTrainer(
            agent=self.dqn_agent,
            env=self.dqn_env,
            checkpoint_dir="checkpoints",
        )
        self.dqn_evaluator = DQNEvaluator(
            agent=self.dqn_agent,
            policy_config=self.policy_runner.default_policy_config,
        )
        self.last_dqn_training_history: List[EpisodeMetrics] = []
        self.last_dqn_eval_report: Optional[DQNEvaluationReport] = None
        self.is_dqn_training: bool = False

        # Module 8: DQN Controller Integration
        self.dqn_controller = DqnAckController(agent=self.dqn_agent)
        self.checkpoint_load_status: Optional[Dict[str, Any]] = None
        # Attempt to load trained checkpoint if available
        for default_ckpt in ["dqn_best.pt", "dqn_latest.pt"]:
            ckpt_p = Path("checkpoints") / default_ckpt
            if ckpt_p.is_file():
                if self.dqn_controller.load_checkpoint(ckpt_p):
                    self.dqn_agent.load_checkpoint(ckpt_p)
                    self.checkpoint_load_status = {
                        "success": True,
                        "filename": default_ckpt,
                        "filepath": str(ckpt_p),
                        "device": str(self.dqn_agent.device),
                        "error": None,
                        "total_steps": self.dqn_agent.total_steps,
                        "episodes_trained": self.dqn_agent.episodes_trained,
                        "epsilon": self.dqn_agent.epsilon,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metadata": self.dqn_controller.checkpoint_metadata,
                    }
                    break

        self.active_scenario: Optional[SensorScenario] = None
        self._history: deque[SensorReading] = deque(maxlen=self.max_history)
        self._applied_event_indices: set[int] = set()

        self._configure_scenario(self.scenario_name)

    def _configure_scenario(self, scenario_name: str) -> None:
        """Bind selected environmental scenario to the simulation manager."""
        self.scenario_name = scenario_name
        builder = SCENARIO_BUILDERS.get(scenario_name, scenario_normal_operation)
        self.active_scenario = builder(seed=self.random_seed)
        self._applied_event_indices.clear()

    def set_random_seed(self, seed: int) -> None:
        """Update random seed and reinitialize engine."""
        self.random_seed = int(seed)
        self.reset()

    def set_scenario(self, scenario_name: str) -> None:
        """Change environmental scenario and reset starting conditions."""
        self._configure_scenario(scenario_name)
        self.reset()

    def pause(self) -> None:
        """Mark simulation as paused (idle between steps)."""
        if self.engine.current_simulation_time > 0:
            self.simulation_status = "Paused"
        else:
            self.simulation_status = "Stopped"

    def resume(self) -> None:
        """Mark simulation as actively running."""
        self.simulation_status = "Running"

    def stop(self) -> None:
        """Stop simulation."""
        self.simulation_status = "Stopped"

    def reset(self) -> None:
        """Clear telemetry history and reset sensor engine to baseline t=0."""
        self.engine.reset()
        self._history.clear()
        self._applied_event_indices.clear()
        self.incident_manager.reset()
        self.incident_manager.detector.sensor_configs = self.get_sensor_configs()
        self.mqtt_manager.reset()
        self.dqn_controller.reset()
        self.simulation_status = "Stopped"

    @property
    def current_time_s(self) -> float:
        """Current monotonic simulation time in seconds."""
        return self.engine.current_simulation_time

    @property
    def readings_history(self) -> List[SensorReading]:
        """Chronological list of all retained readings up to max_history."""
        return list(self._history)

    def get_sensor_configs(self) -> Dict[str, Any]:
        """Map of sensor_id to SensorConfig instances directly from the engine."""
        return {s.sensor_id: getattr(s, "config", None) for s in self.engine.get_sensors()}

    def step(self, dt: Optional[float] = None) -> List[SensorReading]:
        """Advance simulation by dt seconds (default step_size) and process due events."""
        step_dt = self.step_size if dt is None else dt
        current_time = self.engine.current_simulation_time

        # Check and apply scenario events due at or before current_time
        if self.active_scenario:
            for idx, event in enumerate(self.active_scenario.events):
                if idx not in self._applied_event_indices:
                    if current_time >= event.start_time_s:
                        if event.is_active_at(current_time):
                            self.engine.trigger_sensor_event(
                                event.sensor_id,
                                target_value=event.target_value,
                                duration_s=event.duration_s,
                            )
                            self._applied_event_indices.add(idx)

        new_readings = self.engine.advance(step_dt)
        for r in new_readings:
            self._history.append(r)

        new_alerts = self.incident_manager.process_readings(new_readings)
        for alert in new_alerts:
            # Phase 7: DQN Controller Selects Adaptive ACK Strategy & Executes Selected Policy
            decision = self.dqn_controller.select_action(
                alert=alert,
                simulation_time_s=self.engine.current_simulation_time,
            )
            alert.chosen_action = AckAction(decision.action_idx)

            # Apply the selected policy to transport and capture delivery outcome
            self.dqn_controller.apply_policy(
                alert=alert,
                decision=decision,
                mqtt_manager=self.mqtt_manager if self.auto_publish_mqtt else None,
                live_mode=self.auto_publish_mqtt,
            )

        self.mqtt_manager.check_timeouts()

        if self.simulation_status != "Running":
            self.simulation_status = "Paused"
        return new_readings

    def run_steps(self, count: int, dt: Optional[float] = None) -> List[SensorReading]:
        """Execute multiple steps consecutively."""
        all_new: List[SensorReading] = []
        for _ in range(count):
            all_new.extend(self.step(dt))
        if self.simulation_status != "Running":
            self.simulation_status = "Paused"
        return all_new

    def get_kpis(self) -> Dict[str, Any]:
        """Compute live system-level metrics directly from the actual sensor engine."""
        sensors = self.engine.get_sensors()
        statuses = self.engine.get_sensor_statuses()

        normal_cnt = sum(1 for s in statuses.values() if s == NodeOperationalStatus.NORMAL)
        warning_cnt = sum(1 for s in statuses.values() if s == NodeOperationalStatus.WARNING)
        critical_cnt = sum(1 for s in statuses.values() if s == NodeOperationalStatus.CRITICAL)
        unavail_cnt = sum(1 for s in statuses.values() if s == NodeOperationalStatus.UNAVAILABLE)

        return {
            "total_sensors": len(sensors),
            "normal_count": normal_cnt,
            "warning_count": warning_cnt,
            "critical_count": critical_cnt,
            "unavailable_count": unavail_cnt,
            "simulation_time_s": self.engine.current_simulation_time,
            "active_events_count": self.engine.active_event_count,
            "simulation_status": self.simulation_status,
        }

    def get_sensor_inventory(self) -> List[Dict[str, Any]]:
        """Extract sensor inventory table rows derived directly from engine configurations."""
        latest_readings = self.engine.get_latest_readings()
        rows: List[Dict[str, Any]] = []

        for sensor in self.engine.get_sensors():
            reading = latest_readings.get(sensor.sensor_id)
            curr_val = f"{reading.value:.2f}" if reading else f"{sensor.current_value:.2f}"
            seq_num = reading.sequence_number if reading else sensor.sequence_number

            cfg = getattr(sensor, "config", None)
            if cfg:
                if cfg.sensor_type == SensorType.HUMIDITY:
                    warn_str = f"[{cfg.warning_low:.1f}, {cfg.warning_threshold:.1f}]"
                    crit_str = f"[{cfg.critical_low:.1f}, {cfg.critical_threshold:.1f}]"
                else:
                    warn_str = f">= {cfg.warning_threshold:.1f}"
                    crit_str = f">= {cfg.critical_threshold:.1f}"
                host_id = cfg.host_node_id
            else:
                warn_str = "N/A"
                crit_str = f">= {sensor.threshold:.1f}"
                host_id = sensor.host_node_id

            rows.append({
                "Sensor ID": sensor.sensor_id,
                "Sensor Type": sensor.sensor_type.value,
                "Unit": sensor.unit,
                "Current Reading": curr_val,
                "Warning Limit": warn_str,
                "Critical Limit": crit_str,
                "Sampling Interval": f"{sensor.sampling_interval_s:.1f} s",
                "Sequence #": seq_num,
                "Derived Status": sensor.status.value.upper(),
                "Host Node": host_id,
            })

        return rows

    def get_recent_readings_df(self, limit: int = 50) -> pd.DataFrame:
        """Return pandas DataFrame of recent readings (most recent first)."""
        if not self._history:
            return pd.DataFrame(columns=[
                "Sim Time (s)", "Sensor ID", "Value", "Unit", "Status", "Seq #", "Threshold Exceeded"
            ])

        records = []
        for r in reversed(list(self._history)[-limit:]):
            records.append({
                "Sim Time (s)": f"{r.simulation_time_s:.1f}",
                "Sensor ID": r.sensor_id,
                "Value": f"{r.value:.2f}",
                "Unit": r.unit,
                "Status": r.status.value.upper(),
                "Seq #": r.sequence_number,
                "Threshold Exceeded": "YES" if r.is_threshold_exceeded else "NO",
            })
        return pd.DataFrame(records)

    def get_time_series_df(self) -> pd.DataFrame:
        """Return pandas DataFrame formatted for time-series charting."""
        if not self._history:
            return pd.DataFrame(columns=["simulation_time_s", "sensor_id", "value", "unit", "status"])

        data = [{
            "simulation_time_s": r.simulation_time_s,
            "sensor_id": r.sensor_id,
            "value": r.value,
            "unit": r.unit,
            "status": r.status.value,
        } for r in self._history]

        return pd.DataFrame(data)

    def get_incident_kpis(self) -> Dict[str, int]:
        """Return dynamic incident summary counts directly from IncidentManager."""
        return self.incident_manager.get_kpis()

    def get_active_incidents_df(self) -> pd.DataFrame:
        """Return pandas DataFrame of active and acknowledged incidents."""
        records = []
        for inc in self.incident_manager.get_active_incidents():
            records.append({
                "Incident ID": inc.incident_id,
                "Sensor ID": inc.sensor_id,
                "Incident Type": inc.incident_type.replace("_", " ").title(),
                "Severity": inc.severity.value.upper(),
                "Latest Value": f"{inc.last_reading:.2f} {inc.unit}",
                "Trigger Threshold": f"{inc.trigger_threshold:.2f} {inc.unit}",
                "Start Sim Time": f"{inc.trigger_time_s:.1f}s",
                "State": inc.state.value.upper(),
            })
        if not records:
            return pd.DataFrame(columns=[
                "Incident ID", "Sensor ID", "Incident Type", "Severity",
                "Latest Value", "Trigger Threshold", "Start Sim Time", "State"
            ])
        return pd.DataFrame(records)

    def get_alert_history_df(self, limit: int = 50) -> pd.DataFrame:
        """Return pandas DataFrame of recent alert records (most recent first)."""
        alerts = self.incident_manager.get_alerts(limit=limit)
        records = []
        for a in reversed(alerts):
            time_str = a.created_at_utc.split("T")[-1][:8] if "T" in a.created_at_utc else a.created_at_utc
            records.append({
                "Alert ID": a.alert_id,
                "Incident ID": a.incident_id or "N/A",
                "Sensor ID": a.sensor_id,
                "Severity": a.severity.value.upper(),
                "Message": a.message,
                "Sim Time": f"{a.simulation_time_s:.1f}s",
                "Created (UTC)": time_str,
                "Status": "Acknowledged" if a.app_ack_at_utc else a.delivery_status.value.capitalize(),
            })
        if not records:
            return pd.DataFrame(columns=[
                "Alert ID", "Incident ID", "Sensor ID", "Severity",
                "Message", "Sim Time", "Created (UTC)", "Status"
            ])
        return pd.DataFrame(records)

    def acknowledge_incident(self, incident_id: str) -> Any:
        """Operator action to acknowledge an active incident."""
        return self.incident_manager.acknowledge_incident(
            incident_id, simulation_time_s=self.engine.current_simulation_time
        )

    def close_incident(self, incident_id: str) -> Any:
        """Operator action to close a recovered or acknowledged incident."""
        return self.incident_manager.close_incident(
            incident_id, simulation_time_s=self.engine.current_simulation_time
        )

    def get_mqtt_metrics(self) -> Dict[str, Any]:
        """Return MQTT broker and delivery verification statistics."""
        return self.mqtt_manager.get_delivery_metrics()

    def get_mqtt_delivery_df(self) -> pd.DataFrame:
        """Return pandas DataFrame of MQTT delivery verification records."""
        return self.mqtt_manager.get_delivery_df()

    def connect_mqtt(self, timeout_s: Optional[float] = None) -> bool:
        """Attempt to connect MQTT publisher and subscriber to broker."""
        return self.mqtt_manager.connect(timeout_s=timeout_s)

    def disconnect_mqtt(self) -> None:
        """Disconnect MQTT clients from broker."""
        self.mqtt_manager.disconnect()

    def trigger_test_mqtt_alert(self) -> Optional[AlertMessage]:
        """Publish a diagnostic test alert over MQTT to verify delivery verification pipeline."""
        test_alert = AlertMessage(
            alert_id=f"ALT-TEST-{uuid.uuid4().hex[:6].upper()}",
            sensor_id="GAS-001",
            severity=AlertSeverity.WARNING,
            reading_value=120.0,
            threshold_value=100.0,
            unit="ppm",
            incident_type="gas_leak",
            message="Manual MQTT Diagnostic Delivery Verification Test",
            simulation_time_s=self.engine.current_simulation_time,
            priority=TrafficClass.CRITICAL_CONTROL,
        )
        self.mqtt_manager.publish_alert(test_alert, qos=self.mqtt_qos)
        return test_alert

    def set_mqtt_qos(self, qos: int) -> None:
        """Update active MQTT alert publication QoS level."""
        self.mqtt_qos = qos
        self.mqtt_manager.config.default_alert_qos = qos

    def run_policy_comparison_controlled(
        self,
        workload_size: int = 20,
        seed: int = 42,
        packet_loss_rate: float = 0.10,
        base_latency_ms: float = 15.0,
        jitter_ms: float = 5.0,
        ack_loss_rate: float = 0.05,
        ack_timeout_ms: float = 150.0,
        max_retries: int = 3,
        deadline_ms: float = 100.0,
        progress_callback: Any = None,
    ) -> pd.DataFrame:
        """Run Phase 5 controlled evaluation comparing all five ACK policies."""
        # 1. Generate standardized workload
        self.workload_generator.set_seed(seed)
        workload = self.workload_generator.generate_workload(count=workload_size)

        # 2. Configure network and policy limits
        net_cfg = NetworkModelConfig(
            latency_base_ms=base_latency_ms,
            jitter_ms=jitter_ms,
            packet_loss_rate=packet_loss_rate,
            ack_loss_rate=ack_loss_rate,
            random_seed=seed,
        )

        for pol in self.policy_runner.policies.values():
            pol.config.ack_timeout_ms = ack_timeout_ms
            pol.config.max_retries = max_retries
            pol.config.deadline_ms = deadline_ms

        self.policy_runner.run_controlled_comparison(
            workload=workload,
            network_config=net_cfg,
            progress_callback=progress_callback,
        )

        return self.policy_runner.to_summary_dataframe()

    def run_policy_comparison_live(
        self,
        message_count: int = 5,
        ack_timeout_ms: float = 150.0,
        max_retries: int = 2,
        deadline_ms: float = 100.0,
        progress_callback: Any = None,
    ) -> pd.DataFrame:
        """Run Phase 5 live evaluation against connected Mosquitto broker."""
        self.workload_generator.set_seed(int(time.time()))
        workload = self.workload_generator.generate_workload(count=message_count)

        for pol in self.policy_runner.policies.values():
            pol.config.ack_timeout_ms = ack_timeout_ms
            pol.config.max_retries = max_retries
            pol.config.deadline_ms = deadline_ms

        self.policy_runner.run_live_comparison(
            workload=workload,
            mqtt_manager=self.mqtt_manager,
            progress_callback=progress_callback,
        )

        return self.policy_runner.to_summary_dataframe()

    def get_policy_summary_df(self) -> pd.DataFrame:
        """Return pandas DataFrame of the most recent policy comparison summary."""
        return self.policy_runner.to_summary_dataframe()

    def get_policy_results_df(self) -> pd.DataFrame:
        """Return pandas DataFrame of the most recent granular message evaluations."""
        return self.policy_runner.to_results_dataframe()

    def export_policy_summary_csv(self) -> str:
        """Return summary metrics formatted as CSV string."""
        return self.policy_runner.export_summary_csv()

    def export_policy_results_csv(self) -> str:
        """Return granular results formatted as CSV string."""
        return self.policy_runner.export_results_csv()

    # --------------------------------------------------------------------------
    # Phase 6: DQN Training & Evaluation Operations
    # --------------------------------------------------------------------------

    def train_dqn(
        self,
        num_episodes: int = 20,
        steps_per_episode: int = 30,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        batch_size: int = 32,
        memory_capacity: int = 5000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_interval: int = 20,
        reward_config: Optional[RewardConfig] = None,
        progress_callback: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Train DQN agent in controlled environment with live metrics."""
        self.is_dqn_training = True
        try:
            # Reconfigure DQN agent hyperparameters
            self.dqn_agent.config = DQNConfig(
                state_dim=8,
                hidden_dim_1=64,
                hidden_dim_2=32,
                action_dim=5,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                batch_size=batch_size,
                memory_capacity=memory_capacity,
                target_update_interval=target_update_interval,
            )
            self.dqn_agent.epsilon = epsilon_start

            # Reconfigure optimizer
            import torch.optim as optim
            self.dqn_agent.optimizer = optim.Adam(
                self.dqn_agent.online_net.parameters(),
                lr=learning_rate,
            )

            # Reconfigure environment
            self.dqn_env.config.steps_per_episode = steps_per_episode
            if reward_config is not None:
                self.dqn_env.reward_config = reward_config
                self.dqn_env.reward_calculator.config = reward_config

            # Run training
            self.dqn_trainer.train(
                num_episodes=num_episodes,
                progress_callback=progress_callback,
                save_checkpoints=True,
            )
            self.last_dqn_training_history = list(self.dqn_trainer.history)
            return self.dqn_trainer.get_history_df()
        finally:
            self.is_dqn_training = False

    def get_dqn_training_history_df(self) -> pd.DataFrame:
        """Return DataFrame of recorded DQN training history."""
        return self.dqn_trainer.get_history_df()

    def run_dqn_evaluation(
        self,
        workload_count: int = 40,
        seed: int = 42,
        loss_rate: float = 0.12,
        latency_ms: float = 20.0,
        progress_callback: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Evaluate trained DQN against Phase 5 baseline policies on identical workloads."""
        workload_gen = WorkloadGenerator(random_seed=seed)
        test_workload = workload_gen.generate_workload(count=workload_count)

        net_cfg = NetworkModelConfig(
            latency_base_ms=latency_ms,
            packet_loss_rate=loss_rate,
            ack_loss_rate=0.05,
            random_seed=seed,
        )

        report = self.dqn_evaluator.run_comparative_benchmark(
            workload=test_workload,
            network_config=net_cfg,
            progress_callback=progress_callback,
        )
        self.last_dqn_eval_report = report
        return report.to_summary_df()

    def get_dqn_evaluation_summary_df(self) -> pd.DataFrame:
        """Return DataFrame of latest comparative evaluation benchmark."""
        if self.last_dqn_eval_report is None:
            return pd.DataFrame()
        return self.last_dqn_eval_report.to_summary_df()

    def get_available_checkpoints(self) -> List[str]:
        """List available saved .pt checkpoint files in the checkpoints directory."""
        ckpt_dir = Path("checkpoints")
        if not ckpt_dir.exists():
            return []
        return sorted([f.name for f in ckpt_dir.glob("*.pt")])

    def load_dqn_checkpoint(self, filename: str) -> bool:
        """Load selected checkpoint weights into active DQN agent and controller with persistent status."""
        ckpt_path = Path("checkpoints") / filename
        if not ckpt_path.is_file():
            err_msg = f"Checkpoint file not found: {ckpt_path}"
            self.checkpoint_load_status = {
                "success": False,
                "filename": filename,
                "filepath": str(ckpt_path),
                "device": str(self.dqn_agent.device),
                "error": err_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_steps": self.dqn_agent.total_steps,
                "episodes_trained": self.dqn_agent.episodes_trained,
                "epsilon": self.dqn_agent.epsilon,
                "metadata": None,
            }
            return False

        # Attempt to load through controller which verifies schema and architecture dimensions
        ok = self.dqn_controller.load_checkpoint(ckpt_path)
        if ok:
            # Synchronize agent state as well
            self.dqn_agent.load_checkpoint(ckpt_path)
            self.checkpoint_load_status = {
                "success": True,
                "filename": filename,
                "filepath": str(ckpt_path),
                "device": str(self.dqn_agent.device),
                "error": None,
                "total_steps": self.dqn_agent.total_steps,
                "episodes_trained": self.dqn_agent.episodes_trained,
                "epsilon": self.dqn_agent.epsilon,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": self.dqn_controller.checkpoint_metadata,
            }
            return True
        else:
            err_msg = self.dqn_controller.last_error or self.dqn_agent.last_load_error or "Incompatible format or architecture."
            self.checkpoint_load_status = {
                "success": False,
                "filename": filename,
                "filepath": str(ckpt_path),
                "device": str(self.dqn_agent.device),
                "error": err_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_steps": self.dqn_agent.total_steps,
                "episodes_trained": self.dqn_agent.episodes_trained,
                "epsilon": self.dqn_agent.epsilon,
                "metadata": None,
            }
            return False

    def export_dqn_training_csv(self) -> str:
        """Export training history as CSV string."""
        df = self.get_dqn_training_history_df()
        if df.empty:
            return ""
        return df.to_csv(index=False)

    def export_dqn_evaluation_csv(self) -> str:
        """Export comparative evaluation summary as CSV string."""
        df = self.get_dqn_evaluation_summary_df()
        if df.empty:
            return ""
        return df.to_csv(index=False)

    # --------------------------------------------------------------------------
    # Module 8: DQN Controller Integration Methods
    # --------------------------------------------------------------------------

    def get_recent_dqn_decisions(self, limit: int = 50) -> List[DqnDecisionRecord]:
        """Return list of recent live ACK decisions made by the DQN controller."""
        return self.dqn_controller.get_recent_decisions(limit=limit)

    def get_dqn_decisions_df(self, limit: int = 50) -> pd.DataFrame:
        """Return pandas DataFrame of recent live ACK decisions for UI rendering."""
        return self.dqn_controller.get_decisions_df(limit=limit)

    def get_latest_dqn_decision(self) -> Optional[DqnDecisionRecord]:
        """Return the most recent live decision record, if any."""
        return self.dqn_controller.get_latest_decision()

    def load_controller_checkpoint(self, filename: str) -> bool:
        """Load selected checkpoint weights into both DQN agent and active controller."""
        return self.load_dqn_checkpoint(filename)

    def unload_controller_checkpoint(self) -> None:
        """Explicitly unload checkpoint to demonstrate deterministic fallback."""
        self.dqn_controller.unload_checkpoint()
        self.checkpoint_load_status = {
            "success": False,
            "filename": "UNLOADED",
            "filepath": "",
            "device": str(self.dqn_agent.device),
            "error": "Model explicitly unloaded by operator. Operating in Safe Fallback Mode.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_steps": self.dqn_agent.total_steps,
            "episodes_trained": self.dqn_agent.episodes_trained,
            "epsilon": self.dqn_agent.epsilon,
            "metadata": None,
        }

    def set_controller_fallback_policy(self, policy: EvaluationPolicy) -> None:
        """Set fallback policy used when model is unready."""
        self.dqn_controller.fallback_policy = policy

    def set_controller_safety_override(self, enabled: bool) -> None:
        """Enable or disable emergency safety override for Critical Control alerts."""
        self.dqn_controller.enable_safety_override = enabled

    def export_dqn_decisions_csv(self) -> str:
        """Export live decision history as CSV string."""
        df = self.get_dqn_decisions_df(limit=200)
        if df.empty:
            return ""
        return df.to_csv(index=False)

    def get_dqn_controller_kpis(self) -> Dict[str, Any]:
        """Return aggregated operational and delivery performance KPIs from the active controller."""
        return self.dqn_controller.get_kpis()





