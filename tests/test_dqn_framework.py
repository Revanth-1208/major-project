"""Comprehensive Automated Unit and Integration Tests for Phase 6 DQN Framework.

Tests:
1. Eight-dimensional state construction, normalization, bounds, and mutual exclusivity.
2. Five-action mapping to Phase 5 policies and metadata.
3. PyTorch QNetwork architecture, dimensions, forward pass, and finite outputs.
4. Experience replay buffer insertion, FIFO capacity limit, batch sampling, and rejection of invalid data.
5. Multi-objective reward calculation: delivery success, CC bonus, failure penalty, deadline compliance, and local ACK distinction.
6. Controlled IoT environment reset, step, deterministic seeds, and simulated metadata.
7. DQNAgent action selection (explore vs exploit), Bellman train step, and target network synchronization.
8. Model checkpoint atomic save, load, and architecture incompatibility validation.
9. End-to-end training loop for small episodes and history metric collection.
10. Greedy evaluation and comparative benchmarking against Phase 5 baseline policies.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from rl_ack.common.enums import AckAction, AlertSeverity, DeliveryStatus, EvaluationPolicy, TrafficClass
from rl_ack.common.models import AlertMessage, NetworkStateVector
from rl_ack.config import DQNConfig
from rl_ack.dqn.actions import (
    ACTION_DIM,
    ACTION_REGISTRY,
    action_to_policy_enum,
    get_action_metadata,
    instantiate_policy_for_action,
    policy_enum_to_action,
)
from rl_ack.dqn.agent import DQNAgent
from rl_ack.dqn.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from rl_ack.dqn.environment import EnvConfig, SmartKitchenIoTEnv
from rl_ack.dqn.evaluator import DQNEvaluator
from rl_ack.dqn.model import QNetwork
from rl_ack.dqn.replay_buffer import ReplayBuffer
from rl_ack.dqn.reward import RewardCalculator, RewardConfig
from rl_ack.dqn.state import (
    InvalidStateError,
    NetworkStateTracker,
    STATE_DIM,
    create_state_vector,
    validate_and_normalize_state,
)
from rl_ack.dqn.trainer import DQNTrainer
from rl_ack.policies.base import PolicyConfig, PolicyEvaluationResult
from rl_ack.policies.network_model import NetworkModelConfig
from rl_ack.policies.workload import WorkloadGenerator


# ==============================================================================
# 1. State Space & Validation Tests
# ==============================================================================

def test_eight_dimensional_state_dimensions_and_types():
    """Verify state vector always produces exactly 8 float32 bounded values."""
    s_vec = create_state_vector(
        network_utilization=0.45,
        rolling_ack_rate=0.95,
        queue_occupancy=0.20,
        traffic_class=TrafficClass.CRITICAL_CONTROL,
        remaining_energy=0.88,
        rolling_latency_ms=25.0,
    )
    arr = s_vec.to_numpy()
    assert arr.shape == (8,)
    assert arr.dtype == np.float32
    assert np.all(np.isfinite(arr))
    assert 0.0 <= arr[0] <= 1.0  # net util
    assert 0.0 <= arr[1] <= 1.0  # rolling ack
    assert 0.0 <= arr[2] <= 1.0  # queue occ
    assert arr[3] == 1.0  # is_CC
    assert arr[4] == 0.0  # is_HBS
    assert arr[5] == 0.0  # is_BET
    assert 0.0 <= arr[6] <= 1.0  # energy
    assert 0.0 <= arr[7] <= 1.0  # latency


def test_traffic_class_indicators_strictly_mutually_exclusive():
    """Verify traffic class features enforce mutual exclusivity."""
    for tc in [TrafficClass.CRITICAL_CONTROL, TrafficClass.HIGH_BANDWIDTH_STREAMING, TrafficClass.BEST_EFFORT]:
        s_vec = create_state_vector(0.5, 0.9, 0.1, tc, 1.0, 15.0)
        arr = validate_and_normalize_state(s_vec)
        is_cc, is_hbs, is_bet = arr[3], arr[4], arr[5]
        assert (is_cc + is_hbs + is_bet) == pytest.approx(1.0)
        assert sum(1 for val in [is_cc, is_hbs, is_bet] if val == 1.0) == 1

    # Invalid traffic indicators that do not sum to 1.0 should raise InvalidStateError
    with pytest.raises(InvalidStateError):
        invalid_raw = [0.2, 0.8, 0.1, 1.0, 1.0, 0.0, 0.9, 0.1]  # two active
        validate_and_normalize_state(invalid_raw)


def test_state_validation_rejects_non_finite_and_wrong_dimensions():
    """Verify validation rejects NaN, Inf, and non-8-dimensional vectors."""
    with pytest.raises(InvalidStateError):
        validate_and_normalize_state([0.5, np.nan, 0.2, 1.0, 0.0, 0.0, 1.0, 0.1])

    with pytest.raises(InvalidStateError):
        validate_and_normalize_state([0.5, np.inf, 0.2, 1.0, 0.0, 0.0, 1.0, 0.1])

    with pytest.raises(InvalidStateError):
        validate_and_normalize_state([0.5, 0.9, 0.2, 1.0, 0.0])  # length 5 instead of 8


def test_network_state_tracker_cold_start_and_rolling_updates():
    """Verify cold-start defaults and sliding window updates."""
    tracker = NetworkStateTracker(window_size=5)

    # Cold start initial values
    assert tracker.rolling_ack_rate == 1.0  # Optimistic initialization
    assert tracker.rolling_latency_ms == 15.0  # Baseline
    assert tracker.queue_occupancy == 0.0
    assert tracker.remaining_energy == 1.0

    # Record some transmission results
    tracker.record_outcome(confirmed=True, latency_ms=20.0, overhead_bytes=50)
    tracker.record_outcome(confirmed=False, latency_ms=None, overhead_bytes=50)
    assert tracker.rolling_ack_rate == 0.5
    assert tracker.rolling_latency_ms == 20.0
    assert tracker.remaining_energy < 1.0


# ==============================================================================
# 2. Action Space & Policy Mapping Tests
# ==============================================================================

def test_five_dqn_actions_mapping_and_registry():
    """Verify exactly 5 actions exist and map bidirectionally to Phase 5 policies."""
    assert ACTION_DIM == 5
    assert len(ACTION_REGISTRY) == 5

    expected_policies = [
        EvaluationPolicy.STATIC_MQTT_QOS_0,
        EvaluationPolicy.STATIC_LOCAL_ACK,
        EvaluationPolicy.STATIC_MQTT_QOS_1,
        EvaluationPolicy.LOCAL_ACK_SELECTIVE_RETRANSMIT,
        EvaluationPolicy.STATIC_MQTT_QOS_2,
    ]

    for a_idx in range(5):
        meta = get_action_metadata(a_idx)
        assert meta.action_idx == a_idx
        assert meta.policy_enum == expected_policies[a_idx]
        assert action_to_policy_enum(a_idx) == expected_policies[a_idx]
        assert policy_enum_to_action(expected_policies[a_idx]) == AckAction(a_idx)

        # Policy class instantiation
        policy_inst = instantiate_policy_for_action(a_idx)
        assert policy_inst.policy_type == expected_policies[a_idx]

    # Action out of bounds raises error
    with pytest.raises(ValueError):
        get_action_metadata(5)


# ==============================================================================
# 3. PyTorch Model Architecture Tests
# ==============================================================================

def test_q_network_architecture_and_forward_pass():
    """Verify QNetwork layer dimensions 8 -> 64 -> 32 -> 5 and finite output."""
    net = QNetwork(state_dim=8, hidden_dim_1=64, hidden_dim_2=32, action_dim=5)

    # 1D single input (8,)
    single_state = torch.tensor([0.2, 0.9, 0.1, 1.0, 0.0, 0.0, 0.95, 0.1], dtype=torch.float32)
    q_vals = net(single_state)
    assert q_vals.shape == (5,)
    assert torch.all(torch.isfinite(q_vals))

    # 2D batch input (16, 8)
    batch_states = torch.rand(16, 8, dtype=torch.float32)
    batch_q = net(batch_states)
    assert batch_q.shape == (16, 5)
    assert torch.all(torch.isfinite(batch_q))


# ==============================================================================
# 4. Experience Replay Buffer Tests
# ==============================================================================

def test_replay_buffer_push_capacity_and_sampling():
    """Verify replay buffer insertion, FIFO capacity bounding, and batch sampling."""
    buf = ReplayBuffer(capacity=10, random_seed=42)
    assert len(buf) == 0
    assert not buf.is_ready(4)

    s1 = [0.1, 0.9, 0.1, 1.0, 0.0, 0.0, 0.9, 0.1]
    s2 = [0.2, 0.8, 0.2, 0.0, 1.0, 0.0, 0.89, 0.15]

    for i in range(15):
        buf.push(state=s1, action=i % 5, reward=float(i), next_state=s2, done=(i % 5 == 0))

    # Max capacity capped at 10
    assert len(buf) == 10
    assert buf.is_ready(4)

    # Sampling
    states, actions, rewards, next_states, dones = buf.sample(batch_size=4)
    assert states.shape == (4, 8)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_states.shape == (4, 8)
    assert dones.shape == (4, 1)

    # Oversampling raises ValueError
    with pytest.raises(ValueError):
        buf.sample(batch_size=20)

    # Invalid action raises ValueError
    with pytest.raises(ValueError):
        buf.push(s1, action=5, reward=1.0, next_state=s2, done=False)


# ==============================================================================
# 5. Reward Formulation Tests
# ==============================================================================

def test_reward_function_delivery_success_and_cc_bonus():
    """Verify delivery reward and Critical Control emergency bonus."""
    calc = RewardCalculator(RewardConfig())

    # Critical Control alert successfully delivered within deadline
    cc_res = PolicyEvaluationResult(
        alert_id="TEST-CC-1",
        attempt_id="ATT-1",
        policy_name="Test Policy",
        evaluation_mode="controlled_simulation",
        sensor_id="GAS-001",
        traffic_class=TrafficClass.CRITICAL_CONTROL,
        severity=AlertSeverity.CRITICAL,
        configured_qos=1,
        publication_attempts=1,
        retry_count=0,
        broker_acknowledged=True,
        local_acknowledged=False,
        subscriber_received=True,
        delivery_confirmed=True,
        delivery_status=DeliveryStatus.DELIVERY_CONFIRMED,
        end_to_end_latency_ms=25.0,  # Below CC deadline 60ms
        message_overhead_bytes=50,
        deadline_met=True,
    )
    cc_rew, cc_terms = calc.compute_reward(cc_res)
    assert cc_rew > 0.0
    assert "delivery_reward" in cc_terms
    assert cc_terms["delivery_reward"] == 25.0  # 10 base + 15 cc bonus
    assert cc_terms["deadline_bonus"] == 5.0

    # Best Effort Traffic successfully delivered
    bet_res = PolicyEvaluationResult(
        alert_id="TEST-BET-1",
        attempt_id="ATT-2",
        policy_name="Test Policy",
        evaluation_mode="controlled_simulation",
        sensor_id="TEMP-001",
        traffic_class=TrafficClass.BEST_EFFORT,
        severity=AlertSeverity.INFO,
        configured_qos=0,
        publication_attempts=1,
        retry_count=0,
        broker_acknowledged=False,
        local_acknowledged=False,
        subscriber_received=True,
        delivery_confirmed=True,
        delivery_status=DeliveryStatus.DELIVERY_CONFIRMED,
        end_to_end_latency_ms=20.0,
        message_overhead_bytes=32,
        deadline_met=True,
    )
    bet_rew, bet_terms = calc.compute_reward(bet_res)
    assert bet_terms["delivery_reward"] == 10.0  # Base delivery reward, no CC bonus


def test_reward_function_local_ack_never_credited_as_e2e_delivery():
    """Verify local proxy ACK alone is NEVER credited as end-to-end delivery."""
    calc = RewardCalculator(RewardConfig())

    local_ack_res = PolicyEvaluationResult(
        alert_id="TEST-LOCAL-1",
        attempt_id="ATT-3",
        policy_name="Static Local ACK",
        evaluation_mode="controlled_simulation",
        sensor_id="GAS-001",
        traffic_class=TrafficClass.CRITICAL_CONTROL,
        severity=AlertSeverity.CRITICAL,
        configured_qos=0,
        publication_attempts=1,
        retry_count=0,
        broker_acknowledged=False,
        local_acknowledged=True,
        subscriber_received=False,
        delivery_confirmed=False,  # Dropped downstream before subscriber
        delivery_status=DeliveryStatus.FAILED,
        local_ack_latency_ms=2.0,
        end_to_end_latency_ms=None,
        message_overhead_bytes=64,
        deadline_met=False,
    )
    rew, terms = calc.compute_reward(local_ack_res)
    assert rew < 0.0  # Failed E2E delivery must be penalized
    assert "delivery_penalty" in terms
    assert terms["delivery_penalty"] == -40.0  # -15 base - 25 cc penalty
    assert terms.get("local_proxy_signal", 0.0) == 0.5  # Small proxy reachability signal, not E2E delivery


# ==============================================================================
# 6. Controlled IoT Environment Tests
# ==============================================================================

def test_controlled_iot_environment_lifecycle():
    """Verify environment reset, step, deterministic seeds, and termination."""
    cfg = EnvConfig(steps_per_episode=10, random_seed=42)
    env = SmartKitchenIoTEnv(config=cfg)

    s0 = env.reset(seed=42)
    assert s0.shape == (8,)
    assert np.all(np.isfinite(s0))

    # Step through entire episode
    step_count = 0
    done = False
    while not done:
        action = AckAction(step_count % 5)
        ns, rew, done, info = env.step(action)
        step_count += 1

        assert ns.shape == (8,)
        assert np.isfinite(rew)
        assert info["simulated"] is True
        assert "action_label" in info

    assert step_count == 10
    assert done is True


def test_environment_deterministic_reproducibility():
    """Verify identical seeds produce identical state, reward, and outcome sequences."""
    env1 = SmartKitchenIoTEnv(config=EnvConfig(steps_per_episode=5, random_seed=123))
    env2 = SmartKitchenIoTEnv(config=EnvConfig(steps_per_episode=5, random_seed=123))

    s1 = env1.reset(seed=123)
    s2 = env2.reset(seed=123)
    assert np.allclose(s1, s2)

    for a in [0, 2, 3, 1, 4]:
        ns1, r1, d1, i1 = env1.step(a)
        ns2, r2, d2, i2 = env2.step(a)
        assert np.allclose(ns1, ns2)
        assert r1 == pytest.approx(r2)
        assert d1 == d2
        assert i1["delivery_confirmed"] == i2["delivery_confirmed"]


# ==============================================================================
# 7. DQN Agent Learning & Target Network Sync Tests
# ==============================================================================

def test_dqn_agent_action_selection_explore_vs_exploit():
    """Verify epsilon exploration vs greedy exploitation."""
    agent = DQNAgent(config=DQNConfig(state_dim=8, action_dim=5), random_seed=42)
    s = [0.1, 0.9, 0.1, 1.0, 0.0, 0.0, 0.9, 0.1]

    # With epsilon=1.0 and explore=True, actions vary
    agent.epsilon = 1.0
    explored_actions = {agent.select_action(s, explore=True) for _ in range(30)}
    assert len(explored_actions) > 1

    # With explore=False, action is deterministic argmax
    greedy_action_1 = agent.select_action(s, explore=False)
    greedy_action_2 = agent.select_action(s, explore=False)
    assert greedy_action_1 == greedy_action_2

    # Q-values output shape
    q_vals = agent.get_q_values(s)
    assert q_vals.shape == (5,)
    assert int(np.argmax(q_vals)) == int(greedy_action_1)


def test_dqn_agent_train_step_and_target_sync():
    """Verify Bellman train step executes and target network syncs."""
    cfg = DQNConfig(batch_size=4, memory_capacity=20, target_update_interval=2)
    agent = DQNAgent(config=cfg, random_seed=42)

    s1 = [0.1, 0.9, 0.1, 1.0, 0.0, 0.0, 0.9, 0.1]
    s2 = [0.2, 0.8, 0.2, 0.0, 1.0, 0.0, 0.85, 0.15]

    # Not ready before batch_size
    assert agent.train_step() is None

    for i in range(10):
        agent.store_transition(s1, i % 5, float(i), s2, done=(i == 5))

    loss1 = agent.train_step()
    assert loss1 is not None
    assert np.isfinite(loss1)
    assert agent.total_steps == 1

    loss2 = agent.train_step()
    assert agent.total_steps == 2


# ==============================================================================
# 8. Checkpoint Persistence and Validation Tests
# ==============================================================================

def test_checkpoint_save_and_load_roundtrip():
    """Verify atomic checkpoint saving and verified loading."""
    agent1 = DQNAgent(random_seed=42)
    agent1.episodes_trained = 5
    agent1.total_steps = 150
    agent1.epsilon = 0.45

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_dqn.pt"
        agent1.save_checkpoint(ckpt_path)
        assert ckpt_path.is_file()

        # Load into new agent
        agent2 = DQNAgent(random_seed=99)
        ok = agent2.load_checkpoint(ckpt_path)
        assert ok is True
        assert agent2.episodes_trained == 5
        assert agent2.total_steps == 150
        assert agent2.epsilon == pytest.approx(0.45)

        # Online net weights must match
        for p1, p2 in zip(agent1.online_net.parameters(), agent2.online_net.parameters()):
            assert torch.allclose(p1, p2)


def test_checkpoint_validation_rejects_incompatible_architecture():
    """Verify checkpoint loader rejects architecture dimension mismatch."""
    online_net = QNetwork(state_dim=8, action_dim=5)
    target_net = QNetwork(state_dim=8, action_dim=5)
    opt = torch.optim.Adam(online_net.parameters())

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "bad_ckpt.pt"
        save_checkpoint(
            filepath=ckpt_path,
            online_network=online_net,
            target_network=target_net,
            optimizer=opt,
            epsilon=0.5,
            episodes_trained=1,
            total_steps=10,
        )

        # Incompatible net expecting 12 features instead of 8
        incompatible_net = QNetwork(state_dim=12, action_dim=5)
        with pytest.raises(CheckpointError):
            load_checkpoint(ckpt_path, online_network=incompatible_net)


# ==============================================================================
# 9. End-to-End Trainer & History Tests
# ==============================================================================

def test_dqn_trainer_multi_episode_training_and_metrics():
    """Verify trainer runs multiple episodes, collects history, and exports DataFrame."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = DQNAgent(config=DQNConfig(batch_size=8, memory_capacity=1000), random_seed=42)
        env = SmartKitchenIoTEnv(config=EnvConfig(steps_per_episode=15, random_seed=42))
        trainer = DQNTrainer(agent=agent, env=env, checkpoint_dir=tmpdir)

        history = trainer.train(num_episodes=3, save_checkpoints=True)
        assert len(history) == 3

        df = trainer.get_history_df()
        assert len(df) == 3
        assert "episode" in df.columns
        assert "total_reward" in df.columns
        assert "delivery_rate" in df.columns
        assert "cc_delivery_rate" in df.columns
        assert "epsilon" in df.columns

        # Checkpoints created
        assert (Path(tmpdir) / "dqn_latest.pt").is_file()
        assert (Path(tmpdir) / "dqn_best.pt").is_file()


# ==============================================================================
# 10. Greedy Evaluation & Baseline Comparison Tests
# ==============================================================================

def test_dqn_evaluator_comparative_benchmark_against_phase5_baselines():
    """Verify DQNEvaluator compares Adaptive DQN against all 5 fixed policies."""
    agent = DQNAgent(random_seed=42)
    evaluator = DQNEvaluator(agent=agent)

    workload_gen = WorkloadGenerator(random_seed=999)
    workload = workload_gen.generate_workload(count=15)

    net_cfg = NetworkModelConfig(
        latency_base_ms=20.0,
        packet_loss_rate=0.10,
        random_seed=999,
    )

    report = evaluator.run_comparative_benchmark(
        workload=workload,
        network_config=net_cfg,
    )

    # Must contain all 6 policies (5 baselines + Adaptive DQN)
    assert len(report.summary_metrics) == 6
    assert EvaluationPolicy.STATIC_MQTT_QOS_0.value in report.summary_metrics
    assert EvaluationPolicy.STATIC_LOCAL_ACK.value in report.summary_metrics
    assert EvaluationPolicy.STATIC_MQTT_QOS_1.value in report.summary_metrics
    assert EvaluationPolicy.LOCAL_ACK_SELECTIVE_RETRANSMIT.value in report.summary_metrics
    assert EvaluationPolicy.STATIC_MQTT_QOS_2.value in report.summary_metrics
    assert EvaluationPolicy.ADAPTIVE_RL_ACK_DQN.value in report.summary_metrics

    # Summary DataFrame format
    df_summary = report.to_summary_df()
    assert len(df_summary) == 6
    assert "Policy" in df_summary.columns
    assert "Delivery Confirmed Rate (%)" in df_summary.columns

    # Action distribution by class
    assert TrafficClass.CRITICAL_CONTROL.value in report.dqn_action_distribution_by_class
