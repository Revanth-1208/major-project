"""CLI entrypoint for RL-ACK project verification and diagnostic status."""

import sys
import shutil
import importlib
from typing import Dict, Tuple

from rl_ack import __version__
from rl_ack.config import AppConfig


REQUIRED_DEPENDENCIES = [
    ("streamlit", "Streamlit"),
    ("torch", "PyTorch"),
    ("paho.mqtt", "Paho MQTT"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("plotly", "Plotly"),
    ("sqlite3", "SQLite3"),
    ("pytest", "Pytest"),
]


def check_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Verify that all required Python packages and system binaries are present."""
    results: Dict[str, Tuple[bool, str]] = {}
    for mod_name, display_name in REQUIRED_DEPENDENCIES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed (stdlib)")
            results[display_name] = (True, ver)
        except ImportError as e:
            results[display_name] = (False, f"Not installed ({e})")

    # Check Eclipse Mosquitto binary availability
    mosquitto_path = shutil.which("mosquitto")
    if not mosquitto_path:
        # Check standard Windows paths
        standard_windows_path = r"C:\Program Files\mosquitto\mosquitto.exe"
        import os
        if os.path.exists(standard_windows_path):
            results["Mosquitto Broker"] = (True, f"Found at {standard_windows_path}")
        else:
            results["Mosquitto Broker"] = (False, "Binary not found in PATH or standard directory")
    else:
        results["Mosquitto Broker"] = (True, f"Found at {mosquitto_path}")

    return results


def print_system_status() -> bool:
    """Print project configuration and dependency status; return True if all pass."""
    print("=" * 65)
    print(f"  RL-ACK IoT Network Simulation Framework (v{__version__})")
    print("=" * 65)
    print("\n--- Phase 0: System Verification & Architecture Check ---")

    config = AppConfig()
    print(f"Project Root: {config.project_root}")
    print(f"MQTT Broker Default: {config.broker.host}:{config.broker.port}")
    print(f"DQN Architecture Default: {config.dqn.state_dim} inputs -> "
          f"{config.dqn.hidden_dim_1} -> {config.dqn.hidden_dim_2} -> "
          f"{config.dqn.action_dim} Q-values")

    print("\n--- Technology Stack Status ---")
    dep_results = check_dependencies()
    all_ok = True
    for name, (ok, detail) in dep_results.items():
        status_tag = "[OK]  " if ok else "[FAIL]"
        print(f"  {status_tag} {name:<20}: {detail}")
        if not ok:
            all_ok = False

    print("\n--- Architectural Module Boundaries ---")
    module_names = [
        ("rl_ack.sensors", "Module 2: Virtual Sensor Engine"),
        ("rl_ack.topology", "Module 3: System Overview & IoT Topology"),
        ("rl_ack.monitoring", "Module 4: Incident Detection & Monitoring"),
        ("rl_ack.mqtt", "Module 5: MQTT Alert Transmission & Verification"),
        ("rl_ack.policies", "Module 6: Six-Policy Comparative Analysis"),
        ("rl_ack.dqn", "Module 7: DQN Architecture & Training"),
        ("rl_ack.controller", "Module 8: DQN Controller Integration"),
        ("rl_ack.simulation", "Module 9: Interactive Simulation Runner"),
        ("rl_ack.storage", "Module 10: Experiment Data Storage"),
        ("rl_ack.analysis", "Module 11: Performance Analysis & Export"),
    ]
    for mod_path, desc in module_names:
        try:
            m = importlib.import_module(mod_path)
            mod_id = getattr(m, "MODULE_ID", "?")
            print(f"  [REGISTERED] [{mod_id:>2}] {desc}")
        except Exception as e:
            print(f"  [ERROR]      {desc} (Failed to import: {e})")
            all_ok = False

    print("=" * 65)
    if all_ok:
        print("System status: ALL CHECKS PASSED (Phase 0 Project Setup Complete)")
    else:
        print("System status: CHECKS FAILED - Check missing dependencies")
    print("=" * 65)
    return all_ok


def main() -> None:
    """CLI entrypoint."""
    ok = print_system_status()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
