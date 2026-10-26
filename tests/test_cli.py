"""Tests for CLI diagnostic utility and startup verification."""

import subprocess
import sys
from rl_ack.cli import check_dependencies, print_system_status


def test_check_dependencies_finds_required_libraries():
    """Verify that all core project libraries are recognized."""
    results = check_dependencies()
    expected_deps = [
        "Streamlit",
        "PyTorch",
        "Paho MQTT",
        "Pandas",
        "NumPy",
        "Plotly",
        "SQLite3",
        "Pytest",
        "Mosquitto Broker",
    ]
    for dep in expected_deps:
        assert dep in results, f"Missing dependency check: {dep}"
        is_ok, detail = results[dep]
        assert is_ok, f"Dependency {dep} check failed: {detail}"


def test_print_system_status_returns_true(capsys):
    """Verify print_system_status prints report and returns True."""
    success = print_system_status()
    captured = capsys.readouterr()
    assert success is True
    assert "RL-ACK IoT Network Simulation Framework" in captured.out
    assert "ALL CHECKS PASSED" in captured.out


def test_main_cli_subprocess_runs_successfully():
    """Verify main.py runs from shell and exits with code 0."""
    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "RL-ACK IoT Network Simulation Framework" in result.stdout
    assert "ALL CHECKS PASSED" in result.stdout
