"""Top-level script to launch Phase 1 sensor verification demonstration."""

import sys
from pathlib import Path

# Add src/ to path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rl_ack.sensors.demo import run_sensor_demo

if __name__ == "__main__":
    run_sensor_demo()
