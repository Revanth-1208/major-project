"""RL-ACK Project Top-Level Verification Entrypoint."""

import sys
from pathlib import Path

# Add src to Python path if running directly
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rl_ack.cli import main

if __name__ == "__main__":
    main()
