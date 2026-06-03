from __future__ import annotations

import sys
from pathlib import Path

from agent_core.app import main


SCRIPT_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    sys.exit(main(SCRIPT_DIR))
