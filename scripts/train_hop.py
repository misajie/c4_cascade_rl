#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from c4_cascade_rl.cli import train_cmd
if __name__ == "__main__":
    # Default to hop BC unless --algo llm
    sys.argv = [sys.argv[0]] + (sys.argv[1:] if len(sys.argv) > 1 else ["--algo", "bc", "--level", "L0"])
    train_cmd()
