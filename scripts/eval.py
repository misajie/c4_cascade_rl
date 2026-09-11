#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from c4_cascade_rl.cli import eval_cmd
if __name__ == "__main__":
    eval_cmd()
