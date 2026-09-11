#!/usr/bin/env python3
"""Week 1 prepare entry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from c4_cascade_rl.cli import week1_cmd
if __name__ == "__main__":
    week1_cmd()
