#!/usr/bin/env python3
import sys
from rl_cross_context.cli import main
if __name__ == "__main__":
    sys.argv = ["rl_cross_context", "report", *sys.argv[1:]]
    main()
