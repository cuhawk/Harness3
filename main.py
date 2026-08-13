#!/usr/bin/env python3
"""Entry point - see harness_agent/ for the actual implementation, split by domain."""
from __future__ import annotations

from harness_agent.cli import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
        import sys
        sys.exit(0)
