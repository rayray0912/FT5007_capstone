#!/usr/bin/env python3
"""Convenience wrapper so the CLI runs without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mmbacktest.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
