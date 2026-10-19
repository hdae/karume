"""Makes `py_skeleton` importable from the tests (tools/lang is not a package)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
