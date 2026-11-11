"""Make ``src/`` importable when the package is not installed, and force a headless plot backend."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
