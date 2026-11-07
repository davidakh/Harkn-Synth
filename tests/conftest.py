"""Make the project's script folders importable from tests (the modules import
each other as siblings, the same way they do when you run the scripts)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for folder in ("synth", "features", "tools", "model"):
    sys.path.insert(0, str(ROOT / folder))
