"""Make `synth/` importable from tests (the synth modules import each other as
siblings, the same way they do when you run the script directly)."""

import sys
from pathlib import Path

SYNTH_DIR = Path(__file__).resolve().parent.parent / "synth"
sys.path.insert(0, str(SYNTH_DIR))
