"""Compatibility entry point; implementation: src/visual_retrieval/visual_examples.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.visual_examples import (
    image_path,
    save_examples,
)
