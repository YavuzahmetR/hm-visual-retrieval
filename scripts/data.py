"""Compatibility entry point; implementation: src/visual_retrieval/data.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.data import (
    PROJECT_DIR,
    load_manifest,
    image_path,
    main,
)


if __name__ == "__main__":
    main()
