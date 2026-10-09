"""Compatibility entry point; implementation: src/visual_retrieval/resnet_baseline.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.resnet_baseline import (
    PROJECT_DIR,
    image_path,
    load_manifest,
    evaluate,
    save_examples,
    ValidationImages,
    main,
)


if __name__ == "__main__":
    main()
