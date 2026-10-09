"""Compatibility entry point; implementation: src/visual_retrieval/evaluate_test.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.evaluate_test import (
    encode_batch,
    load_encoder,
    make_loader,
    PROJECT_DIR,
    load_manifest,
    evaluate,
    ProjectionHead,
    project_features,
    SELECTED_EPOCH,
    WORKERS,
    MODEL_DIR,
    OUTPUT_DIR,
    validate_cache,
    get_test_features,
    print_comparison,
    main,
)


if __name__ == "__main__":
    main()
