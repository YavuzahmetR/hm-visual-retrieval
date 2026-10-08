"""Compatibility entry point; implementation: src/visual_retrieval/evaluate_robust_head_test.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.evaluate_robust_head_test import (
    PROJECT_DIR,
    HEAD_SHA256,
    _sha256,
    validate_embeddings,
    check_assets,
    evaluate,
    ProjectionHead,
    project_validation,
    TEST_CACHE,
    read_inputs,
    make_comparison,
    print_comparison,
    main,
)


if __name__ == "__main__":
    main()
