"""Compatibility entry point; implementation: src/visual_retrieval/train_projection_head.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.train_projection_head import (
    PROJECT_DIR,
    load_manifest,
    evaluate,
    SEED,
    EPOCHS,
    FAMILIES_PER_BATCH,
    MARGIN,
    LEARNING_RATE,
    RESULTS_DIR,
    RUN_DIR,
    ProjectionHead,
    load_cache,
    family_batches,
    hard_triplets,
    project_validation,
    main,
)


if __name__ == "__main__":
    main()
