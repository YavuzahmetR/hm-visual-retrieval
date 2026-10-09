"""Compatibility entry point; implementation: src/visual_retrieval/continue_projection_head.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.continue_projection_head import (
    PROJECT_DIR,
    load_manifest,
    evaluate,
    FAMILIES_PER_BATCH,
    ProjectionHead,
    family_batches,
    hard_triplets,
    load_cache,
    project_validation,
    RESULTS_DIR,
    SOURCE_DIR,
    RUN_DIR,
    MAX_EPOCH,
    PATIENCE,
    observe_score,
    save_checkpoint,
    main,
)


if __name__ == "__main__":
    main()
