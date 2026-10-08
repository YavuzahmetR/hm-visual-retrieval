"""Compatibility entry point; implementation: src/visual_retrieval/train_robust_head.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.train_robust_head import (
    PROJECT_DIR,
    image_path,
    load_manifest,
    RESNET_SHA256,
    read_uploaded_image,
    CONDITIONS,
    MODEL_SHA256,
    PROTOCOL_VERSION,
    TRANSFORM_CONFIG,
    VALIDATION_SHA256,
    load_models,
    load_reusable_run,
    load_validation_assets,
    make_query_image,
    sample_queries,
    stable_seed,
    evaluate,
    evaluate_queries,
    hard_triplets,
    project_validation,
    PREPROCESSING,
    read_assets,
    training_batch,
    batches,
    validate_head,
    robustness_mean,
    save_checkpoint,
    main,
)


if __name__ == "__main__":
    main()
