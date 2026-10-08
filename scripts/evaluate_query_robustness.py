"""Compatibility entry point; implementation: src/visual_retrieval/evaluate_query_robustness.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.evaluate_query_robustness import (
    PROJECT_DIR,
    image_path,
    load_manifest,
    HEAD_SHA256,
    RESNET_SHA256,
    read_uploaded_image,
    validate_embeddings,
    evaluate_queries,
    metrics_from_rankings,
    ProjectionHead,
    KS,
    PROTOCOL_VERSION,
    CACHE_TORCHVISION_VERSION,
    CONDITIONS,
    TRANSFORM_CONFIG,
    VALIDATION_SHA256,
    MODEL_SHA256,
    stable_seed,
    sample_queries,
    make_query_image,
    load_validation_assets,
    load_models,
    query_tensor,
    encode_queries,
    load_query_rankings,
    load_reusable_run,
    save_preview,
    print_summary,
    main,
)


if __name__ == "__main__":
    main()
