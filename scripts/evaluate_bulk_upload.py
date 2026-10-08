"""Compatibility entry point; implementation: src/visual_retrieval/evaluate_bulk_upload.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.evaluate_bulk_upload import (
    PROJECT_DIR,
    image_path,
    load_manifest,
    HEAD_SHA256,
    RESNET_SHA256,
    _sha256,
    load_demo,
    rank_results,
    read_uploaded_image,
    validate_embeddings,
    validate_gallery_files,
    validate_gallery_rows,
    metrics_from_rankings,
    ProjectionHead,
    check_assets,
    load_labels,
    thumbnail,
    write_html,
    main,
)


if __name__ == "__main__":
    main()
