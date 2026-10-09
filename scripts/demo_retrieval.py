"""Compatibility entry point; implementation: src/visual_retrieval/demo_retrieval.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.demo_retrieval import (
    PROJECT_DIR,
    load_manifest,
    ProjectionHead,
    HEAD_SHA256,
    ROBUST_HEAD_SHA256,
    ROBUST_SOURCE_SHA256,
    ROBUST_GALLERY_SHA256,
    RESNET_SHA256,
    GALLERY_SHA256,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_PIXELS,
    validate_gallery_rows,
    validate_embeddings,
    rank_results,
    read_uploaded_image,
    DemoResources,
    _sha256,
    validate_gallery_files,
    load_demo,
)
