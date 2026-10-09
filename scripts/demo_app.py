"""Compatibility entry point; implementation: src/visual_retrieval/demo_app.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.demo_app import (
    image_path,
    load_manifest,
    DemoResources,
    load_demo,
    rank_results,
    read_uploaded_image,
    CATALOG_SCHEMA,
    CATALOG_METADATA,
    has_catalog_metadata,
    ensure_catalog_metadata,
    cached_resources,
    uploaded_query,
    show_local_image,
    article_name,
    show_results,
    main,
)


if __name__ == "__main__":
    main()
