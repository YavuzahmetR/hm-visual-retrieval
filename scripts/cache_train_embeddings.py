"""Compatibility entry point; implementation: src/visual_retrieval/cache_train_embeddings.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.cache_train_embeddings import (
    PROJECT_DIR,
    image_path,
    load_manifest,
    BATCH_SIZE,
    EMBEDDING_DIM,
    RESULTS_DIR,
    ProductImages,
    make_loader,
    load_encoder,
    encode_batch,
    benchmark,
    save_progress,
    build_cache,
    main,
)


if __name__ == "__main__":
    main()
