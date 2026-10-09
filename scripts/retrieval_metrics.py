"""Compatibility entry point; implementation: src/visual_retrieval/retrieval_metrics.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.retrieval_metrics import (
    metrics_from_rankings,
    evaluate_queries,
    evaluate,
)
