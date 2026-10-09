"""Compatibility entry point; implementation: src/visual_retrieval/report_validation.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_retrieval.report_validation import (
    PROJECT_DIR,
    image_path,
    load_manifest,
    RESULTS_DIR,
    RUN_DIR,
    load_rankings,
    plot_history,
    plot_comparison,
    main,
)


if __name__ == "__main__":
    main()
