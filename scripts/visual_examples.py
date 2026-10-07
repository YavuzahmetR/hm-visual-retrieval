"""Save a few validation queries and their top cosine matches."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from data import image_path
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_examples(
    table: pd.DataFrame, root: Path, rankings: np.ndarray, output: Path
) -> None:
    query_indices = np.random.default_rng(42).choice(
        len(table), size=min(3, len(table)), replace=False
    )
    figure, axes = plt.subplots(len(query_indices), 6, figsize=(15, 9), squeeze=False)
    rows = []
    for row_number, query_index in enumerate(query_indices):
        query = table.iloc[query_index]
        candidates = [query_index, *rankings[query_index, :5]]
        for column, gallery_index in enumerate(candidates):
            candidate = table.iloc[gallery_index]
            ax = axes[row_number, column]
            with Image.open(image_path(root, candidate["article_id"])) as opened:
                ax.imshow(opened.convert("RGB"))
            same_family = query["product_code"] == candidate["product_code"]
            heading = (
                "QUERY"
                if column == 0
                else f"#{column} | {'same family' if same_family else 'other family'}"
            )
            ax.set_title(
                f"{heading}\n{candidate['article_id']}\n{candidate['product_type_name']}",
                fontsize=9,
            )
            ax.axis("off")
            if column:
                rows.append(
                    {
                        "query_id": query["article_id"],
                        "rank": column,
                        "match_id": candidate["article_id"],
                        "same_product_family": same_family,
                    }
                )
    figure.suptitle("Frozen ResNet18 — validation examples", fontsize=14)
    figure.tight_layout()
    figure.savefig(output / "validation_examples.png", dpi=140)
    plt.close(figure)
    pd.DataFrame(rows).to_csv(output / "validation_example_matches.csv", index=False)
