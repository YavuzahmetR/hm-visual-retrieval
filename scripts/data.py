"""Read the saved split and resolve image paths without changing the split."""

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_manifest() -> tuple[pd.DataFrame, Path]:
    config = json.loads((PROJECT_DIR / "project_config.json").read_text("utf-8"))
    table = pd.read_csv(
        PROJECT_DIR / config["manifest"],
        dtype={"article_id": str, "product_code": str},
    )
    required = {"article_id", "product_code", "split", "image_path"}
    if not required.issubset(table.columns):
        raise ValueError(
            f"Manifest is missing columns: {required - set(table.columns)}"
        )
    if table["article_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate article IDs.")
    if table[list(required)].isna().any().any():
        raise ValueError("Manifest contains missing IDs, split labels or paths.")
    if set(table["split"]) != {"train", "validation", "test"}:
        raise ValueError("Expected the original train/validation/test split.")
    if table.groupby("product_code")["split"].nunique().max() != 1:
        raise ValueError("A product family occurs in multiple splits.")
    return table, Path(config["data_root"]).resolve()


def image_path(data_root: Path, article_id: str) -> Path:
    # IDs must retain their leading zeros; never convert them to integers.
    if not re.fullmatch(r"[0-9]{10}", article_id):
        raise ValueError(f"Invalid article ID: {article_id!r}")
    return data_root / "images" / article_id[:3] / f"{article_id}.jpg"


def main() -> None:
    table, root = load_manifest()
    table["local_image_path"] = [
        str(image_path(root, value)) for value in table["article_id"]
    ]
    present = table["local_image_path"].map(lambda value: Path(value).is_file())
    print(
        table.groupby("split").agg(
            images=("article_id", "size"), groups=("product_code", "nunique")
        )
    )
    print(f"Local images present: {present.sum()}/{len(table)}", flush=True)
    output = PROJECT_DIR / "results"
    output.mkdir(exist_ok=True)
    table.to_csv(output / "local_manifest.csv", index=False)
    if not present.all():
        raise SystemExit("Image download/extraction is not complete yet.")


if __name__ == "__main__":
    main()
