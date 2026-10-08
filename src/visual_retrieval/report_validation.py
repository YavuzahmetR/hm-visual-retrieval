import json

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

from visual_retrieval.data import PROJECT_DIR, image_path, load_manifest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = PROJECT_DIR / "results"
RUN_DIR = RESULTS_DIR / "projection_head_v2"


def load_rankings(directory, validation):
    rows = pd.read_csv(
        directory / "validation_embedding_rows.csv",
        dtype={"article_id": str, "product_code": str},
    )
    expected = validation[["article_id", "product_code"]]

    if not rows.equals(expected):
        raise ValueError(
            "Ranking satırları validation ile eşleşmiyor."
        )

    ranks = np.load(
        directory / "validation_top10_indices.npy",
        allow_pickle=False,
    )

    if ranks.shape != (len(validation), 10):
        raise ValueError("Ranking boyutu geçersiz.")

    if not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError("Ranking indeksleri tamsayı olmalı.")

    if (
        np.any(ranks < 0)
        or np.any(ranks >= len(validation))
    ):
        raise ValueError(
            "Ranking indeksleri galeri sınırlarını aşıyor."
        )

    if np.any(
        ranks == np.arange(len(validation))[:, None]
    ):
        raise ValueError(
            "Ranking içinde ürünün kendisi var."
        )

    return ranks


def plot_history(history, report, config):
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4)
    )

    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
    )
    axes[0].set_ylabel("Train triplet loss")

    axes[1].plot(
        history["epoch"],
        history["val_map@10"],
        marker="o",
        label="Projection head",
    )

    axes[1].axhline(
        report["frozen_baseline_metrics"]["map@10"],
        color="gray",
        linestyle="--",
        label="Frozen ResNet18",
    )

    axes[1].scatter(
        report["best_epoch"],
        report["validation_metrics"]["map@10"],
        color="green",
        s=90,
        zorder=5,
        label=f"Selected epoch {report['best_epoch']}",
    )
    axes[1].set_ylabel("Validation mAP@10")

    for ax in axes:
        ax.axvline(
            config["starting_epoch"] + 0.5,
            color="orange",
            linestyle=":",
            label="AdamW restarted before epoch 6",
        )
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    figure.suptitle(
        "Projection head: training and validation"
    )
    figure.tight_layout()
    figure.savefig(
        RUN_DIR / "learning_curve.png",
        dpi=140,
    )
    plt.close(figure)


def plot_comparison(
    validation, root, frozen, projected, best_epoch
):
    # İlk rapordaki aynı üç sorgu.
    # Örnekleri yeni sonuçlara göre seçmiyoruz.
    queries = np.random.default_rng(42).choice(
        len(validation),
        size=3,
        replace=False,
    )

    figure, axes = plt.subplots(
        6, 6,
        figsize=(16, 18),
        squeeze=False,
    )

    models = [
        ("Frozen ResNet18", frozen),
        (
            f"Projection head - epoch {best_epoch}",
            projected,
        ),
    ]

    for query_number, query_index in enumerate(queries):
        query = validation.iloc[query_index]

        for model_number, (model_name, ranks) in enumerate(models):
            row_number = query_number * 2 + model_number

            candidates = [
                query_index,
                *ranks[query_index, :5],
            ]

            for column, candidate_index in enumerate(candidates):
                candidate = validation.iloc[candidate_index]
                ax = axes[row_number, column]

                with Image.open(
                    image_path(
                        root, candidate["article_id"]
                    )
                ) as opened:
                    ax.imshow(opened.convert("RGB"))

                same_family = (
                    candidate["product_code"]
                    == query["product_code"]
                )

                if column == 0:
                    heading = f"{model_name}\nQUERY"
                    color = "navy"
                else:
                    match_label = (
                        "SAME FAMILY"
                        if same_family
                        else "other family"
                    )
                    heading = f"#{column} | {match_label}"
                    color = (
                        "darkgreen"
                        if same_family
                        else "black"
                    )

                ax.set_title(
                    f"{heading}\n"
                    f"{candidate['article_id']}\n"
                    f"{candidate['product_type_name']}",
                    fontsize=8,
                    color=color,
                )
                ax.axis("off")

    figure.suptitle(
        "Same validation queries: frozen encoder vs learned head",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        RUN_DIR / "validation_comparison.png",
        dpi=140,
    )
    plt.close(figure)


def main():
    manifest, root = load_manifest()

    validation = manifest.loc[
        manifest["split"] == "validation"
    ].reset_index(drop=True)

    report = json.loads(
        (RUN_DIR / "metrics.json").read_text("utf-8")
    )
    config = json.loads(
        (RUN_DIR / "config.json").read_text("utf-8")
    )
    history = pd.read_csv(
        RUN_DIR / "history.csv"
    )

    frozen = load_rankings(
        RESULTS_DIR, validation
    )
    projected = load_rankings(
        RUN_DIR, validation
    )

    plot_history(history, report, config)

    plot_comparison(
        validation,
        root,
        frozen,
        projected,
        report["best_epoch"],
    )

    print(
        "Eğitim grafiği:",
        RUN_DIR / "learning_curve.png",
    )
    print(
        "Görsel karşılaştırma:",
        RUN_DIR / "validation_comparison.png",
    )


if __name__ == "__main__":
    main()