import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cache_train_embeddings import (
    encode_batch,
    load_encoder,
    make_loader,
)
from data import PROJECT_DIR, load_manifest
from retrieval_metrics import evaluate
from train_projection_head import ProjectionHead
from train_projection_head import project_validation as project_features

SELECTED_EPOCH = 15
WORKERS = 2

MODEL_DIR = PROJECT_DIR / "results" / "projection_head_v2"
OUTPUT_DIR = PROJECT_DIR / "results" / "test_evaluation"


def validate_cache(features, saved_rows, expected_rows):
    if not saved_rows.equals(expected_rows):
        raise ValueError(
            "Test cache satırları manifestle eşleşmiyor."
        )

    if features.shape != (len(expected_rows), 512):
        raise ValueError("Test embedding boyutu geçersiz.")

    if (
        features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError("Test embedding sayıları geçersiz.")

    if not np.allclose(
        np.linalg.norm(features, axis=1), 1, atol=1e-5
    ):
        raise ValueError(
            "Test embedding'leri normalize edilmemiş."
        )


def get_test_features(test, root):
    feature_path = OUTPUT_DIR / "frozen_embeddings.npy"
    rows_path = OUTPUT_DIR / "embedding_rows.csv"
    metadata_path = OUTPUT_DIR / "cache_metadata.json"

    expected_rows = test[["article_id", "product_code"]]

    weights_path = (
        PROJECT_DIR
        / "model_cache/hub/checkpoints/resnet18-f37072fd.pth"
    )

    # Cache'in aynı encoder ve aynı ürün sırasına ait olduğunu doğrular.
    protocol = (
        "ResNet18 ImageNet1K V1|weights.transforms|float32|L2|"
        + str(torch.__version__)
        + "|"
        + hashlib.sha256(
            weights_path.read_bytes()
        ).hexdigest()
    )

    signature = hashlib.sha256(
        (
            protocol + expected_rows.to_csv(index=False)
        ).encode("utf-8")
    ).hexdigest()

    if feature_path.exists():
        metadata = json.loads(
            metadata_path.read_text("utf-8")
        )

        if metadata["signature"] != signature:
            raise ValueError(
                "Test cache farklı veri veya encoder'a ait."
            )

        saved_rows = pd.read_csv(
            rows_path,
            dtype={
                "article_id": str,
                "product_code": str,
            },
        )

        features = np.load(
            feature_path,
            allow_pickle=False,
        )

        validate_cache(
            features, saved_rows, expected_rows
        )

        print(
            "Kaydedilmiş test vektörleri kullanılıyor.",
            flush=True,
        )
        return features

    encoder, transform = load_encoder()

    loader = make_loader(
        test, root, transform, WORKERS
    )

    features = np.empty(
        (len(test), 512),
        dtype=np.float32,
    )
    completed = 0

    with torch.inference_mode():
        for images in tqdm(
            loader,
            desc="Test image embeddings",
        ):
            vectors = encode_batch(
                encoder, images
            ).cpu().numpy()

            end = completed + len(vectors)
            features[completed:end] = vectors
            completed = end

    if completed != len(test):
        raise RuntimeError(
            "Bütün test görselleri işlenmedi."
        )

    validate_cache(
        features, expected_rows, expected_rows
    )

    temporary = (
        OUTPUT_DIR / "frozen_embeddings.tmp.npy"
    )
    np.save(
        temporary,
        features,
        allow_pickle=False,
    )

    expected_rows.to_csv(
        rows_path,
        index=False,
    )

    metadata_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "protocol": protocol,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(feature_path)
    return features


def print_comparison(report):
    table = pd.DataFrame({
        "Frozen ResNet18": report["frozen_metrics"],
        "Projection head": report["projection_metrics"],
        "Delta": report["metric_deltas"],
    })

    table.index.name = "metric"

    print(
        "\n"
        + table.loc[
            ["map@10", "recall@10", "hit_rate@10"]
        ].round(4).to_string()
    )


def main():
    checkpoint_path = MODEL_DIR / "best.pt"

    checkpoint_hash = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    report_path = OUTPUT_DIR / "metrics.json"

    # Tamamlanan final değerlendirme yeniden hesaplanmaz.
    if report_path.exists():
        report = json.loads(
            report_path.read_text("utf-8")
        )

        if (
            report["selection"]["checkpoint_sha256"]
            != checkpoint_hash
        ):
            raise ValueError(
                "Kayıtlı test sonucu başka checkpoint'e ait."
            )

        print(
            "Final test değerlendirmesi zaten tamamlanmış."
        )
        print_comparison(report)
        return

    started = time.perf_counter()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadı.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["epoch"] != SELECTED_EPOCH:
        raise ValueError(
            "Checkpoint seçilen 15. epoch'a ait değil."
        )

    manifest, root = load_manifest()

    test = manifest.loc[
        manifest["split"] == "test"
    ].reset_index(drop=True)

    groups = test["product_code"].to_numpy()

    print(
        f"Final test: {len(test)} ürün, "
        f"{test['product_code'].nunique()} aile",
        flush=True,
    )
    print(
        f"Seçilen model: epoch {SELECTED_EPOCH}",
        flush=True,
    )

    head = ProjectionHead().to("cuda")
    head.load_state_dict(checkpoint["state_dict"])
    head.requires_grad_(False).eval()

    OUTPUT_DIR.mkdir(exist_ok=True)

    frozen_features = get_test_features(
        test, root
    )

    print(
        "Frozen ResNet18 test retrieval ölçülüyor...",
        flush=True,
    )

    frozen_metrics, frozen_rankings = evaluate(
        frozen_features,
        groups,
    )

    print(
        "Projection head test retrieval ölçülüyor...",
        flush=True,
    )

    projected_features = project_features(
        head,
        torch.from_numpy(
            frozen_features
        ).to("cuda"),
    )

    projection_metrics, projection_rankings = evaluate(
        projected_features,
        groups,
    )

    np.save(
        OUTPUT_DIR / "projection_embeddings.npy",
        projected_features,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "frozen_top10_indices.npy",
        frozen_rankings,
        allow_pickle=False,
    )
    np.save(
        OUTPUT_DIR / "projection_top10_indices.npy",
        projection_rankings,
        allow_pickle=False,
    )

    deltas = {
        key: projection_metrics[key] - frozen_metrics[key]
        for key in frozen_metrics
    }

    report = {
        "split": "test",
        "queries": len(test),
        "product_families": int(
            test["product_code"].nunique()
        ),
        "gallery": (
            "complete test split; query itself excluded"
        ),
        "relevance": (
            "same product_code, different article_id"
        ),
        "ap_denominator": (
            "min(K, number of relevant gallery images)"
        ),
        "selection": {
            "run": MODEL_DIR.name,
            "epoch": SELECTED_EPOCH,
            "criterion": "validation map@10",
            "validation_map@10": (
                checkpoint["validation_metrics"]["map@10"]
            ),
            "checkpoint_sha256": checkpoint_hash,
        },
        "frozen_metrics": frozen_metrics,
        "projection_metrics": projection_metrics,
        "metric_deltas": deltas,
        "torch_version": str(torch.__version__),
        "seconds": time.perf_counter() - started,
    }

    comparison = pd.DataFrame({
        "frozen_resnet18": frozen_metrics,
        "projection_head": projection_metrics,
        "delta": deltas,
    })
    comparison.index.name = "metric"

    comparison.to_csv(
        OUTPUT_DIR / "comparison.csv"
    )

    temporary_report = (
        OUTPUT_DIR / "metrics.tmp.json"
    )
    temporary_report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    temporary_report.replace(report_path)

    print_comparison(report)
    print("\nSonuçlar:", OUTPUT_DIR)


if __name__ == "__main__":
    main()