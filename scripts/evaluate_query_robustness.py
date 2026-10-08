"""Compare frozen/head retrieval on controlled validation query changes.

This is a synthetic robustness benchmark, not measured real-world upload success.
No training, downloads, gallery extraction or test-split evaluation takes place.
"""

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from data import PROJECT_DIR, image_path, load_manifest
from demo_retrieval import (
    HEAD_SHA256,
    RESNET_SHA256,
    read_uploaded_image,
    validate_embeddings,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from PIL import __version__ as PILLOW_VERSION
from retrieval_metrics import evaluate_queries, metrics_from_rankings
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm
from train_projection_head import ProjectionHead

KS = (1, 5, 10)
# Dönüşüm, inference veya metrik mantığı değişirse bu sürümü artır.
PROTOCOL_VERSION = 1
CACHE_TORCHVISION_VERSION = "0.21.0+cu124"
CONDITIONS = ("clean", "lighting", "rotation", "crop", "blur", "perspective")
# Tek seferde yalnızca bir etki uygulanır; renk tonunu değiştirmiyoruz.
TRANSFORM_CONFIG = {
    "lighting": {"brightness": [0.7, 1.3], "contrast": [0.85, 1.15]},
    "rotation": {"degrees": 15, "fill": [245, 245, 245]},
    "crop": {"side_fraction": [0.9, 1.0], "preserve_aspect_ratio": True},
    "blur": {"kernel_size": 5, "sigma": [0.5, 1.2]},
    "perspective": {"distortion_scale": 0.1, "p": 1.0, "fill": [245, 245, 245]},
}
# Bu dört cache tamamlanmış validation deneyine aittir. Değişirse duruyoruz.
VALIDATION_SHA256 = {
    "results/validation_embeddings.npy": "999a52dc4d3dc02894d54b1aa3051f5e14c598496063cddb131c458ebcbfbb21",
    "results/validation_top10_indices.npy": "2880c46f406aa4bc02eb4279419bf36c84aec5cf8885d3da32c96696ceead08f",
    "results/projection_head_v2/validation_embeddings.npy": "4f07411a4b95841110b57ccc35e5fb5ba8aacb009a8c670a1fa532390b95f7c2",
    "results/projection_head_v2/validation_top10_indices.npy": "661e5698c375a3062b6b8debb854e8c30fb37c0854223e548fd50f42e5e77207",
}
MODEL_SHA256 = {
    "models/projection_head_epoch15.pt": HEAD_SHA256,
    "model_cache/hub/checkpoints/resnet18-f37072fd.pth": RESNET_SHA256,
}


def stable_seed(seed: int, identity: str, condition: str) -> int:
    """Python hash() yerine süreçler arasında değişmeyen bir seed üretir."""
    text = f"{seed}|{identity}|{condition}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def sample_queries(table: pd.DataFrame, families: int, seed: int) -> pd.DataFrame:
    """Aileleri rastgele sıralar; aile başına yalnızca bir article seçer."""
    grouped = table.groupby("product_code", sort=True).indices
    if not 1 <= families <= len(grouped):
        raise ValueError(f"Choose between 1 and {len(grouped)} families.")
    if any(len(indices) < 2 for indices in grouped.values()):
        raise ValueError("Every family needs at least two different articles.")
    order = np.random.default_rng(seed).permutation(sorted(grouped))
    chosen = []
    for family in order[:families]:
        # Pilot büyütülünce eski ailelerin kaynak article'ları değişmesin.
        rng = np.random.default_rng(stable_seed(seed, family, "source"))
        chosen.append(int(rng.choice(grouped[family])))
    queries = table.iloc[chosen][["article_id", "product_code"]].reset_index(drop=True)
    queries["gallery_index"] = chosen
    return queries


def make_query_image(
    image: Image.Image, condition: str, article_id: str, seed: int
) -> Image.Image:
    """Aynı article/koşul aynı dönüşümü alır; global RNG durumu korunur."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "clean":
        return image.copy()
    # Bulanıklık miktarı kaynak görsel çözünürlüğüne bağlı olmasın.
    image = TF.resize(image, 256, interpolation=InterpolationMode.BILINEAR)
    settings = TRANSFORM_CONFIG[condition]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(
            stable_seed(seed, article_id, condition)
        )
        if condition == "lighting":
            return transforms.ColorJitter(**settings)(image)
        if condition == "rotation":
            return transforms.RandomRotation(
                settings["degrees"],
                interpolation=InterpolationMode.BILINEAR,
                fill=tuple(settings["fill"]),
            )(image)
        if condition == "crop":
            # Kareye zorlamak yerine orijinal en/boy oranını korur.
            low, high = settings["side_fraction"]
            fraction = float(torch.empty(1).uniform_(low, high).item())
            width, height = image.size
            crop_w, crop_h = (
                max(1, int(width * fraction)),
                max(1, int(height * fraction)),
            )
            left = int(torch.randint(width - crop_w + 1, (1,)).item())
            top = int(torch.randint(height - crop_h + 1, (1,)).item())
            return image.crop((left, top, left + crop_w, top + crop_h))
        if condition == "blur":
            return transforms.GaussianBlur(**settings)(image)
        return transforms.RandomPerspective(
            distortion_scale=settings["distortion_scale"],
            p=settings["p"],
            interpolation=InterpolationMode.BILINEAR,
            fill=tuple(settings["fill"]),
        )(image)


def load_validation_assets() -> tuple[pd.DataFrame, Path, dict, dict]:
    """Doğru model ve cache'leri doğrular; yeni embedding hesaplamaz."""
    manifest, data_root = load_manifest()
    table = manifest.loc[manifest["split"] == "validation"].reset_index(drop=True)
    if len(table) != 8110 or table["product_code"].nunique() != 2306:
        raise ValueError(
            "Expected the original 8,110-item / 2,306-family validation split."
        )
    expected_rows = table[["article_id", "product_code"]]
    for name in (
        "results/validation_embedding_rows.csv",
        "results/projection_head_v2/validation_embedding_rows.csv",
    ):
        rows = pd.read_csv(PROJECT_DIR / name, dtype=str)
        if not rows.equals(expected_rows):
            raise ValueError(f"Gallery row order differs from the manifest: {name}")
    for name, expected_hash in {**VALIDATION_SHA256, **MODEL_SHA256}.items():
        with (PROJECT_DIR / name).open("rb") as source:
            actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Saved model or validation cache has changed: {name}")
    config = json.loads(
        (PROJECT_DIR / "results/projection_head_v2/config.json").read_text("utf-8")
    )
    if config["torch_version"] != str(torch.__version__):
        raise ValueError(
            "PyTorch version differs from the saved validation experiment."
        )
    if torchvision.__version__ != CACHE_TORCHVISION_VERSION:
        raise ValueError(
            "Torchvision version differs from the original cached preprocessing."
        )

    galleries, rankings = {}, {}
    for model, base, dimension in (
        ("frozen", "results/validation", 512),
        ("projection", "results/projection_head_v2/validation", 128),
    ):
        gallery = np.load(PROJECT_DIR / f"{base}_embeddings.npy", allow_pickle=False)
        ranks = np.load(PROJECT_DIR / f"{base}_top10_indices.npy", allow_pickle=False)
        validate_embeddings(gallery, len(table), dimension)
        if ranks.shape != (len(table), 10) or not np.issubdtype(
            ranks.dtype, np.integer
        ):
            raise ValueError("Saved top-10 rankings have an invalid shape or dtype.")
        if (ranks < 0).any() or (ranks >= len(table)).any():
            raise ValueError("Saved rankings contain an invalid gallery index.")
        if (ranks == np.arange(len(table))[:, None]).any():
            raise ValueError("Saved rankings contain self-matches.")
        galleries[model], rankings[model] = gallery, ranks
    return table, data_root, galleries, rankings


def load_models(
    device: str,
) -> tuple[nn.Module, nn.Module, Callable[[Image.Image], torch.Tensor]]:
    """Yerel ağırlıkları yükler; weights=None indirmeyi engeller."""
    encoder = resnet18(weights=None)
    encoder.load_state_dict(
        torch.load(
            PROJECT_DIR / "model_cache/hub/checkpoints/resnet18-f37072fd.pth",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    encoder.fc = nn.Identity()
    encoder.requires_grad_(False).to(device).eval()
    checkpoint = torch.load(
        PROJECT_DIR / "models/projection_head_epoch15.pt",
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint["epoch"] != 15:
        raise ValueError("Expected the validation-selected epoch-15 head.")
    head = ProjectionHead()
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.requires_grad_(False).to(device).eval()
    return encoder, head, ResNet18_Weights.IMAGENET1K_V1.transforms()


def query_tensor(
    article_id: str,
    data_root: Path,
    condition: str,
    seed: int,
    preprocess: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    """Demo upload'ı ile aynı JPG/PNG decoder ve ImageNet preprocessing."""
    try:
        payload = image_path(data_root, article_id).read_bytes()
        image = read_uploaded_image(payload)
        return preprocess(make_query_image(image, condition, article_id, seed))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot process article {article_id}: {error}") from error


def encode_queries(
    queries: pd.DataFrame,
    data_root: Path,
    condition: str,
    seed: int,
    models: tuple[nn.Module, nn.Module, Callable[[Image.Image], torch.Tensor]],
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Her batch bir kez ResNet'ten geçer; iki temsil aynı fotoğraftan çıkar."""
    encoder, head, preprocess = models
    frozen_batches, projected_batches = [], []
    rows = list(queries.itertuples(index=False))
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), batch_size), desc=condition):
            images = torch.stack(
                [
                    query_tensor(row.article_id, data_root, condition, seed, preprocess)
                    for row in rows[start : start + batch_size]
                ]
            ).to(device)
            frozen = F.normalize(encoder(images), dim=1)
            projected = head(frozen)
            frozen_batches.append(frozen.cpu().numpy())
            projected_batches.append(projected.cpu().numpy())
    return {
        "frozen": np.concatenate(frozen_batches),
        "projection": np.concatenate(projected_batches),
    }


def load_query_rankings(
    path: Path, queries: pd.DataFrame, gallery_size: int
) -> np.ndarray:
    """Kaydedilmiş sonuçların satır/boyut ve self-match kontrolleri."""
    ranks = np.load(path, allow_pickle=False)
    if (
        ranks.shape != (len(queries), max(KS))
        or not np.issubdtype(ranks.dtype, np.integer)
        or (ranks < 0).any()
        or (ranks >= gallery_size).any()
    ):
        raise ValueError(f"Invalid saved query rankings: {path}")
    if (ranks == queries["gallery_index"].to_numpy()[:, None]).any():
        raise ValueError(f"Saved rankings contain a source article: {path}")
    if any(len(np.unique(row)) != len(row) for row in ranks):
        raise ValueError(f"Saved rankings contain duplicate results: {path}")
    return ranks


def load_reusable_run(
    directory: Path, config: dict, queries: pd.DataFrame, gallery_size: int
) -> tuple[np.ndarray, dict, dict]:
    """Uyumlu pilotun embedding ve sıralamalarını article ID ile eşleştirir."""
    if not (directory / "metrics.json").is_file():
        raise ValueError("--reuse-from must point to a completed robustness run.")
    old_config = json.loads((directory / "config.json").read_text("utf-8"))
    # Yalnızca sorgu örnekleminin büyüklüğü/içeriği farklı olabilir.
    sample_fields = {"families", "query_rows_sha256", "query_images_sha256"}
    old_protocol = {
        key: value for key, value in old_config.items() if key not in sample_fields
    }
    new_protocol = {
        key: value for key, value in config.items() if key not in sample_fields
    }
    if old_protocol != new_protocol:
        raise ValueError(
            "Reuse run has different models, gallery, transforms or settings."
        )
    old_queries = pd.read_csv(
        directory / "queries.csv", dtype={"article_id": str, "product_code": str}
    )
    rows_hash = hashlib.sha256(old_queries.to_csv(index=False).encode()).hexdigest()
    if (
        list(old_queries.columns) != list(queries.columns)
        or len(old_queries) != old_config["families"]
        or old_queries["article_id"].duplicated().any()
        or rows_hash != old_config["query_rows_sha256"]
    ):
        raise ValueError("Reuse query rows differ from their saved config.")
    # Başındaki sıfırları koruyarak kaynak satırları yeni sorgu listesinde bulur.
    positions = pd.Index(queries["article_id"]).get_indexer(old_queries["article_id"])
    if (positions < 0).any() or not queries.iloc[positions].reset_index(
        drop=True
    ).equals(old_queries):
        raise ValueError(
            "Reuse articles/families are not aligned with the new queries."
        )
    for article in old_queries["article_id"]:
        if old_config["query_images_sha256"].get(article) != config[
            "query_images_sha256"
        ].get(article):
            raise ValueError(f"Reuse source image has changed: {article}")

    features, rankings = {}, {}
    for condition in CONDITIONS[1:]:
        with np.load(
            directory / f"{condition}_query_embeddings.npz", allow_pickle=False
        ) as cache:
            features[condition] = {
                model: cache[model] for model in ("frozen", "projection")
            }
        rankings[condition] = {}
        for model, dimension in (("frozen", 512), ("projection", 128)):
            validate_embeddings(features[condition][model], len(old_queries), dimension)
            rankings[condition][model] = load_query_rankings(
                directory / f"{condition}_{model}_top10_indices.npy",
                old_queries,
                gallery_size,
            )
    return positions, features, rankings


def save_preview(queries: pd.DataFrame, data_root: Path, seed: int, path: Path) -> None:
    """İlk üç sorgunun modelin gördüğü 224x224 hâllerini gösterir."""
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    rows = list(queries.head(3).itertuples(index=False))
    figure = Figure(figsize=(15, 3 * len(rows)))
    FigureCanvasAgg(figure)  # Önizlemeyi pencere açmadan PNG dosyasına yazar.
    axes = figure.subplots(len(rows), len(CONDITIONS), squeeze=False)
    for i, row in enumerate(rows):
        for j, condition in enumerate(CONDITIONS):
            tensor = query_tensor(
                row.article_id, data_root, condition, seed, preprocess
            )
            # Sadece önizleme için ImageNet normalizasyonunu geri alır.
            mean = torch.tensor(preprocess.mean)[:, None, None]
            std = torch.tensor(preprocess.std)[:, None, None]
            preview = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            axes[i, j].imshow(preview)
            axes[i, j].set_title(f"{condition}\n{row.article_id}", fontsize=9)
            axes[i, j].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    figure.clear()


def print_summary(comparison: pd.DataFrame) -> None:
    columns = [
        "model",
        "condition",
        "map@10",
        "recall@10",
        "hit_rate@5",
        "hit_rate@10",
        "delta_map@10",
    ]
    print("\n" + comparison[columns].round(4).to_string(index=False), flush=True)
    print("\nDelta: same model's transformed score minus its sampled clean score.")
    print("Controlled validation robustness only; not real-world upload accuracy.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, default=Path("results/query_robustness_pilot_v1")
    )
    parser.add_argument(
        "--reuse-from",
        type=Path,
        help="Reuse queries from a compatible completed pilot; encode only new sources.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate assets; no inference or output files.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.seed < 0:
        parser.error("Batch size must be positive and seed must be nonnegative.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Choose --device cpu explicitly.")

    started = time.perf_counter()
    table, data_root, galleries, saved_rankings = load_validation_assets()
    queries = sample_queries(table, args.families, args.seed)
    image_hashes = {}
    for article in queries["article_id"]:
        path = image_path(data_root, article)
        if not path.is_file():
            raise FileNotFoundError(f"Missing query image: {article}")
        with path.open("rb") as source:
            image_hashes[article] = hashlib.file_digest(source, "sha256").hexdigest()
    print(
        f"Queries: {len(queries)} families, one source article each; gallery: {len(table)}."
    )
    print(
        f"Conditions: clean + 5 independent changes; device: {args.device}; float32; no training."
    )
    output = args.output if args.output.is_absolute() else PROJECT_DIR / args.output
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "split": "validation",
        "families": args.families,
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "ks": list(KS),
        "query_rows_sha256": hashlib.sha256(
            queries.to_csv(index=False).encode()
        ).hexdigest(),
        "query_images_sha256": image_hashes,
        "gallery_rows_sha256": hashlib.sha256(
            table[["article_id", "product_code"]].to_csv(index=False).encode()
        ).hexdigest(),
        "model_sha256": MODEL_SHA256,
        "gallery_sha256": VALIDATION_SHA256,
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision.__version__,
        "pillow_version": PILLOW_VERSION,
        "numpy_version": np.__version__,
        "transforms": TRANSFORM_CONFIG,
        "preprocessing": "RGB; resized short edge 256 before each change; then ImageNet1K V1 weights.transforms; float32; L2",
        "relevance": "same product_code; source article_id excluded in every condition",
        "ap_denominator": "min(K, relevant gallery articles after source exclusion)",
    }
    reused_indices = np.array([], dtype=np.int64)
    reused_features, reused_rankings = {}, {}
    reuse_directory = None
    if args.reuse_from is not None:
        reuse_directory = (
            args.reuse_from
            if args.reuse_from.is_absolute()
            else PROJECT_DIR / args.reuse_from
        )
        if reuse_directory.resolve() == output.resolve():
            raise ValueError("Reuse source and output must be separate directories.")
        reused_indices, reused_features, reused_rankings = load_reusable_run(
            reuse_directory, config, queries, len(table)
        )
    new_mask = np.ones(len(queries), dtype=bool)
    new_mask[reused_indices] = False
    new_indices = np.flatnonzero(new_mask)
    new_queries = queries.iloc[new_indices].reset_index(drop=True)
    print(
        f"Reusable sources: {len(reused_indices)}; remaining sources: {len(new_queries)}."
    )
    if args.check_only:
        print(
            "Assets, reuse caches and row order verified. No inference or output files created."
        )
        return

    config_path = output / "config.json"
    if output.exists():
        if (
            not config_path.is_file()
            or json.loads(config_path.read_text("utf-8")) != config
        ):
            raise FileExistsError(
                "Output belongs to another protocol. Choose a new --output directory."
            )
        if (output / "metrics.json").is_file():
            print(
                "Completed run found; displaying saved comparison without new inference."
            )
            print_summary(pd.read_csv(output / "comparison.csv"))
            return
    else:
        output.mkdir(parents=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        queries.to_csv(output / "queries.csv", index=False)

    if not (output / "query_preview.png").is_file():
        save_preview(queries, data_root, args.seed, output / "query_preview.png")
    query_groups = queries["product_code"].to_numpy()
    gallery_groups = table["product_code"].to_numpy()
    selected_indices = queries["gallery_index"].to_numpy()
    family_sizes = table["product_code"].value_counts()
    relevant_counts = queries["product_code"].map(family_sizes).to_numpy() - 1
    comparison_rows, query_scores, timings = [], [], []
    clean_metrics, models = {}, None
    for condition in CONDITIONS:
        condition_started = time.perf_counter()
        inference_seconds, cache_reused = 0.0, True
        encoded_count, reused_count = 0, 0
        if condition != "clean":
            cache_path = output / f"{condition}_query_embeddings.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as cache:
                    features = {model: cache[model] for model in galleries}
                reused_count = len(queries)
            else:
                features = {
                    model: np.empty((len(queries), dimension), dtype=np.float32)
                    for model, dimension in (("frozen", 512), ("projection", 128))
                }
                reused_count = len(reused_indices)
                for model in galleries:
                    if reused_count:
                        features[model][reused_indices] = reused_features[condition][
                            model
                        ]
                if len(new_queries):
                    if models is None:
                        models = load_models(args.device)
                    encode_started = time.perf_counter()
                    new_features = encode_queries(
                        new_queries,
                        data_root,
                        condition,
                        args.seed,
                        models,
                        args.device,
                        args.batch_size,
                    )
                    inference_seconds = time.perf_counter() - encode_started
                    encoded_count, cache_reused = len(new_queries), False
                    for model in galleries:
                        features[model][new_indices] = new_features[model]
                for model, dim in (("frozen", 512), ("projection", 128)):
                    validate_embeddings(features[model], len(queries), dim)
                temporary = output / f"{condition}_query_embeddings.tmp.npz"
                np.savez(temporary, **features)
                temporary.replace(cache_path)
            for model, dim in (("frozen", 512), ("projection", 128)):
                validate_embeddings(features[model], len(queries), dim)

        for model, gallery in galleries.items():
            if condition == "clean":
                # Skor matrisi yeniden hesaplanmaz: eski top-10 satırlarını okur.
                ranks = saved_rankings[model][selected_indices]
                metrics, per_query = metrics_from_rankings(
                    ranks, query_groups, gallery_groups, relevant_counts, KS
                )
                clean_metrics[model] = metrics
            else:
                ranks_path = output / f"{condition}_{model}_top10_indices.npy"
                if ranks_path.is_file():
                    ranks = load_query_rankings(ranks_path, queries, len(table))
                else:
                    ranks = np.empty((len(queries), max(KS)), dtype=np.int64)
                    if len(reused_indices):
                        ranks[reused_indices] = reused_rankings[condition][model]
                    if len(new_indices):
                        _, new_ranks, _ = evaluate_queries(
                            features[model][new_indices],
                            gallery,
                            query_groups[new_indices],
                            gallery_groups,
                            new_queries["article_id"].to_numpy(),
                            table["article_id"].to_numpy(),
                            KS,
                        )
                        ranks[new_indices] = new_ranks
                metrics, per_query = metrics_from_rankings(
                    ranks, query_groups, gallery_groups, relevant_counts, KS
                )
            ranks_path = output / f"{condition}_{model}_top10_indices.npy"
            temporary = ranks_path.with_suffix(".tmp.npy")
            np.save(temporary, ranks, allow_pickle=False)
            temporary.replace(ranks_path)
            deltas = {
                f"delta_{name}": value - clean_metrics[model][name]
                for name, value in metrics.items()
            }
            comparison_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "queries": len(queries),
                    **metrics,
                    **deltas,
                }
            )
            scores = queries.copy()
            scores["model"], scores["condition"] = model, condition
            for name, values in per_query.items():
                scores[name] = values
            query_scores.append(scores)
        timings.append(
            {
                "condition": condition,
                "cache_reused": cache_reused,
                "reused_queries": reused_count,
                "encoded_queries": encoded_count,
                "inference_seconds": inference_seconds,
                "total_seconds": time.perf_counter() - condition_started,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    pd.concat(query_scores, ignore_index=True).to_csv(
        output / "per_query_metrics.csv", index=False
    )
    report = {
        "benchmark": "controlled validation query robustness, not real-world upload success",
        "source_queries": len(queries),
        "transformed_queries": len(queries) * 5,
        "reused_source_queries": len(reused_indices),
        "reuse_from": str(reuse_directory) if reuse_directory is not None else None,
        "gallery_articles": len(table),
        "clean_source": "saved top-10 rankings, same sampled source articles",
        "timings": timings,
        "elapsed_seconds": time.perf_counter() - started,
        "comparison": comparison_rows,
        "note": "Multiple changes of one source are correlated; they are not independent new products.",
    }
    temporary_report = output / "metrics.tmp.json"
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_report.replace(output / "metrics.json")
    print_summary(comparison)
    print(f"\nSaved results and query preview: {output}")


if __name__ == "__main__":
    main()
