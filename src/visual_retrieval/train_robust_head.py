"""Continue the 128D head with online augmentation; the CNN stays frozen.

This is one augmented continuation, not a controlled augmentation ablation.
Only train images update weights. Existing validation vectors are reused.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from visual_retrieval.data import PROJECT_DIR, image_path, load_manifest
from visual_retrieval.demo_retrieval import RESNET_SHA256, read_uploaded_image
from visual_retrieval.evaluate_query_robustness import (
    CONDITIONS,
    MODEL_SHA256,
    PROTOCOL_VERSION,
    TRANSFORM_CONFIG,
    VALIDATION_SHA256,
    load_models,
    load_reusable_run,
    load_validation_assets,
    make_query_image,
    sample_queries,
    stable_seed,
)
from visual_retrieval.retrieval_metrics import evaluate, evaluate_queries
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from visual_retrieval.train_projection_head import hard_triplets, project_validation

PREPROCESSING = "ResNet18_Weights.IMAGENET1K_V1.transforms()"


def read_assets(benchmark: Path) -> tuple:
    """Train satırlarını ve tamamlanmış validation cache'lerini okur; inference yok."""
    manifest, data_root = load_manifest()
    train = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    if len(train) != 64912 or train["product_code"].nunique() != 18447:
        raise ValueError(
            "Expected the original 64,912-item / 18,447-family train split."
        )
    missing = [a for a in train["article_id"] if not image_path(data_root, a).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing train images, first examples: {missing[:5]}")
    validation, _, galleries, _ = load_validation_assets()
    config = json.loads((benchmark / "config.json").read_text("utf-8"))
    queries = sample_queries(validation, 2306, 42)
    gallery_hash = hashlib.sha256(
        validation[["article_id", "product_code"]].to_csv(index=False).encode()
    ).hexdigest()
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "split": "validation",
        "families": 2306,
        "seed": 42,
        "ks": [1, 5, 10],
        "transforms": TRANSFORM_CONFIG,
        "model_sha256": MODEL_SHA256,
        "gallery_sha256": VALIDATION_SHA256,
        "gallery_rows_sha256": gallery_hash,
        "torch_version": str(torch.__version__),
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Robustness cache does not match the original validation protocol."
        )
    positions, features, _ = load_reusable_run(
        benchmark, config, queries, len(validation)
    )
    if not np.array_equal(positions, np.arange(len(queries))):
        raise ValueError(
            "Expected all 2,306 cached query rows in their original order."
        )
    report = json.loads((benchmark / "metrics.json").read_text("utf-8"))
    baseline = [row for row in report["comparison"] if row["model"] == "projection"]
    if {row["condition"] for row in baseline} != set(CONDITIONS) or len(baseline) != 6:
        raise ValueError("Expected the six completed epoch-15 conditions.")
    # Temiz genel validation sonucu zaten checkpoint'te kayıtlı; tekrar hesaplamıyoruz.
    parent = torch.load(
        PROJECT_DIR / "models/projection_head_epoch15.pt",
        map_location="cpu",
        weights_only=True,
    )
    return (
        train,
        data_root,
        validation,
        queries,
        galleries["frozen"],
        features,
        baseline,
        parent,
    )


def training_batch(
    articles: list[str], data_root: Path, epoch: int, seed: int, preprocess
) -> torch.Tensor:
    """Görsellerden batch üretir: %30 temiz, %70 bir veya iki hafif değişiklik."""
    tensors = []
    for article in articles:
        image = read_uploaded_image(image_path(data_root, article).read_bytes())
        identity = f"{article}:epoch{epoch}"
        rng = np.random.default_rng(stable_seed(seed, identity, "train"))
        if rng.random() < 0.7:
            changes = rng.choice(CONDITIONS[1:], size=2, replace=False)
            image = make_query_image(image, str(changes[0]), identity, seed)
            if rng.random() < 0.25:
                image = make_query_image(image, str(changes[1]), identity, seed)
        tensors.append(preprocess(image))
    return torch.stack(tensors)


def batches(families: list, rng: np.random.Generator, families_per_batch: int):
    """Her aileyi bir kez ziyaret eder; pozitifler iki farklı article'dır."""
    count = math.ceil(len(families) / families_per_batch)
    for groups in np.array_split(rng.permutation(len(families)), count):
        selected_articles = []
        for group_id in groups:
            pair = rng.choice(families[group_id], 2, replace=False)
            selected_articles.append(pair)
        indices = np.concatenate(selected_articles)
        yield indices, np.repeat(groups, 2)


def validate_head(
    head: nn.Module,
    gallery_features: torch.Tensor,
    query_features: dict,
    validation: pd.DataFrame,
    queries: pd.DataFrame,
) -> tuple[dict, list[dict], np.ndarray]:
    """Aynı frozen512 vektörlerden yeni128 üretir; yalnızca yeni head'i değerlendirir."""
    gallery = project_validation(head, gallery_features)
    clean, _ = evaluate(gallery, validation["product_code"].to_numpy())
    comparison = []
    for condition in CONDITIONS:
        if condition == "clean":
            projected = gallery[queries["gallery_index"].to_numpy()]
        else:
            projected = project_validation(head, query_features[condition])
        metrics, _, _ = evaluate_queries(
            projected,
            gallery,
            queries["product_code"].to_numpy(),
            validation["product_code"].to_numpy(),
            queries["article_id"].to_numpy(),
            validation["article_id"].to_numpy(),
        )
        comparison.append({"condition": condition, **metrics})
    return clean, comparison, gallery


def robustness_mean(comparison: list[dict]) -> float:
    """Beş değişikliğin mAP@10 ortalaması; temiz sorgu bu ortalamaya dahil değil."""
    return float(
        np.mean([row["map@10"] for row in comparison if row["condition"] != "clean"])
    )


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    """Ağırlık ve eğitim durumunu geçici dosya üzerinden kaydeder."""
    temporary = path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/robust_head_v1"))
    parser.add_argument(
        "--benchmark", type=Path, default=Path("results/query_robustness_full_v1")
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--families-per-batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pilot-batches",
        type=int,
        default=0,
        help="0: all families; >0: short hardware pilot",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Read-only; no inference or training"
    )
    args = parser.parse_args()
    if not (
        1 <= args.epochs <= 5
        and args.patience > 0
        and 2 <= args.families_per_batch <= 64
    ):
        parser.error(
            "Choose 1-5 epochs, positive patience and 2-64 families per batch."
        )
    if (
        not math.isfinite(args.lr)
        or args.lr <= 0
        or args.pilot_batches < 0
        or args.seed < 0
    ):
        parser.error(
            "Learning rate must be positive; pilot batches and seed cannot be negative."
        )
    output, benchmark = args.output.resolve(), args.benchmark.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; no overwrite: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training experiment.")

    train, data_root, validation, queries, frozen, features, baseline, parent = (
        read_assets(benchmark)
    )
    minimum_clean = float(parent["validation_metrics"]["map@10"])
    best_score = robustness_mean(baseline)
    print(
        f"Train: {len(train)} articles / 18,447 families; batch: {args.families_per_batch * 2} images."
    )
    print(
        f"Epoch-15 references: clean all-image mAP@10={minimum_clean:.6f}; robustness mean={best_score:.6f}."
    )
    if args.check_only:
        print(
            "Train paths and validation caches verified. No inference, training or outputs."
        )
        return

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    encoder, head, preprocess = load_models("cuda")
    # CNN ve BN running statistics sabit; yalnızca head optimizer'a girer.
    encoder.eval()
    head.requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    gallery_tensor = torch.from_numpy(frozen).to("cuda")
    query_tensors = {
        c: torch.from_numpy(features[c]["frozen"]).to("cuda") for c in CONDITIONS[1:]
    }
    families = list(train.groupby("product_code", sort=True).indices.values())
    if any(len(family) < 2 for family in families):
        raise ValueError("Every train family must have two distinct articles.")
    total_batches = math.ceil(len(families) / args.families_per_batch)
    used_batches = min(total_batches, args.pilot_batches or total_batches)
    config = {
        **vars(args),
        "output": str(output),
        "benchmark": str(benchmark),
        "head_dim": 128,
        "parent_epoch": 15,
        "parent_sha256": MODEL_SHA256["models/projection_head_epoch15.pt"],
        "encoder_sha256": RESNET_SHA256,
        "encoder_frozen": True,
        "bn_statistics": "eval/fixed",
        "preprocessing": PREPROCESSING,
        "precision": "float32",
        "margin": 0.2,
        "weight_decay": 1e-4,
        "augmentation": {
            "clean_probability": 0.3,
            "changes": TRANSFORM_CONFIG,
            "second_change_probability_given_augmented": 0.25,
        },
        "selection": "maximize mean transformed validation mAP@10, subject to clean all-image mAP@10 >= epoch15",
        "minimum_clean_map10": minimum_clean,
        "resume_mode": "epoch15 weights, new AdamW",
        "comparison_limit": "No unaugmented continuation control; augmentation effect is not isolated.",
        "torch_version": str(torch.__version__),
        "benchmark_query_sha256": json.loads(
            (benchmark / "config.json").read_text("utf-8")
        )["query_rows_sha256"],
        "pilot": bool(args.pilot_batches),
    }
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    comparison = [{**row, "model": "epoch15"} for row in baseline]
    history, waiting, best_epoch = [], 0, 0
    print("Training ONLY the head; CNN frozen; no test split; no model promotion.")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        encoder.eval()
        head.train()
        losses, sample_count = 0.0, 0
        progress = tqdm(total=used_batches, desc=f"epoch {epoch}/{args.epochs}")
        for step, (indices, labels) in enumerate(
            batches(families, rng, args.families_per_batch)
        ):
            if step >= used_batches:
                break
            images = training_batch(
                train.iloc[indices]["article_id"].tolist(),
                data_root,
                epoch,
                args.seed,
                preprocess,
            ).to("cuda")
            # no_grad kullanılır: head'in backward'ı bu frozen vektörleri kullanabilir.
            with torch.no_grad():
                embeddings = F.normalize(encoder(images), dim=1)
            optimizer.zero_grad(set_to_none=True)
            projected = head(embeddings)
            positive, negative = hard_triplets(
                projected, torch.from_numpy(labels).to("cuda")
            )
            loss = criterion(projected, projected[positive], projected[negative])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite train loss; experiment stopped.")
            loss.backward()
            optimizer.step()
            losses += loss.item() * len(indices)
            sample_count += len(indices)
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        progress.close()
        train_seconds = time.perf_counter() - started
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        clean, changed, gallery = validate_head(
            head, gallery_tensor, query_tensors, validation, queries
        )
        score = robustness_mean(changed)
        eligible = clean["map@10"] >= minimum_clean
        improved = eligible and score > best_score
        if improved:
            best_score, best_epoch, waiting = score, epoch, 0
        else:
            waiting += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": losses / sample_count,
                "train_seconds": train_seconds,
                "peak_cuda_mb": peak_mb,
                "clean_map@10": clean["map@10"],
                "robustness_mean_map@10": score,
                "clean_guard_passed": eligible,
                "selected": improved,
            }
        )
        comparison.extend({"model": f"aug_epoch{epoch}", **row} for row in changed)
        checkpoint = {
            "format": "robust_head_v1",
            "state_dict": head.state_dict(),
            "epoch": epoch,
            "parent_epoch": 15,
            "encoder_sha256": RESNET_SHA256,
            "preprocessing": PREPROCESSING,
            "config": config,
            "optimizer_state_dict": optimizer.state_dict(),
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "validation_metrics": clean,
            "robustness_metrics": changed,
            "robustness_mean_map10": score,
        }
        save_checkpoint(output / "last.pt", checkpoint)
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            np.save(output / "validation_embeddings.npy", gallery, allow_pickle=False)
            validation[["article_id", "product_code"]].to_csv(
                output / "validation_embedding_rows.csv", index=False
            )
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
        summary = {
            "selected_additional_epoch": best_epoch,
            "keep_epoch15": best_epoch == 0,
            "best_robustness_mean_map10": best_score,
            "minimum_clean_map10": minimum_clean,
            "test_evaluated": False,
            "pilot": bool(args.pilot_batches),
            "history": history,
        }
        (output / "metrics.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"Clean mAP@10: {clean['map@10']:.6f}; robustness mean: {score:.6f}; selected: {improved}; peak GPU: {peak_mb:.0f} MB."
        )
        print(
            f"Train measurement: {train_seconds:.1f}s / {used_batches} batches. Linear full-epoch estimate: {train_seconds / used_batches * total_batches / 60:.1f} minutes (approximate)."
        )
        if waiting >= args.patience:
            print("Early stopping: no eligible improvement.")
            break
    print(
        "Keep the published epoch-15 model."
        if best_epoch == 0
        else f"Candidate saved: {output / 'best.pt'}; demo/model unchanged."
    )
    if args.pilot_batches:
        print(
            "Hardware pilot only; do not publish/select its checkpoint as the final model."
        )


if __name__ == "__main__":
    main()
