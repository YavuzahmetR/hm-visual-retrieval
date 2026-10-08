import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from visual_retrieval.data import PROJECT_DIR, load_manifest
from visual_retrieval.retrieval_metrics import evaluate
from visual_retrieval.train_projection_head import (
    FAMILIES_PER_BATCH,
    ProjectionHead,
    family_batches,
    hard_triplets,
    load_cache,
    project_validation,
)

RESULTS_DIR = PROJECT_DIR / "results"
SOURCE_DIR = RESULTS_DIR / "projection_head_v1"
RUN_DIR = RESULTS_DIR / "projection_head_v2"

MAX_EPOCH = 20
PATIENCE = 3


# Yeni en iyi skor gelirse bekleme sayacı sıfırlanır.
def observe_score(score, best_score, waiting):
    improved = score > best_score

    if improved:
        return True, score, 0

    return False, best_score, waiting + 1


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadı.")

    if RUN_DIR.exists():
        raise FileExistsError(
            "projection_head_v2 zaten mevcut; üzerine yazılmadı."
        )

    checkpoint = torch.load(
        SOURCE_DIR / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    start_epoch = int(checkpoint["epoch"])
    source_config = checkpoint["config"]

    if start_epoch >= MAX_EPOCH:
        raise ValueError(
            "Başlangıç epoch'u MAX_EPOCH'tan küçük olmalı."
        )

    if source_config["families_per_batch"] != FAMILIES_PER_BATCH:
        raise ValueError(
            "Batch ayarı önceki deneyle eşleşmiyor."
        )

    print("Cache dosyaları okunuyor...", flush=True)

    manifest, _ = load_manifest()
    train_array, train_rows = load_cache("train", manifest)
    val_array, val_rows = load_cache("validation", manifest)

    families = list(
        train_rows.groupby(
            "product_code", sort=True
        ).indices.values()
    )

    if (
        len(families) < 2
        or any(len(rows) < 2 for rows in families)
    ):
        raise ValueError(
            "En az iki aile ve aile başına iki ürün gerekiyor."
        )

    seed = source_config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed + start_epoch)

    train_features = torch.from_numpy(train_array).to("cuda")
    val_features = torch.from_numpy(val_array).to("cuda")

    # Öğrenilmiş ağırlıkları yükler.
    head = ProjectionHead().to("cuda")
    head.load_state_dict(checkpoint["state_dict"])

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=source_config["learning_rate"],
        weight_decay=source_config["weight_decay"],
    )

    criterion = nn.TripletMarginLoss(
        margin=source_config["margin"],
        p=2,
    )

    # v1'de optimizer kaydı yok; yeni optimizer kullanılır.
    # Sonraki checkpoint'ler bu bilgileri de içerecek.
    mode = "weights_only_new_optimizer"

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        mode = "model_optimizer_and_rng_restored"

    config = {
        **source_config,
        "epochs": MAX_EPOCH,
        "patience": PATIENCE,
        "parent_run": SOURCE_DIR.name,
        "starting_epoch": start_epoch,
        "resume_mode": mode,
        "sampling_seed": seed + start_epoch,
        "torch_version": str(torch.__version__),
    }

    original_report = json.loads(
        (SOURCE_DIR / "metrics.json").read_text("utf-8")
    )
    baseline = original_report["frozen_baseline_metrics"]

    starting_score = float(
        checkpoint["validation_metrics"]["map@10"]
    )
    best_score = starting_score
    best_epoch = start_epoch
    waiting = 0

    previous_history = pd.read_csv(
        SOURCE_DIR / "history.csv"
    )
    history = previous_history.loc[
        previous_history["epoch"] <= start_epoch
    ].to_dict("records")

    RUN_DIR.mkdir()

    (RUN_DIR / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(history).to_csv(
        RUN_DIR / "history.csv",
        index=False,
    )

    # Devam eğitimi iyileştirmezse mevcut en iyi model korunur.
    for name in (
        "best.pt",
        "metrics.json",
        "validation_embeddings.npy",
        "validation_top10_indices.npy",
        "validation_embedding_rows.csv",
    ):
        shutil.copy2(
            SOURCE_DIR / name,
            RUN_DIR / name,
        )

    print(
        f"Başlangıç: epoch {start_epoch}, "
        f"mAP@10={starting_score:.4f}"
    )
    print(
        f"Hedef: en fazla epoch {MAX_EPOCH}; "
        f"patience={PATIENCE}"
    )
    print(f"Devam biçimi: {mode}", flush=True)

    total_batches = math.ceil(
        len(families) / FAMILIES_PER_BATCH
    )

    for epoch in range(start_epoch + 1, MAX_EPOCH + 1):
        started = time.perf_counter()
        head.train()

        loss_sum = 0.0
        sample_count = 0

        progress = tqdm(
            family_batches(families, rng),
            total=total_batches,
            desc=f"Epoch {epoch}/{MAX_EPOCH}",
        )

        for indices, labels in progress:
            indices = torch.as_tensor(
                indices,
                dtype=torch.long,
                device="cuda",
            )
            labels = torch.as_tensor(
                labels,
                dtype=torch.long,
                device="cuda",
            )

            projected = head(train_features[indices])

            positive, negative = hard_triplets(
                projected, labels
            )

            loss = criterion(
                projected,
                projected[positive],
                projected[negative],
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError("Loss geçersiz.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            value = loss.item()
            loss_sum += value * len(indices)
            sample_count += len(indices)

            progress.set_postfix(loss=f"{value:.4f}")

        train_seconds = time.perf_counter() - started

        print(
            "Validation retrieval ölçülüyor...",
            flush=True,
        )
        started = time.perf_counter()

        val_embeddings = project_validation(
            head, val_features
        )

        metrics, rankings = evaluate(
            val_embeddings,
            val_rows["product_code"].to_numpy(),
        )

        val_seconds = time.perf_counter() - started

        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            **{
                f"val_{key}": value
                for key, value in metrics.items()
            },
            "train_seconds": train_seconds,
            "validation_seconds": val_seconds,
        })

        pd.DataFrame(history).to_csv(
            RUN_DIR / "history.csv",
            index=False,
        )

        improved, best_score, waiting = observe_score(
            metrics["map@10"],
            best_score,
            waiting,
        )

        if improved:
            best_epoch = epoch

        # Bu kez optimizer ve rastgelelik durumları da kaydedilir.
        saved = {
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "epoch": epoch,
            "validation_metrics": metrics,
            "config": config,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "no_improvement_epochs": waiting,
        }

        save_checkpoint(
            RUN_DIR / "last.pt",
            saved,
        )

        if improved:
            save_checkpoint(
                RUN_DIR / "best.pt",
                saved,
            )

            np.save(
                RUN_DIR / "validation_embeddings.npy",
                val_embeddings,
                allow_pickle=False,
            )
            np.save(
                RUN_DIR / "validation_top10_indices.npy",
                rankings,
                allow_pickle=False,
            )

            report = {
                "best_epoch": epoch,
                "validation_metrics": metrics,
                "frozen_baseline_metrics": baseline,
                "map@10_delta": (
                    best_score - baseline["map@10"]
                ),
                "starting_map@10": starting_score,
                "gain_over_start": best_score - starting_score,
                "test_evaluated": False,
            }

            (RUN_DIR / "metrics.json").write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )

        print(
            f"Epoch {epoch} | "
            f"loss={loss_sum / sample_count:.4f} | "
            f"mAP@10={metrics['map@10']:.4f} | "
            f"Recall@10={metrics['recall@10']:.4f} | "
            f"HitRate@10={metrics['hit_rate@10']:.4f} | "
            f"iyileşmeyen epoch={waiting}/{PATIENCE}",
            flush=True,
        )

        if waiting >= PATIENCE:
            print(
                "Early stopping: üç epoch boyunca "
                "yeni en iyi skor yok."
            )
            break

    print(f"\nEn iyi epoch: {best_epoch}")
    print(f"En iyi mAP@10: {best_score:.4f}")
    print(
        "5 epoch sonucuna göre fark: "
        f"{best_score - starting_score:+.4f}"
    )
    print("Sonuçlar:", RUN_DIR)


if __name__ == "__main__":
    main()