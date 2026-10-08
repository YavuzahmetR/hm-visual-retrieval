import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from visual_retrieval.data import PROJECT_DIR, load_manifest
from visual_retrieval.retrieval_metrics import evaluate

SEED = 42
EPOCHS = 5
FAMILIES_PER_BATCH = 64
MARGIN = 0.2
LEARNING_RATE = 1e-3

RESULTS_DIR = PROJECT_DIR / "results"
RUN_DIR = RESULTS_DIR / "projection_head_v1"


# Öğrenilecek tek katman: 512 boyut -> 128 boyut.
# Sonuçların uzunluğunu tekrar 1 yapıyoruz.
class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 128, bias=False)

    def forward(self, embeddings):
        return F.normalize(self.linear(embeddings), dim=1)


# Vektörlerin doğru ürünlerle aynı sırada olduğunu kontrol eder.
def load_cache(split, manifest):
    rows = pd.read_csv(
        RESULTS_DIR / f"{split}_embedding_rows.csv",
        dtype={"article_id": str, "product_code": str},
    )

    expected = manifest.loc[
        manifest["split"] == split,
        ["article_id", "product_code"],
    ].reset_index(drop=True)

    if not rows.equals(expected):
        raise ValueError(
            f"{split}: cache satırları manifestle eşleşmiyor."
        )

    embeddings = np.load(
        RESULTS_DIR / f"{split}_embeddings.npy",
        allow_pickle=False,
    )

    if embeddings.shape != (len(rows), 512):
        raise ValueError(f"{split}: embedding boyutu geçersiz.")

    if (
        embeddings.dtype != np.float32
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError(f"{split}: embedding sayıları geçersiz.")

    if not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1, atol=1e-5
    ):
        raise ValueError(
            f"{split}: embedding'ler normalize edilmemiş."
        )

    return embeddings, rows


# Her aile epoch içinde bir kez ziyaret edilir.
# Her ziyaret için iki farklı varyant rastgele seçilir.
def family_batches(families, rng):
    order = rng.permutation(len(families))
    batch_count = math.ceil(
        len(families) / FAMILIES_PER_BATCH
    )

    for group_ids in np.array_split(order, batch_count):
        selected_articles = []
        for group_id in group_ids:
            pair = rng.choice(families[group_id], size=2, replace=False)
            selected_articles.append(pair)
        indices = np.concatenate(selected_articles)

        labels = np.repeat(group_ids, 2)
        yield indices, labels


# Her ürün için:
# positive = batch'teki aynı aileden başka ürün,
# negative = batch'teki başka ailelerden en yakın ürün.
def hard_triplets(embeddings, labels):
    with torch.no_grad():
        distances = torch.cdist(embeddings, embeddings)

        same_family = labels[:, None] == labels[None, :]

        # Ürünün kendisi positive olamaz.
        same_family.fill_diagonal_(False)

        other_family = labels[:, None] != labels[None, :]

        positive_indices = distances.masked_fill(
            ~same_family, -torch.inf
        ).argmax(dim=1)

        negative_indices = distances.masked_fill(
            ~other_family, torch.inf
        ).argmin(dim=1)

    return positive_indices, negative_indices


# Validation vektörlerini öğrenilmiş katmandan geçirir.
def project_validation(head, features):
    head.eval()
    batches = []

    with torch.inference_mode():
        for batch in features.split(2048):
            batches.append(head(batch).cpu().numpy())

    return np.concatenate(batches)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA bulunamadı; GPU ayarını kontrol edelim."
        )

    if RUN_DIR.exists():
        raise FileExistsError(
            "projection_head_v1 zaten mevcut. "
            "Eski deneyin üzerine yazılmadı."
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)

    print(
        "Kaydedilmiş train ve validation vektörleri okunuyor...",
        flush=True,
    )

    manifest, _ = load_manifest()

    train_array, train_rows = load_cache("train", manifest)
    val_array, val_rows = load_cache("validation", manifest)

    # Aynı product_code'a ait satır numaralarını bir araya toplar.
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

    # Önceki baseline sonucunu dosyadan okur.
    baseline = json.loads(
        (
            RESULTS_DIR / "resnet_validation_metrics.json"
        ).read_text("utf-8")
    )["metrics"]

    # Küçük vektör tablolarını GPU'ya bir kez taşıyoruz.
    train_features = torch.from_numpy(train_array).to("cuda")
    val_features = torch.from_numpy(val_array).to("cuda")

    head = ProjectionHead().to("cuda")

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    criterion = nn.TripletMarginLoss(
        margin=MARGIN,
        p=2,
    )

    RUN_DIR.mkdir()

    config = {
        "seed": SEED,
        "epochs": EPOCHS,
        "families_per_batch": FAMILIES_PER_BATCH,
        "variants_per_family": 2,
        "margin": MARGIN,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 1e-4,
        "head": "Linear(512, 128, bias=False) + L2 normalization",
        "encoder": (
            "Frozen ResNet18 ImageNet1K V1; "
            "cached 512-D embeddings"
        ),
        "selection_metric": "validation map@10",
        "relevance": "same product_code; self-match excluded",
        "ap_denominator": (
            "min(K, number of relevant gallery images)"
        ),
        "torch_version": str(torch.__version__),
    }

    (RUN_DIR / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        f"Train: {len(train_rows)} ürün, "
        f"{len(families)} aile"
    )
    print(
        "Öğrenilecek parametre:",
        sum(p.numel() for p in head.parameters()),
    )
    print(
        f"Frozen baseline mAP@10: {baseline['map@10']:.4f}",
        flush=True,
    )

    history = []
    best_score = -float("inf")

    total_batches = math.ceil(
        len(families) / FAMILIES_PER_BATCH
    )

    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        head.train()

        loss_sum = 0.0
        sample_count = 0

        batches = family_batches(families, rng)

        progress = tqdm(
            batches,
            total=total_batches,
            desc=f"Epoch {epoch}/{EPOCHS}",
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
                raise RuntimeError(
                    "Loss geçersiz; eğitim durduruldu."
                )

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

        # Önceki baseline ile aynı galeri ve aynı metrikler.
        metrics, rankings = evaluate(
            val_embeddings,
            val_rows["product_code"].to_numpy(),
        )

        val_seconds = time.perf_counter() - started

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            **{
                f"val_{key}": value
                for key, value in metrics.items()
            },
            "train_seconds": train_seconds,
            "validation_seconds": val_seconds,
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            RUN_DIR / "history.csv",
            index=False,
        )

        # Son epoch'u değil, en iyi validation epoch'unu saklar.
        if metrics["map@10"] > best_score:
            best_score = metrics["map@10"]

            torch.save(
                {
                    "state_dict": {
                        name: value.detach().cpu().clone()
                        for name, value
                        in head.state_dict().items()
                    },
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "config": config,
                },
                RUN_DIR / "best.pt",
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

            val_rows.to_csv(
                RUN_DIR / "validation_embedding_rows.csv",
                index=False,
            )

            report = {
                "best_epoch": epoch,
                "validation_metrics": metrics,
                "frozen_baseline_metrics": baseline,
                "map@10_delta": (
                    best_score - baseline["map@10"]
                ),
                "test_evaluated": False,
            }

            (RUN_DIR / "metrics.json").write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )

        print(
            f"Epoch {epoch} | "
            f"loss={row['train_loss']:.4f} | "
            f"mAP@10={metrics['map@10']:.4f} | "
            f"Recall@10={metrics['recall@10']:.4f} | "
            f"HitRate@10={metrics['hit_rate@10']:.4f}",
            flush=True,
        )

    print(
        f"\nEn iyi projection mAP@10: {best_score:.4f}"
    )
    print(
        f"Frozen baseline mAP@10: {baseline['map@10']:.4f}"
    )
    print(
        f"Fark: {best_score - baseline['map@10']:+.4f}"
    )
    print("Sonuçlar:", RUN_DIR)


if __name__ == "__main__":
    main()
