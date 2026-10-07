import argparse
import hashlib
import json
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from data import PROJECT_DIR, image_path, load_manifest


BATCH_SIZE = 32
EMBEDDING_DIM = 512
RESULTS_DIR = PROJECT_DIR / "results"


# Fotoğrafı açar ve ResNet'in beklediği dönüşümü uygular.
class ProductImages(Dataset):
    def __init__(self, table, root, transform):
        self.article_ids = table["article_id"].tolist()
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, index):
        path = image_path(self.root, self.article_ids[index])

        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def make_loader(table, root, transform, workers):
    dataset = ProductImages(table, root, transform)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        timeout=120 if workers > 0 else 0,
    )


# Önceki validation hesabıyla aynı encoder ve preprocessing.
def load_encoder():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadı. GPU ayarını kontrol edelim.")

    torch.hub.set_dir(str(PROJECT_DIR / "model_cache" / "hub"))

    weights_path = (
        PROJECT_DIR
        / "model_cache"
        / "hub"
        / "checkpoints"
        / "resnet18-f37072fd.pth"
    )

    if not weights_path.is_file():
        raise FileNotFoundError(f"Model ağırlığı bulunamadı: {weights_path}")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Önceden indirilmiş ResNet18 yükleniyor...", flush=True)

    weights = ResNet18_Weights.IMAGENET1K_V1
    encoder = resnet18(weights=weights)

    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False)
    encoder = encoder.to("cuda").eval()

    return encoder, weights.transforms()


def encode_batch(encoder, images):
    images = images.to("cuda", non_blocking=True)
    embeddings = encoder(images)

    return torch.nn.functional.normalize(embeddings, dim=1)


# Küçük bir train örneği üzerinde iki yükleme yöntemini karşılaştırır.
def benchmark(train, root, encoder, transform):
    sample_indices = np.linspace(
        0, len(train) - 1, num=min(512, len(train)), dtype=int
    )
    sample = train.iloc[sample_indices]
    measurements = []

    with torch.inference_mode():
        for workers in (0, 2):
            print(f"\nnum_workers={workers} ölçülüyor...", flush=True)

            loader = make_loader(sample, root, transform, workers)

            # İlk batch GPU hazırlığı ve worker başlangıcını da içerir.
            setup_started = time.perf_counter()
            iterator = iter(loader)
            first_batch = next(iterator)

            encode_batch(encoder, first_batch)
            torch.cuda.synchronize()

            startup_seconds = time.perf_counter() - setup_started

            # Devam eden batch'lerin gerçek işlem hızını ölçüyoruz.
            started = time.perf_counter()
            image_count = 0

            for images in tqdm(
                iterator,
                total=len(loader) - 1,
                desc=f"Benchmark workers={workers}",
            ):
                encode_batch(encoder, images)
                image_count += len(images)

            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            images_per_second = image_count / seconds

            estimated_minutes = (
                startup_seconds + len(train) / images_per_second
            ) / 60

            measurements.append((images_per_second, workers))

            print(f"Başlangıç: {startup_seconds:.1f} saniye")
            print(f"Hız: {images_per_second:.1f} görsel/saniye")
            print(f"Tahmini tam işlem: {estimated_minutes:.1f} dakika")

            del iterator, loader

    best_workers = max(measurements)[1]
    print(f"\nBu ölçümde daha hızlı seçenek: --workers {best_workers}")
    print("Benchmark bitti; tam train cache henüz oluşturulmadı.")


# İlerleme bilgisini önce geçici dosyaya, sonra gerçek dosyaya yazar.
def save_progress(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_cache(train, root, encoder, transform, workers):
    RESULTS_DIR.mkdir(exist_ok=True)

    final_path = RESULTS_DIR / "train_embeddings.npy"
    partial_path = RESULTS_DIR / "train_embeddings.partial.npy"
    progress_path = RESULTS_DIR / "train_cache_progress.json"
    rows_path = RESULTS_DIR / "train_embedding_rows.csv"

    if final_path.exists():
        raise FileExistsError(
            "train_embeddings.npy zaten mevcut. Üzerine yazılmadı."
        )

    rows = train[["article_id", "product_code"]].reset_index(drop=True)

    # Devam ederken aynı ürünleri, aynı sırayla işlediğimizi doğrular.
    protocol = (
        f"resnet18-imagenet1k-v1|weights.transforms|"
        f"float32|l2-normalized|torch={torch.__version__}"
    )
    signature = hashlib.sha256(
        (protocol + rows.to_csv(index=False)).encode("utf-8")
    ).hexdigest()

    shape = (len(rows), EMBEDDING_DIM)

    if partial_path.exists() != progress_path.exists():
        raise RuntimeError(
            "Geçici cache ile ilerleme dosyası birlikte bulunamadı. "
            "Hata çıktısını paylaş; dosyaları şimdilik silme."
        )

    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))

        if state["signature"] != signature:
            raise RuntimeError("Mevcut cache farklı veri veya modele ait.")

        completed = state["completed_rows"]
        cache = np.load(partial_path, mmap_mode="r+", allow_pickle=False)

        if (
            cache.shape != shape
            or cache.dtype != np.float32
            or not 0 <= completed <= len(rows)
        ):
            raise RuntimeError("Geçici cache boyutu veya ilerlemesi geçersiz.")

        print(f"Kaldığı yerden devam: {completed}/{len(rows)}", flush=True)

    else:
        cache = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )

        completed = 0
        state = {
            "signature": signature,
            "protocol": protocol,
            "shape": list(shape),
            "completed_rows": 0,
            "complete": False,
        }
        save_progress(progress_path, state)

    remaining = train.iloc[completed:]
    loader = make_loader(remaining, root, transform, workers)

    with torch.inference_mode():
        for images in tqdm(loader, desc="Train embeddings"):
            vectors = encode_batch(encoder, images).cpu().numpy()

            if not np.isfinite(vectors).all():
                raise RuntimeError("Embedding içinde geçersiz sayı oluştu.")

            if not np.allclose(
                np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5
            ):
                raise RuntimeError("Embedding normalizasyonu geçersiz.")

            end = completed + len(vectors)
            cache[completed:end] = vectors

            # Veriler kaydedildikten sonra ilerleme güncellenir.
            cache.flush()
            completed = end
            state["completed_rows"] = completed
            save_progress(progress_path, state)

    # Windows'ta dosyayı yeniden adlandırmadan önce memmap'i kapatır.
    del cache

    rows.to_csv(rows_path, index=False)
    partial_path.replace(final_path)

    state["complete"] = True
    save_progress(progress_path, state)

    print(f"\nTamamlandı: {shape}")
    print(f"Embedding dosyası: {final_path}")
    print(f"Satırların ürün kimlikleri: {rows_path}")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--build", action="store_true")
    parser.add_argument("--workers", type=int, choices=(0, 2), default=0)
    args = parser.parse_args()

    table, root = load_manifest()

    # Yalnızca train; validation ve test hesapları burada yapılmaz.
    train = table.loc[table["split"] == "train"].reset_index(drop=True)

    print(f"Train görselleri: {len(train)}", flush=True)
    encoder, transform = load_encoder()

    if args.benchmark:
        benchmark(train, root, encoder, transform)
    else:
        build_cache(train, root, encoder, transform, args.workers)


# Windows'taki paralel veri yükleyicileri için gerekli.
if __name__ == "__main__":
    main()