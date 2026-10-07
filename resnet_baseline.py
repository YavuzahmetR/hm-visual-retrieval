"""Frozen ResNet18 retrieval baseline on the original validation split only."""

import json
import time

import numpy as np
import torch
from data import PROJECT_DIR, image_path, load_manifest
from PIL import Image
from retrieval_metrics import evaluate
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm
from visual_examples import save_examples


class ValidationImages(Dataset):
    def __init__(self, table, root, transform):
        self.ids = table["article_id"].tolist()
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(image_path(self.root, self.ids[index])) as opened:
            return self.transform(opened.convert("RGB"))


def main() -> None:
    started = time.perf_counter()
    print(
        "Loading original validation split; no training or test evaluation.", flush=True
    )
    table, root = load_manifest()
    validation = table.loc[table["split"] == "validation"].reset_index(drop=True)
    missing = [
        value
        for value in validation["article_id"]
        if not image_path(root, value).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} validation images missing; finish extraction first."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; stopping instead of switching to CPU.")
    torch.hub.set_dir(str(PROJECT_DIR / "model_cache/hub"))
    weights = ResNet18_Weights.IMAGENET1K_V1
    if not (
        PROJECT_DIR / "model_cache/hub/checkpoints/resnet18-f37072fd.pth"
    ).is_file():
        raise FileNotFoundError(
            "Run setup to download the official ResNet18 checkpoint first."
        )
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).to("cuda").eval()
    loader = DataLoader(
        ValidationImages(validation, root, weights.transforms()),
        batch_size=32,
        shuffle=False,
        num_workers=0,  # Simple and safe for both Windows scripts and notebooks.
        pin_memory=True,
    )
    features = []
    with torch.inference_mode():
        for images in tqdm(loader, desc="Validation embeddings", mininterval=5):
            batch_features = torch.nn.functional.normalize(
                encoder(images.to("cuda", non_blocking=True)), dim=1
            )
            features.append(batch_features.cpu().numpy())
    embeddings = np.concatenate(features)
    print(
        "Evaluating cosine retrieval against the complete validation gallery...",
        flush=True,
    )
    metrics, rankings = evaluate(embeddings, validation["product_code"].to_numpy())
    output = PROJECT_DIR / "results"
    output.mkdir(exist_ok=True)
    np.save(output / "validation_embeddings.npy", embeddings, allow_pickle=False)
    np.save(output / "validation_top10_indices.npy", rankings, allow_pickle=False)
    validation[["article_id", "product_code"]].to_csv(
        output / "validation_embedding_rows.csv", index=False
    )
    report = {
        "model": "ResNet18 ImageNet1K V1; frozen; fc=Identity",
        "split": "validation",
        "queries": len(validation),
        "relevance": "same product_code, excluding the query image",
        "ap_denominator": "min(K, number of relevant gallery images)",
        "seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    (output / "resnet_validation_metrics.json").write_text(
        json.dumps(report, indent=2), "utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    save_examples(validation, root, rankings, output)
    print("Visual examples saved to results/validation_examples.png", flush=True)


if __name__ == "__main__":
    main()
