"""Verify each stage separately before running a larger image job."""

import json
import time
from pathlib import Path


def main() -> None:
    started = time.perf_counter()
    print("1/5 Importing PyTorch and Torchvision...", flush=True)
    import torch
    from data import PROJECT_DIR, image_path, load_manifest
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18

    torch.hub.set_dir(str(PROJECT_DIR / "model_cache" / "hub"))
    print(f"2/5 PyTorch {torch.__version__}; checking CUDA...", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; stopping rather than silently switching to CPU."
        )
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    probe = torch.ones(4, device=device)
    torch.cuda.synchronize()
    print(f"GPU arithmetic check: {(probe * 2).sum().item()}", flush=True)

    print("3/5 Loading cached ImageNet ResNet18 weights...", flush=True)
    weights = ResNet18_Weights.IMAGENET1K_V1
    # The setup downloads the official checkpoint separately, with a timeout.
    cache = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not cache.is_file():
        raise FileNotFoundError(f"Model weights have not been downloaded: {cache}")
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).to(device).eval()

    print("4/5 Reading a saved validation image...", flush=True)
    table, root = load_manifest()
    article_id = table.loc[table["split"] == "validation", "article_id"].iloc[0]
    path = image_path(root, article_id)
    with Image.open(path) as opened:
        batch = weights.transforms()(opened.convert("RGB")).unsqueeze(0).to(device)

    print("5/5 Computing a normalized 512-dimensional embedding...", flush=True)
    with torch.inference_mode():
        embedding = torch.nn.functional.normalize(encoder(batch), dim=1)
    torch.cuda.synchronize()
    norm = embedding.norm(dim=1).item()
    if (
        embedding.shape != (1, 512)
        or not torch.isfinite(embedding).all()
        or abs(norm - 1) > 1e-5
    ):
        raise RuntimeError("Embedding shape, finiteness or normalization check failed.")
    report = {
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "article_id": article_id,
        "input_shape": list(batch.shape),
        "embedding_shape": list(embedding.shape),
        "embedding_norm": norm,
        "seconds": time.perf_counter() - started,
    }
    output = PROJECT_DIR / "results"
    output.mkdir(exist_ok=True)
    (output / "resnet_smoke.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
