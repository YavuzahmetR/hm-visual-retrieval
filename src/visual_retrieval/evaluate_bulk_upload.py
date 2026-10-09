"""Search local upload photos; calculate retrieval metrics only with known labels."""

import argparse
import base64
import hashlib
import io
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from visual_retrieval.data import PROJECT_DIR, image_path, load_manifest
from visual_retrieval.demo_retrieval import (
    HEAD_SHA256,
    RESNET_SHA256,
    _sha256,
    load_demo,
    rank_results,
    read_uploaded_image,
    validate_embeddings,
    validate_gallery_files,
    validate_gallery_rows,
)
from PIL import Image
from visual_retrieval.retrieval_metrics import metrics_from_rankings
from torch.nn import functional as F
from visual_retrieval.train_projection_head import ProjectionHead


def check_assets(checkpoint_path: Path) -> pd.DataFrame:
    """Modeli çalıştırmadan galeri sırasını, cache'i ve ağırlık dosyalarını kontrol et."""
    manifest, _ = load_manifest()
    catalog = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    expected = catalog[["article_id", "product_code"]]
    if len(catalog) != 8065 or catalog["product_code"].nunique() != 2306:
        raise ValueError("The original 8,065-product test gallery is required.")
    cache = PROJECT_DIR / "results/test_evaluation"
    rows = pd.read_csv(cache / "embedding_rows.csv", dtype=str)
    validate_gallery_rows(rows, expected)
    for name, dim in (
        ("frozen_embeddings.npy", 512),
        ("projection_embeddings.npy", 128),
    ):
        validate_embeddings(np.load(cache / name, mmap_mode="r"), len(rows), dim)
    validate_gallery_files(cache)
    weights = PROJECT_DIR / "model_cache/hub/checkpoints/resnet18-f37072fd.pth"
    published = PROJECT_DIR / "models/projection_head_epoch15.pt"
    if _sha256(weights) != RESNET_SHA256 or _sha256(published) != HEAD_SHA256:
        raise ValueError("Published encoder/head weights have changed.")
    protocol = "ResNet18 ImageNet1K V1|weights.transforms|float32|L2|"
    protocol += str(torch.__version__) + "|" + RESNET_SHA256
    signature = hashlib.sha256(
        (protocol + expected.to_csv(index=False)).encode()
    ).hexdigest()
    metadata = json.loads((cache / "cache_metadata.json").read_text("utf-8"))
    if metadata.get("protocol") != protocol or metadata.get("signature") != signature:
        raise ValueError("Gallery encoder, preprocessing, environment or rows differ.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if _sha256(checkpoint_path) != HEAD_SHA256 and (
        checkpoint.get("format") != "robust_head_v1"
        or checkpoint.get("encoder_sha256") != RESNET_SHA256
        or checkpoint.get("preprocessing")
        != "ResNet18_Weights.IMAGENET1K_V1.transforms()"
    ):
        raise ValueError(
            "Candidate must be a robust_head_v1 head for the unchanged frozen encoder/preprocessing."
        )
    head = ProjectionHead()
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    return catalog


def load_labels(
    path: Path | None, files: list[Path], catalog: pd.DataFrame
) -> pd.DataFrame | None:
    """Dosya adından etiket tahmin etme; ground truth kullanıcı tarafından verilmelidir."""
    if path is None:
        return None
    labels = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"filename", "product_code"}.issubset(labels.columns):
        raise ValueError("Labels require filename and product_code columns.")
    if "source_article_id" not in labels:
        labels["source_article_id"] = ""
    if labels["filename"].duplicated().any() or set(labels["filename"]) != {
        p.name for p in files
    }:
        raise ValueError("Labels must contain exactly one row for every input photo.")
    if not labels["product_code"].str.fullmatch(r"[0-9]{7}").all():
        raise ValueError(
            "product_code must retain all seven digits, including leading zeros."
        )
    if not labels["source_article_id"].str.fullmatch(r"(?:[0-9]{10})?").all():
        raise ValueError("source_article_id must be empty or contain ten digits.")
    by_article = catalog.set_index("article_id")["product_code"]
    counts = catalog["product_code"].value_counts()
    for row in labels.itertuples(index=False):
        if row.product_code not in counts:
            raise ValueError(
                f"{row.filename}: labelled family is absent from the gallery."
            )
        if (
            row.source_article_id
            and by_article.get(row.source_article_id) != row.product_code
        ):
            raise ValueError(
                f"{row.filename}: source article is absent or belongs to another family."
            )
        if counts[row.product_code] - bool(row.source_article_id) < 1:
            raise ValueError(f"{row.filename}: no relevant gallery item remains.")
    return labels.set_index("filename")


def thumbnail(image: Image.Image) -> str:
    """Rapor için küçük kopya üret; inference her zaman orijinal fotoğrafı kullanır."""
    small = image.copy()
    small.thumbnail((240, 240))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def write_html(
    path: Path,
    results: pd.DataFrame,
    query_thumbnails: dict[str, str],
    data_root: Path,
    model_label: str = "Projection head",
) -> None:
    """İnternet veya ek sunucu gerektirmeyen yerel bir görsel rapor oluştur."""
    sections, gallery_thumbnails = [], {}
    for query_number, (filename, matches) in enumerate(
        results.groupby("filename", sort=False), 1
    ):
        cards = []
        for row in matches.itertuples(index=False):
            if row.article_id not in gallery_thumbnails:
                try:
                    image = read_uploaded_image(
                        image_path(data_root, row.article_id).read_bytes()
                    )
                    gallery_thumbnails[row.article_id] = thumbnail(image)
                except (OSError, ValueError):
                    gallery_thumbnails[row.article_id] = ""
            source = gallery_thumbnails[row.article_id]
            picture = (
                f'<img src="{source}" alt="Product photo">'
                if source
                else "<p>Image unavailable</p>"
            )
            relation = ""
            if pd.notna(row.same_family) and row.same_family != "":
                relation = (
                    "Same family variant"
                    if str(row.same_family).lower() == "true"
                    else "Other family"
                )
            cards.append(
                f"<article>{picture}<b>#{row.rank} {escape(str(row.prod_name))}"
                f" · {escape(str(row.colour_group_name))}</b>"
                f"<p>Cosine: {row.cosine:.4f}<br>{relation}</p>"
                f"<details><summary>Product IDs</summary>Article: {row.article_id}"
                f"<br>Family: {row.product_code}</details></article>"
            )
        sections.append(
            f'<section><h2>Query {query_number:02d}</h2><img class="query" '
            f'src="{query_thumbnails[filename]}" alt="Query photo">'
            f'<div class="results">{"".join(cards)}</div></section>'
        )
    document = '<!doctype html><html lang="en"><meta charset="utf-8">'
    document += "<title>H&M upload preview</title><style>"
    document += (
        "body{font-family:Arial;margin:24px;background:#f4f4f4}section{margin:32px 0}"
    )
    document += ".results{display:flex;flex-wrap:wrap;gap:12px}article{width:240px;background:white;padding:12px}"
    document += "img{max-width:240px;height:240px;object-fit:contain;display:block}.query{margin-bottom:12px}"
    document += "</style><h1>H&M Visual Retrieval · Upload preview</h1>"
    document += f"<h2>{escape(model_label)}</h2>"
    document += "<p>Cosine is a ranking score, not a probability of correctness. "
    document += "Family labels, when supplied, are user-provided ground truth.</p>"
    document += "".join(sections) + "</html>"
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Folder containing JPG/PNG photos; not recursive.",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "results/bulk_upload_v1"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "models/projection_head_epoch15.pt",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional CSV: filename,product_code,source_article_id.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, choices=(5, 10), default=5)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate assets/labels without inference or outputs.",
    )
    args = parser.parse_args()
    if not args.input.is_dir() or args.batch_size < 1:
        parser.error(
            "--input must be an existing folder and --batch-size must be positive."
        )
    files = sorted(
        p
        for p in args.input.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        parser.error("The input folder contains no JPG/PNG photos.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device cpu.")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        parser.error(
            "Output must be a new or empty directory; existing results are preserved."
        )
    catalog = check_assets(args.checkpoint)
    labels = load_labels(args.labels, files, catalog)
    print(
        f"Photos: {len(files)}; gallery: {len(catalog)}; top K: {args.top_k}; device: {args.device}."
    )
    if args.check_only:
        print("Assets and labels verified; no image encoding or output files created.")
        return

    # CLI checkpoint'i açıkça seçer; demodaki varsayılanın değişmesi bunu etkilemez.
    resources = load_demo(args.device, head_version="epoch15")
    if _sha256(args.checkpoint) != HEAD_SHA256:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        resources.head.load_state_dict(checkpoint["state_dict"], strict=True)
        # Yeni head için eski 128D cache kullanılamaz; yalnızca frozen 512D cache geçerlidir.
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(catalog), 1024):
                frozen = torch.from_numpy(
                    np.array(resources.frozen[start : start + 1024], copy=True)
                ).to(args.device)
                chunks.append(resources.head(frozen).cpu().numpy())
        resources.projection = np.concatenate(chunks)
        validate_embeddings(resources.projection, len(catalog), 128)

    predictions, errors, rankings, query_names, query_thumbnails, hashes = (
        [],
        [],
        [],
        [],
        {},
        {},
    )
    article_indices = {value: i for i, value in enumerate(catalog["article_id"])}
    for start in range(0, len(files), args.batch_size):
        tensors, names = [], []
        for path in files[start : start + args.batch_size]:
            try:
                content = path.read_bytes()
                image = read_uploaded_image(content)
                tensors.append(resources.transform(image))
                names.append(path.name)
                query_thumbnails[path.name] = thumbnail(image)
                hashes[path.name] = hashlib.sha256(content).hexdigest()
            except (OSError, ValueError) as error:
                errors.append({"filename": path.name, "error": str(error)})
        if not tensors:
            continue
        with torch.inference_mode():
            batch = torch.stack(tensors).to(args.device)
            features = (
                resources.head(F.normalize(resources.encoder(batch), dim=1))
                .cpu()
                .numpy()
            )
        for filename, feature in zip(names, features):
            source = (
                labels.loc[filename, "source_article_id"] if labels is not None else ""
            )
            family = labels.loc[filename, "product_code"] if labels is not None else ""
            matches = rank_results(
                feature,
                resources.projection,
                resources.catalog,
                args.top_k,
                source or None,
            )
            rankings.append(
                [article_indices[article] for article in matches["article_id"]]
            )
            query_names.append(filename)
            matches.insert(0, "filename", filename)
            matches["query_product_code"] = family
            matches["source_article_id"] = source
            matches["same_family"] = matches["product_code"] == family if family else ""
            predictions.append(matches)
        print(
            f"Processed {min(start + args.batch_size, len(files))}/{len(files)} photos.",
            flush=True,
        )

    summary = {
        "input_photos": len(files),
        "valid_photos": len(query_names),
        "skipped_photos": len(errors),
        "coverage": len(query_names) / len(files),
        "labelled": labels is not None,
        "labels_sha256": _sha256(args.labels) if args.labels else None,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint": str(args.checkpoint.resolve()),
        "query_sha256": hashes,
        "device": args.device,
        "top_k": args.top_k,
        "gallery": "Original frozen encoder; 8,065-product test gallery; supplied head.",
        "note": "External upload evaluation is separate from published test metrics. Cosine is not accuracy.",
    }
    if labels is not None and query_names:
        selected = labels.loc[query_names]
        counts = catalog["product_code"].value_counts()
        relevant = (
            selected["product_code"].map(counts).to_numpy()
            - selected["source_article_id"].ne("").to_numpy()
        )
        ks = (1, 5) if args.top_k == 5 else (1, 5, 10)
        summary["metrics"], _ = metrics_from_rankings(
            np.array(rankings),
            selected["product_code"].to_numpy(),
            catalog["product_code"].to_numpy(),
            relevant,
            ks,
        )
        summary["metric_queries"] = len(query_names)
        summary["relevance"] = (
            "Same product_code; known source article excluded when supplied."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(errors, columns=["filename", "error"]).to_csv(
        args.output / "errors.csv", index=False
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if predictions:
        results = pd.concat(predictions, ignore_index=True)
        results.to_csv(args.output / "predictions.csv", index=False)
        write_html(
            args.output / "preview.html",
            results,
            query_thumbnails,
            resources.data_root,
            "Epoch 15 head"
            if _sha256(args.checkpoint) == HEAD_SHA256
            else "Augmented head (selected)",
        )
    print(
        json.dumps(
            summary.get("metrics", {"note": "No labelled metrics calculated."}),
            indent=2,
        )
    )
    print(
        f"Saved {len(query_names)} valid searches; {len(errors)} skipped images: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
