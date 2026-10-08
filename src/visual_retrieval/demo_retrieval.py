"""Inference and cached search for the selected robust head or original epoch-15 head."""

import hashlib
import io
import json
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from visual_retrieval.data import PROJECT_DIR, load_manifest
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from visual_retrieval.train_projection_head import ProjectionHead

HEAD_SHA256 = "375f8d7d9d96974a6cb0df6df3c71925ebae168a5fe683d5e8389637c51cd6f4"
ROBUST_HEAD_SHA256 = "d5b8a02a6e255e6ff231ce54241e615d40d8ac568baeb85d6e2ee039275e6ddd"
ROBUST_SOURCE_SHA256 = (
    "a5568a607380831e8a58366de548b5149b346ecf8ecce82c54472706f1167d7d"
)
ROBUST_GALLERY_SHA256 = (
    "3be7395c5ce314f8e76b87189f742c946bddb116c2825583c756d18e90c2b743"
)
RESNET_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
GALLERY_SHA256 = {
    "frozen_embeddings.npy": "4fe53e7ed666af4f56387819118c72735248e791b48041b3516e18643334dd21",
    "projection_embeddings.npy": "3102adb2e81362edd55c4c89f400133ef659317e7c81b66988e09db1f6c9a2aa",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_PIXELS = 20_000_000


def validate_gallery_rows(rows: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Preserve string IDs and the original manifest/embedding row order."""
    if list(rows.columns) != ["article_id", "product_code"]:
        raise ValueError("Gallery rows must contain article_id and product_code.")
    if rows.empty or rows.isna().any().any():
        raise ValueError("Gallery rows are empty or contain missing IDs.")
    for column, digits in (("article_id", 10), ("product_code", 7)):
        if not rows[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must be read as strings.")
        if not rows[column].str.fullmatch(rf"[0-9]{{{digits}}}").all():
            raise ValueError(f"Invalid {column}; leading zeros must be preserved.")
    if rows["article_id"].duplicated().any():
        raise ValueError("Gallery contains duplicate article IDs.")
    if not rows.equals(expected):
        raise ValueError("Gallery row order does not match the saved test split.")


def validate_embeddings(features: np.ndarray, row_count: int, dim: int) -> None:
    """Check cached vectors without extracting embeddings or scoring a split."""
    if not isinstance(features, np.ndarray) or row_count < 1 or dim < 1:
        raise ValueError("Expected a nonempty embedding matrix.")
    if features.shape != (row_count, dim) or features.dtype != np.float32:
        raise ValueError(f"Expected float32 embeddings with shape {(row_count, dim)}.")
    if not np.isfinite(features).all() or not np.allclose(
        np.linalg.norm(features, axis=1), 1.0, atol=1e-5, rtol=0
    ):
        raise ValueError("Embeddings must be finite and L2-normalized.")


def rank_results(
    query: np.ndarray,
    gallery: np.ndarray,
    rows: pd.DataFrame,
    k: int = 5,
    exclude_article_id: str | None = None,
) -> pd.DataFrame:
    """Search an already validated gallery; ties follow original gallery order."""
    query = np.asarray(query, dtype=np.float32)
    if gallery.ndim != 2 or len(gallery) != len(rows):
        raise ValueError("Gallery embeddings and rows are not aligned.")
    if query.shape != (gallery.shape[1],):
        raise ValueError("Query and gallery embedding dimensions differ.")
    if not np.isfinite(query).all() or not np.isclose(
        np.linalg.norm(query), 1.0, atol=1e-5, rtol=0
    ):
        raise ValueError("Query must be finite and L2-normalized.")

    candidates = np.ones(len(rows), dtype=bool)
    if exclude_article_id is not None:
        matches = rows["article_id"].to_numpy() == exclude_article_id
        if not matches.any():
            raise ValueError("The query article is absent from this gallery.")
        candidates &= ~matches
    if (
        isinstance(k, bool)
        or not isinstance(k, (int, np.integer))
        or not 1 <= k <= candidates.sum()
    ):
        raise ValueError("K must be a positive integer within the available gallery.")

    scores = gallery @ query
    if not np.isfinite(scores).all():
        raise ValueError("Cosine scores contain invalid values.")
    indices = np.flatnonzero(candidates)
    order = indices[np.argsort(-scores[indices], kind="stable")[:k]]
    results = rows.iloc[order].reset_index(drop=True).copy()
    results["cosine"] = np.clip(scores[order], -1.0, 1.0)
    results["rank"] = np.arange(1, len(results) + 1)
    return results


def read_uploaded_image(data: bytes) -> Image.Image:
    """Decode JPEG/PNG in memory, apply EXIF orientation and return loaded RGB."""
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload must be a nonempty JPG/PNG no larger than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are supported.")
                if source.width * source.height > MAX_UPLOAD_PIXELS:
                    raise ValueError("Image exceeds the 20-million-pixel limit.")
                source.load()
                return ImageOps.exif_transpose(source).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("The uploaded image cannot be decoded safely.") from error


@dataclass
class DemoResources:
    catalog: pd.DataFrame
    data_root: Path
    frozen: np.ndarray
    projection: np.ndarray
    encoder: nn.Module
    head: nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    device: str
    head_version: str = "epoch15"
    _inference_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def search_catalog(
        self, article_id: str, mode: str = "projection", k: int = 5
    ) -> pd.DataFrame:
        if mode not in {"projection", "frozen"}:
            raise ValueError("Unknown retrieval model.")
        indices = np.flatnonzero(self.catalog["article_id"].to_numpy() == article_id)
        if len(indices) != 1:
            raise ValueError("Choose an article_id from the test catalog.")
        gallery = self.projection if mode == "projection" else self.frozen
        return rank_results(gallery[indices[0]], gallery, self.catalog, k, article_id)

    def embed_image(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Encode just this query; gallery arrays remain on CPU and unchanged."""
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._inference_lock, torch.inference_mode():
            frozen = F.normalize(self.encoder(batch), dim=1)
            projected = self.head(frozen)
            return frozen[0].cpu().numpy(), projected[0].cpu().numpy()


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_gallery_files(cache_dir: Path) -> None:
    """Reject changed bytes anywhere in the completed experiment's gallery."""
    for name, expected_hash in GALLERY_SHA256.items():
        if _sha256(cache_dir / name) != expected_hash:
            raise ValueError(f"{name} differs from the saved epoch-15 demo gallery.")


def _load_test_catalog() -> tuple[pd.DataFrame, Path, pd.DataFrame]:
    """Read the unchanged test split and locate local images."""
    manifest, data_root = load_manifest()
    test_catalog = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    expected = test_catalog[["article_id", "product_code"]]
    if len(expected) != 8065 or expected["product_code"].nunique() != 2306:
        raise ValueError("The original 8,065-product test gallery is required.")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Image data root is missing: {data_root}")

    return test_catalog, data_root, expected


def _load_frozen_gallery(test_catalog: pd.DataFrame, expected: pd.DataFrame):
    """Validate row order, names and the saved 512D encoder vectors."""
    cache_dir = PROJECT_DIR / "results/test_evaluation"
    rows = pd.read_csv(
        cache_dir / "embedding_rows.csv",
        dtype={"article_id": str, "product_code": str},
    )
    validate_gallery_rows(rows, expected)
    # The verified manifest order also aligns names/descriptions with embeddings.
    catalog = test_catalog[
        [
            "article_id",
            "product_code",
            "prod_name",
            "colour_group_name",
            "graphical_appearance_name",
            "detail_desc",
        ]
    ].copy()
    display_fields = [
        "prod_name",
        "colour_group_name",
        "graphical_appearance_name",
        "detail_desc",
    ]
    catalog[display_fields] = catalog[display_fields].fillna("")
    frozen = np.load(
        cache_dir / "frozen_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    validate_embeddings(frozen, len(rows), 512)
    if (
        _sha256(cache_dir / "frozen_embeddings.npy")
        != GALLERY_SHA256["frozen_embeddings.npy"]
    ):
        raise ValueError("Frozen gallery differs from the original encoder cache.")

    return catalog, rows, frozen, cache_dir


def _load_checkpoint(head_version: str, expected: pd.DataFrame, cache_dir: Path):
    """Match local weights and cache metadata to the published experiment."""
    weights_path = PROJECT_DIR / "model_cache/hub/checkpoints/resnet18-f37072fd.pth"
    head_path = (
        PROJECT_DIR
        / "models"
        / (
            "projection_head_robust.pt"
            if head_version == "robust"
            else "projection_head_epoch15.pt"
        )
    )
    weights_hash, head_hash = _sha256(weights_path), _sha256(head_path)
    expected_head_hash = ROBUST_HEAD_SHA256 if head_version == "robust" else HEAD_SHA256
    if weights_hash != RESNET_SHA256 or head_hash != expected_head_hash:
        raise ValueError("Local encoder/head weights differ from the published model.")
    protocol = (
        "ResNet18 ImageNet1K V1|weights.transforms|float32|L2|"
        + str(torch.__version__)
        + "|"
        + weights_hash
    )
    signature = hashlib.sha256(
        (protocol + expected.to_csv(index=False)).encode("utf-8")
    ).hexdigest()
    metadata = json.loads((cache_dir / "cache_metadata.json").read_text("utf-8"))
    if metadata.get("protocol") != protocol or metadata.get("signature") != signature:
        raise ValueError(
            "Frozen cache encoder, preprocessing, environment or rows differ."
        )
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    return checkpoint, weights_path, head_hash, weights_hash, signature


def _load_projected_gallery(
    head_version: str, checkpoint: dict, weights_hash: str, head_hash: str,
    signature: str, cache_dir: Path, rows: pd.DataFrame,
    expected: pd.DataFrame, frozen: np.ndarray,
):
    """Validate the selected 128D gallery and its head without rebuilding it."""
    if head_version == "robust":
        projection_dir = PROJECT_DIR / "results/robust_head_v1/test_evaluation"
        robust_rows = pd.read_csv(projection_dir / "embedding_rows.csv", dtype=str)
        validate_gallery_rows(robust_rows, expected)
        report = json.loads((projection_dir / "metrics.json").read_text("utf-8"))
        selection = report.get("selection", {})
        if (
            checkpoint.get("format") != "robust_head_v1"
            or checkpoint.get("encoder_sha256") != weights_hash
            or checkpoint.get("preprocessing")
            != "ResNet18_Weights.IMAGENET1K_V1.transforms()"
            or checkpoint.get("source_checkpoint_sha256") != ROBUST_SOURCE_SHA256
            or checkpoint.get("epoch") != 5
            or checkpoint.get("parent_epoch") != 15
            or report.get("split") != "test"
            or report.get("queries") != len(rows)
            or report.get("cache_signature") != signature
            or selection.get("checkpoint_sha256") != ROBUST_SOURCE_SHA256
            or selection.get("additional_epoch") != checkpoint["epoch"]
        ):
            raise ValueError("Robust head and its saved test gallery do not match.")
        expected_projection_hash = ROBUST_GALLERY_SHA256
    else:
        projection_dir = cache_dir
        report = json.loads((cache_dir / "metrics.json").read_text("utf-8"))
        selection = report.get("selection", {})
        if (
            checkpoint.get("epoch") != 15
            or report.get("split") != "test"
            or selection.get("epoch") != 15
            or selection.get("checkpoint_sha256") != head_hash
        ):
            raise ValueError("Saved test report does not match the epoch-15 head.")
        expected_projection_hash = GALLERY_SHA256["projection_embeddings.npy"]

    projection_path = projection_dir / "projection_embeddings.npy"
    if _sha256(projection_path) != expected_projection_hash:
        raise ValueError(
            "Projection gallery differs from the selected head's saved cache."
        )
    projected = np.load(projection_path, mmap_mode="r", allow_pickle=False)
    validate_embeddings(projected, len(rows), 128)
    head = ProjectionHead()
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.requires_grad_(False).eval()

    # Check a few saved pairs against the head; do not regenerate the gallery.
    sample = np.linspace(0, len(rows) - 1, 8, dtype=int)
    with torch.inference_mode():
        sample_projection = head(
            torch.from_numpy(np.array(frozen[sample], copy=True))
        ).numpy()
    if not np.allclose(sample_projection, projected[sample], atol=1e-5, rtol=1e-5):
        raise ValueError("Sampled projection cache rows do not match this head.")

    return head, projected


def _load_encoder(weights_path: Path, device: str) -> nn.Module:
    """Build the frozen encoder from verified local weights."""
    # weights=None prevents an implicit torchvision download.
    encoder = resnet18(weights=None)
    encoder.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True), strict=True
    )
    encoder.fc = nn.Identity()
    encoder.requires_grad_(False).to(device).eval()
    return encoder


def load_demo(device: str = "cpu", head_version: str = "robust") -> DemoResources:
    """Load local assets only; never download weights or rebuild missing caches."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda for demo inference.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose CPU for the demo.")
    if head_version not in {"robust", "epoch15"}:
        raise ValueError("Choose robust or epoch15 for the projection head.")

    test_catalog, data_root, expected = _load_test_catalog()
    catalog, rows, frozen, cache_dir = _load_frozen_gallery(test_catalog, expected)
    checkpoint, weights_path, head_hash, weights_hash, signature = _load_checkpoint(
        head_version, expected, cache_dir
    )
    head, projected = _load_projected_gallery(
        head_version, checkpoint, weights_hash, head_hash, signature,
        cache_dir, rows, expected, frozen,
    )
    encoder = _load_encoder(weights_path, device)
    head.to(device)
    return DemoResources(
        catalog=catalog,
        data_root=data_root,
        frozen=frozen,
        projection=projected,
        encoder=encoder,
        head=head,
        transform=ResNet18_Weights.IMAGENET1K_V1.transforms(),
        device=device,
        head_version=head_version,
    )
