"""Fast demo checks using synthetic embeddings and tiny in-memory images."""

import hashlib
import struct
import zlib
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import demo_retrieval
from demo_retrieval import (
    rank_results,
    read_uploaded_image,
    validate_embeddings,
    validate_gallery_rows,
)


@pytest.fixture
def rows():
    return pd.DataFrame(
        {
            "article_id": ["0108775015", "0108775016", "0208775015", "0308775015"],
            "product_code": ["0108775", "0108775", "0208775", "0308775"],
        }
    )


@pytest.fixture
def gallery():
    return np.array([[1, 0], [0.8, 0.6], [0.8, 0.6], [-1, 0]], dtype=np.float32)


def image_bytes(mode="RGB", file_format="PNG", exif=None):
    output = BytesIO()
    image = Image.new(mode, (3, 2), color=0)
    options = {} if exif is None else {"exif": exif}
    image.save(output, format=file_format, **options)
    return output.getvalue()


def test_catalog_search_excludes_only_its_article_and_preserves_leading_zeros(
    rows, gallery
):
    result = rank_results(
        gallery[0], gallery, rows, k=3, exclude_article_id="0108775015"
    )

    assert result["article_id"].tolist() == ["0108775016", "0208775015", "0308775015"]
    assert result["product_code"].tolist() == ["0108775", "0208775", "0308775"]
    assert result["rank"].tolist() == [1, 2, 3]
    assert result["cosine"].tolist() == pytest.approx([0.8, 0.8, -1])
    assert result.index.tolist() == [0, 1, 2]
    assert rows.columns.tolist() == ["article_id", "product_code"]


def test_external_search_keeps_all_articles_and_stable_ties(rows, gallery):
    result = rank_results(gallery[0], gallery, rows, k=4)

    assert result["article_id"].tolist() == rows["article_id"].tolist()
    assert result["cosine"].tolist() == pytest.approx([1, 0.8, 0.8, -1])


def test_top_one_is_selected_by_cosine_instead_of_family_label(rows, gallery):
    result = rank_results(np.array([-1, 0], dtype=np.float32), gallery, rows, k=1)

    assert result["article_id"].tolist() == ["0308775015"]
    assert result["cosine"].tolist() == pytest.approx([1])


def test_legacy_cached_catalog_gets_names_in_its_original_row_order(rows, monkeypatch):
    import demo_app

    manifest = rows.assign(
        split="test",
        prod_name=["Trousers", "Trousers", "Shirt", "Dress"],
        colour_group_name=["Blue", "Black", "White", "Red"],
        graphical_appearance_name="Solid",
        detail_desc=None,
    )
    resources = SimpleNamespace(catalog=rows.copy())
    monkeypatch.setattr(demo_app, "load_manifest", lambda: (manifest, None))

    assert not demo_app.has_catalog_metadata(resources)
    assert demo_app.ensure_catalog_metadata(resources) is resources
    assert demo_app.has_catalog_metadata(resources)
    assert resources.catalog["article_id"].tolist() == rows["article_id"].tolist()
    assert resources.catalog["prod_name"].tolist() == manifest["prod_name"].tolist()
    assert resources.catalog["colour_group_name"].tolist() == [
        "Blue",
        "Black",
        "White",
        "Red",
    ]
    assert resources.catalog["detail_desc"].eq("").all()

    def unexpected_reload():
        pytest.fail("A named catalog should remain cached across UI reruns.")

    monkeypatch.setattr(demo_app, "load_manifest", unexpected_reload)
    assert demo_app.ensure_catalog_metadata(resources) is resources


def test_legacy_cached_catalog_with_different_row_order_is_rejected(rows, monkeypatch):
    import demo_app

    manifest = rows.assign(
        split="test",
        prod_name="Product",
        colour_group_name="Blue",
        graphical_appearance_name="Solid",
        detail_desc="Description",
    )
    resources = SimpleNamespace(catalog=rows.iloc[::-1].reset_index(drop=True))
    monkeypatch.setattr(demo_app, "load_manifest", lambda: (manifest, None))
    with pytest.raises(ValueError, match="gallery rows"):
        demo_app.ensure_catalog_metadata(resources)


def test_legacy_cached_catalog_requires_real_product_metadata(rows, monkeypatch):
    import demo_app

    resources = SimpleNamespace(catalog=rows.copy())
    monkeypatch.setattr(
        demo_app, "load_manifest", lambda: (rows.assign(split="test"), None)
    )
    with pytest.raises(ValueError, match="metadata is missing"):
        demo_app.ensure_catalog_metadata(resources)


@pytest.mark.parametrize("k", [0, -1, 5, True, 1.5, "2", None])
def test_search_rejects_invalid_k(rows, gallery, k):
    with pytest.raises((TypeError, ValueError)):
        rank_results(gallery[0], gallery, rows, k=k)


def test_search_rejects_more_results_than_remain_after_exclusion(rows, gallery):
    with pytest.raises(ValueError):
        rank_results(gallery[0], gallery, rows, k=4, exclude_article_id="0108775015")


@pytest.mark.parametrize("article_id", ["9999999999", "", 108775015])
def test_catalog_exclusion_requires_a_known_string_id(rows, gallery, article_id):
    with pytest.raises((TypeError, ValueError)):
        rank_results(gallery[0], gallery, rows, k=1, exclude_article_id=article_id)


@pytest.mark.parametrize(
    "query",
    [
        None,
        np.array([], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
        np.array([1, 0, 0], dtype=np.float32),
        np.array([np.nan, 0], dtype=np.float32),
        np.array([np.inf, 0], dtype=np.float32),
        np.array([0, 0], dtype=np.float32),
        np.array([2, 0], dtype=np.float32),
    ],
    ids=[
        "none",
        "empty",
        "matrix",
        "wrong-dim",
        "nan",
        "infinite",
        "zero",
        "not-normalized",
    ],
)
def test_search_rejects_invalid_query(rows, gallery, query):
    with pytest.raises((TypeError, ValueError)):
        rank_results(query, gallery, rows, k=1)


def test_search_rejects_cache_row_count_mismatch(rows, gallery):
    with pytest.raises(ValueError):
        rank_results(gallery[0], gallery[:-1], rows, k=1)


def test_search_handles_more_than_ten_thousand_candidates_without_losing_id_alignment():
    size = 10_001
    features = np.tile(np.array([0, 1], dtype=np.float32), (size, 1))
    features[-1] = [1, 0]
    table = pd.DataFrame(
        {
            "article_id": [f"{index:010d}" for index in range(size)],
            "product_code": [f"{index // 2:07d}" for index in range(size)],
        }
    )

    result = rank_results(np.array([1, 0], dtype=np.float32), features, table)

    assert result["article_id"].tolist() == [
        "0000010000",
        "0000000000",
        "0000000001",
        "0000000002",
        "0000000003",
    ]
    assert result["cosine"].tolist() == pytest.approx([1, 0, 0, 0, 0])


def test_valid_cache_and_ordered_rows_are_accepted(rows, gallery):
    assert validate_embeddings(gallery, row_count=4, dim=2) is None
    assert validate_gallery_rows(rows, rows.copy()) is None


@pytest.mark.parametrize(
    "features, row_count, dim",
    [
        (None, 4, 2),
        (np.ones(4, dtype=np.float32), 4, 2),
        (np.empty((0, 2), dtype=np.float32), 0, 2),
        (np.array([[1, 0]], dtype=np.float64), 1, 2),
        (np.array([[1, 0]], dtype=np.int64), 1, 2),
        (np.array([[1, 0]], dtype=np.float32), 2, 2),
        (np.array([[1, 0]], dtype=np.float32), 1, 3),
        (np.array([[np.nan, 0]], dtype=np.float32), 1, 2),
        (np.array([[np.inf, 0]], dtype=np.float32), 1, 2),
        (np.array([[0, 0]], dtype=np.float32), 1, 2),
        (np.array([[2, 0]], dtype=np.float32), 1, 2),
    ],
    ids=[
        "none",
        "vector",
        "empty",
        "float64",
        "integer",
        "row-count",
        "dimension",
        "nan",
        "infinite",
        "zero",
        "not-normalized",
    ],
)
def test_cache_validation_rejects_corrupt_or_incompatible_embeddings(
    features, row_count, dim
):
    with pytest.raises((TypeError, ValueError)):
        validate_embeddings(features, row_count=row_count, dim=dim)


@pytest.mark.parametrize(
    "problem",
    [
        "reordered",
        "missing-column",
        "extra-column",
        "duplicate",
        "missing-id",
        "numeric-id",
        "short-id",
        "unicode-id",
        "short-family",
        "different-family",
        "empty",
    ],
)
def test_gallery_validation_rejects_schema_ids_or_alignment_changes(rows, problem):
    invalid = rows.copy()
    if problem == "reordered":
        invalid = invalid.iloc[::-1].reset_index(drop=True)
    elif problem == "missing-column":
        invalid = invalid.drop(columns="product_code")
    elif problem == "extra-column":
        invalid["split"] = "test"
    elif problem == "duplicate":
        invalid.loc[1, "article_id"] = invalid.loc[0, "article_id"]
    elif problem == "missing-id":
        invalid.loc[0, "article_id"] = None
    elif problem == "numeric-id":
        invalid["article_id"] = invalid["article_id"].astype(int)
    elif problem == "short-id":
        invalid.loc[0, "article_id"] = "108775015"
    elif problem == "unicode-id":
        invalid.loc[0, "article_id"] = "０１０８７７５０１５"
    elif problem == "short-family":
        invalid.loc[0, "product_code"] = "108775"
    elif problem == "different-family":
        invalid.loc[0, "product_code"] = "9999999"
    elif problem == "empty":
        invalid = invalid.iloc[:0]

    with pytest.raises((TypeError, ValueError)):
        validate_gallery_rows(invalid, rows)


@pytest.mark.parametrize(
    "mode, file_format",
    [("RGB", "JPEG"), ("RGB", "PNG"), ("L", "PNG"), ("RGBA", "PNG")],
)
def test_upload_decodes_supported_images_to_loaded_rgb(mode, file_format):
    decoded = read_uploaded_image(image_bytes(mode, file_format))

    assert decoded.mode == "RGB"
    assert decoded.size == (3, 2)
    assert decoded.getpixel((0, 0)) == (0, 0, 0)
    assert len(decoded.tobytes()) == 18


def test_upload_applies_exif_orientation_before_retrieval():
    exif = Image.Exif()
    exif[274] = 6

    decoded = read_uploaded_image(image_bytes(file_format="JPEG", exif=exif))

    assert decoded.mode == "RGB"
    assert decoded.size == (2, 3)


@pytest.mark.parametrize(
    "data",
    [
        None,
        "a.png",
        b"",
        b"not an image",
        image_bytes(file_format="GIF"),
        image_bytes(file_format="PNG")[:40],
        image_bytes(file_format="JPEG")[:-20],
    ],
)
def test_upload_rejects_invalid_unsupported_or_truncated_images(data):
    with pytest.raises((TypeError, ValueError)):
        read_uploaded_image(data)


def test_upload_rejects_encoded_data_above_ten_megabytes():
    with pytest.raises(ValueError):
        read_uploaded_image(b"x" * (10 * 1024 * 1024 + 1))


def test_upload_rejects_excessive_pixel_dimensions_before_decoding():
    # The PNG header can advertise an oversized image without allocating its pixels.
    original = image_bytes()
    header = struct.pack(">IIBBBBB", 5001, 4000, 8, 2, 0, 0, 0)
    oversized = (
        original[:16]
        + header
        + struct.pack(">I", zlib.crc32(b"IHDR" + header))
        + original[33:]
    )

    with pytest.raises(ValueError):
        read_uploaded_image(oversized)


def test_gallery_file_validation_rejects_unsampled_normalized_row_swap(
    tmp_path, monkeypatch
):
    angles = np.linspace(0, 1, 10, dtype=np.float32)
    projection = np.column_stack((np.cos(angles), np.sin(angles)))
    np.save(tmp_path / "frozen_embeddings.npy", np.eye(10, dtype=np.float32))
    np.save(tmp_path / "projection_embeddings.npy", projection)
    trusted_hashes = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("frozen_embeddings.npy", "projection_embeddings.npy")
    }
    monkeypatch.setattr(demo_retrieval, "GALLERY_SHA256", trusted_hashes)
    assert demo_retrieval.validate_gallery_files(tmp_path) is None

    # Rows 4 and 8 escaped the previous eight probes of this ten-row gallery.
    tampered = projection.copy()
    tampered[[4, 8]] = tampered[[8, 4]]
    assert validate_embeddings(tampered, row_count=10, dim=2) is None
    np.save(tmp_path / "projection_embeddings.npy", tampered)

    with pytest.raises(ValueError):
        demo_retrieval.validate_gallery_files(tmp_path)
