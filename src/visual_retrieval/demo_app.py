"""Local H&M visual retrieval demo using the verified test gallery."""

import numpy as np
import pandas as pd
import streamlit as st
import torch
from visual_retrieval.data import image_path, load_manifest
from visual_retrieval.demo_retrieval import DemoResources, load_demo, rank_results, read_uploaded_image
from PIL import Image

CATALOG_SCHEMA = 3
CATALOG_METADATA = (
    "prod_name",
    "colour_group_name",
    "graphical_appearance_name",
    "detail_desc",
)


def has_catalog_metadata(resources: DemoResources) -> bool:
    """Invalidate older cached resources before the UI accesses product names."""
    return set(CATALOG_METADATA).issubset(resources.catalog.columns)


def ensure_catalog_metadata(resources: DemoResources) -> DemoResources:
    """Support a loader still imported by an open, older Streamlit process."""
    if has_catalog_metadata(resources):
        return resources
    manifest, _ = load_manifest()
    test_catalog = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    ids = ["article_id", "product_code"]
    missing = set(CATALOG_METADATA) - set(test_catalog.columns)
    if missing:
        raise ValueError(
            f"Product metadata is missing from the manifest: {sorted(missing)}"
        )
    if not resources.catalog[ids].reset_index(drop=True).equals(test_catalog[ids]):
        raise ValueError("Cached gallery rows do not match the named test catalog.")
    catalog = test_catalog[ids + list(CATALOG_METADATA)].copy()
    catalog[list(CATALOG_METADATA)] = catalog[list(CATALOG_METADATA)].fillna("")
    resources.catalog = catalog
    return resources


@st.cache_resource(
    max_entries=4,
    show_spinner="Loading model and saved gallery...",
    validate=has_catalog_metadata,
)
def cached_resources(
    device: str, head_version: str, catalog_schema: int
) -> DemoResources:
    """Cache the selected model and gallery with product names and descriptions."""
    if catalog_schema != CATALOG_SCHEMA:
        raise ValueError("Unsupported demo catalog schema.")
    return ensure_catalog_metadata(load_demo(device=device, head_version=head_version))


@st.cache_data(max_entries=8, show_spinner="Encoding uploaded image...")
def uploaded_query(
    payload: bytes, device: str, head_version: str, _resources: DemoResources
) -> tuple[Image.Image, tuple[np.ndarray, np.ndarray]]:
    # Model değişince eski head'in sorgu embedding'i cache'ten gelmemeli.
    if _resources.head_version != head_version:
        raise ValueError(
            "Upload cache key does not match the selected projection head."
        )
    image = read_uploaded_image(payload)
    embeddings = _resources.embed_image(image)
    # Keep the cached preview small; encoding uses the full decoded image.
    image.thumbnail((1024, 1024))
    return image, embeddings


def show_local_image(
    resources: DemoResources, article_id: str, width: int | str = "stretch"
) -> None:
    try:
        with Image.open(image_path(resources.data_root, article_id)) as source:
            preview = source.convert("RGB")
        st.image(preview, width=width)
    except (OSError, ValueError) as error:
        st.warning(f"Image unavailable for {article_id}: {error}")


def article_name(product_name: str, colour: str, pattern: str) -> str:
    """Display the recorded name with variant attributes, not a category label."""
    parts = [product_name or "Product name unavailable", colour]
    if pattern and pattern not in {"Solid", "Unknown"}:
        parts.append(pattern)
    return " · ".join(part for part in parts if part)


def show_results(
    results: pd.DataFrame,
    resources: DemoResources,
    label: str,
    query_family: str | None = None,
) -> None:
    st.subheader(label)
    for column, row in zip(st.columns(len(results)), results.itertuples(index=False)):
        with column:
            variant_name = article_name(
                row.prod_name, row.colour_group_name, row.graphical_appearance_name
            )
            st.markdown(f"**#{row.rank} · {variant_name}**")
            show_local_image(resources, row.article_id)
            st.caption(f"Article ID: {row.article_id}")
            st.caption(f"Product: {row.prod_name} · Family ID: {row.product_code}")
            st.caption(f"Cosine similarity: {row.cosine:.4f}")
            if query_family is not None:
                st.caption(
                    "Same family variant"
                    if row.product_code == query_family
                    else "Other family"
                )
            if row.detail_desc:
                with st.expander("Product details"):
                    st.write(row.detail_desc)


def main() -> None:
    st.set_page_config(page_title="H&M Visual Retrieval", page_icon="🔎", layout="wide")
    st.title("H&M Visual Retrieval")
    st.write("Find product variants with a photo or an item from the catalog.")
    st.caption(
        "Product-family variant retrieval: the same product_code, a different article_id."
    )

    with st.sidebar:
        st.header("Search settings")
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        device = st.selectbox("Query inference device", devices, format_func=str.upper)
        st.caption(
            "CPU is the default. CUDA accelerates uploaded-photo encoding; catalog search uses saved embeddings."
        )
        mode = st.radio(
            "Retrieval model", ["Projection head", "Frozen ResNet18", "Compare both"]
        )
        head_choice = st.selectbox(
            "Projection head version",
            ["Augmented head (selected)", "Epoch 15 head"],
        )
        head_version = (
            "robust" if head_choice == "Augmented head (selected)" else "epoch15"
        )
        source = st.radio("Query source", ["Catalog item", "Upload a photo"])
        st.caption(
            "Cosine similarity is a ranking score, not a probability of correctness."
        )

    try:
        resources = cached_resources(
            device, head_version, catalog_schema=CATALOG_SCHEMA
        )
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Cannot load the local demo assets: {error}")
        st.info(
            "Check project_config.json, the saved test gallery, the selected head checkpoint, and local ResNet18 weights."
        )
        st.stop()

    st.caption(
        f"Gallery: {len(resources.catalog):,} test products · Frozen ImageNet ResNet18 · {head_choice}"
    )
    query_family = None
    article_id = None
    query_embeddings = None
    preview = None
    if source == "Catalog item":
        product_labels = {
            row.article_id: (
                f"{article_name(row.prod_name, row.colour_group_name, row.graphical_appearance_name)}"
                f" · {row.article_id}"
            )
            for row in resources.catalog.itertuples(index=False)
        }
        article_id = st.selectbox(
            "Search products",
            resources.catalog["article_id"].tolist(),
            format_func=product_labels.__getitem__,
        )
        query_product = resources.catalog.loc[
            resources.catalog["article_id"] == article_id
        ].iloc[0]
        query_family = query_product["product_code"]
    else:
        st.info(
            "Uploaded photos are exploratory. Published test metrics evaluate catalog variants and do not measure performance on external photos."
        )
        upload = st.file_uploader(
            "Upload a JPG or PNG (up to 10 MB)", type=["jpg", "jpeg", "png"]
        )
        if upload is None:
            st.stop()
        if upload.size > 10 * 1024 * 1024:
            st.error("Please choose an image no larger than 10 MB.")
            st.stop()
        try:
            payload = upload.getvalue()
            preview, query_embeddings = uploaded_query(
                payload, device, head_version, resources
            )
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Cannot read or encode this image: {error}")
            st.stop()
        st.caption("Uploaded images are processed in memory and are not saved to disk.")

    query_column, details_column = st.columns([1, 4])
    with query_column:
        st.subheader("Query")
        if article_id is not None:
            show_local_image(resources, article_id, width=220)
        else:
            st.image(preview, width=220)
    with details_column:
        if article_id is not None:
            variant_name = article_name(
                query_product["prod_name"],
                query_product["colour_group_name"],
                query_product["graphical_appearance_name"],
            )
            st.subheader(variant_name)
            st.write(f"Article: **{variant_name}** · ID: `{article_id}`")
            st.write(
                f"Product: **{query_product['prod_name']}** · Family ID: `{query_family}`"
            )
            if query_product["detail_desc"]:
                st.write(query_product["detail_desc"])
            st.caption(
                "The query article is excluded from the results. Same-family labels use catalog metadata."
            )
        else:
            st.write("Uploaded photo")
        st.write("Top five results, ordered by cosine similarity.")

    modes = (
        ["projection", "frozen"]
        if mode == "Compare both"
        else ["projection" if mode == "Projection head" else "frozen"]
    )
    for model_mode in modes:
        try:
            if article_id is not None:
                results = resources.search_catalog(article_id, mode=model_mode, k=5)
            else:
                index = 1 if model_mode == "projection" else 0
                gallery = (
                    resources.projection
                    if model_mode == "projection"
                    else resources.frozen
                )
                results = rank_results(
                    query_embeddings[index], gallery, resources.catalog, k=5
                )
        except (ValueError, RuntimeError) as error:
            st.error(f"Search could not complete: {error}")
            st.stop()
        label = (
            (
                "Augmented projection head · 128D"
                if head_version == "robust"
                else "Epoch 15 projection head · 128D"
            )
            if model_mode == "projection"
            else "Frozen ResNet18 baseline · 512D"
        )
        show_results(results, resources, label, query_family)


if __name__ == "__main__":
    main()
