"""Retrieval metrics: self-match excluded; relevant = same product family."""

import numpy as np


def metrics_from_rankings(
    rankings: np.ndarray,
    query_groups: np.ndarray,
    gallery_groups: np.ndarray,
    relevant_counts: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Score existing rankings; also return each query's metrics for analysis."""
    if (
        not ks
        or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in ks)
        or min(ks) < 1
    ):
        raise ValueError("K must contain positive integers.")
    if (
        rankings.ndim != 2
        or query_groups.ndim != 1
        or gallery_groups.ndim != 1
        or relevant_counts.shape != query_groups.shape
        or len(rankings) != len(query_groups)
        or not len(query_groups)
        or max(ks) > rankings.shape[1]
    ):
        raise ValueError("Rankings, query labels and relevant counts are not aligned.")
    if (
        not np.issubdtype(rankings.dtype, np.integer)
        or (rankings < 0).any()
        or (rankings >= len(gallery_groups)).any()
        or not np.isfinite(relevant_counts).all()
        or (relevant_counts < 1).any()
    ):
        raise ValueError("Invalid rankings or queries without relevant gallery items.")
    if any(len(np.unique(row)) != len(row) for row in rankings):
        raise ValueError("A ranking contains a duplicate gallery item.")

    # Her satır bir sorgu, her sütun bir sıralama konumudur.
    # [:, None] aile etiketini o sorgunun tüm sonuçlarıyla karşılaştırır.
    relevant = gallery_groups[rankings] == query_groups[:, None]
    precisions = np.cumsum(relevant, axis=1) / np.arange(1, rankings.shape[1] + 1)
    per_query = {}
    for k in ks:
        found = relevant[:, :k].sum(axis=1)
        per_query[f"recall@{k}"] = found / relevant_counts
        per_query[f"hit_rate@{k}"] = (found > 0).astype(float)
        per_query[f"map@{k}"] = (precisions[:, :k] * relevant[:, :k]).sum(
            axis=1
        ) / np.minimum(relevant_counts, k)
    metrics = {name: float(values.mean()) for name, values in per_query.items()}
    return metrics, per_query


def evaluate_queries(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_groups: np.ndarray,
    gallery_groups: np.ndarray,
    query_article_ids: np.ndarray,
    gallery_article_ids: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> tuple[dict[str, float], np.ndarray, dict[str, np.ndarray]]:
    """Evaluate separate query photos, excluding each known source article.

    The unchanged gallery may contain many more items than the query set.
    Relevant = same family, different article; stable ties use gallery order.
    """
    for features, groups, ids in (
        (query_embeddings, query_groups, query_article_ids),
        (gallery_embeddings, gallery_groups, gallery_article_ids),
    ):
        if (
            features.ndim != 2
            or not len(features)
            or groups.shape != (len(features),)
            or ids.shape != (len(features),)
        ):
            raise ValueError(
                "Embedding rows, family labels and article IDs must align."
            )
        if not np.isfinite(features).all() or not np.allclose(
            np.linalg.norm(features, axis=1), 1, atol=1e-5, rtol=0
        ):
            raise ValueError("Embeddings must be finite and L2-normalized.")
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("Query and gallery embedding dimensions differ.")
    if (
        not ks
        or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in ks)
        or min(ks) < 1
        or max(ks) >= len(gallery_embeddings)
    ):
        raise ValueError("K must be positive and smaller than the gallery size.")
    if len(np.unique(gallery_article_ids)) != len(gallery_article_ids):
        raise ValueError("Gallery article IDs must be unique.")

    index_by_id = {article: i for i, article in enumerate(gallery_article_ids)}
    if any(article not in index_by_id for article in query_article_ids):
        raise ValueError("Every source article must exist in the gallery.")
    source_indices = np.array([index_by_id[article] for article in query_article_ids])
    if not np.array_equal(gallery_groups[source_indices], query_groups):
        raise ValueError("Source article and query family labels disagree.")
    families, counts = np.unique(gallery_groups, return_counts=True)
    count_by_family = dict(zip(families, counts))
    relevant_counts = np.array([count_by_family[group] - 1 for group in query_groups])
    if (relevant_counts < 1).any():
        raise ValueError("Every query needs another article from the same family.")

    rankings = np.empty((len(query_embeddings), max(ks)), dtype=np.int64)
    for start in range(0, len(query_embeddings), 256):
        stop = min(start + 256, len(query_embeddings))
        # Normalize vektörlerin iç çarpımı cosine skorudur.
        # 256'lık blok, tüm sorguların skor matrisini bellekte tutmamayı sağlar.
        scores = query_embeddings[start:stop] @ gallery_embeddings.T
        scores[np.arange(stop - start), source_indices[start:stop]] = -np.inf
        rankings[start:stop] = np.argsort(-scores, axis=1, kind="stable")[:, : max(ks)]
    metrics, per_query = metrics_from_rankings(
        rankings, query_groups, gallery_groups, relevant_counts, ks
    )
    return metrics, rankings, per_query


def evaluate(
    embeddings: np.ndarray, groups: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)
) -> tuple[dict, np.ndarray]:
    """Return full-gallery metrics and rankings, using bounded query blocks.

    Recall@K = retrieved relevant / all relevant items in the gallery.
    HitRate@K = fraction of queries with at least one relevant result.
    AP@K denominator = min(K, total relevant); mAP is the query mean.
    The query's own image is always excluded.
    """
    if embeddings.ndim != 2 or len(embeddings) != len(groups):
        raise ValueError("Expected an embedding matrix aligned with group labels.")
    if not ks or min(ks) < 1 or max(ks) >= len(groups):
        raise ValueError("K must be positive and smaller than the gallery size.")
    if not np.isfinite(embeddings).all() or not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1, atol=1e-5
    ):
        raise ValueError("Embeddings must be finite and L2-normalized.")
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    relevant_counts = counts[inverse] - 1
    if (relevant_counts < 1).any():
        raise ValueError("Each query needs another image from its product family.")
    rankings = np.empty((len(groups), max(ks)), dtype=np.int64)
    for start in range(0, len(groups), 256):
        stop = min(start + 256, len(groups))
        scores = embeddings[start:stop] @ embeddings.T
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        # Stable ordering resolves exact ties by gallery row index.
        rankings[start:stop] = np.argsort(-scores, axis=1, kind="stable")[:, : max(ks)]
    relevant = groups[rankings] == groups[:, None]
    precisions = np.cumsum(relevant, axis=1) / np.arange(1, max(ks) + 1)
    result = {}
    for k in ks:
        found = relevant[:, :k].sum(axis=1)
        result[f"recall@{k}"] = float(np.mean(found / relevant_counts))
        result[f"hit_rate@{k}"] = float(np.mean(found > 0))
        result[f"map@{k}"] = float(
            np.mean(
                (precisions[:, :k] * relevant[:, :k]).sum(axis=1)
                / np.minimum(relevant_counts, k)
            )
        )
    return result, rankings
