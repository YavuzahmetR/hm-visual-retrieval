"""Retrieval metrics: self-match excluded; relevant = same product family."""

import numpy as np


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
