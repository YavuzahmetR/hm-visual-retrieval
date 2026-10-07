"""Tiny controlled galleries test metric denominators and self exclusion."""

import numpy as np
import pytest
from data import image_path
from retrieval_metrics import evaluate


def test_perfect_pairs_exclude_self():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    metrics, rankings = evaluate(embeddings, np.array(["a", "a", "b", "b"]), ks=(1, 3))
    assert metrics["recall@1"] == 1
    assert metrics["map@1"] == 1
    assert not (rankings == np.arange(4)[:, None]).any()


def test_recall_differs_from_hit_rate_for_three_variants():
    embeddings = np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    metrics, _ = evaluate(embeddings, np.array(["a"] * 3 + ["b"] * 3), ks=(1, 2))
    assert metrics["recall@1"] == 0.5
    assert metrics["hit_rate@1"] == 1
    assert metrics["recall@2"] == 1
    assert metrics["map@2"] == 1


def test_unmatched_families_are_rejected():
    with pytest.raises(ValueError, match="another image"):
        evaluate(np.eye(3), np.array(["a", "b", "c"]), ks=(1,))


def test_leading_zero_and_traversal_protection(tmp_path):
    assert image_path(tmp_path, "0108775015") == tmp_path / "images/010/0108775015.jpg"
    with pytest.raises(ValueError):
        image_path(tmp_path, "../../file")
