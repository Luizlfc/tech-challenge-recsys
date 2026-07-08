"""Unit tests for the `preprocess` DVC stage helpers."""
import pandas as pd

from recsys.pipeline.preprocess import binarize_positive, filter_min_interactions


def test_binarize_positive_keeps_only_ratings_above_threshold():
    ratings = pd.DataFrame(
        {"userId": [1, 1, 2], "movieId": [10, 20, 10], "rating": [5.0, 2.0, 4.0], "timestamp": [1, 2, 3]}
    )
    result = binarize_positive(ratings, rating_threshold=4.0)
    assert set(result["movieId"]) == {10}
    assert len(result) == 2


def test_filter_min_interactions_drops_sparse_users():
    df = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 3, 3],
            "movieId": [10, 20, 30, 10, 20, 30],
            "timestamp": [1, 2, 3, 4, 5, 6],
        }
    )
    result = filter_min_interactions(df, min_interactions=2)
    assert set(result["userId"]) == {1, 3}
