"""Unit tests for the scikit-learn baseline recommenders."""
import numpy as np
import pandas as pd

from recsys.models.knn_baseline import ItemKNNRecommender
from recsys.models.popularity_baseline import PopularityRecommender
from recsys.models.svd_baseline import SVDRecommender


def _tiny_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2, 3],
            "item_idx": [0, 1, 1, 2, 0, 2, 1],
            "label": [1, 1, 1, 1, 1, 1, 1],
        }
    )


def test_popularity_recommender_scores_more_popular_items_higher():
    strategy = PopularityRecommender()
    strategy.fit(_tiny_train_df(), num_users=4, num_items=3)
    scores = strategy.score_items(user_idx=0, candidate_items=np.array([0, 1, 2]))
    assert scores[1] >= scores[0] >= scores[2]


def test_item_knn_recommender_fits_and_scores():
    strategy = ItemKNNRecommender(n_neighbors=2)
    strategy.fit(_tiny_train_df(), num_users=4, num_items=3)
    scores = strategy.score_items(user_idx=0, candidate_items=np.array([0, 1, 2]))
    assert scores.shape == (3,)


def test_svd_recommender_fits_and_scores():
    strategy = SVDRecommender(n_components=2, seed=0)
    strategy.fit(_tiny_train_df(), num_users=4, num_items=3)
    scores = strategy.score_items(user_idx=0, candidate_items=np.array([0, 1, 2]))
    assert scores.shape == (3,)


def test_recommend_top_k_excludes_given_items():
    strategy = PopularityRecommender()
    strategy.fit(_tiny_train_df(), num_users=4, num_items=3)
    top_k = strategy.recommend_top_k(user_idx=0, k=2, exclude_items={1})
    assert 1 not in top_k
