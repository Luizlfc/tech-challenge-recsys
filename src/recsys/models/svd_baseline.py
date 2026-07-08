"""Matrix-factorization baseline via TruncatedSVD."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from recsys.data.interaction_matrix import build_sparse_matrix
from recsys.models.base import RecommenderStrategy


class SVDRecommender(RecommenderStrategy):
    """Scores candidates via the dot product of latent user/item factors."""

    name = "svd"

    def __init__(self, n_components: int = 50, seed: int = 42) -> None:
        super().__init__()
        self.n_components = n_components
        self.seed = seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int) -> None:
        """Factorize the sparse user-item matrix into latent user/item vectors."""
        self.num_users, self.num_items = num_users, num_items
        matrix = build_sparse_matrix(train_df, num_users, num_items)

        n_components = min(self.n_components, min(matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        self.user_factors = svd.fit_transform(matrix)
        self.item_factors = svd.components_.T

    def score_items(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        """Dot product between the user vector and each candidate item vector."""
        user_vector = self.user_factors[user_idx]
        return self.item_factors[candidate_items] @ user_vector

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, path)

    def load(self, path: Path) -> None:
        self.__dict__.update(joblib.load(path))
