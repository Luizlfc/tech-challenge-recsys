"""Item-based k-NN collaborative filtering baseline."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from recsys.data.interaction_matrix import build_sparse_matrix
from recsys.models.base import RecommenderStrategy


class ItemKNNRecommender(RecommenderStrategy):
    """Scores candidates via item-item similarity weighted by the user's history."""

    name = "item_knn"

    def __init__(self, n_neighbors: int = 20, metric: str = "cosine") -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.similarity_matrix: csr_matrix | None = None
        self.user_item_matrix: csr_matrix | None = None

    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int) -> None:
        """Fit a sparse top-N item-item cosine similarity matrix."""
        self.num_users, self.num_items = num_users, num_items
        self.user_item_matrix = build_sparse_matrix(train_df, num_users, num_items)

        n_neighbors = min(self.n_neighbors + 1, num_items)
        model = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        model.fit(self.user_item_matrix.T)
        distances, indices = model.kneighbors(self.user_item_matrix.T)

        self.similarity_matrix = self._build_similarity_matrix(indices[:, 1:], 1.0 - distances[:, 1:], num_items)

    @staticmethod
    def _build_similarity_matrix(indices: np.ndarray, similarities: np.ndarray, num_items: int) -> csr_matrix:
        """Assemble the sparse item-item similarity matrix from kneighbors output."""
        rows = np.repeat(np.arange(num_items), indices.shape[1])
        return csr_matrix((similarities.ravel(), (rows, indices.ravel())), shape=(num_items, num_items))

    def score_items(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        """Score = sum of similarity between candidates and the user's known items."""
        user_row = self.user_item_matrix[user_idx]
        all_scores = (user_row @ self.similarity_matrix).toarray().ravel()
        return all_scores[candidate_items]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, path)

    def load(self, path: Path) -> None:
        self.__dict__.update(joblib.load(path))
