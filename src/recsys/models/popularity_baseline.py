"""Non-personalized popularity baseline."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from recsys.models.base import RecommenderStrategy


class PopularityRecommender(RecommenderStrategy):
    """Recommends the globally most-interacted-with items to every user."""

    name = "popularity"

    def __init__(self) -> None:
        super().__init__()
        self.item_scores: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int) -> None:
        """Score items by their (min-max scaled) positive interaction count."""
        self.num_users, self.num_items = num_users, num_items
        positives = train_df[train_df["label"] == 1]
        counts = positives["item_idx"].value_counts().reindex(range(num_items), fill_value=0)
        scaler = MinMaxScaler()
        self.item_scores = scaler.fit_transform(counts.to_numpy().reshape(-1, 1)).ravel()

    def score_items(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        """Every user gets the same non-personalized popularity score."""
        return self.item_scores[candidate_items]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, path)

    def load(self, path: Path) -> None:
        self.__dict__.update(joblib.load(path))
