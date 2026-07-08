"""Strategy interface shared by every recommender (baselines and the MLP)."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class RecommenderStrategy(ABC):
    """Common contract so `train.py`/`evaluate.py` can treat all models the same way."""

    name: str = "base"

    def __init__(self) -> None:
        self.num_users: int = 0
        self.num_items: int = 0

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int) -> None:
        """Fit the recommender on the (user_idx, item_idx, label) training rows."""

    @abstractmethod
    def score_items(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        """Return a relevance score for each candidate item, for the given user."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a previously fitted model from disk."""

    def recommend_top_k(self, user_idx: int, k: int, exclude_items: set[int] | None = None) -> list[int]:
        """Rank all candidate items for a user and return the top-k item indices."""
        exclude_items = exclude_items or set()
        candidates = np.array([item for item in range(self.num_items) if item not in exclude_items])
        if len(candidates) == 0:
            return []

        scores = self.score_items(user_idx, candidates)
        top_k_positions = np.argsort(-scores)[:k]
        return candidates[top_k_positions].tolist()
