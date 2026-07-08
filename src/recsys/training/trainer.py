"""Template Method: a fixed fit -> evaluate skeleton shared by every strategy."""
from dataclasses import dataclass

import pandas as pd

from recsys.evaluation.evaluator import evaluate_model
from recsys.models.base import RecommenderStrategy


@dataclass
class TrainingData:
    """Bundles the artifacts every training run needs."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    num_users: int
    num_items: int


class Trainer:
    """Runs the same fit -> evaluate skeleton regardless of which strategy is injected."""

    def __init__(self, strategy: RecommenderStrategy, data: TrainingData, k_values: list[int]) -> None:
        self.strategy = strategy
        self.data = data
        self.k_values = k_values

    def run(self) -> dict[str, float]:
        """Fit the strategy on train and evaluate it against the validation split."""
        self.strategy.fit(self.data.train_df, self.data.num_users, self.data.num_items)
        return self.evaluate()

    def evaluate(self) -> dict[str, float]:
        """Rank validation items, excluding what the strategy already saw in train."""
        train_positives = self.data.train_df[self.data.train_df["label"] == 1]
        exclude_by_user = train_positives.groupby("user_idx")["item_idx"].apply(set).to_dict()
        val_positives = self.data.val_df[self.data.val_df["label"] == 1]
        return evaluate_model(self.strategy, val_positives, exclude_by_user, self.k_values)
