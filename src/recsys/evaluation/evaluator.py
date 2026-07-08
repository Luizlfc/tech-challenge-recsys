"""Aggregate ranking metrics across users for a fitted RecommenderStrategy."""
import numpy as np
import pandas as pd

from recsys.evaluation.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recsys.models.base import RecommenderStrategy

METRIC_FUNCTIONS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "hit_rate": hit_rate_at_k,
    "mrr": mrr_at_k,
}


def evaluate_model(
    strategy: RecommenderStrategy,
    test_df: pd.DataFrame,
    exclude_items_by_user: dict[int, set[int]],
    k_values: list[int],
) -> dict[str, float]:
    """Rank items for every test user and average each metric across users."""
    max_k = max(k_values)
    scores: dict[str, list[float]] = {f"{metric}@{k}": [] for metric in METRIC_FUNCTIONS for k in k_values}

    for user_idx, item_idx in zip(test_df["user_idx"], test_df["item_idx"], strict=True):
        exclude = exclude_items_by_user.get(user_idx, set()) - {item_idx}
        ranked = strategy.recommend_top_k(user_idx, max_k, exclude_items=exclude)
        relevant = {item_idx}

        for k in k_values:
            for metric_name, metric_fn in METRIC_FUNCTIONS.items():
                scores[f"{metric_name}@{k}"].append(metric_fn(ranked, relevant, k))

    return {name: float(np.mean(values)) if values else 0.0 for name, values in scores.items()}
