"""Smoke tests for the PyTorch NeuralCFModel and its strategy adapter."""
import numpy as np
import pandas as pd
import torch

from recsys.models.mlp_recommender import NeuralCFModel
from recsys.models.mlp_wrapper import MLPRecommenderStrategy


def test_neural_cf_forward_pass_shape():
    model = NeuralCFModel(num_users=5, num_items=5, embedding_dim=4, hidden_layers=[8, 4], dropout=0.0)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 1, 2])
    logits = model(users, items)
    assert logits.shape == (3,)


def _tiny_labeled_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2],
            "item_idx": [0, 1, 1, 2, 0, 2],
            "label": [1, 0, 1, 0, 1, 0],
        }
    )


def test_mlp_recommender_strategy_trains_one_epoch_without_error():
    strategy = MLPRecommenderStrategy(
        embedding_dim=4, hidden_layers=[8, 4], epochs=1, batch_size=2, early_stopping_patience=1
    )
    strategy.fit(_tiny_labeled_df(), num_users=3, num_items=3)
    scores = strategy.score_items(user_idx=0, candidate_items=np.array([0, 1, 2]))
    assert len(scores) == 3
    assert len(strategy.training_history) == 1
