"""Neural collaborative-filtering style MLP recommender."""
import torch
from torch import nn


class NeuralCFModel(nn.Module):
    """Concatenates user/item embeddings and scores them through an MLP tower."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_layers: list[int] = (128, 64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = self._build_mlp(embedding_dim * 2, list(hidden_layers), dropout)
        self.output_layer = nn.Linear(list(hidden_layers)[-1], 1)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_layers: list[int], dropout: float) -> nn.Sequential:
        """Build the Linear -> ReLU -> Dropout tower from the hidden layer sizes."""
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers += [nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            current_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Return raw logits (apply sigmoid externally / use BCEWithLogitsLoss)."""
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        x = torch.cat([user_vec, item_vec], dim=-1)
        x = self.mlp(x)
        return self.output_layer(x).squeeze(-1)
