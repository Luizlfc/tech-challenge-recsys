"""Adapter exposing NeuralCFModel through the RecommenderStrategy interface."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from recsys.data.interaction_dataset import InteractionDataset
from recsys.models.base import RecommenderStrategy
from recsys.models.mlp_recommender import NeuralCFModel
from recsys.utils.logger import get_logger

logger = get_logger(__name__)


class MLPRecommenderStrategy(RecommenderStrategy):
    """Trains and serves a NeuralCFModel behind the common recommender interface."""

    name = "mlp"

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_layers: list[int] = (128, 64, 32),
        dropout: float = 0.2,
        lr: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        weight_decay: float = 0.0,
        early_stopping_patience: int = 3,
        val_df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_layers = list(hidden_layers)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.val_df = val_df
        self.model: NeuralCFModel | None = None
        self.training_history: list[dict[str, float]] = []

    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int) -> None:
        """Build the model and run the early-stopping training loop."""
        self.num_users, self.num_items = num_users, num_items
        self.model = NeuralCFModel(num_users, num_items, self.embedding_dim, self.hidden_layers, self.dropout)

        train_loader = DataLoader(InteractionDataset(train_df), batch_size=self.batch_size, shuffle=True)
        val_loader = self._build_val_loader()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        self._train_with_early_stopping(train_loader, val_loader, optimizer, criterion)

    def _build_val_loader(self) -> DataLoader | None:
        """Build the validation DataLoader, if a validation set was provided."""
        if self.val_df is None:
            return None
        return DataLoader(InteractionDataset(self.val_df), batch_size=self.batch_size)

    def _train_with_early_stopping(
        self, train_loader: DataLoader, val_loader: DataLoader | None, optimizer, criterion
    ) -> None:
        """Run the training loop, stopping early once val loss stops improving."""
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            train_loss = self._run_epoch(train_loader, optimizer, criterion, train_mode=True)
            val_loss = self._run_epoch(val_loader, optimizer, criterion, train_mode=False) if val_loader else train_loss
            self.training_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            logger.info("Epoch %d: train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss, epochs_without_improvement = val_loss, 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

    def _run_epoch(self, loader: DataLoader, optimizer, criterion, train_mode: bool) -> float:
        """Run one training or validation epoch and return the average loss."""
        self.model.train(train_mode)
        total_loss, total_examples = 0.0, 0

        for user_idx, item_idx, label in loader:
            with torch.set_grad_enabled(train_mode):
                logits = self.model(user_idx, item_idx)
                loss = criterion(logits, label.float())
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * len(label)
            total_examples += len(label)

        return total_loss / max(total_examples, 1)

    def score_items(self, user_idx: int, candidate_items: np.ndarray) -> np.ndarray:
        """Score every candidate item for a user with a single forward pass."""
        self.model.eval()
        with torch.no_grad():
            users = torch.full((len(candidate_items),), user_idx, dtype=torch.long)
            items = torch.tensor(candidate_items, dtype=torch.long)
            logits = self.model(users, items)
        return torch.sigmoid(logits).numpy()

    def save(self, path: Path) -> None:
        """Persist model weights and architecture metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "num_users": self.num_users,
                "num_items": self.num_items,
                "embedding_dim": self.embedding_dim,
                "hidden_layers": self.hidden_layers,
                "dropout": self.dropout,
            },
            path,
        )

    def load(self, path: Path) -> None:
        """Rebuild the model architecture and restore its weights."""
        checkpoint = torch.load(path, weights_only=False)
        self.num_users = checkpoint["num_users"]
        self.num_items = checkpoint["num_items"]
        self.embedding_dim = checkpoint["embedding_dim"]
        self.hidden_layers = checkpoint["hidden_layers"]
        self.dropout = checkpoint["dropout"]
        self.model = NeuralCFModel(self.num_users, self.num_items, self.embedding_dim, self.hidden_layers, self.dropout)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
