"""PyTorch Dataset wrapping labeled (user_idx, item_idx, label) interaction rows."""
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """Exposes user/item/label rows as tensors for the MLP training loop."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.user_idx = torch.tensor(df["user_idx"].to_numpy(), dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].to_numpy(), dtype=torch.long)
        self.label = torch.tensor(df["label"].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_idx[index], self.item_idx[index], self.label[index]
