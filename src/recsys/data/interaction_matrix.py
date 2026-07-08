"""Shared sparse user-item matrix construction for the sklearn baselines."""
import pandas as pd
from scipy.sparse import csr_matrix


def build_sparse_matrix(df: pd.DataFrame, num_users: int, num_items: int) -> csr_matrix:
    """Build a binary user-item sparse matrix from positive interaction rows."""
    positives = df[df.get("label", 1) == 1] if "label" in df.columns else df
    rows = positives["user_idx"].to_numpy()
    cols = positives["item_idx"].to_numpy()
    data = [1.0] * len(positives)
    return csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
