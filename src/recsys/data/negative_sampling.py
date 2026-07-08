"""Negative sampling for implicit-feedback training data."""
import numpy as np
import pandas as pd


def sample_negatives(
    positives: pd.DataFrame,
    num_items: int,
    user_full_history: dict[int, set[int]],
    ratio: int,
    seed: int,
) -> pd.DataFrame:
    """Sample `ratio` negative items per positive row, excluding the user's full history."""
    rng = np.random.default_rng(seed)
    all_items = np.arange(num_items)
    rows = []

    for user_idx, group in positives.groupby("user_idx"):
        seen = user_full_history.get(user_idx, set())
        candidates = all_items[~np.isin(all_items, list(seen))]
        if len(candidates) == 0:
            continue
        n_needed = len(group) * ratio
        replace = len(candidates) < n_needed
        sample_size = n_needed if replace else min(n_needed, len(candidates))
        sampled = rng.choice(candidates, size=sample_size, replace=replace)
        rows.extend((user_idx, int(item_idx)) for item_idx in sampled)

    return pd.DataFrame(rows, columns=["user_idx", "item_idx"])
