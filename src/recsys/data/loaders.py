"""Raw/processed data loading helpers."""
from pathlib import Path

import pandas as pd


def load_raw_ratings(raw_dir: Path) -> pd.DataFrame:
    """Load the MovieLens ratings.csv file."""
    ratings_path = raw_dir / "ml-latest-small" / "ratings.csv"
    return pd.read_csv(ratings_path)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file."""
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
