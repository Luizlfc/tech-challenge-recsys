"""DVC stage 1: clean the raw MovieLens ratings into binary positive interactions."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recsys.config import get_settings  # noqa: E402
from recsys.data.loaders import load_raw_ratings, save_parquet  # noqa: E402
from recsys.utils.io import load_params  # noqa: E402
from recsys.utils.logger import get_logger  # noqa: E402
from recsys.utils.seeding import set_all_seeds  # noqa: E402

logger = get_logger(__name__)


def binarize_positive(ratings: pd.DataFrame, rating_threshold: float) -> pd.DataFrame:
    """Keep only interactions considered a positive signal (rating >= threshold)."""
    positive = ratings[ratings["rating"] >= rating_threshold].copy()
    return positive[["userId", "movieId", "timestamp"]]


def filter_min_interactions(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """Drop users and items with fewer than `min_interactions` positive interactions."""
    item_counts = df["movieId"].value_counts()
    df = df[df["movieId"].isin(item_counts[item_counts >= min_interactions].index)]

    user_counts = df["userId"].value_counts()
    df = df[df["userId"].isin(user_counts[user_counts >= min_interactions].index)]
    return df.reset_index(drop=True)


def preprocess(raw_dir: Path, output_path: Path, min_interactions: int, rating_threshold: float) -> pd.DataFrame:
    """Run the full preprocessing step and persist the cleaned interactions."""
    ratings = load_raw_ratings(raw_dir)
    logger.info("Loaded %d raw ratings", len(ratings))

    positive = binarize_positive(ratings, rating_threshold)
    logger.info("%d interactions kept as positive (rating >= %.1f)", len(positive), rating_threshold)

    cleaned = filter_min_interactions(positive, min_interactions)
    logger.info("%d interactions remain after filtering (min_interactions=%d)", len(cleaned), min_interactions)

    save_parquet(cleaned, output_path)
    logger.info("Saved processed interactions to %s", output_path)
    return cleaned


def main() -> None:
    """Entrypoint for the `preprocess` DVC stage."""
    settings = get_settings()
    params = load_params()["preprocess"]
    set_all_seeds(load_params()["seed"])

    preprocess(
        raw_dir=settings.raw_dir,
        output_path=settings.processed_dir / "interactions.parquet",
        min_interactions=params["min_interactions_per_user"],
        rating_threshold=params["rating_threshold"],
    )


if __name__ == "__main__":
    main()
