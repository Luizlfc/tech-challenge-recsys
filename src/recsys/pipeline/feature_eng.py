"""DVC stage 2: build id maps, leave-one-out split and negative-sampled train/val sets."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recsys.config import get_settings  # noqa: E402
from recsys.data.loaders import load_parquet, save_parquet  # noqa: E402
from recsys.data.negative_sampling import sample_negatives  # noqa: E402
from recsys.utils.io import load_params, write_json  # noqa: E402
from recsys.utils.logger import get_logger  # noqa: E402
from recsys.utils.seeding import set_all_seeds  # noqa: E402

logger = get_logger(__name__)


def build_id_maps(df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Map raw MovieLens userId/movieId to contiguous 0-based indices."""
    user_map = {raw_id: idx for idx, raw_id in enumerate(sorted(df["userId"].unique()))}
    item_map = {raw_id: idx for idx, raw_id in enumerate(sorted(df["movieId"].unique()))}
    return user_map, item_map


def apply_id_maps(df: pd.DataFrame, user_map: dict[int, int], item_map: dict[int, int]) -> pd.DataFrame:
    """Attach user_idx/item_idx columns derived from the id maps."""
    df = df.copy()
    df["user_idx"] = df["userId"].map(user_map)
    df["item_idx"] = df["movieId"].map(item_map)
    return df


def leave_one_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per user, hold out the most recent interaction for test and the next for val."""
    ordered = df.sort_values(["user_idx", "timestamp"])
    ranked = ordered.groupby("user_idx").cumcount(ascending=False)

    test = ordered[ranked == 0]
    val = ordered[ranked == 1]
    train = ordered[ranked >= 2]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_user_history(df: pd.DataFrame) -> dict[int, set[int]]:
    """Map each user_idx to the full set of items they've interacted with."""
    return df.groupby("user_idx")["item_idx"].apply(set).to_dict()


def build_popularity(train: pd.DataFrame, item_map: dict[int, int]) -> pd.DataFrame:
    """Count positive interactions per item in the training split."""
    counts = train["item_idx"].value_counts().rename_axis("item_idx").reset_index(name="count")
    all_items = pd.DataFrame({"item_idx": list(item_map.values())})
    return all_items.merge(counts, on="item_idx", how="left").fillna(0)


def add_labels(positives: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    """Combine positive/negative rows into a single labeled, shuffled dataset."""
    positives = positives[["user_idx", "item_idx"]].assign(label=1)
    negatives = negatives.assign(label=0)
    combined = pd.concat([positives, negatives], ignore_index=True)
    return combined.sample(frac=1.0, random_state=0).reset_index(drop=True)


def main() -> None:
    """Entrypoint for the `feature_eng` DVC stage."""
    settings = get_settings()
    all_params = load_params()
    params = all_params["feature_eng"]
    set_all_seeds(all_params["seed"])

    interactions = load_parquet(settings.processed_dir / "interactions.parquet")
    user_map, item_map = build_id_maps(interactions)
    indexed = apply_id_maps(interactions, user_map, item_map)

    train_pos, val_pos, test_pos = leave_one_out_split(indexed)
    logger.info("Split sizes -> train: %d, val: %d, test: %d", len(train_pos), len(val_pos), len(test_pos))

    full_history = build_user_history(indexed)
    ratio = params["negative_ratio"]
    seed = params["seed"]

    train_negatives = sample_negatives(train_pos, len(item_map), full_history, ratio, seed)
    val_negatives = sample_negatives(val_pos, len(item_map), full_history, ratio, seed + 1)

    train_df = add_labels(train_pos, train_negatives)
    val_df = add_labels(val_pos, val_negatives)
    test_df = test_pos[["user_idx", "item_idx"]].assign(label=1)

    popularity = build_popularity(train_pos, item_map)

    features_dir = settings.features_dir
    save_parquet(train_df, features_dir / "train.parquet")
    save_parquet(val_df, features_dir / "val.parquet")
    save_parquet(test_df, features_dir / "test.parquet")
    save_parquet(popularity, features_dir / "popularity.parquet")
    write_json({str(k): v for k, v in user_map.items()}, features_dir / "user_id_map.json")
    write_json({str(k): v for k, v in item_map.items()}, features_dir / "item_id_map.json")

    logger.info("Feature engineering complete: %d users, %d items", len(user_map), len(item_map))


if __name__ == "__main__":
    main()
