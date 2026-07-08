"""Unit tests for the `feature_eng` DVC stage helpers."""
import pandas as pd

from recsys.pipeline.feature_eng import apply_id_maps, build_id_maps, build_user_history, leave_one_out_split


def test_build_id_maps_creates_contiguous_indices():
    df = pd.DataFrame({"userId": [5, 5, 9], "movieId": [100, 200, 100]})
    user_map, item_map = build_id_maps(df)
    assert user_map == {5: 0, 9: 1}
    assert item_map == {100: 0, 200: 1}


def test_apply_id_maps_adds_idx_columns():
    df = pd.DataFrame({"userId": [5, 9], "movieId": [100, 200]})
    user_map, item_map = build_id_maps(df)
    result = apply_id_maps(df, user_map, item_map)
    assert list(result["user_idx"]) == [0, 1]
    assert list(result["item_idx"]) == [0, 1]


def test_leave_one_out_split_holds_out_last_two_per_user():
    df = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 0, 1, 1, 1],
            "item_idx": [1, 2, 3, 4, 5, 6, 7],
            "timestamp": [1, 2, 3, 4, 1, 2, 3],
        }
    )
    train, val, test = leave_one_out_split(df)
    assert test[test["user_idx"] == 0]["item_idx"].item() == 4
    assert val[val["user_idx"] == 0]["item_idx"].item() == 3
    assert set(train[train["user_idx"] == 0]["item_idx"]) == {1, 2}
    assert len(train[train["user_idx"] == 1]) == 1


def test_build_user_history_maps_user_to_item_set():
    df = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [1, 2, 3]})
    history = build_user_history(df)
    assert history[0] == {1, 2}
    assert history[1] == {3}
