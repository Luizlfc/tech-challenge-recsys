"""Unit tests for the ModelFactory (Factory pattern)."""
import pytest

from recsys.models.factory import ModelFactory
from recsys.models.knn_baseline import ItemKNNRecommender
from recsys.models.popularity_baseline import PopularityRecommender


def test_create_returns_expected_type():
    strategy = ModelFactory.create("popularity")
    assert isinstance(strategy, PopularityRecommender)


def test_create_passes_kwargs_to_constructor():
    strategy = ModelFactory.create("item_knn", n_neighbors=5)
    assert isinstance(strategy, ItemKNNRecommender)
    assert strategy.n_neighbors == 5


def test_create_unknown_model_raises():
    with pytest.raises(ValueError):
        ModelFactory.create("does_not_exist")


def test_register_adds_new_strategy():
    class DummyStrategy(PopularityRecommender):
        pass

    ModelFactory.register("dummy", DummyStrategy)
    assert isinstance(ModelFactory.create("dummy"), DummyStrategy)
