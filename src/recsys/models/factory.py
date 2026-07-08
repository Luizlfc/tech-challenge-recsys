"""Factory Method for constructing RecommenderStrategy instances by name."""
from recsys.models.base import RecommenderStrategy
from recsys.models.knn_baseline import ItemKNNRecommender
from recsys.models.mlp_wrapper import MLPRecommenderStrategy
from recsys.models.popularity_baseline import PopularityRecommender
from recsys.models.svd_baseline import SVDRecommender


class ModelFactory:
    """Creates recommender strategies by name so callers never import concrete classes."""

    _registry: dict[str, type[RecommenderStrategy]] = {
        "popularity": PopularityRecommender,
        "item_knn": ItemKNNRecommender,
        "svd": SVDRecommender,
        "mlp": MLPRecommenderStrategy,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> RecommenderStrategy:
        """Instantiate the strategy registered under `name`."""
        if name not in cls._registry:
            raise ValueError(f"Unknown model '{name}'. Available: {list(cls._registry)}")
        return cls._registry[name](**kwargs)

    @classmethod
    def register(cls, name: str, strategy_cls: type[RecommenderStrategy]) -> None:
        """Register a new strategy class under `name` (open for extension)."""
        cls._registry[name] = strategy_cls
