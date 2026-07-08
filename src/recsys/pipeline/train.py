"""DVC stage 3: train the MLP and all baselines, tracking each run in MLflow."""
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recsys.config import get_settings  # noqa: E402
from recsys.data.loaders import load_parquet  # noqa: E402
from recsys.mlflow_utils.tracking import configure_mlflow, mlflow_safe_name, start_run  # noqa: E402
from recsys.models.factory import ModelFactory  # noqa: E402
from recsys.training.trainer import Trainer, TrainingData  # noqa: E402
from recsys.utils.io import load_params, read_json, write_json  # noqa: E402
from recsys.utils.logger import get_logger  # noqa: E402
from recsys.utils.seeding import set_all_seeds  # noqa: E402

logger = get_logger(__name__)

MODEL_SAVE_PATHS = {
    "popularity": Path("baselines/popularity.pkl"),
    "item_knn": Path("baselines/item_knn.pkl"),
    "svd": Path("baselines/svd.pkl"),
    "mlp": Path("mlp/model.pt"),
}


def build_model_kwargs(name: str, params: dict, val_df) -> dict:
    """Assemble constructor kwargs for a given model name from params.yaml."""
    if name == "mlp":
        return {**params["train"]["mlp"], "val_df": val_df}
    if name == "item_knn":
        return dict(params["train"]["baselines"]["knn"])
    if name == "svd":
        return {**params["train"]["baselines"]["svd"], "seed": params["seed"]}
    return {}


def train_one_model(name: str, data: TrainingData, params: dict, k_values: list[int]):
    """Train a single strategy end-to-end and log its params/metrics to MLflow."""
    kwargs = build_model_kwargs(name, params, data.val_df)
    strategy = ModelFactory.create(name, **kwargs)
    trainer = Trainer(strategy, data, k_values)

    with start_run(run_name=name, tags={"dvc_stage": "train", "strategy": name}):
        val_metrics = trainer.run()
        loggable_kwargs = {k: v for k, v in kwargs.items() if k != "val_df"}
        mlflow.log_params({f"{name}.{k}": v for k, v in loggable_kwargs.items()})
        mlflow.log_metrics({mlflow_safe_name(f"val_{k}"): v for k, v in val_metrics.items()})
        logger.info("[%s] val metrics: %s", name, val_metrics)

    return strategy


def main() -> None:
    """Entrypoint for the `train` DVC stage."""
    settings = get_settings()
    params = load_params()
    set_all_seeds(params["seed"])
    configure_mlflow(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)

    train_df = load_parquet(settings.features_dir / "train.parquet")
    val_df = load_parquet(settings.features_dir / "val.parquet")
    user_map = read_json(settings.features_dir / "user_id_map.json")
    item_map = read_json(settings.features_dir / "item_id_map.json")
    data = TrainingData(train_df, val_df, len(user_map), len(item_map))
    k_values = params["evaluate"]["k_values"]

    trained = {}
    for name in params["train"]["models_to_run"]:
        strategy = train_one_model(name, data, params, k_values)
        save_path = settings.models_dir / MODEL_SAVE_PATHS[name]
        strategy.save(save_path)
        trained[name] = str(save_path)

    write_json(trained, settings.models_dir / "trained_models.json")
    logger.info("Training complete for models: %s", list(trained))


if __name__ == "__main__":
    main()
