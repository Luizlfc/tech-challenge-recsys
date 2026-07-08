"""DVC stage 4: evaluate all trained models, pick a winner, and manage the MLflow registry."""
import sys
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recsys.config import get_settings  # noqa: E402
from recsys.data.loaders import load_parquet  # noqa: E402
from recsys.evaluation.evaluator import evaluate_model  # noqa: E402
from recsys.mlflow_utils.registry import promote_to_production, register_best_model  # noqa: E402
from recsys.mlflow_utils.tracking import configure_mlflow, mlflow_safe_name, start_run  # noqa: E402
from recsys.models.factory import ModelFactory  # noqa: E402
from recsys.utils.io import load_params, read_json, write_json  # noqa: E402
from recsys.utils.logger import get_logger  # noqa: E402
from recsys.utils.seeding import set_all_seeds  # noqa: E402

logger = get_logger(__name__)


def build_seen_items(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[int, set[int]]:
    """Union of items each user interacted with in train+val (excluded from ranking)."""
    seen = pd.concat([train_df[train_df["label"] == 1], val_df[val_df["label"] == 1]])
    return seen.groupby("user_idx")["item_idx"].apply(set).to_dict()


def load_trained_model(name: str, path: str):
    """Recreate a strategy instance and load its saved weights."""
    strategy = ModelFactory.create(name)
    strategy.load(Path(path))
    return strategy


def evaluate_all_models(
    trained_models: dict[str, str], test_df: pd.DataFrame, seen_items: dict[int, set[int]], k_values: list[int]
) -> dict[str, dict[str, float]]:
    """Evaluate every trained model on the held-out test split."""
    results = {}
    for name, path in trained_models.items():
        strategy = load_trained_model(name, path)
        results[name] = evaluate_model(strategy, test_df, seen_items, k_values)
        logger.info("[%s] test metrics: %s", name, results[name])
    return results


def log_model_artifact(model_path: Path) -> str:
    """Log the winning model's checkpoint file to MLflow and return a registry-ready URI."""
    mlflow.log_artifact(str(model_path), artifact_path="model")
    run_id = mlflow.active_run().info.run_id
    return f"runs:/{run_id}/model/{model_path.name}"


def main() -> None:
    """Entrypoint for the `evaluate` DVC stage."""
    settings = get_settings()
    params = load_params()
    set_all_seeds(params["seed"])
    configure_mlflow(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)

    train_df = load_parquet(settings.features_dir / "train.parquet")
    val_df = load_parquet(settings.features_dir / "val.parquet")
    test_df = load_parquet(settings.features_dir / "test.parquet")
    seen_items = build_seen_items(train_df, val_df)
    trained_models = read_json(settings.models_dir / "trained_models.json")

    k_values = params["evaluate"]["k_values"]
    promotion_metric = params["evaluate"]["promotion_metric"]
    results = evaluate_all_models(trained_models, test_df, seen_items, k_values)

    comparison = pd.DataFrame(results).T
    write_json(results, settings.metrics_dir / "eval_metrics.json")
    comparison.to_csv(settings.metrics_dir / "eval_table.csv")

    best_name = max(results, key=lambda name: results[name][promotion_metric])
    logger.info("Best model by %s: %s (%.4f)", promotion_metric, best_name, results[best_name][promotion_metric])

    with start_run(run_name="evaluate", tags={"dvc_stage": "evaluate"}):
        mlflow.log_artifact(str(settings.metrics_dir / "eval_table.csv"))
        for name, metrics in results.items():
            mlflow.log_metrics({mlflow_safe_name(f"{name}.{metric}"): value for metric, value in metrics.items()})

        model_uri = log_model_artifact(Path(trained_models[best_name]))
        version = register_best_model(model_uri, settings.mlflow_registry_model_name, tags={"strategy": best_name})
        promoted = promote_to_production(
            settings.mlflow_registry_model_name,
            version.version,
            results[best_name][promotion_metric],
            params["evaluate"]["promotion_threshold"],
        )
        mlflow.set_tag("promoted_to_production", str(promoted))

    logger.info("Model '%s' registered as version %s (promoted=%s)", best_name, version.version, promoted)


if __name__ == "__main__":
    main()
