"""Standalone script to (re)apply the Staging -> Production promotion gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlflow.tracking import MlflowClient  # noqa: E402

from recsys.config import get_settings  # noqa: E402
from recsys.mlflow_utils.registry import promote_to_production  # noqa: E402
from recsys.mlflow_utils.tracking import configure_mlflow, mlflow_safe_name  # noqa: E402
from recsys.utils.io import load_params  # noqa: E402
from recsys.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def get_latest_staging_version(client: MlflowClient, model_name: str):
    """Return the most recent version currently in the Staging stage."""
    versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not versions:
        raise RuntimeError(f"No version of '{model_name}' is in Staging.")
    return versions[0]


def main() -> None:
    """Promote the latest Staging model version to Production if it meets the threshold."""
    settings = get_settings()
    params = load_params()
    configure_mlflow(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)

    client = MlflowClient()
    version = get_latest_staging_version(client, settings.mlflow_registry_model_name)
    run = client.get_run(version.run_id)
    metric_name = params["evaluate"]["promotion_metric"]
    strategy_name = version.tags.get("strategy", "unknown")
    logged_key = mlflow_safe_name(f"{strategy_name}.{metric_name}")
    metric_value = run.data.metrics.get(logged_key)

    if metric_value is None:
        raise RuntimeError(f"Metric '{logged_key}' not found on run {version.run_id}")

    promoted = promote_to_production(
        settings.mlflow_registry_model_name, version.version, metric_value, params["evaluate"]["promotion_threshold"]
    )
    logger.info("Version %s (%s=%.4f) promoted=%s", version.version, metric_name, metric_value, promoted)


if __name__ == "__main__":
    main()
