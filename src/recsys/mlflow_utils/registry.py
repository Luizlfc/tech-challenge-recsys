"""MLflow Model Registry helpers: register a model and gate Staging -> Production."""
import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient


def register_best_model(model_uri: str, model_name: str, tags: dict[str, str]) -> ModelVersion:
    """Register a model version under `model_name` and move it to Staging."""
    client = MlflowClient()
    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version.version, key, value)
    client.transition_model_version_stage(model_name, version.version, stage="Staging")
    return version


def promote_to_production(model_name: str, version: str, metric_value: float, threshold: float) -> bool:
    """Promote a Staging version to Production only if it clears the quality threshold."""
    if metric_value < threshold:
        return False
    client = MlflowClient()
    client.transition_model_version_stage(model_name, version, stage="Production", archive_existing_versions=True)
    return True
