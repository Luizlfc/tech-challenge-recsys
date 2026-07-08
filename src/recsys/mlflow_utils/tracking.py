"""MLflow tracking helpers."""
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow


def mlflow_safe_name(name: str) -> str:
    """MLflow metric/param names disallow '@' - map precision@5 -> precision_at_5."""
    return name.replace("@", "_at_")


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at the tracking server and select/create the experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
    """Context manager wrapping mlflow.start_run with a run name and tags."""
    with mlflow.start_run(run_name=run_name, tags=tags):
        yield
