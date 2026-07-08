"""Application settings loaded from environment variables / .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration shared across pipeline stages."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recsys-movielens"
    mlflow_registry_model_name: str = "movielens-recommender"
    random_seed: int = 42
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    metrics_dir: Path = Path("metrics")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"


def get_settings() -> Settings:
    """Return a freshly loaded Settings instance."""
    return Settings()
