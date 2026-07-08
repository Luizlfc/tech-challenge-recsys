"""Small filesystem helpers shared across pipeline stages."""
import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it doesn't exist yet."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dict."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a dict to a JSON file, creating parent directories as needed."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_params(path: Path = Path("params.yaml")) -> dict[str, Any]:
    """Load the DVC params.yaml file."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
