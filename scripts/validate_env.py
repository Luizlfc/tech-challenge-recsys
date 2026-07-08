"""Validate that the local environment has everything needed to run the pipeline."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

REQUIRED_MODULES = ["torch", "sklearn", "mlflow", "pandas", "numpy", "scipy", "pydantic"]
REQUIRED_BINARIES = ["dvc", "docker"]


def validate_modules() -> list[str]:
    """Return the list of required Python modules that fail to import."""
    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


def validate_binaries() -> list[str]:
    """Return the list of required CLI binaries not found on PATH."""
    return [binary for binary in REQUIRED_BINARIES if shutil.which(binary) is None]


def validate_env_file() -> bool:
    """Check that a .env file exists (copied from .env.example)."""
    return Path(".env").exists()


def main() -> int:
    """Run all validations and print a report; return a process exit code."""
    missing_modules = validate_modules()
    missing_binaries = validate_binaries()
    has_env_file = validate_env_file()

    if missing_modules:
        logger.error("Missing Python packages: %s", ", ".join(missing_modules))
    if missing_binaries:
        logger.warning("Missing CLI binaries (needed for DVC/Docker steps): %s", ", ".join(missing_binaries))
    if not has_env_file:
        logger.warning(".env not found - copy .env.example to .env before running the pipeline")

    if not missing_modules:
        logger.info("All required Python packages are importable.")

    return 1 if missing_modules else 0


if __name__ == "__main__":
    raise SystemExit(main())
