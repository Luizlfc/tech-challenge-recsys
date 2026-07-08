"""Shared pytest fixtures."""
import pytest

from recsys.utils.seeding import set_all_seeds


@pytest.fixture(autouse=True)
def _fixed_seed() -> None:
    """Ensure every test runs with deterministic RNG state."""
    set_all_seeds(42)
