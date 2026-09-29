"""Smoke tests for the initial package structure."""

from importlib import import_module

import pytest

PACKAGES = (
    "bot",
    "collectors",
    "domain",
    "jobs",
    "letters",
    "ml",
    "nlp",
    "processing",
    "scoring",
    "storage",
)


@pytest.mark.parametrize("package_name", PACKAGES, ids=PACKAGES)
def test_package_can_be_imported(package_name: str) -> None:
    """Each application package is importable from the src directory."""
    module = import_module(package_name)

    assert module.__name__ == package_name
