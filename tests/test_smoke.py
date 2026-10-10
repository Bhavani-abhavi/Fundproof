"""Trivial smoke test: the package imports and has a version."""

import uwcopilot


def test_package_has_version() -> None:
    assert uwcopilot.__version__ == "0.1.0"
