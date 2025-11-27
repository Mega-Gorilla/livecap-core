"""Shared fixtures for benchmark tests."""

from __future__ import annotations

import os
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def clean_github_env():
    """Ensure GitHub Actions environment variables are cleared for all benchmark tests.

    This prevents tests from accidentally writing to the real GITHUB_STEP_SUMMARY
    when ProgressReporter is initialized during test execution.
    """
    with mock.patch.dict(
        os.environ,
        {"GITHUB_ACTIONS": "", "GITHUB_STEP_SUMMARY": ""},
    ):
        yield
