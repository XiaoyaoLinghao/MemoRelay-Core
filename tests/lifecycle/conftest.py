"""Retention tests: isolated store per test (synthetic data only)."""

from __future__ import annotations

import pytest

from knowledge_weaver.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")
