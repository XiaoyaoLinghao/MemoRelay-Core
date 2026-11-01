"""Portability tests: isolated source/target stores (synthetic data only)."""

from __future__ import annotations

import pytest

from knowledge_weaver.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "source" / "knowledge.db")
