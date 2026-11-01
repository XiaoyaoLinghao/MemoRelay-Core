"""Retrieval fixtures: isolated store, bindings, recall requests, test embedder."""
from __future__ import annotations

import math

import pytest

from knowledge_weaver.embeddings.index import configure_model
from knowledge_weaver.store import Store

from ..state.conftest import record  # noqa: F401 - re-exported claim builder


@pytest.fixture(autouse=True)
def reset_embedding_config():
    configure_model(None)
    yield
    configure_model(None)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")


@pytest.fixture
def binding():
    return {
        "client_id": "client:test",
        "workspace_id": "owner:demo",
        "project_ids": ["project:songta"],
        "actor_ids": ["*"],
    }


def recall_request(**overrides):
    request = {
        "schema_version": "1.0",
        "request_id": "req:test:1",
        "session_id": "session:test",
        "turn_id": "turn:1",
        "query": "端口",
        "project_ids": ["project:songta"],
        "intent": "current",
        "budget_tokens": 2000,
        "limit": 8,
        "deadline_ms": 7000,
        "cursor": None,
    }
    request.update(overrides)
    return request


class ConceptEmbeddingProvider:
    """Tiny concept embedder: real 'semantics' without a model download."""

    model_id = "test:concept"
    model_revision = "1"
    dimension = 8
    normalization = "l2"
    CONCEPTS = {
        0: ["端口", "port", "4200", "4100", "监听", "listen"],
        1: ["备份", "backup", "每周", "weekly"],
        2: ["nas", "存储", "storage", "共享"],
    }

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [0.0] * self.dimension
            for index, keywords in self.CONCEPTS.items():
                if any(keyword in lowered for keyword in keywords):
                    vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class FailingEmbeddingProvider:
    model_id = "test:broken"
    model_revision = "1"
    dimension = 8
    normalization = "l2"

    def embed(self, texts):
        raise RuntimeError("synthetic embedding outage")
