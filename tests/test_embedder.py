"""Tests for the embedder module."""

import os
import httpx
import pytest
from unittest.mock import patch, MagicMock
from knowledge_weaver.embedder import EmbeddingClient, get_embedder


class TestEmbedderFromEnv:
    """test_embedder_from_env: set env vars, create embedder, verify config."""

    def test_creates_client_from_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-key-123")
        monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

        client = get_embedder()
        assert client is not None
        assert client.base_url == "https://api.example.com/v1"
        assert client.api_key == "sk-test-key-123"
        assert client.model == "BAAI/bge-large-zh-v1.5"
        assert client.dimension == 1024

    def test_strips_trailing_slash_from_base_url(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com/v1/")
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "test-model")

        client = get_embedder()
        assert client is not None
        assert client.base_url == "https://api.example.com/v1"

    def test_custom_dimension(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")

        client = get_embedder()
        assert client is not None
        assert client.dimension == 1536


class TestEmbedderNoEnv:
    """test_embedder_no_env: no env vars → embedder should be None/disabled."""

    def test_returns_none_when_base_url_missing(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "test-model")

        assert get_embedder() is None

    def test_returns_none_when_api_key_missing(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com/v1")
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        monkeypatch.setenv("EMBEDDING_MODEL", "test-model")

        assert get_embedder() is None

    def test_returns_none_when_model_missing(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-key")
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

        assert get_embedder() is None

    def test_returns_none_when_all_missing(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

        assert get_embedder() is None


class TestEmbedText:
    """test_embed_text: mock httpx, verify correct API call format."""

    def test_single_text_embedding(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.test.com/v1/embeddings",
            method="POST",
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "test-model",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

        client = EmbeddingClient(
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            model="test-model",
            dimension=3,
        )
        result = client.embed("测试文本")

        assert result == [0.1, 0.2, 0.3]

        # Verify the request format
        request = httpx_mock.get_requests()[-1]
        assert request.method == "POST"
        assert str(request.url) == "https://api.test.com/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer sk-test"
        assert request.headers["Content-Type"] == "application/json"

        import json
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["input"] == ["测试文本"]

    def test_http_error_returns_empty_list(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.test.com/v1/embeddings",
            method="POST",
            status_code=401,
            json={"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
        )

        client = EmbeddingClient(
            base_url="https://api.test.com/v1",
            api_key="sk-bad",
            model="test-model",
        )
        result = client.embed("test")

        assert result == []

    def test_connection_error_returns_empty_list(self, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))

        client = EmbeddingClient(
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            model="test-model",
        )
        result = client.embed("test")

        assert result == []


class TestEmbedBatch:
    """test_embed_batch: batch embedding multiple texts."""

    def test_batch_embedding(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.test.com/v1/embeddings",
            method="POST",
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                ],
                "model": "test-model",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )

        client = EmbeddingClient(
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            model="test-model",
            dimension=2,
        )
        results = client.embed_batch(["text1", "text2"])

        assert len(results) == 2
        assert results[0] == [0.1, 0.2]
        assert results[1] == [0.3, 0.4]

        # Verify batch request format
        request = httpx_mock.get_requests()[-1]
        import json
        body = json.loads(request.content)
        assert body["input"] == ["text1", "text2"]

    def test_empty_batch_returns_empty(self):
        client = EmbeddingClient(
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            model="test-model",
        )
        results = client.embed_batch([])
        assert results == []

    def test_batch_http_error_returns_empty(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.test.com/v1/embeddings",
            method="POST",
            status_code=500,
            json={"error": "internal server error"},
        )

        client = EmbeddingClient(
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            model="test-model",
        )
        results = client.embed_batch(["text1", "text2"])

        # On error, each input gets an empty embedding vector
        assert results == [[], []]


class TestEmbedDimension:
    """test_embed_dimension: verify dimension parameter handling."""

    def test_default_dimension_is_1024(self):
        client = EmbeddingClient(
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            model="test-model",
        )
        assert client.dimension == 1024

    def test_custom_dimension(self):
        client = EmbeddingClient(
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            model="text-embedding-3-small",
            dimension=1536,
        )
        assert client.dimension == 1536

    def test_dimension_from_env_var(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "3072")

        client = get_embedder()
        assert client is not None
        assert client.dimension == 3072
