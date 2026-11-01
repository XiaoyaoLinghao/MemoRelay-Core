"""F10/F11/F12 regression: vector generation keys, late-commit fencing, retries."""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from knowledge_weaver.embeddings.http_provider import EmbeddingUnavailable, HttpEmbeddingProvider
from knowledge_weaver.embeddings.index import (
    HashEmbeddingProvider,
    load_vectors,
    run_embed_job,
    run_embed_worker_once,
)
from knowledge_weaver.jobs import JobError, claim as claim_job, enqueue
from knowledge_weaver.lifecycle.retention import sweep
from knowledge_weaver.retrieval.embeddings_index import collect_vectors

from ..state.conftest import record


class CountingProvider(HashEmbeddingProvider):
    def __init__(self, dimension: int = 2) -> None:
        super().__init__(dimension=dimension)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return super().embed(texts)


def _vector_count(store, revision_id):
    with store.read() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM claim_vectors WHERE revision_id = ?", (revision_id,)
        ).fetchone()["c"]


def test_model_revision_change_invalidates_cache(store):
    recorded = record(store, display_text="端口4200", value=4200)
    first = CountingProvider()
    collect_vectors(store, {recorded["revision_id"]}, first)
    assert first.calls == 1
    second = CountingProvider()
    second.model_revision = "2"
    collect_vectors(store, {recorded["revision_id"]}, second)
    assert second.calls == 1, "a new model revision must not reuse the old vector"
    vectors = load_vectors(
        store,
        {recorded["revision_id"]},
        model_id=second.model_id,
        dimension=2,
        model_revision="2",
        normalization="l2",
    )
    assert recorded["revision_id"] in vectors


def test_changed_revision_text_invalidates_cache(store):
    recorded = record(store, display_text="端口4200", value=4200)
    first = CountingProvider()
    collect_vectors(store, {recorded["revision_id"]}, first)
    with store.transaction() as conn:
        conn.execute(
            "UPDATE claim_revisions SET display_text = ? WHERE revision_id = ?",
            ("端口改为4300", recorded["revision_id"]),
        )
    second = CountingProvider()
    collect_vectors(store, {recorded["revision_id"]}, second)
    assert second.calls == 1, "a stale text hash must force re-embedding"


def test_normalization_change_invalidates_cache(store):
    recorded = record(store, display_text="端口4200", value=4200)
    first = CountingProvider()
    collect_vectors(store, {recorded["revision_id"]}, first)
    assert first.calls == 1
    second = CountingProvider()
    second.normalization = "none"
    collect_vectors(store, {recorded["revision_id"]}, second)
    assert second.calls == 1


def test_late_embedding_commit_is_fenced_after_expiry(store):
    old = record(
        store,
        display_text="旧端口4100",
        value=4100,
        predicate="server.port",
        subject_id="subject:port",
        event_time={"value": "2024-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    record(
        store,
        display_text="新端口4200",
        value=4200,
        predicate="server.port",
        subject_id="subject:port",
        event_time={"value": "2025-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    enqueue(store, "embed", old["revision_id"])
    job = claim_job(store, worker_id="worker:test", kinds=("embed",))[0]

    class ExpiringProvider(HashEmbeddingProvider):
        def embed(self, texts):
            # Retention runs while the provider is computing the vector.
            sweep(store, now="2026-10-30T00:00:00Z")
            return super().embed(texts)

    with pytest.raises(JobError) as excinfo:
        run_embed_job(store, job, ExpiringProvider(dimension=2))
    assert excinfo.value.code == "STALE_CURSOR"
    assert _vector_count(store, old["revision_id"]) == 0
    with store.read() as conn:
        row = conn.execute(
            "SELECT state, last_error_code FROM jobs WHERE kind = 'embed'"
        ).fetchone()
    assert row["state"] == "failed"
    assert row["last_error_code"] == "STALE_CURSOR"


def _closed_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_embedding_outage_records_bounded_retry(store):
    recorded = record(store, display_text="端口4200", value=4200)
    enqueue(store, "embed", recorded["revision_id"])
    provider = HttpEmbeddingProvider(
        f"http://127.0.0.1:{_closed_port()}", model_id="test:embedding", dimension=2, timeout=0.5
    )
    result = run_embed_worker_once(store, provider)
    assert result["state"] == "retry_wait", result
    assert result["error"] == "EMBEDDING_MISMATCH"
    with store.read() as conn:
        job = conn.execute(
            "SELECT state, attempts, last_error_code FROM jobs WHERE kind = 'embed'"
        ).fetchone()
    assert (job["state"], job["attempts"], job["last_error_code"]) == (
        "retry_wait",
        1,
        "EMBEDDING_MISMATCH",
    )


class _StubEmbeddingServer:
    def __init__(self, *, status=200, payload=None) -> None:
        self.status = status
        self.payload = payload
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps(outer.payload if outer.payload is not None else {}).encode()
                self.send_response(outer.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # pragma: no cover
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc_info):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"


def test_embedding_shape_mismatch_is_permanent(store):
    recorded = record(store, display_text="端口4200", value=4200)
    enqueue(store, "embed", recorded["revision_id"])
    with _StubEmbeddingServer(
        payload={"data": [{"embedding": [1.0, 0.0, 0.0]}]}
    ) as server:
        provider = HttpEmbeddingProvider(server.url, model_id="test:embedding", dimension=2)
        result = run_embed_worker_once(store, provider)
    assert result["state"] == "failed", result
    assert result["error"] == "EMBEDDING_MISMATCH"


def test_embedding_rate_limit_is_retryable(store):
    recorded = record(store, display_text="端口4200", value=4200)
    enqueue(store, "embed", recorded["revision_id"])
    with _StubEmbeddingServer(status=429, payload={"error": "slow down"}) as server:
        provider = HttpEmbeddingProvider(server.url, model_id="test:embedding", dimension=2)
        result = run_embed_worker_once(store, provider)
    assert result["state"] == "retry_wait", result


def test_embedding_unavailable_carries_retryability():
    assert EmbeddingUnavailable("outage").retryable is True
    assert EmbeddingUnavailable("bad shape", retryable=False).retryable is False
