"""F01 regression: the recall query is outbound data and must be protected.

The embedding/model endpoint is a synthetic loopback server; all secrets are
synthetic test canaries. The test records the exact JSON bodies the product
would send so a leak is observable.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from knowledge_weaver.embeddings.http_provider import HttpEmbeddingProvider
from knowledge_weaver.retrieval import execute_recall

from .conftest import recall_request


class _CapturingEmbeddingServer:
    def __init__(self) -> None:
        self.bodies: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.bodies.append(payload)
                data = json.dumps(
                    {"data": [{"embedding": [1.0, 0.0]} for _ in payload["input"]]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # pragma: no cover - silence test server
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

    def provider(self) -> HttpEmbeddingProvider:
        return HttpEmbeddingProvider(
            f"http://127.0.0.1:{self.server.server_port}",
            model_id="test:embedding",
            dimension=2,
        )


@pytest.fixture
def clean_store(store):
    """A stored fact with clean text so the only secret input is the query."""
    from .conftest import record

    record(store, display_text="端口4200", value=4200)
    return store


def test_query_secret_is_not_sent_to_embedding_endpoint(clean_store, binding):
    canary = "T20_QUERY_SECRET_72"
    with _CapturingEmbeddingServer() as capture:
        result = execute_recall(
            clean_store,
            recall_request(query=f"端口 密码: {canary}"),
            binding=binding,
            embedder=capture.provider(),
        )
    assert capture.bodies, "the test must actually exercise HTTP outbound"
    serialized = json.dumps(capture.bodies, ensure_ascii=False)
    assert canary not in serialized, capture.bodies
    assert "[REDACTED:PASSWORD_ASSIGNMENT]" in serialized
    assert result["items"], "masking the query must not break keyword recall"


def test_url_and_cookie_query_values_are_protected_before_egress(clean_store, binding):
    canary_user = "T20_URL_USER_31"
    canary_cookie = "T20_COOKIE_COOKIE_55"
    query = (
        f"端口 https://{canary_user}:{canary_user}-pw@example.invalid/x "
        f"Cookie: session={canary_cookie}"
    )
    with _CapturingEmbeddingServer() as capture:
        execute_recall(
            clean_store, recall_request(query=query), binding=binding, embedder=capture.provider()
        )
    assert capture.bodies
    serialized = json.dumps(capture.bodies, ensure_ascii=False)
    assert canary_user not in serialized
    assert canary_cookie not in serialized


def test_withhold_kind_query_keeps_semantic_part_local(clean_store, binding):
    private_key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "T20KEYMATERIAL0123456789abcdef\n"
        "-----END RSA PRIVATE KEY-----"
    )
    with _CapturingEmbeddingServer() as capture:
        result = execute_recall(
            clean_store,
            recall_request(query=f"端口 {private_key}"),
            binding=binding,
            embedder=capture.provider(),
        )
        assert capture.bodies == [], "a withhold-kind query must not leave the process"
    assert "embedding_unavailable" in result["incomplete_reasons"]
    assert result["status"] == "degraded"
    assert result["items"], "keyword recall must still work locally"
