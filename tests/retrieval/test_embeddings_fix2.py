"""F11 round-3 regression: recall inline embeddings share the late-commit guard.

Migrated from the independent round-2 counterexamples: during a real
``execute_recall`` an inline ``collect_vectors`` embedding call returns after
the revision was redacted, retracted or expired, and the late vector must not be
written. Only synthetic data under pytest ``tmp_path`` is used.
"""

from __future__ import annotations

from knowledge_weaver.embeddings.index import HashEmbeddingProvider
from knowledge_weaver.lifecycle.retention import sweep
from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.retrieval.embeddings_index import collect_vectors
from knowledge_weaver.store import receive_batch

from ..state.conftest import record
from ..store.conftest import make_batch, make_event
from .conftest import recall_request


def _vector_count(store, revision_id: str) -> int:
    with store.read() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM claim_vectors WHERE revision_id = ?", (revision_id,)
        ).fetchone()["c"]


def _revision_status(store, revision_id: str) -> str:
    with store.read() as conn:
        return conn.execute(
            "SELECT status FROM claim_revisions WHERE revision_id = ?", (revision_id,)
        ).fetchone()["status"]


def _redact_event(event: dict) -> dict:
    return make_event(
        native_id=event["source"]["native_event_id"],
        text="端口 [REDACTED]",
        revision=2,
        operation="redact",
        parent_revision=1,
    )


def _retract_event(event: dict) -> dict:
    return make_event(
        native_id=event["source"]["native_event_id"],
        text="",
        revision=2,
        operation="retract",
        parent_revision=1,
    )


class _InvalidateDuringEmbed(HashEmbeddingProvider):
    """Runs a real concurrent invalidation once, while recall is embedding."""

    def __init__(self, action, *, dimension: int = 2) -> None:
        super().__init__(dimension=dimension)
        self._action = action
        self.raced = False

    def embed(self, texts):
        if not self.raced:
            self.raced = True
            self._action()
        return super().embed(texts)


def test_inline_redact_race_does_not_commit_late_vector(store, binding):
    """F11: redact during inline embedding leaves no late vector for the scrubbed body."""
    event = make_event(native_id="message:fix2-inline-redact", text="端口原配置")
    receive_batch(store, make_batch([event]))
    claim = record(store, display_text=event["body"]["text"], value=4200, evidence_event=event)
    valid_event = make_event(native_id="message:fix2-inline-redact-valid", text="备份周期为每周")
    receive_batch(store, make_batch([valid_event]))
    valid = record(
        store,
        display_text="备份周期为每周",
        value="weekly",
        predicate="backup_interval",
        evidence_event=valid_event,
    )
    embedder = _InvalidateDuringEmbed(
        lambda: receive_batch(store, make_batch([_redact_event(event)], batch_id="batch:fix2:redact"))
    )
    execute_recall(store, recall_request(query="端口"), binding=binding, embedder=embedder)
    assert _vector_count(store, claim["revision_id"]) == 0
    # The valid sibling candidate in the same inline batch is still cached.
    assert _vector_count(store, valid["revision_id"]) == 1


def test_inline_retract_race_does_not_commit_late_vector(store, binding):
    """F11 [retract]: a retracted source support may not gain a vector afterwards."""
    event = make_event(native_id="message:fix2-inline-retract", text="端口原配置")
    receive_batch(store, make_batch([event]))
    claim = record(store, display_text=event["body"]["text"], value=4200, evidence_event=event)
    embedder = _InvalidateDuringEmbed(
        lambda: receive_batch(store, make_batch([_retract_event(event)], batch_id="batch:fix2:retract"))
    )
    execute_recall(store, recall_request(query="端口"), binding=binding, embedder=embedder)
    assert _vector_count(store, claim["revision_id"]) == 0


def test_inline_expired_superseded_history_does_not_commit_late_vector(store, binding):
    """F11 [expired]: a real sweep expires replaced history while embedding runs."""
    old = record(
        store,
        display_text="旧端口4100",
        value=4100,
        event_time={"value": "2024-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    new = record(
        store,
        display_text="新端口4200",
        value=4200,
        supersedes=[old["revision_id"]],
        event_time={"value": "2025-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )

    def expire_history():
        result = sweep(store, now="2026-10-30T00:00:00Z")
        assert result["claim_revisions"] == 1, result

    embedder = _InvalidateDuringEmbed(expire_history)
    execute_recall(
        store, recall_request(intent="history", query="端口"), binding=binding, embedder=embedder
    )
    assert _revision_status(store, old["revision_id"]) == "expired"
    assert _vector_count(store, old["revision_id"]) == 0
    # The still-current replacement revision keeps its vector.
    assert _vector_count(store, new["revision_id"]) == 1


def test_valid_inline_embedding_is_still_cached_and_recallable(store, binding):
    """F10/F11: the commit guard must not block a normal valid inline vector."""
    claim = record(store, display_text="松塔项目服务器端口为4200", value=4200)
    result = execute_recall(
        store, recall_request(query="端口4200"), binding=binding, embedder=HashEmbeddingProvider(2)
    )
    assert _vector_count(store, claim["revision_id"]) == 1
    assert "embedding_unavailable" not in result["incomplete_reasons"], result
    assert result["items"] and result["items"][0]["claim_id"] == claim["claim_id"], result


def test_collect_vectors_skips_already_invalid_revision(store):
    """F11: direct inline collection may not resurrect an expired revision either."""
    old = record(
        store,
        display_text="旧端口4100",
        value=4100,
        event_time={"value": "2024-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    record(
        store,
        display_text="新端口4200",
        value=4200,
        supersedes=[old["revision_id"]],
        event_time={"value": "2025-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    sweep(store, now="2026-10-30T00:00:00Z")
    assert _revision_status(store, old["revision_id"]) == "expired"
    vectors = collect_vectors(store, {old["revision_id"]}, HashEmbeddingProvider(2))
    assert vectors == {}
    assert _vector_count(store, old["revision_id"]) == 0


def test_collect_vectors_rejects_late_vector_when_text_changed(store):
    """F11: the shared guard re-checks the body hash inside the write transaction."""
    claim = record(store, display_text="端口4100", value=4100)

    def change_text():
        with store.transaction() as conn:
            conn.execute(
                "UPDATE claim_revisions SET display_text = ? WHERE revision_id = ?",
                ("端口改为4200", claim["revision_id"]),
            )

    embedder = _InvalidateDuringEmbed(change_text)
    vectors = collect_vectors(store, {claim["revision_id"]}, embedder)
    assert vectors == {}
    assert _vector_count(store, claim["revision_id"]) == 0
