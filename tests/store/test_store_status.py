"""T02: health/status schemas and restore state visibility."""
from __future__ import annotations

from knowledge_weaver import contracts
from knowledge_weaver.store import health, receive_batch, status

from .conftest import assert_schema, make_batch, make_event


def test_health_matches_contract_and_is_ready(store):
    view = health(store)
    assert_schema("Health", view)
    assert view["status"] == "ready"
    assert view["database_integrity"] == "ok"
    assert view["supported_contracts"] == ["1.0"]
    assert view["watermark"] == 0


def test_health_reports_restore_in_progress(store):
    store.set_restore_in_progress(True)
    assert health(store)["status"] == "restoring"
    store.set_restore_in_progress(False)
    assert health(store)["status"] == "ready"


def test_status_matches_contract_and_counts_pending_extraction(store):
    assert_schema("Status", status(store))
    view = status(store)
    extract = [p for p in view["pipelines"] if p["operation"] == "extract"][0]
    assert extract["pending"] == 0
    receive_batch(store, make_batch([make_event()]))
    view = status(store)
    extract = [p for p in view["pipelines"] if p["operation"] == "extract"][0]
    # A complete accepted event enqueues exactly one persistent extract job.
    assert extract["pending"] == 1
    assert extract["state"] == "idle"


# ---------------------------------------------------------------------------
# F16: per-operation pipeline status comes from the durable job ledger
# ---------------------------------------------------------------------------


def _pipeline(view, operation):
    return [item for item in view["pipelines"] if item["operation"] == operation][0]


def test_failed_embedding_is_visible_in_status(store):
    from knowledge_weaver.jobs import claim, enqueue, fail
    from knowledge_weaver.store.status import status

    enqueue(store, "embed", "synthetic:revision")
    job = claim(store, worker_id="worker:test", kinds=("embed",))[0]
    fail(store, job["job_id"], job["lease_token"], "EMBEDDING_MISMATCH", retryable=False)
    pipeline = _pipeline(status(store), "embed")
    assert pipeline["failed"] == 1
    assert pipeline["state"] == "degraded"
    assert pipeline["last_error_code"] == "EMBEDDING_MISMATCH"
    assert pipeline["pending"] == 0
    assert pipeline["last_success_at"] is None


def test_retrying_job_is_degraded_and_counts_pending(store):
    from knowledge_weaver.jobs import claim, enqueue, fail
    from knowledge_weaver.store.status import status

    enqueue(store, "embed", "synthetic:revision")
    job = claim(store, worker_id="worker:test", kinds=("embed",))[0]
    fail(store, job["job_id"], job["lease_token"], "EMBEDDING_MISMATCH", retryable=True)
    pipeline = _pipeline(status(store), "embed")
    assert pipeline["pending"] == 1
    assert pipeline["state"] == "degraded"
    assert pipeline["last_error_code"] == "EMBEDDING_MISMATCH"


def test_extract_failure_is_not_cleared_by_embed_success(store):
    from knowledge_weaver.embeddings.index import HashEmbeddingProvider, run_embed_worker_once
    from knowledge_weaver.jobs import claim, enqueue, fail
    from knowledge_weaver.store.status import status

    from ..state.conftest import record

    enqueue(store, "extract", "event:synthetic@1", input_version=1)
    extract_job = claim(store, worker_id="worker:test", kinds=("extract",))[0]
    fail(store, extract_job["job_id"], extract_job["lease_token"], "MODEL_UNAVAILABLE", retryable=False)

    # A different operation succeeds afterwards; the extract failure stays.
    recorded = record(store, display_text="端口4200", value=4200)
    enqueue(store, "embed", recorded["revision_id"])
    result = run_embed_worker_once(store, HashEmbeddingProvider(dimension=2))
    assert result["state"] == "succeeded", result

    view = status(store)
    extract = _pipeline(view, "extract")
    assert extract["failed"] == 1
    assert extract["state"] == "degraded"
    assert extract["last_error_code"] == "MODEL_UNAVAILABLE"


def test_status_matches_contract_schema(store):
    import pytest as _pytest

    from knowledge_weaver import contracts
    from knowledge_weaver.store.status import status

    view = status(store)
    assert contracts.validate_instance("Status", view) == []
    for pipeline in view["pipelines"]:
        assert set(pipeline) == {
            "operation",
            "state",
            "pending",
            "failed",
            "oldest_pending_at",
            "last_success_at",
            "last_error_code",
        }
