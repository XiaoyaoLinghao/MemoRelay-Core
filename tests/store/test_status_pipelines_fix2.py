"""AC-52 coverage gap: every pipeline operation records its own failure, pending
work and last success; idle time or another operation's success never clears an
unrecovered failure; diagnostics survive a new session.

Codex round-2: "AC52未覆盖全部工序/新会话诊断". The previous nodes covered KW
extract/embed only. This module checks the whole status view
(capture/deliver/extract/project/embed/cleanup/export), the persistent job-ledger
semantics for the job-backed operations, and a fresh Store session reading the
same database (new-conversation diagnostics). It also completes one real failure
recovery and shows the counters returning to healthy.

Frozen expected (docs/baseline-20261030/acceptance-cases.json AC-52):
"各工序分别记录故障、pending和last_success；空闲/他工序成功不清除未恢复失败。"
"""

from __future__ import annotations

import pytest

from tests.extraction.test_extraction import MockExtractor, _claim, _proposal
from tests.state.conftest import record
from tests.store.conftest import make_batch, make_event

from knowledge_weaver.embeddings.index import HashEmbeddingProvider, configure_model
from knowledge_weaver.extraction import load_event_group, run_worker_once, sanitized_parts
from knowledge_weaver.jobs import claim, commit, enqueue, fail
from knowledge_weaver.store import Store, receive_batch, status as status_view

OPERATIONS = (
    "capture",
    "deliver",
    "extract",
    "project",
    "embed",
    "cleanup",
    "export",
)
FIELDS = (
    "state",
    "pending",
    "failed",
    "oldest_pending_at",
    "last_success_at",
    "last_error_code",
)


@pytest.fixture
def store(tmp_path):
    configure_model(None)
    return Store(tmp_path / "kw" / "knowledge.db")


def _pipelines(store):
    return {entry["operation"]: entry for entry in status_view(store)["pipelines"]}


def _event(native_id, text):
    return make_event(native_id=native_id, text=text)


def _fail_job(store, *, kind, input_key, error_code="MODEL_UNAVAILABLE", retryable=False, worker="worker:test"):
    enqueue(store, kind, input_key)
    job = claim(store, worker_id=worker, kinds=(kind,))[0]
    outcome = fail(store, job["job_id"], job["lease_token"], error_code, retryable=retryable)
    return job, outcome


def test_every_operation_reports_its_own_state_pending_and_last_success(store):
    """Clause: 各工序分别记录故障、pending和last_success."""
    pipelines = _pipelines(store)
    assert tuple(pipelines) == OPERATIONS, tuple(pipelines)
    for operation, entry in pipelines.items():
        for field in FIELDS:
            assert field in entry, (operation, field, entry)
        # Operation entries are independent records, not one shared dict.
        assert entry is not pipelines["capture"] or operation == "capture"
        assert entry["pending"] == 0 and entry["failed"] == 0, entry
        assert entry["state"] == "idle", entry
        assert entry["last_success_at"] is None and entry["last_error_code"] is None, entry

    # capture/deliver are receipt-backed and record last_success on ingest.
    receive_batch(store, make_batch([_event("message:ac52:1", "松塔项目端口4200")]))
    after = _pipelines(store)
    assert after["capture"]["last_success_at"] is not None, after["capture"]
    assert after["capture"]["pending"] == 0, after["capture"]
    # deliver shows the durable receipts whose semantic processing is pending.
    assert after["deliver"]["last_success_at"] is not None, after["deliver"]
    assert after["deliver"]["pending"] >= 1, after["deliver"]
    assert after["deliver"]["state"] == "running", after["deliver"]
    for operation in ("capture", "deliver"):
        assert after[operation]["failed"] == 0, after[operation]

    # extract and embed are job-backed: queued work is visibly pending, not lost.
    configure_model("test:ac52", model_revision="1", dimension=2)
    recorded = record(store, display_text="端口4200", value=4200)
    jobs = _pipelines(store)
    assert jobs["extract"]["pending"] >= 1 and jobs["extract"]["state"] == "idle", jobs["extract"]
    assert jobs["extract"]["oldest_pending_at"] is not None, jobs["extract"]
    assert jobs["embed"]["pending"] == 1 and jobs["embed"]["state"] == "idle", jobs["embed"]
    assert jobs["embed"]["last_success_at"] is None, jobs["embed"]
    assert recorded["revision_id"]


def test_unrecovered_failure_survives_idle_and_other_operation_success(store):
    """Clause: 空闲/他工序成功不清除未恢复失败."""
    event = _event("message:ac52:fail", "松塔项目端口4200")
    receive_batch(store, make_batch([event]))
    _fail_job(store, kind="extract", input_key=f"{event['event_id']}@1", error_code="MODEL_UNAVAILABLE")
    degraded = _pipelines(store)["extract"]
    assert degraded["failed"] == 1, degraded
    assert degraded["state"] == "degraded", degraded
    assert degraded["last_error_code"] == "MODEL_UNAVAILABLE", degraded
    failed_before = degraded["failed"]
    error_before = degraded["last_error_code"]

    # Another operation (capture + a successful embed) and plain idle status
    # reads must not clear the unrecovered extraction failure.
    configure_model("test:ac52", model_revision="1", dimension=2)
    record(store, display_text="备用端口4300", value=4300, native_id="message:ac52:embed")
    from knowledge_weaver.embeddings.index import run_embed_worker_once

    assert run_embed_worker_once(store, HashEmbeddingProvider(dimension=2, model_id="test:ac52"))[
        "state"
    ] == "succeeded"
    receive_batch(store, make_batch([_event("message:ac52:capture", "无新消息也要可诊断")]))
    for _ in range(2):  # idle status reads
        status_view(store)
    after = _pipelines(store)["extract"]
    assert after["failed"] == failed_before, after
    assert after["state"] == "degraded", after
    assert after["last_error_code"] == error_before, after
    assert after["last_success_at"] is None, after

    # New session / new conversation: a fresh Store on the same database sees
    # the same diagnostic chain (no success resets the ledger).
    restarted = Store(store.db_path)
    session = _pipelines(restarted)["extract"]
    assert session["failed"] == failed_before, session
    assert session["state"] == "degraded" and session["last_error_code"] == error_before, session


def test_one_real_failure_recovery_returns_the_pipeline_to_healthy(store):
    """Actions clause: 完成一次故障恢复."""
    event = _event("message:ac52:recover", "松塔项目端口4200")
    receive_batch(store, make_batch([event]))
    input_key = f"{event['event_id']}@1"
    enqueue(store, "extract", input_key)
    job = claim(store, worker_id="worker:retry", kinds=("extract",))[0]
    retry = fail(
        store,
        job["job_id"],
        job["lease_token"],
        "MODEL_UNAVAILABLE",
        retryable=True,
    )
    assert retry["state"] == "retry_wait", retry
    assert _pipelines(store)["extract"]["state"] == "degraded"

    # The retry eventually runs (virtual clock far enough past the backoff) and
    # succeeds through the real extraction pipeline.  The retry run_after is
    # derived from the *real* fail() clock plus the bounded backoff, so the
    # virtual clock must stay relative: a hard-coded date becomes flaky once the
    # wall clock advances past it.
    part = sanitized_parts(load_event_group(store, event["event_id"], 1))[0]
    proposal = _proposal(
        _claim(
            part,
            predicate="server.port",
            facet="desired",
            value=4200,
            display_text="端口4200",
            assertion_kind="explicit_statement",
            quote=part["body"]["text"] if isinstance(part, dict) else part.text,
        )
    )
    import datetime as _dt

    later = (
        _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=1)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    result = run_worker_once(store, MockExtractor(proposal), now=later)
    assert result["state"] == "succeeded", result
    healthy = _pipelines(store)["extract"]
    assert healthy["failed"] == 0, healthy
    assert healthy["pending"] == 0, healthy
    assert healthy["state"] == "idle", healthy
    assert healthy["last_success_at"] is not None, healthy
    assert healthy["last_error_code"] is None, healthy


@pytest.mark.parametrize("kind", ["project", "cleanup", "export"])
def test_each_job_backed_operation_has_an_observable_ledger(store, kind):
    """Clause: 各工序分别记录故障、pending和last_success (whole status view)."""
    enqueue(store, kind, f"{kind}:pending:1")
    pending = _pipelines(store)[kind]
    assert pending["pending"] == 1 and pending["state"] == "idle", pending
    assert pending["oldest_pending_at"] is not None, pending

    job = claim(store, worker_id="worker:test", kinds=(kind,))[0]
    running = _pipelines(store)[kind]
    assert running["state"] == "running", running

    fail(store, job["job_id"], job["lease_token"], "SYNTHETIC_FAILURE", retryable=False)
    failed = _pipelines(store)[kind]
    assert failed["failed"] == 1, failed
    assert failed["state"] == "degraded", failed
    assert failed["last_error_code"] == "SYNTHETIC_FAILURE", failed

    # A later success of the same operation records last_success but never clears
    # the still-unrecovered failure.
    enqueue(store, kind, f"{kind}:success:1")
    success_job = claim(store, worker_id="worker:test", kinds=(kind,))[0]
    commit(
        store,
        success_job["job_id"],
        success_job["lease_token"],
        expected_input_version=int(success_job["input_version"]),
    )
    mixed = _pipelines(store)[kind]
    assert mixed["failed"] == 1, mixed
    assert mixed["state"] == "degraded", mixed
    assert mixed["last_success_at"] is not None, mixed
    assert mixed["last_error_code"] == "SYNTHETIC_FAILURE", mixed
