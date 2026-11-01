"""T06: persistent jobs, leases, fencing, retries and crash recovery."""
from __future__ import annotations

import pytest

from knowledge_weaver.jobs import (
    JobError,
    claim,
    commit,
    enqueue,
    fail,
    heartbeat,
    recover,
    status_counts,
)
from knowledge_weaver.store import Store

NOW = "2026-10-30T01:00:00Z"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")


def _mutation(conn, marker="m1"):
    conn.execute(
        "INSERT INTO watermark_log (kind, ref, created_at) VALUES ('test.mutation', ?, ?)",
        (marker, NOW),
    )


def _mutations(store):
    with store.read() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM watermark_log WHERE kind = 'test.mutation'"
        ).fetchone()["c"]


def test_enqueue_is_idempotent(store):
    first = enqueue(store, "extract", "evt:a@1", input_version=1)
    second = enqueue(store, "extract", "evt:a@1", input_version=1)
    assert first["job_id"] == second["job_id"]
    third = enqueue(store, "extract", "evt:a@1", input_version=2)
    assert third["job_id"] != first["job_id"]
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


def test_enqueue_rejects_unknown_kind(store):
    with pytest.raises(JobError):
        enqueue(store, "summarize-everything", "x")


def test_claim_journals_attempt_and_excludes_second_worker(store):
    enqueue(store, "extract", "evt:a@1")
    claimed = claim(store, worker_id="worker:1", lease_seconds=60, now=NOW)
    assert len(claimed) == 1
    job = claimed[0]
    assert job["state"] == "running"
    assert job["lease_token"].startswith("lease:")
    other = claim(store, worker_id="worker:2", now=NOW)
    assert other == []
    with store.read() as conn:
        attempts = conn.execute("SELECT * FROM job_attempts").fetchall()
    assert len(attempts) == 1
    assert attempts[0]["outcome"] is None


def test_heartbeat_requires_lease_token(store):
    enqueue(store, "extract", "evt:a@1")
    job = claim(store, worker_id="worker:1", now=NOW)[0]
    assert heartbeat(store, job["job_id"], job["lease_token"], now=NOW) is True
    assert heartbeat(store, job["job_id"], "lease:wrong", now=NOW) is False


def test_expired_lease_reclaimed_and_stale_worker_cannot_commit(store):
    enqueue(store, "extract", "evt:a@1")
    old = claim(store, worker_id="worker:old", lease_seconds=60, now=NOW)[0]
    recovered = recover(store, now="2026-10-30T01:01:01Z")
    assert recovered["retry_wait"] == 1
    assert recovered["interrupted_attempts"] == 1
    new = claim(store, worker_id="worker:new", lease_seconds=60, now="2026-10-30T01:01:12Z")[0]
    assert new["lease_token"] != old["lease_token"]
    # The old worker's late result is fenced out.
    with pytest.raises(JobError) as excinfo:
        commit(
            store,
            old["job_id"],
            old["lease_token"],
            expected_input_version=1,
            mutate=lambda conn: _mutation(conn, "stale"),
            now="2026-10-30T01:01:03Z",
        )
    assert excinfo.value.code == "STALE_CURSOR"
    assert _mutations(store) == 0
    # The new holder commits successfully.
    commit(
        store,
        new["job_id"],
        new["lease_token"],
        expected_input_version=1,
        mutate=lambda conn: _mutation(conn, "fresh"),
        now="2026-10-30T01:01:04Z",
    )
    assert _mutations(store) == 1
    with store.read() as conn:
        assert conn.execute("SELECT state FROM jobs").fetchone()["state"] == "succeeded"


def test_generation_fence_rejects_stale_result(store):
    enqueue(store, "project", "claim:x@1")
    job = claim(store, worker_id="worker:1", now=NOW)[0]
    with pytest.raises(JobError):
        commit(
            store,
            job["job_id"],
            job["lease_token"],
            expected_input_version=1,
            mutate=lambda conn: _mutation(conn),
            fence_check=lambda conn: (False, "claim generation changed"),
            now=NOW,
        )
    assert _mutations(store) == 0
    with store.read() as conn:
        row = conn.execute("SELECT * FROM jobs").fetchone()
        attempt = conn.execute("SELECT * FROM job_attempts").fetchone()
    assert row["state"] == "failed"
    assert row["last_error_code"] == "STALE_CURSOR"
    assert attempt["outcome"] == "stale"


def test_input_version_change_is_rejected(store):
    enqueue(store, "extract", "evt:b@1", input_version=1)
    job = claim(store, worker_id="worker:1", now=NOW)[0]
    with pytest.raises(JobError):
        commit(store, job["job_id"], job["lease_token"], expected_input_version=2, mutate=lambda conn: _mutation(conn), now=NOW)
    assert _mutations(store) == 0


def test_commit_rolls_back_mutation_on_exception(store):
    enqueue(store, "extract", "evt:c@1")
    job = claim(store, worker_id="worker:1", now=NOW)[0]

    def broken(conn):
        _mutation(conn)
        raise RuntimeError("projection bug")

    with pytest.raises(RuntimeError):
        commit(store, job["job_id"], job["lease_token"], expected_input_version=1, mutate=broken, now=NOW)
    assert _mutations(store) == 0
    with store.read() as conn:
        row = conn.execute("SELECT * FROM jobs").fetchone()
        attempt = conn.execute("SELECT * FROM job_attempts").fetchone()
    assert row["state"] == "running"  # caller decides fail/retry
    assert row["lease_token"] == job["lease_token"]
    assert attempt["outcome"] is None


def test_retry_backoff_and_terminal_failure(store):
    enqueue(store, "extract", "evt:d@1")
    job = claim(store, worker_id="worker:1", now=NOW)[0]
    result = fail(store, job["job_id"], job["lease_token"], "MODEL_UNAVAILABLE", now=NOW)
    assert result["state"] == "retry_wait"
    # Not ready before the backoff elapses.
    assert claim(store, worker_id="worker:2", now=NOW) == []
    job2 = claim(store, worker_id="worker:2", now="2026-10-30T01:00:11Z")[0]
    fail(store, job2["job_id"], job2["lease_token"], "MODEL_UNAVAILABLE", now="2026-10-30T01:00:11Z")
    job3 = claim(store, worker_id="worker:3", now="2026-10-30T01:02:00Z")[0]
    result = fail(store, job3["job_id"], job3["lease_token"], "MODEL_UNAVAILABLE", now="2026-10-30T01:02:00Z")
    assert result["state"] == "failed"
    with store.read() as conn:
        assert conn.execute("SELECT state FROM jobs").fetchone()["state"] == "failed"


def test_permanent_error_is_terminal(store):
    enqueue(store, "extract", "evt:e@1")
    job = claim(store, worker_id="worker:1", now=NOW)[0]
    result = fail(store, job["job_id"], job["lease_token"], "UNSUPPORTED_SCHEMA", retryable=False, now=NOW)
    assert result["state"] == "failed"


def test_fail_requires_current_lease(store):
    enqueue(store, "extract", "evt:f@1")
    job = claim(store, worker_id="worker:1", now=NOW)[0]
    with pytest.raises(JobError):
        fail(store, job["job_id"], "lease:stale", "MODEL_UNAVAILABLE", now=NOW)


def test_status_counts(store):
    enqueue(store, "extract", "evt:g@1")
    enqueue(store, "extract", "evt:g@2")
    claim(store, worker_id="worker:1", now=NOW, limit=1)
    counts = status_counts(store)
    assert counts["extract"]["pending"] == 1
    assert counts["extract"]["running"] == 1


def test_recover_leaves_succeeded_jobs_alone(store):
    enqueue(store, "extract", "evt:h@1")
    job = claim(store, worker_id="worker:1", now=NOW)[0]
    commit(store, job["job_id"], job["lease_token"], expected_input_version=1, now=NOW)
    assert recover(store, now="2026-10-31T00:00:00Z") == {"retry_wait": 0, "failed": 0, "interrupted_attempts": 0}
