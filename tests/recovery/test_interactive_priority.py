"""Interactive extraction priority: 3 user claims : 1 background claim."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from knowledge_weaver.jobs import claim, commit, enqueue, fail
from knowledge_weaver.store import Store, receive_batch
from tests.store.conftest import make_batch, make_event

NOW = "2026-10-30T01:00:00Z"
FUTURE = "2026-10-30T03:00:00Z"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")


def _receive_user_events(store, count, *, prefix="message", batch_id="batch:priority:1"):
    events = [
        make_event(
            native_id=f"{prefix}:{index}",
            role="user",
            origin_kind="user_input",
            text=f"user note {index}",
        )
        for index in range(count)
    ]
    result = receive_batch(store, make_batch(events, batch_id=batch_id))
    assert [receipt["disposition"] for receipt in result["receipts"]] == ["accepted"] * count
    return events


def _age_jobs(store, job_ids):
    """Give synthetic background jobs older FIFO timestamps than live receipts."""
    with store.transaction() as conn:
        for index, job_id in enumerate(job_ids):
            conn.execute(
                "UPDATE jobs SET created_at = ? WHERE job_id = ?",
                (f"2026-09-01T00:{index // 60:02d}:{index % 60:02d}Z", job_id),
            )


def test_real_event_receipt_creates_claimable_user_extract(store):
    event = make_event(native_id="message:real-user", origin_kind="user_input")
    result = receive_batch(store, make_batch([event]))
    assert result["receipts"][0]["disposition"] == "accepted"

    claimed = claim(store, worker_id="worker:1", kinds=("extract",), now=NOW)
    assert len(claimed) == 1
    job = claimed[0]
    assert job["kind"] == "extract"
    assert job["input_key"] == f"{event['event_id']}@1"
    assert job["input_version"] == event["event_revision"]
    assert job["state"] == "running"


def test_new_user_job_precedes_sixteen_older_background_jobs(store):
    background = [
        enqueue(store, "extract", f"tool:call:{index}@1") for index in range(16)
    ]
    _age_jobs(store, [job["job_id"] for job in background])
    event = make_event(native_id="message:new-user", origin_kind="user_input")
    receive_batch(store, make_batch([event]))

    claimed = claim(store, worker_id="worker:1", kinds=("extract",), now=NOW)
    assert len(claimed) == 1
    assert claimed[0]["input_key"] == f"{event['event_id']}@1"


def test_sustained_user_stream_does_not_starve_background(store):
    background = [
        enqueue(store, "extract", f"tool:call:{index}@1") for index in range(2)
    ]
    _age_jobs(store, [job["job_id"] for job in background])
    _receive_user_events(store, 8, prefix="message:sustained", batch_id="batch:priority:2")
    background_keys = {job["input_key"] for job in background}

    claimed = claim(
        store, worker_id="worker:1", kinds=("extract",), limit=12, now=NOW
    )
    assert len(claimed) == 10
    background_positions = [
        index
        for index, job in enumerate(claimed)
        if job["input_key"] in background_keys
    ]
    # Exactly three user claims between each background claim: never starved.
    assert background_positions == [3, 7]
    assert [
        job["job_id"] for job in claimed if job["input_key"] in background_keys
    ] == [job["job_id"] for job in background]
    assert sum(
        1 for job in claimed if job["input_key"] not in background_keys
    ) == 8


def test_background_arriving_after_three_user_claims_is_served_next(store):
    events = _receive_user_events(
        store, 4, prefix="message:persistent", batch_id="batch:priority:6"
    )
    user_keys = {f"{event['event_id']}@1" for event in events}

    first_three = []
    for worker in ("worker:1", "worker:2", "worker:3"):
        first_three.extend(
            claim(store, worker_id=worker, kinds=("extract",), now=NOW)
        )
    assert len(first_three) == 3
    assert all(job["input_key"] in user_keys for job in first_three)

    background = enqueue(store, "extract", "tool:late@1")
    fourth = claim(store, worker_id="worker:4", kinds=("extract",), now=NOW)
    assert [job["job_id"] for job in fourth] == [background["job_id"]]


def test_future_held_jobs_are_excluded_before_priority(store):
    held_background = enqueue(store, "extract", "tool:held@1", run_after=FUTURE)
    events = _receive_user_events(
        store, 1, prefix="message:held", batch_id="batch:priority:3"
    )
    user_key = f"{events[0]['event_id']}@1"
    with store.transaction() as conn:
        conn.execute(
            "UPDATE jobs SET run_after = ? WHERE input_key = ?", (FUTURE, user_key)
        )
        user_job = conn.execute(
            "SELECT * FROM jobs WHERE input_key = ?", (user_key,)
        ).fetchone()

    assert claim(store, worker_id="worker:1", kinds=("extract",), now=NOW) == []

    ready_background = enqueue(store, "extract", "tool:ready@1")
    claimed = claim(store, worker_id="worker:1", kinds=("extract",), now=NOW)
    assert [job["job_id"] for job in claimed] == [ready_background["job_id"]]

    later = claim(
        store,
        worker_id="worker:1",
        kinds=("extract",),
        limit=2,
        now="2026-10-30T03:00:01Z",
    )
    assert {job["job_id"] for job in later} == {
        held_background["job_id"],
        user_job["job_id"],
    }


def test_input_version_idempotency_keeps_one_user_job(store):
    event = make_event(native_id="message:idempotent", origin_kind="user_input")
    batch = make_batch([event])
    first = receive_batch(store, batch)
    second = receive_batch(store, batch)
    assert first["receipts"][0]["disposition"] == "accepted"
    assert second["receipts"][0]["disposition"] == "duplicate"

    key = f"{event['event_id']}@1"
    with store.read() as conn:
        rows = conn.execute("SELECT * FROM jobs WHERE kind = 'extract'").fetchall()
    assert len(rows) == 1
    assert rows[0]["input_key"] == key
    assert rows[0]["input_version"] == 1

    again = enqueue(store, "extract", key, input_version=1)
    assert again["job_id"] == rows[0]["job_id"]

    claimed = claim(
        store, worker_id="worker:1", kinds=("extract",), limit=2, now=NOW
    )
    assert [job["job_id"] for job in claimed] == [rows[0]["job_id"]]


def test_succeeded_and_failed_jobs_are_never_replayed(store):
    events = _receive_user_events(
        store, 1, prefix="message:terminal", batch_id="batch:priority:4"
    )
    user_key = f"{events[0]['event_id']}@1"
    background = enqueue(store, "extract", "tool:terminal@1")

    user_claim = claim(store, worker_id="worker:1", kinds=("extract",), now=NOW)
    assert user_claim[0]["input_key"] == user_key
    commit(
        store,
        user_claim[0]["job_id"],
        user_claim[0]["lease_token"],
        expected_input_version=1,
        now=NOW,
    )

    background_claim = claim(
        store, worker_id="worker:2", kinds=("extract",), now=NOW
    )
    assert background_claim[0]["job_id"] == background["job_id"]
    fail(
        store,
        background_claim[0]["job_id"],
        background_claim[0]["lease_token"],
        "UNSUPPORTED_SCHEMA",
        retryable=False,
        now=NOW,
    )

    assert claim(
        store, worker_id="worker:3", kinds=("extract",), limit=10, now=NOW
    ) == []
    with store.read() as conn:
        states = {
            row["input_key"]: row["state"]
            for row in conn.execute("SELECT input_key, state FROM jobs")
        }
    assert states[user_key] == "succeeded"
    assert states[background["input_key"]] == "failed"


def test_mixed_and_other_kinds_preserve_fifo(store):
    old_project = enqueue(store, "project", "claim:old@1")
    new_project = enqueue(store, "project", "claim:new@1")
    _age_jobs(store, [old_project["job_id"]])
    events = _receive_user_events(
        store, 1, prefix="message:mixed", batch_id="batch:priority:5"
    )
    user_key = f"{events[0]['event_id']}@1"

    mixed = claim(
        store,
        worker_id="worker:1",
        kinds=("extract", "project"),
        now=NOW,
    )
    assert [job["job_id"] for job in mixed] == [old_project["job_id"]]

    extract_only = claim(
        store, worker_id="worker:2", kinds=("extract",), now=NOW
    )
    assert [job["input_key"] for job in extract_only] == [user_key]

    project_only = claim(
        store, worker_id="worker:3", kinds=("project",), now=NOW
    )
    assert [job["job_id"] for job in project_only] == [new_project["job_id"]]


def test_default_kind_filter_preserves_fifo(store):
    background = enqueue(store, "extract", "tool:default@1")
    _age_jobs(store, [background["job_id"]])
    _receive_user_events(
        store, 1, prefix="message:default", batch_id="batch:priority:7"
    )

    claimed = claim(store, worker_id="worker:1", now=NOW)
    assert [job["job_id"] for job in claimed] == [background["job_id"]]


def test_two_concurrent_workers_cannot_double_claim(store):
    second_store = Store(store.db_path)
    event = make_event(native_id="message:concurrent", origin_kind="user_input")
    receive_batch(store, make_batch([event]))
    barrier = threading.Barrier(2)

    def worker(candidate_store, worker_id):
        barrier.wait(timeout=10)
        return claim(
            candidate_store, worker_id=worker_id, kinds=("extract",), now=NOW
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(worker, store, "worker:a"),
            pool.submit(worker, second_store, "worker:b"),
        ]
        results = [future.result(timeout=10) for future in futures]

    claimed = [job for result in results for job in result]
    assert len(claimed) == 1
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_attempts").fetchone()[0] == 1
        row = conn.execute("SELECT state, lease_owner FROM jobs").fetchone()
    assert row["state"] == "running"
    assert row["lease_owner"] == claimed[0]["lease_owner"]


def test_embedding_claims_between_users_do_not_reset_fairness(store):
    events=_receive_user_events(store,4,prefix="interleaved")
    background=enqueue(store,"extract","tool:waiting@1")
    for i in range(3):
        job=claim(store,worker_id="user",kinds=("extract",),now=NOW)[0]
        assert job["input_key"] in {e["event_id"]+"@1" for e in events}
        enqueue(store,"embed",f"revision:{i}")
        assert claim(store,worker_id="embed",kinds=("embed",),now=NOW)
    assert claim(store,worker_id="fair",kinds=("extract",),now=NOW)[0]["job_id"]==background["job_id"]
