"""AC-38 coverage gap: complete multi-part combinations.

Codex round-2 rejected the previous AC-38 mapping because "全部多片撤回与迟到提交
组合缺少完整证据". This module exercises the real inbox/retention/rejection paths
with synthetic multi-part groups only:

* 不截断后推进完成游标 - a group only becomes complete exactly when the last
  missing part arrives; no extraction job exists before that, and the watermark
  only advances with durable writes;
* 脱敏先于分块 - KW's independent secret re-check runs on each part before it is
  stored; a part with raw secret content is durably withheld (REDACTION_REQUIRED),
  and only an explicit sanitized re-submission (the real F03 recovery shape) is
  accepted and makes the group complete;
* 不完整不宣称提炼完成 - an incomplete group has no extraction job and cannot be
  loaded as extraction input (PARENT_PENDING);
* 冲突拒绝 - a re-sent part with different content or different group metadata
  is rejected as EVENT_CONFLICT and never silently overwrites stored evidence;
* 各修订可追溯 - every revision keeps its receipt/source/history trail, and a
  multi-part retract withdraws *all* old parts and its dependent projection while
  a late extraction result cannot resurrect the claim.

Frozen expected (docs/baseline-20261030/acceptance-cases.json AC-38):
"不截断后推进完成游标；脱敏先于分块；不完整不宣称提炼完成；冲突拒绝；各修订可
追溯。"
"""

from __future__ import annotations

import pytest

from tests.extraction.test_extraction import MockExtractor, _claim, _proposal
from tests.retrieval.conftest import recall_request
from tests.state.conftest import evidence_ref, record
from tests.store.conftest import make_batch, make_event

from knowledge_weaver import contracts
from knowledge_weaver.embeddings.index import HashEmbeddingProvider
from knowledge_weaver.extraction import (
    ExtractionError,
    load_event_group,
    run_extraction_job,
    run_worker_once,
    sanitized_parts,
)
from knowledge_weaver.jobs import claim as claim_job
from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.retrieval.embeddings_index import collect_vectors
from knowledge_weaver.store import Store, fetch_source, get_current, receive_batch

BINDING = {
    "client_id": "client:test",
    "workspace_id": "owner:demo",
    "project_ids": ["project:songta"],
    "actor_ids": ["*"],
}
CANARY = "T20_R3_MULTIPART_SECRET_4471"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")


def _parts(native_id, texts, *, revision=1, operation="create", parent_revision=None):
    count = len(texts)
    return [
        make_event(
            native_id=native_id,
            text=text,
            part_index=index,
            part_count=count,
            revision=revision,
            operation=operation,
            parent_revision=parent_revision,
        )
        for index, text in enumerate(texts)
    ]


def _group_state(store, event_id, revision=1):
    with store.read() as conn:
        row = conn.execute(
            "SELECT group_state, lifecycle FROM event_revisions WHERE event_id = ? AND revision = ?",
            (event_id, revision),
        ).fetchone()
    return None if row is None else dict(row)


def _extract_jobs(store):
    with store.read() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT kind, input_key, state FROM jobs WHERE kind = 'extract' "
                "ORDER BY input_key"
            ).fetchall()
        ]


def _recall(store, *, query="端口", intent="current"):
    return execute_recall(store, recall_request(query=query, intent=intent), binding=BINDING)


def test_out_of_order_parts_only_advance_completion_when_all_present(store):
    """Clause: 不截断后推进完成游标 / 不完整不宣称提炼完成."""
    first, second = _parts("message:mp-order", ["片一：端口4200", "片零：库端口"])

    receipt = receive_batch(store, make_batch([first], batch_id="batch:mp:1"))
    assert receipt["receipts"][0]["disposition"] == "accepted"
    state = _group_state(store, first["event_id"])
    assert state["group_state"] == "incomplete", state
    assert _extract_jobs(store) == [], "no extraction may be queued for an incomplete group"
    recall = _recall(store)
    assert recall["items"] == [], recall

    receipt = receive_batch(store, make_batch([second], batch_id="batch:mp:2"))
    assert receipt["receipts"][0]["disposition"] == "accepted"
    state = _group_state(store, first["event_id"])
    assert state["group_state"] == "complete", state

    jobs = _extract_jobs(store)
    assert [(job["input_key"], job["state"]) for job in jobs] == [
        (f"{first['event_id']}@1", "pending")
    ], jobs
    # Completion happened by the real group transition: a later duplicate part
    # never queues a second job (stable completion cursor).
    duplicate = receive_batch(store, make_batch([second], batch_id="batch:mp:3"))
    assert duplicate["receipts"][0]["disposition"] == "duplicate"
    assert len(_extract_jobs(store)) == 1

    part = sanitized_parts(load_event_group(store, first["event_id"], 1))[0]
    extractor = MockExtractor(
        _proposal(
            _claim(
                part,
                predicate="server.port",
                facet="desired",
                value=4200,
                display_text="端口4200",
                assertion_kind="explicit_statement",
                quote=first["body"]["text"],
            )
        )
    )
    result = run_worker_once(store, extractor)
    assert result["state"] == "succeeded" and result["claims_written"] == 1, result


def test_missing_middle_part_is_never_treated_as_extraction_input(store):
    """Clause: 不截断后推进完成游标 / 不完整不宣称提炼完成."""
    parts = _parts("message:mp-gap", ["片零：端口4100", "片一：端口4200", "片二：端口4300"])
    receive_batch(store, make_batch([parts[0]], batch_id="batch:gap:0"))
    receive_batch(store, make_batch([parts[2]], batch_id="batch:gap:2"))
    state = _group_state(store, parts[0]["event_id"])
    assert state["group_state"] == "incomplete", state
    assert _extract_jobs(store) == []
    with pytest.raises(ExtractionError) as error:
        load_event_group(store, parts[0]["event_id"], 1)
    assert error.value.code == "PARENT_PENDING", error.value

    receive_batch(store, make_batch([parts[1]], batch_id="batch:gap:1"))
    state = _group_state(store, parts[0]["event_id"])
    assert state["group_state"] == "complete", state
    assert len(_extract_jobs(store)) == 1


@pytest.mark.parametrize(
    "field", ["body", "content_revision", "event_time", "project_ids"]
)
def test_conflicting_part_is_rejected_and_never_overwrites(store, field):
    """Clause: 冲突拒绝."""
    part0, part1 = _parts("message:mp-conflict", ["片零：端口4200", "片一：备份每周"])
    receive_batch(store, make_batch([part0], batch_id="batch:conflict:0"))
    stored_before = fetch_source(store, part0["event_id"], 1, 0)
    assert stored_before["availability"] == "full"

    tampered = make_event(
        native_id="message:mp-conflict",
        text="片零：被篡改的正文" if field == "body" else "片零：端口4200",
        part_index=0,
        part_count=2,
    )
    if field == "content_revision":
        tampered["body"]["content_revision"] = "native:2"
    elif field == "event_time":
        tampered["event_time"]["value"] = "2025-01-01T00:00:00Z"
    elif field == "project_ids":
        tampered["project_ids"] = ["project:other"]
    receipt = receive_batch(store, make_batch([tampered], batch_id="batch:conflict:x"))["receipts"][0]
    assert receipt["disposition"] == "rejected", receipt
    assert receipt["error_code"] == "EVENT_CONFLICT", receipt

    stored_after = fetch_source(store, part0["event_id"], 1, 0)
    assert stored_after["text"] == stored_before["text"], stored_after
    assert _group_state(store, part0["event_id"])["group_state"] == "incomplete"
    assert _extract_jobs(store) == []


def test_secret_part_is_withheld_before_storage_and_sanitized_reread_completes_group(store):
    """Clause: 脱敏先于分块 (KW re-check) - the raw part never becomes readable
    evidence, and only the explicit sanitized re-submission is accepted."""
    secret_text = f"密码: {CANARY}"
    raw = make_event(native_id="message:mp-secret", text=secret_text, part_index=0, part_count=2)
    secret_check = lambda text: bool(contracts.secret_kinds(text))  # noqa: E731

    receipt = receive_batch(
        store, make_batch([raw], batch_id="batch:sec:0"), secret_check=secret_check
    )["receipts"][0]
    assert receipt["disposition"] == "withheld", receipt
    assert receipt["error_code"] == "REDACTION_REQUIRED", receipt
    blocked = fetch_source(store, raw["event_id"], 1, 0)
    assert blocked["availability"] == "withheld" and blocked["text"] is None, blocked
    assert CANARY.encode() not in store.db_path.read_bytes()

    # Real F03 recovery shape: the adapter re-reads the source, redacts the full
    # message and re-submits the part with the same content revision.
    clean = make_event(
        native_id="message:mp-secret",
        text="[REDACTED:PASSWORD_ASSIGNMENT]",
        part_index=0,
        part_count=2,
    )
    clean_receipt = receive_batch(
        store, make_batch([clean], batch_id="batch:sec:1"), secret_check=secret_check
    )["receipts"][0]
    assert clean_receipt["disposition"] == "accepted", clean_receipt
    upgraded = fetch_source(store, raw["event_id"], 1, 0)
    assert upgraded["availability"] == "full", upgraded
    assert upgraded["text"] == "[REDACTED:PASSWORD_ASSIGNMENT]", upgraded
    assert _group_state(store, raw["event_id"])["group_state"] == "incomplete"
    assert _extract_jobs(store) == []

    tail = make_event(native_id="message:mp-secret", text="片一：端口4200", part_index=1, part_count=2)
    receive_batch(store, make_batch([tail], batch_id="batch:sec:2"), secret_check=secret_check)
    assert _group_state(store, raw["event_id"])["group_state"] == "complete"
    assert len(_extract_jobs(store)) == 1
    assert CANARY.encode() not in store.db_path.read_bytes()


def test_multipart_retract_withdraws_all_parts_and_blocks_late_extraction(store):
    """Actions clause: 单空片 retract 须撤回所有旧片证据，推进依赖投影并阻止迟到
    结果复活；Clause: 各修订可追溯."""
    parts = _parts("message:mp-retract", ["片零：端口4200", "片一：补充说明"])
    receive_batch(store, make_batch(parts, batch_id="batch:retr:create"))
    assert _group_state(store, parts[0]["event_id"])["group_state"] == "complete"
    late_proposal = _proposal(
        _claim(
            sanitized_parts(load_event_group(store, parts[0]["event_id"], 1))[0],
            predicate="server.port",
            facet="desired",
            value=4200,
            display_text="端口4200",
            assertion_kind="explicit_statement",
            quote=parts[0]["body"]["text"],
        )
    )

    # An extraction worker already leased the job, and a dependent claim/vector
    # exists; the retract then happens before the worker commits.
    job = claim_job(store, worker_id="worker:late", kinds=("extract",))[0]
    recorded = record(
        store,
        display_text="端口4200",
        value=4200,
        evidence_event=parts[0],
    )
    collect_vectors(store, {recorded["revision_id"]}, HashEmbeddingProvider(dimension=2))

    retract = make_event(
        native_id="message:mp-retract",
        revision=2,
        operation="retract",
        parent_revision=1,
        text="",
    )
    retract["body"]["sha256"] = contracts.EMPTY_BODY_SHA256
    receipt = receive_batch(store, make_batch([retract], batch_id="batch:retr:retract"))["receipts"][0]
    assert receipt["disposition"] == "accepted", receipt

    # All old parts of the multi-part revision are withdrawn, not only part 0.
    with store.read() as conn:
        rows = conn.execute(
            "SELECT part_index, availability, body_text FROM event_parts "
            "WHERE event_id = ? AND revision = 1",
            (parts[0]["event_id"],),
        ).fetchall()
        lifecycle = conn.execute(
            "SELECT lifecycle FROM event_revisions WHERE event_id = ? AND revision = 1",
            (parts[0]["event_id"],),
        ).fetchone()
        links = conn.execute(
            "SELECT availability FROM evidence_links WHERE revision_id = ?",
            (recorded["revision_id"],),
        ).fetchall()
        vector_count = conn.execute(
            "SELECT COUNT(*) AS c FROM claim_vectors WHERE revision_id = ?",
            (recorded["revision_id"],),
        ).fetchone()["c"]
    assert {row["part_index"] for row in rows} == {0, 1}
    assert {row["availability"] for row in rows} == {"retracted"}
    assert lifecycle["lifecycle"] == "retracted", dict(lifecycle)
    assert {link["availability"] for link in links} == {"retracted"}
    assert vector_count == 0, "retracted evidence may not keep a derived vector"
    assert get_current(store, recorded["claim_id"])["state"] == "revoked"
    assert fetch_source(store, parts[0]["event_id"], 1, 0)["availability"] == "retracted"
    assert fetch_source(store, parts[0]["event_id"], 1, 1)["availability"] == "retracted"

    # The late extraction result is fenced: loading the now-retracted source
    # fails and no claim revision is written or resurrected.
    with pytest.raises(ExtractionError) as error:
        run_extraction_job(store, job, MockExtractor(late_proposal))
    assert error.value.code == "SOURCE_EXPIRED", error.value
    with store.read() as conn:
        revisions = conn.execute(
            "SELECT COUNT(*) AS c FROM claim_revisions WHERE claim_id = ?",
            (recorded["claim_id"],),
        ).fetchone()["c"]
    assert revisions == 1, "a late extraction must not append a resurrecting revision"
    assert get_current(store, recorded["claim_id"])["state"] == "revoked"

    # Revision history stays traceable: create revision 1 + retract revision 2.
    history = _recall(store, query="端口", intent="history")
    revision_ids = {item["revision_id"] for item in history["items"]}
    assert recorded["revision_id"] in revision_ids, history
    with store.read() as conn:
        receipts = conn.execute(
            "SELECT revision, part_index, disposition FROM receipts WHERE event_id = ? "
            "ORDER BY revision, part_index",
            (parts[0]["event_id"],),
        ).fetchall()
        revisions = conn.execute(
            "SELECT revision, operation, group_state FROM event_revisions WHERE event_id = ? "
            "ORDER BY revision",
            (parts[0]["event_id"],),
        ).fetchall()
    assert [(row["revision"], row["part_index"]) for row in receipts] == [(1, 0), (1, 1), (2, 0)]
    assert [(row["revision"], row["operation"]) for row in revisions] == [
        (1, "create"),
        (2, "retract"),
    ]


def test_edit_revision_is_traceable_alongside_multipart_history(store):
    """Clause: 各修订可追溯."""
    parts = _parts("message:mp-edit", ["片零：端口4100", "片一：补充说明"])
    receive_batch(store, make_batch(parts, batch_id="batch:edit:create"))
    edit = _parts(
        "message:mp-edit",
        ["片零：端口改为4200"],
        revision=2,
        operation="edit",
        parent_revision=1,
    )[0]
    edit["body"]["content_revision"] = "native:2"
    receipt = receive_batch(store, make_batch([edit], batch_id="batch:edit:edit"))["receipts"][0]
    assert receipt["disposition"] == "accepted", receipt

    with store.read() as conn:
        rows = conn.execute(
            "SELECT revision, operation, group_state, lifecycle FROM event_revisions "
            "WHERE event_id = ? ORDER BY revision",
            (parts[0]["event_id"],),
        ).fetchall()
    assert [(row["revision"], row["operation"]) for row in rows] == [
        (1, "create"),
        (2, "edit"),
    ], [dict(row) for row in rows]
    assert all(row["group_state"] == "complete" for row in rows), [dict(row) for row in rows]

    # Each revision keeps its own durable source view and receipts.
    assert fetch_source(store, parts[0]["event_id"], 1, 0)["text"] == "片零：端口4100"
    assert fetch_source(store, parts[0]["event_id"], 1, 1)["text"] == "片一：补充说明"
    assert fetch_source(store, parts[0]["event_id"], 2, 0)["text"] == "片零：端口改为4200"
    with store.read() as conn:
        receips = conn.execute(
            "SELECT revision, part_index FROM receipts WHERE event_id = ? ORDER BY revision, part_index",
            (parts[0]["event_id"],),
        ).fetchall()
    assert [(row["revision"], row["part_index"]) for row in receips] == [(1, 0), (1, 1), (2, 0)]
