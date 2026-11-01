"""AC-17 coverage gaps: the real 179/180/181 completed-task boundary, the
long-term key conclusion K that must survive task-detail expiry, and the
summary marking that its source detail is expired.

Codex round-2 rejected the previous AC-17 node because it "只计算180天日期";
this module runs the **real** retention sweep path and explicitly asserts that
the expiry actually happened (event part purged, lifecycle tombstoned, source
availability expired), not merely that a date was computed.

Also migrated here: the Codex-corrected expiry semantics from
``artifacts/T20-round2/run-20261030-codex/test_followups_v2.py`` -
``test_retention_exact_boundaries_keep_conclusion_not_detail``: when only the
source detail expires, the valid conclusion and its vector stay, while the
detail body leaves FTS/recall.

Frozen expected (docs/baseline-20261030/acceptance-cases.json AC-17):
"t_done+179 保留 D180；t_done+180 起可清理；t_done+181 D180 不可检索；有效关键
结论 K 保留，不因任务细节到期而丢失；摘要标明源详情已到期。全部满足为通过。"
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from tests.retrieval.conftest import recall_request
from tests.state.conftest import evidence_ref, record
from tests.store.conftest import make_batch, make_event

from knowledge_weaver.embeddings.index import HashEmbeddingProvider
from knowledge_weaver.lifecycle import EXPIRED_BY_DAYS, is_tombstoned, lifecycle_status, sweep
from knowledge_weaver.lifecycle.retention import register_lifecycle
from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.retrieval.embeddings_index import collect_vectors
from knowledge_weaver.store import Store, fetch_source, receive_batch

BINDING = {
    "client_id": "client:test",
    "workspace_id": "owner:demo",
    "project_ids": ["project:songta"],
    "actor_ids": ["*"],
}
T_DONE = "2026-01-01T00:00:00Z"
DETAIL = "T20_R3_D180_PROCESS_DETAIL_55123"
DETAIL2 = "T20_R3_D30_PROCESS_DETAIL_55124"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")


def _day(anchor: str, days: int) -> str:
    parsed = dt.datetime.fromisoformat(anchor.replace("Z", "+00:00"))
    return (parsed + dt.timedelta(days=days)).astimezone(dt.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _recall(store, query, *, intent="current"):
    return execute_recall(store, recall_request(query=query, intent=intent), binding=BINDING)


def _item_for(recall_result, claim_id):
    hits = [item for item in recall_result["items"] if item["claim_id"] == claim_id]
    assert len(hits) == 1, recall_result
    return hits[0]


def _seed_completed_task_fixture(store):
    """D180 detail + long-term K + derived summary (created t_done+100)."""
    detail_event = make_event(
        native_id="message:ac17-detail",
        text=f"{DETAIL}；任务流程细节；当前端口4200",
        event_time={"value": T_DONE, "precision": "instant", "timezone": "UTC"},
    )
    receive_batch(store, make_batch([detail_event]))
    key_conclusion = record(
        store,
        display_text="当前端口4200",
        value=4200,
        predicate="server.port",
        evidence_event=detail_event,
        event_time={"value": T_DONE, "precision": "instant", "timezone": "UTC"},
    )
    summary_created_at = _day(T_DONE, 100)
    summary = record(
        store,
        display_text="任务摘要：功能验收通过，流程细节见来源",
        value={"summary": "功能验收通过", "source_detail": "D180"},
        predicate="task.summary",
        evidence_event=detail_event,
        event_time={"value": summary_created_at, "precision": "instant", "timezone": "UTC"},
    )
    provider = HashEmbeddingProvider(dimension=2)
    collect_vectors(store, {key_conclusion["revision_id"]}, provider)
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{detail_event['event_id']}#1@0",
        class_="completed_task",
        business_anchor=T_DONE,
    )
    return {
        "event": detail_event,
        "key": key_conclusion,
        "summary": summary,
        "source_subject": f"{detail_event['event_id']}#1@0",
    }


def _assert_detail_retained_and_readable(store, fixture):
    source = fetch_source(store, fixture["event"]["event_id"], 1, 0)
    assert source["availability"] == "full", source
    assert DETAIL in (source["text"] or ""), source
    with store.read() as conn:
        lifecycle = conn.execute(
            "SELECT * FROM lifecycle_records WHERE subject_kind = 'event_part' AND subject_id = ?",
            (fixture["source_subject"],),
        ).fetchone()
    assert lifecycle["state"] == "active", dict(lifecycle)
    assert is_tombstoned(store, "event_part", fixture["source_subject"]) is False

    current = _recall(store, "端口4200")
    key_item = _item_for(current, fixture["key"]["claim_id"])
    assert key_item["source_detail_expired"] is False, key_item
    assert all(link["availability"] == "full" for link in key_item["sources"]), key_item
    summary_item = _item_for(_recall(store, "任务摘要"), fixture["summary"]["claim_id"])
    assert summary_item["source_detail_expired"] is False, summary_item


def _assert_detail_expired_and_purged(store, fixture):
    # The expiry really happened in the durable store, not just in a date calc.
    source = fetch_source(store, fixture["event"]["event_id"], 1, 0)
    assert source["availability"] == "expired", source
    assert source["text"] is None, source
    with store.read() as conn:
        part = conn.execute(
            "SELECT body_text, availability FROM event_parts WHERE event_id = ? AND revision = 1 "
            "AND part_index = 0",
            (fixture["event"]["event_id"],),
        ).fetchone()
        lifecycle = conn.execute(
            "SELECT state FROM lifecycle_records WHERE subject_kind = 'event_part' AND subject_id = ?",
            (fixture["source_subject"],),
        ).fetchone()
        tombstone = conn.execute(
            "SELECT generation FROM tombstones WHERE subject_kind = 'event_part' AND subject_id = ?",
            (fixture["source_subject"],),
        ).fetchone()
    assert part["availability"] == "expired" and part["body_text"] is None, dict(part)
    assert lifecycle["state"] == "tombstoned", dict(lifecycle)
    assert tombstone is not None, "retention must leave a minimal tombstone"
    assert is_tombstoned(store, "event_part", fixture["source_subject"]) is True


def _assert_key_conclusion_and_summary_retained(store, fixture):
    # K is still a queryable, retained knowledge item with its own source marked
    # expired; the vector of the still-valid conclusion is kept (Codex-corrected
    # semantics: only the detail expires, the conclusion does not).
    current = _recall(store, "端口4200")
    key_item = _item_for(current, fixture["key"]["claim_id"])
    assert key_item["source_detail_expired"] is True, key_item
    assert key_item["sources"], key_item
    assert all(link["availability"] == "expired" for link in key_item["sources"]), key_item
    assert key_item["value"] == 4200, key_item

    # Summary states that its source detail is expired instead of claiming the
    # original text is still readable.
    summary_item = _item_for(_recall(store, "任务摘要"), fixture["summary"]["claim_id"])
    assert summary_item["source_detail_expired"] is True, summary_item
    assert all(link["availability"] == "expired" for link in summary_item["sources"]), summary_item

    with store.read() as conn:
        revision = conn.execute(
            "SELECT status, display_text FROM claim_revisions WHERE revision_id = ?",
            (fixture["key"]["revision_id"],),
        ).fetchone()
        vector_count = conn.execute(
            "SELECT COUNT(*) AS c FROM claim_vectors WHERE revision_id = ?",
            (fixture["key"]["revision_id"],),
        ).fetchone()["c"]
    # Retention expired the *source part*, not the valid conclusion revision.
    assert revision["status"] in ("current", "conflict", "uncertain"), dict(revision)
    assert revision["display_text"] == "当前端口4200", dict(revision)
    assert vector_count == 1, "valid conclusion vector must be retained when only its source detail expires"


def test_completed_task_179_180_181_day_boundary_keeps_conclusion(store):
    """Clauses: t_done+179 保留 D180; t_done+180 起可清理; t_done+181 D180 不可
    检索; 有效关键结论 K 保留; 摘要标明源详情已到期."""
    fixture = _seed_completed_task_fixture(store)

    # t_done + 179: still retained, nothing expired yet.
    swept_179 = sweep(store, now=_day(T_DONE, 179))
    assert swept_179["event_parts"] == 0, swept_179
    assert swept_179["expired"] == 0, swept_179
    _assert_detail_retained_and_readable(store, fixture)

    # t_done + 180: the completed-task detail may be cleaned from this instant on.
    swept_180 = sweep(store, now=_day(T_DONE, 180))
    assert swept_180["event_parts"] == 1, swept_180
    assert swept_180["tombstoned"] >= 1, swept_180
    _assert_detail_expired_and_purged(store, fixture)
    _assert_key_conclusion_and_summary_retained(store, fixture)

    # t_done + 181: D180 is not retrievable; K is still retained and its vector
    # survives; a repeated sweep is idempotent.
    swept_181 = sweep(store, now=_day(T_DONE, 181))
    assert swept_181["event_parts"] == 0, swept_181
    with store.read() as conn:
        matched = conn.execute(
            "SELECT 1 FROM claim_fts WHERE claim_fts MATCH ?", (DETAIL,)
        ).fetchone()
    assert matched is None
    detail_recall = _recall(store, DETAIL)
    assert detail_recall["status"] == "empty", detail_recall
    assert detail_recall["items"] == [], detail_recall

    current = _recall(store, "端口4200")
    _item_for(current, fixture["key"]["claim_id"])
    with store.read() as conn:
        vector_count = conn.execute(
            "SELECT COUNT(*) AS c FROM claim_vectors WHERE revision_id = ?",
            (fixture["key"]["revision_id"],),
        ).fetchone()["c"]
    assert vector_count == 1

    # The source API keeps the honest expired marker and never the old body.
    source = fetch_source(store, fixture["event"]["event_id"], 1, 0)
    assert source["availability"] == "expired" and source["text"] is None, source


@pytest.mark.parametrize(
    "days,class_",
    [(30, "temporary_detail"), (180, "completed_task")],
)
def test_migrated_expiry_retains_conclusion_not_detail(store, days, class_):
    """Migrated corrected expiry semantics (Codex followups-v2):

    source detail expires exactly at the class boundary; the valid conclusion and
    its vector stay; only the expired revision/detail leaves recall/FTS.
    """
    detail = f"R3_DETAIL_DISTINCT_{days}_7719"
    event = make_event(native_id=f"message:expiry-{days}", text=f"{detail}; 当前端口4200")
    receive_batch(store, make_batch([event]))
    conclusion = record(store, display_text="当前端口4200", value=4200, evidence_event=event)
    collect_vectors(store, {conclusion["revision_id"]}, HashEmbeddingProvider(dimension=2))
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#1@0",
        class_=class_,
        business_anchor="2026-01-01T00:00:00Z",
    )
    anchor = "2026-01-01T00:00:00Z"
    for delta in (days - 1, days, days + 1):
        sweep(store, now=_day(anchor, delta))
        source = fetch_source(store, event["event_id"], 1, 0)
        if delta < days:
            assert source["availability"] == "full", (delta, source)
            continue
        assert source["availability"] == "expired" and source["text"] is None, (delta, source)
        query = _recall(store, "当前端口4200")
        assert query["items"], query
        assert detail not in json.dumps(query, ensure_ascii=False)
        assert all(
            item["source_detail_expired"] and all(s["availability"] == "expired" for s in item["sources"])
            for item in query["items"]
        )
        with store.read() as conn:
            assert conn.execute("SELECT COUNT(*) AS c FROM claim_vectors").fetchone()["c"] == 1
            assert (
                conn.execute("SELECT 1 FROM claim_fts WHERE claim_fts MATCH ?", (detail,)).fetchone()
                is None
            )


def test_temporary_detail_30_day_boundary_has_no_cleanup_inquiry_or_chat_notification(store):
    """AC-16 clauses: t_c+29 保留; t_c+30 起达到可清理点; t_c+31 原文与索引不可
    检索; 摘要/来源标记说明源详情已到期、不可直接查看; 无逐项询问或聊天清理
    通知; 边界约定须与全部类别一致。Runs the real sweep path and inspects the
    durable store for any user-visible prompt/question/notification surface."""
    anchor = "2026-01-01T00:00:00Z"
    detail = "R3_AC16_D30_TEMP_DETAIL_6612"
    event = make_event(
        native_id="message:ac16-temp",
        text=f"{detail}; 当前端口4200",
        event_time={"value": anchor, "precision": "instant", "timezone": "UTC"},
    )
    receive_batch(store, make_batch([event]))
    conclusion = record(
        store,
        display_text="当前端口4200",
        value=4200,
        evidence_event=event,
        event_time={"value": anchor, "precision": "instant", "timezone": "UTC"},
    )
    collect_vectors(store, {conclusion["revision_id"]}, HashEmbeddingProvider(dimension=2))
    summary_time = _day(anchor, 20)
    summary = record(
        store,
        display_text="摘要：临时调试已结束，详情见来源",
        value={"summary": "临时调试已结束"},
        predicate="task.summary",
        evidence_event=event,
        event_time={"value": summary_time, "precision": "instant", "timezone": "UTC"},
    )
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#1@0",
        class_="temporary_detail",
        business_anchor=anchor,
    )

    def _counts():
        with store.read() as conn:
            return {
                table: conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
                for table in ("events", "event_revisions", "event_parts", "receipts", "claims", "jobs")
            }

    before = _counts()

    # t_c + 29: still retained.
    swept_29 = sweep(store, now=_day(anchor, 29))
    assert swept_29["event_parts"] == 0 and swept_29["expired"] == 0, swept_29
    source = fetch_source(store, event["event_id"], 1, 0)
    assert source["availability"] == "full" and detail in (source["text"] or ""), source
    assert _recall(store, "端口4200")["items"], "valid conclusion must stay retrievable"
    assert _recall(store, "任务摘要")["items"]

    # t_c + 30: the cleanup point is reached; detail body/index gone, summary and
    # source mark the expired detail as not directly readable.
    swept_30 = sweep(store, now=_day(anchor, 30))
    assert swept_30["event_parts"] == 1 and swept_30["tombstoned"] >= 1, swept_30
    source = fetch_source(store, event["event_id"], 1, 0)
    assert source["availability"] == "expired" and source["text"] is None, source
    detail_recall = _recall(store, detail)
    assert detail_recall["items"] == [], detail_recall
    with store.read() as conn:
        assert conn.execute("SELECT 1 FROM claim_fts WHERE claim_fts MATCH ?", (detail,)).fetchone() is None
    conclusion_item = _item_for(_recall(store, "端口4200"), conclusion["claim_id"])
    assert conclusion_item["source_detail_expired"] is True, conclusion_item
    summary_item = _item_for(_recall(store, "任务摘要"), summary["claim_id"])
    assert summary_item["source_detail_expired"] is True, summary_item
    assert all(link["availability"] == "expired" for link in summary_item["sources"]), summary_item

    # t_c + 31: idempotent; the detail stays unretrievable.
    swept_31 = sweep(store, now=_day(anchor, 31))
    assert swept_31["event_parts"] == 0, swept_31
    assert fetch_source(store, event["event_id"], 1, 0)["text"] is None

    # Clause: 无逐项询问或聊天清理通知. The sweep surface is a pure counter
    # dict, the lifecycle view is a pure status dict, and no user-visible
    # prompt/question/notification table or record is created.
    assert set(swept_30) == {"classified", "expired", "tombstoned", "claim_revisions", "event_parts"}, swept_30
    assert all(isinstance(value, int) for value in swept_30.values()), swept_30
    status_view = lifecycle_status(store)
    assert set(status_view) == {"by_class", "due_now", "tombstones"}, status_view
    assert all(
        isinstance(state_counts, dict)
        and all(isinstance(count, int) for count in state_counts.values())
        for state_counts in status_view["by_class"].values()
    ), status_view["by_class"]
    assert isinstance(status_view["due_now"], int) and isinstance(status_view["tombstones"], int)
    with store.read() as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
    user_facing_markers = ("notif", "inquir", "question", "chat", "prompt", "inbox")
    assert not [
        table for table in tables if any(marker in table.lower() for marker in user_facing_markers)
    ], tables
    after = _counts()
    # Cleaning only purges derived state; it never writes a new user-visible
    # message/receipt/inquiry into the append-only evidence store.
    assert after["events"] == before["events"], (before, after)
    assert after["receipts"] == before["receipts"], (before, after)
    assert after["claims"] == before["claims"], (before, after)
    # AC-52 round 4: a real sweep records its own run in the diagnostic operation
    # ledger (kind=cleanup, model_version='kw:operation-ledger:1'). That is not
    # new queue work and not a user-visible message, so the original "no new jobs"
    # intent still holds once ledger rows are excluded.
    ledger_version = "kw:operation-ledger:1"
    with store.read() as conn:
        non_ledger_jobs = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE COALESCE(model_version, '') != ?",
            (ledger_version,),
        ).fetchone()["c"]
    assert non_ledger_jobs == before["jobs"], (before, after)

    # Clause: 边界约定须与全部类别一致 (one parsed-datetime boundary convention
    # for every retention class; the sweep uses expires_at <= now, never per-item
    # prompting).
    assert EXPIRED_BY_DAYS == {
        "temporary_detail": 30,
        "completed_task": 180,
        "superseded_history": 365,
        "revoked_history": 365,
    }, EXPIRED_BY_DAYS
