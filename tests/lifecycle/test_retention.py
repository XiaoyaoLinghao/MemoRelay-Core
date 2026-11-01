"""T13: 30/180/365 business-anchor retention, expiry, tombstones and bootstrap."""
from __future__ import annotations

from knowledge_weaver import contracts
from knowledge_weaver.lifecycle import (
    EXPIRED_BY_DAYS,
    classify_claim_revisions,
    is_tombstoned,
    lifecycle_status,
    register_lifecycle,
    sweep,
)
from knowledge_weaver.store import Store, current_items, get_current, receive_batch
from knowledge_weaver.store.claims import record_claim_revision

from ..state.conftest import create_user_event, current_item, evidence_ref, record


def _claim_with_event(store, text, value, *, predicate="server.port"):
    event = create_user_event(store, text)
    return record_claim_revision(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        subject_id="subject:songta-server",
        predicate=predicate,
        facet="desired",
        qualifiers={},
        value=value,
        display_text=text,
        authority="user_confirmed",
        assertion_kind="explicit_statement",
        event_time=event["event_time"],
        evidence=[evidence_ref(event)],
    )


def _force_status_time(store, revision_id, *, status, changed_at):
    with store.transaction() as conn:
        conn.execute(
            "UPDATE claim_revisions SET status = ?, status_changed_at = ? WHERE revision_id = ?",
            (status, changed_at, revision_id),
        )


def test_current_fact_has_no_expiry(store):
    claim = _claim_with_event(store, "端口4200", 4200)
    classify_claim_revisions(store, now="2026-10-30T00:00:00Z")
    with store.read() as conn:
        row = conn.execute(
            "SELECT * FROM lifecycle_records WHERE subject_id = ?", (claim["revision_id"],)
        ).fetchone()
    assert row["class"] == "current_fact"
    assert row["expires_at"] is None
    assert sweep(store, now="2030-01-01T00:00:00Z")["expired"] == 0


def test_superseded_history_expires_after_365_days_from_replacement(store):
    first = _claim_with_event(store, "端口4100", 4100)
    current = record(
        store,
        display_text="端口改为4200",
        value=4200,
        supersedes=[first["revision_id"]],
        event_time={"value": "2026-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    classify_claim_revisions(store, now="2026-01-01T00:00:01Z")
    with store.read() as conn:
        row = conn.execute(
            "SELECT * FROM lifecycle_records WHERE subject_id = ?", (first["revision_id"],)
        ).fetchone()
    assert row["class"] == "superseded_history"
    assert row["expires_at"].startswith("2027-01-01")
    # Not due yet; access/rebuild does not renew or expire it.
    current_items(store, workspace_id="owner:demo")
    assert sweep(store, now="2026-12-31T00:00:00Z")["expired"] == 0
    result = sweep(store, now="2027-01-02T00:00:00Z")
    assert result["expired"] == 1
    assert is_tombstoned(store, "claim_revision", first["revision_id"])
    with store.read() as conn:
        revision = conn.execute(
            "SELECT status, display_text, value_json FROM claim_revisions WHERE revision_id = ?",
            (first["revision_id"],),
        ).fetchone()
        fts = conn.execute(
            "SELECT COUNT(*) AS c FROM claim_fts WHERE revision_id = ?", (first["revision_id"],)
        ).fetchone()
    assert revision["status"] == "expired"
    assert revision["display_text"] == ""
    assert revision["value_json"] == "null"
    assert fts["c"] == 0
    # The newer current revision is unaffected.
    assert current_item(store)["value"] == 4200
    assert get_current(store, current["claim_id"])["state"] == "current"


def test_revoked_history_expires_after_365_days(store):
    claim = _claim_with_event(store, "端口4200", 4200)
    event = create_user_event(store, "无关")
    with store.transaction() as conn:
        revision = conn.execute(
            "SELECT source_actors_json FROM claim_revisions WHERE revision_id = ?",
            (claim["revision_id"],),
        ).fetchone()
    _force_status_time(store, claim["revision_id"], status="revoked", changed_at="2026-01-01T00:00:00Z")
    classify_claim_revisions(store, now="2026-01-01T00:00:01Z")
    with store.read() as conn:
        row = conn.execute(
            "SELECT * FROM lifecycle_records WHERE subject_id = ?", (claim["revision_id"],)
        ).fetchone()
    assert row["class"] == "revoked_history"
    assert row["expires_at"].startswith("2027-01-01")
    sweep(store, now="2027-01-02T00:00:00Z")
    assert is_tombstoned(store, "claim_revision", claim["revision_id"])


def test_temporary_detail_expires_after_30_days_and_purges_event_body(store):
    event = create_user_event(store, "调试失败细节：尝试端口4300，连接被拒绝")
    subject = f"{event['event_id']}#1@0"
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=subject,
        class_="temporary_detail",
        business_anchor="2026-01-01T00:00:00Z",
        reason="failed debug attempt",
    )
    assert sweep(store, now="2026-01-20T00:00:00Z")["expired"] == 0
    result = sweep(store, now="2026-02-01T00:00:00Z")
    assert result["event_parts"] == 1
    assert is_tombstoned(store, "event_part", subject)
    with store.read() as conn:
        part = conn.execute(
            "SELECT body_text, availability FROM event_parts WHERE event_id = ?", (event["event_id"],)
        ).fetchone()
    assert part["body_text"] is None
    assert part["availability"] == "expired"


def test_completed_task_detail_uses_180_day_anchor(store):
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id="evt:example#1@0",
        class_="completed_task",
        business_anchor="2026-01-01T00:00:00Z",
    )
    with store.read() as conn:
        row = conn.execute("SELECT expires_at FROM lifecycle_records").fetchone()
    assert row["expires_at"].startswith("2026-06-30")
    assert EXPIRED_BY_DAYS["completed_task"] == 180


def test_unknown_state_stays_unclassified_and_is_not_deleted(store):
    from ..store.conftest import make_batch, make_event

    event = make_event(native_id="message:unclassified", text="时间未知的事实")
    receive_batch(store, make_batch([event]))
    claim = record_claim_revision(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        subject_id="subject:songta-server",
        predicate="server.port",
        facet="desired",
        qualifiers={},
        value=4200,
        display_text="时间未知的事实",
        authority="user_confirmed",
        assertion_kind="explicit_statement",
        event_time={"value": None, "precision": "unknown", "timezone": "UTC"},
        evidence=[evidence_ref(event)],
    )
    classify_claim_revisions(store, now="2026-01-01T00:00:00Z")
    with store.read() as conn:
        row = conn.execute(
            "SELECT * FROM lifecycle_records WHERE subject_id = ?", (claim["revision_id"],)
        ).fetchone()
    assert row["class"] == "unclassified"
    assert row["expires_at"] is None
    assert sweep(store, now="2035-01-01T00:00:00Z")["expired"] == 0


def test_mixed_summaries_and_repeated_sweeps_do_not_renew(store):
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id="evt:a#1@0",
        class_="temporary_detail",
        business_anchor="2026-01-01T00:00:00Z",
    )
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id="evt:b#1@0",
        class_="temporary_detail",
        business_anchor="2026-02-01T00:00:00Z",
    )
    first = sweep(store, now="2026-02-05T00:00:00Z")
    assert first["expired"] == 1  # only the January detail is due
    second = sweep(store, now="2026-03-05T00:00:00Z")
    assert second["expired"] == 1  # February detail; the expired one is not renewed
    assert is_tombstoned(store, "event_part", "evt:a#1@0")
    assert is_tombstoned(store, "event_part", "evt:b#1@0")


def test_bootstrap_pending_import_is_not_expired(store):
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id="evt:imported#1@0",
        class_="temporary_detail",
        business_anchor="2025-01-01T00:00:00Z",
        state="bootstrap_pending",
    )
    assert sweep(store, now="2030-01-01T00:00:00Z")["expired"] == 0
    with store.read() as conn:
        row = conn.execute("SELECT state FROM lifecycle_records").fetchone()
    assert row["state"] == "bootstrap_pending"


def test_lifecycle_status_reports_due_and_tombstones(store):
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id="evt:due#1@0",
        class_="temporary_detail",
        business_anchor="2026-01-01T00:00:00Z",
    )
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id="evt:later#1@0",
        class_="completed_task",
        business_anchor="2026-10-01T00:00:00Z",
    )
    summary = lifecycle_status(store)
    assert summary["due_now"] == 1
    sweep(store, now="2026-02-01T00:00:00Z")
    summary = lifecycle_status(store)
    assert summary["tombstones"] == 1
    assert summary["due_now"] == 0


# ---------------------------------------------------------------------------
# F09: expiring a source detail must propagate into evidence and task views
# ---------------------------------------------------------------------------


def test_source_expiry_marks_retained_claim_evidence_expired(store):
    event = create_user_event(store, "旧过程证据对应有效结论")
    record(
        store,
        display_text="有效结论",
        value="active",
        evidence_event=event,
    )
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#1@0",
        class_="completed_task",
        business_anchor="2020-01-01T00:00:00Z",
    )
    sweep(store, now="2026-10-30T00:00:00Z")
    items = current_items(store, workspace_id="owner:demo")
    assert items
    for item in items:
        assert item["source_detail_expired"] is True
        assert all(source["availability"] != "full" for source in item["sources"])
    with store.read() as conn:
        availability = conn.execute(
            "SELECT availability FROM evidence_links"
        ).fetchone()["availability"]
    assert availability == "expired"
    # The retained conclusion itself stays explainable (not silently deleted).
    assert any(item["text"] == "有效结论" for item in items)


def test_source_expiry_downgrades_task_verification_view(store):
    event = create_user_event(store, "功能验收通过")
    claim = record_claim_revision(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        subject_id="subject:task:deploy",
        predicate="task.status",
        facet="observed",
        qualifiers={},
        value={"status": "verified", "conditions": ["functional"], "satisfied": ["functional"]},
        display_text="功能验收通过",
        authority="tool_verified",
        assertion_kind="observation",
        event_time=event["event_time"],
        evidence=[evidence_ref(event)],
    )
    from knowledge_weaver.store import list_task_views

    views = list_task_views(store, workspace_id="owner:demo")
    assert views and views[0]["status"] == "verified_done"
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#1@0",
        class_="completed_task",
        business_anchor="2020-01-01T00:00:00Z",
    )
    sweep(store, now="2026-10-30T00:00:00Z")
    views = list_task_views(store, workspace_id="owner:demo")
    assert not any(view["status"] == "verified_done" for view in views), views
    current = get_current(store, claim["claim_id"])
    assert current["state"] in ("uncertain", "expired", "revoked")
