"""Anchor-candidate query-plan regression tests.

``load_event_group`` offers scoped existing fields as extraction anchors.  The
keyword branch used to start from every claim in the workspace/project and run
the FTS ``MATCH`` subquery once per claim (``SCAN claim_fts`` inside the join
loop), which is quadratic on a large store.  These tests pin the corrected
execution shape against a real store:

* the candidate query must start from the keyword index (EXPLAIN QUERY PLAN),
  not from ``claims``;
* a relevant but older field still ranks, while keyword matches from another
  workspace/project, and keyword matches on a superseded revision, never take a
  candidate slot or leak their text;
* the bounded recent fallback still applies when nothing matches.

Only EXPLAIN order is asserted (no wall-clock threshold): the plan is the
algorithmic property under test and is stable across machines.
"""
from __future__ import annotations

import sqlite3

from knowledge_weaver.extraction import load_event_group
from knowledge_weaver.store import receive_batch
from knowledge_weaver.store.claims import record_claim_revision_in_transaction

from ..store.conftest import make_batch, make_event

ANCHOR = "anchoralpha"
WORKSPACE = "owner:demo"
PROJECT = "project:songta"
EVENT_TIME = {
    "value": "2026-10-30T09:00:00+08:00",
    "precision": "instant",
    "timezone": "Asia/Shanghai",
}
LATER_EVENT_TIME = {
    "value": "2026-10-31T09:00:00+08:00",
    "precision": "instant",
    "timezone": "Asia/Shanghai",
}


def _receive(store, *, native_id, text=ANCHOR, workspace_id=WORKSPACE, project_ids=(PROJECT,)):
    event = make_event(
        native_id=native_id,
        text=text,
        workspace_id=workspace_id,
        project_ids=project_ids,
    )
    result = receive_batch(
        store,
        make_batch([event], batch_id="batch:" + native_id.replace("-", "")),
    )
    assert [receipt["disposition"] for receipt in result["receipts"]] == ["accepted"]
    return event


def _evidence(event):
    text = event["body"]["text"]
    return [
        {
            "event_id": event["event_id"],
            "event_revision": int(event["event_revision"]),
            "part_index": 0,
            "start_byte": 0,
            "end_byte": len(text.encode("utf-8")),
            "body_sha256": event["body"]["sha256"],
        }
    ]


def _write(
    conn,
    *,
    evidence,
    subject,
    predicate,
    display_text,
    value,
    project_id=PROJECT,
    workspace_id=WORKSPACE,
    event_time=EVENT_TIME,
):
    return record_claim_revision_in_transaction(
        conn,
        workspace_id=workspace_id,
        project_id=project_id,
        subject_id=subject,
        predicate=predicate,
        facet="desired",
        qualifiers={},
        value=value,
        display_text=display_text,
        authority="user_confirmed",
        assertion_kind="explicit_statement",
        event_time=event_time,
        evidence=evidence,
    )


class _RecordingConnection(sqlite3.Connection):
    """Record (sql, parameters) so the exact production query can be EXPLAINed."""

    executed: list[tuple[str, tuple]] = []

    def execute(self, sql, parameters=()):
        type(self).executed.append((sql, parameters))
        return super().execute(sql, parameters)


def test_candidate_query_plan_starts_from_fts(store, monkeypatch):
    event = _receive(store, native_id="message:plan-source")
    evidence = _evidence(event)
    with store.transaction() as conn:
        _write(
            conn,
            evidence=evidence,
            subject="subject:plan-anchor",
            predicate="server.port",
            display_text=ANCHOR,
            value=4200,
        )
        _write(
            conn,
            evidence=evidence,
            subject="subject:plan-other",
            predicate="server.mode",
            display_text="unrelated field",
            value="manual",
        )

    real_connect = sqlite3.connect

    def recording_connect(*args, **kwargs):
        kwargs.setdefault("factory", _RecordingConnection)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)
    _RecordingConnection.executed.clear()
    load_event_group(store, event["event_id"], 1)

    fts_statements = [
        (sql, params)
        for sql, params in _RecordingConnection.executed
        if "claim_fts MATCH" in sql and "bm25(claim_fts)" in sql
    ]
    assert len(fts_statements) == 1, "expected exactly one keyword candidate query"
    sql, params = fts_statements[0]

    with store.read() as conn:
        plan = conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    details = [row["detail"] for row in plan]
    fts_steps = [detail for detail in details if "claim_fts" in detail]

    # The keyword virtual table is the outer loop and the only scanned table.
    # The old defect showed ``SCAN claim_fts`` nested after a claims scan, once
    # per claim; that repeating step would either not be first or not be unique.
    assert len(fts_steps) == 1, details
    assert details[0] == fts_steps[0], details
    assert details[0].split()[0] == "SCAN", details
    assert "VIRTUAL TABLE" in details[0], details

    # Every other table is reached through its selected-key equality (SEARCH),
    # never through an outer scan of claims/current_claims/claim_revisions.
    scans = [detail for detail in details if detail.split()[0] == "SCAN"]
    assert scans == [fts_steps[0]], details


def test_relevant_old_field_ranks_and_scoped_foreign_matches_do_not_leak(store):
    source = _receive(store, native_id="message:rank-source")
    other_project_source = _receive(
        store,
        native_id="message:other-project",
        text="other project source",
        project_ids=("project:other",),
    )
    other_workspace_source = _receive(
        store,
        native_id="message:other-workspace",
        text="other workspace source",
        workspace_id="owner:other",
    )

    foreign_ids: set[str] = set()
    with store.transaction() as conn:
        relevant = _write(
            conn,
            evidence=_evidence(source),
            subject="subject:anchor",
            predicate="server.port",
            display_text=ANCHOR,
            value=4200,
        )
        for index in range(49):
            _write(
                conn,
                evidence=_evidence(source),
                subject=f"subject:filler{index:03d}",
                predicate="filler.field",
                display_text=(
                    f"{ANCHOR} filler{index:03d} "
                    + " ".join(f"term{term:03d}" for term in range(12))
                ),
                value=1000 + index,
            )
        for index in range(10):
            written = _write(
                conn,
                evidence=_evidence(other_project_source),
                subject=f"subject:otherproject{index:03d}",
                predicate="server.port",
                display_text=ANCHOR,
                value=5000 + index,
                project_id="project:other",
            )
            foreign_ids.add(written["claim_id"])
        for index in range(10):
            written = _write(
                conn,
                evidence=_evidence(other_workspace_source),
                subject=f"subject:otherworkspace{index:03d}",
                predicate="server.port",
                display_text=ANCHOR,
                value=6000 + index,
                workspace_id="owner:other",
            )
            foreign_ids.add(written["claim_id"])

        # A superseded revision still exists in the keyword index, but only the
        # current selected revision may represent the claim.
        stale = _write(
            conn,
            evidence=_evidence(source),
            subject="subject:stale",
            predicate="stale.field",
            display_text=f"{ANCHOR} obsolete",
            value="old value",
            event_time=EVENT_TIME,
        )
        stale_revision_two = _write(
            conn,
            evidence=_evidence(source),
            subject="subject:stale",
            predicate="stale.field",
            display_text="fresh current field",
            value="new value",
            event_time=LATER_EVENT_TIME,
        )
        assert stale["claim_id"] == stale_revision_two["claim_id"]

    group = load_event_group(store, source["event_id"], 1)
    candidates = group["claim_candidates"]
    candidate_ids = {candidate["claim_id"] for candidate in candidates}

    # 50 in-scope keyword matches fill the FTS branch; the recent fallback is
    # not consulted, so a superseded-only match must not appear at all.
    assert len(candidates) == 48
    assert relevant["claim_id"] in candidate_ids
    assert candidates[0]["claim_id"] == relevant["claim_id"]
    assert candidates[0]["label"] == ANCHOR
    assert all(candidate["project_id"] == PROJECT for candidate in candidates)
    assert foreign_ids.isdisjoint(candidate_ids)
    assert stale["claim_id"] not in candidate_ids
    assert all("anchoralpha obsolete" not in candidate["label"] for candidate in candidates)


def test_recent_fallback_still_returns_scoped_current_revisions(store):
    source = _receive(store, native_id="message:fallback-source", text="zzzunmatchedzzz")
    with store.transaction() as conn:
        current = _write(
            conn,
            evidence=_evidence(source),
            subject="subject:fallback",
            predicate="server.mode",
            display_text=ANCHOR,
            value="manual",
        )
        foreign = _write(
            conn,
            evidence=_evidence(source),
            subject="subject:fallback-foreign",
            predicate="server.mode",
            display_text=ANCHOR,
            value="manual",
            project_id="project:other",
        )

    group = load_event_group(store, source["event_id"], 1)
    candidate_ids = {candidate["claim_id"] for candidate in group["claim_candidates"]}
    assert current["claim_id"] in candidate_ids
    assert foreign["claim_id"] not in candidate_ids
