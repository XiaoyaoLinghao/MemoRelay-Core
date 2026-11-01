"""AC-08 coverage gaps: current truth, retained history/conflict, expired source
detail, and most importantly the two Codex-flagged semantic jumps:

* "摘要不把 C_old 当现行" - the user-visible current view (recall/current_items)
  after a correction must never present C_old as current, including after an
  asynchronous extraction/summarisation pass has run;
* "更正优先于异步提炼" - a late asynchronous extraction of an older source (the
  C_old statement) must not overwrite or degrade the corrected current truth, and
  the projection generation / revision history must stay explainable.

Frozen expected (docs/baseline-20261030/acceptance-cases.json AC-08):
"默认返回 C_new 为当前真相；C_old 保留为已替代并带来源/时间；C_conflict 保留
并在相关时提示；摘要不把 C_old 当现行；已到期源详情标明不可直接查看，不声称原
文仍可查；更正优先于异步提炼。全部满足为通过。"

Every assertion below maps to one of those clauses (see the module-level list in
runtime/t20-fix2/acceptance-staging/README.md).
"""

from __future__ import annotations

import json

import pytest

from tests.extraction.test_extraction import MockExtractor, _claim, _proposal
from tests.retrieval.conftest import recall_request
from tests.state.conftest import create_user_event, evidence_ref, record
from tests.store.conftest import make_batch, make_event

from knowledge_weaver import contracts
from knowledge_weaver.embeddings.index import HashEmbeddingProvider
from knowledge_weaver.extraction import (
    enqueue_extraction,
    load_event_group,
    run_worker_once,
    sanitized_parts,
)
from knowledge_weaver.lifecycle import sweep
from knowledge_weaver.lifecycle.retention import register_lifecycle
from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.store import (
    Store,
    current_items,
    get_current,
    receive_batch,
)
from knowledge_weaver.store.claims import apply_correction

BINDING = {
    "client_id": "client:test",
    "workspace_id": "owner:demo",
    "project_ids": ["project:songta"],
    "actor_ids": ["*"],
}

T1 = {"value": "2026-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"}
T2 = {"value": "2026-06-01T00:00:00Z", "precision": "instant", "timezone": "UTC"}
OLD_TEXT = "项目使用 SQLite"
NEW_TEXT = "改用本地 PostgreSQL"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")


def _correction(
    claim_id,
    expected_revision,
    generation,
    evidence,
    *,
    operation_id="op:ac08:corr:1",
    new_value="PostgreSQL",
    display_text=NEW_TEXT,
    reason="用户明确更正",
):
    return {
        "schema_version": "1.0",
        "operation_id": operation_id,
        "claim_id": claim_id,
        "expected_revision_id": expected_revision,
        "expected_projection_generation": generation,
        "new_value": new_value,
        "display_text": display_text,
        "evidence": [evidence],
        "supersedes_revision_ids": [expected_revision],
        "reason": reason,
    }


def _extract_claim(store, event, *, value, display_text=OLD_TEXT, predicate="project.db"):
    """Run the real extraction pipeline once for one received event."""
    part = sanitized_parts(load_event_group(store, event["event_id"], 1))[0]
    proposal = _proposal(
        _claim(
            part,
            predicate=predicate,
            facet="desired",
            value=value,
            display_text=display_text,
            assertion_kind="explicit_statement",
            quote=part.text,
        )
    )
    result = run_worker_once(store, MockExtractor(proposal))
    assert result["claimed"] == 1, result
    assert result["state"] == "succeeded", result
    return result


def _current(store, query="PostgreSQL", embedder=None):
    return execute_recall(
        store, recall_request(query=query, intent="current"), binding=BINDING, embedder=embedder
    )


def _history(store, query="项目 SQLite PostgreSQL"):
    return execute_recall(store, recall_request(query=query, intent="history"), binding=BINDING)


def _seed_corrected_claim(store):
    """C_old (t1, source A) -> C_new correction (source B); returns ids."""
    old_event = make_event(native_id="message:ac08-old", text=OLD_TEXT, event_time=T1)
    receive_batch(store, make_batch([old_event]))
    _extract_claim(store, old_event, value="SQLite")
    item = current_items(store, workspace_id="owner:demo")[0]
    claim_id = item["claim_id"]
    old_revision = item["revision_id"]
    generation = get_current(store, claim_id)["projection_generation"]
    new_event = create_user_event(store, NEW_TEXT, native_id="message:ac08-new")
    new_event["event_time"] = T2
    result = apply_correction(
        store, _correction(claim_id, old_revision, generation, evidence_ref(new_event))
    )
    assert result["status"] == "applied", result
    return {
        "claim_id": claim_id,
        "old_revision": old_revision,
        "old_event": old_event,
        "new_event": new_event,
        "correction_revision": result["revision_id"],
        "generation_before": generation,
        "generation_after": result["projection_generation"],
    }


def test_correction_is_current_and_old_revision_is_retained_with_source_and_time(store):
    """Clauses: C_new current; C_old retained as superseded with source/time."""
    ids = _seed_corrected_claim(store)

    current = get_current(store, ids["claim_id"])
    assert current["state"] == "current", current
    assert current["selected_revision_ids"] == [ids["correction_revision"]], current
    assert ids["generation_after"] > ids["generation_before"]

    items = current_items(store, workspace_id="owner:demo")
    assert [(item["value"], item["state"]) for item in items] == [("PostgreSQL", "current")], items

    # The old revision stays in history as superseded, with its evidence source
    # and business time intact.
    with store.read() as conn:
        old = conn.execute(
            "SELECT * FROM claim_revisions WHERE revision_id = ?", (ids["old_revision"],)
        ).fetchone()
        links = conn.execute(
            "SELECT * FROM evidence_links WHERE revision_id = ?", (ids["old_revision"],)
        ).fetchall()
    assert old["status"] == "superseded", dict(old)
    assert old["event_time_value"] == T1["value"]
    assert [link["event_id"] for link in links] == [ids["old_event"]["event_id"]]

    history = _history(store)
    history_revisions = {item["revision_id"]: item for item in history["items"]}
    assert ids["correction_revision"] in history_revisions, history
    assert ids["old_revision"] in history_revisions, history
    assert history_revisions[ids["old_revision"]]["state"] == "superseded"
    assert history_revisions[ids["old_revision"]]["sources"], history_revisions[ids["old_revision"]]


def test_user_visible_current_view_never_presents_old_value_as_current(store):
    """Clause: 摘要不把 C_old 当现行 (current projection + recall summary view)."""
    ids = _seed_corrected_claim(store)

    current = _current(store, embedder=HashEmbeddingProvider(dimension=2))
    assert current["status"] == "ok", current
    assert current["items"], current
    values = [item["value"] for item in current["items"]]
    assert "PostgreSQL" in values, current["items"]
    assert "SQLite" not in values, current["items"]
    assert all(item["state"] == "current" for item in current["items"]), current["items"]

    # Querying by the OLD phrasing must not promote the old value back to current.
    by_old = _current(store, query="SQLite")
    assert by_old["status"] == "empty", by_old
    assert by_old["items"] == [], by_old["items"]


def test_conflicting_source_is_retained_and_surfaced(store):
    """Clause: C_conflict 保留并在相关时提示."""
    ids = _seed_corrected_claim(store)
    # A second, independent conflicting source on a related claim: equal
    # authority and equal business time -> conflict, both candidates visible.
    record(
        store,
        display_text="数据库主机 db-old",
        value="db-old",
        predicate="database.host",
        event_time=T1,
        native_id="message:ac08-conflict-1",
    )
    record(
        store,
        display_text="数据库主机 db-new",
        value="db-new",
        predicate="database.host",
        event_time=T1,
        native_id="message:ac08-conflict-2",
    )
    conflict_items = [
        item for item in current_items(store, workspace_id="owner:demo") if item["predicate"] == "database.host"
    ]
    assert len(conflict_items) == 1 and conflict_items[0]["state"] == "conflict"
    assert conflict_items[0]["conflict_revision_ids"], conflict_items[0]

    # The correction of the other claim must not delete the conflict: it stays
    # visible ("相关时提示") with both candidate revisions.
    recall = _current(store, query="数据库主机")
    conflict_hits = [item for item in recall["items"] if item["predicate"] == "database.host"]
    assert len(conflict_hits) == 1, recall["items"]
    assert conflict_hits[0]["state"] == "conflict"
    assert conflict_hits[0]["conflict_revision_ids"], conflict_hits[0]

    corrected = get_current(store, ids["claim_id"])
    assert corrected["state"] == "current"
    assert corrected["selected_revision_ids"] == [ids["correction_revision"]]

    # History keeps every conflicting candidate revision traceable.
    with store.read() as conn:
        rows = conn.execute(
            "SELECT revision_id, status FROM claim_revisions r JOIN claims c "
            "ON c.claim_id = r.claim_id WHERE c.predicate = 'database.host'"
        ).fetchall()
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"conflict"}


def test_expired_source_detail_is_marked_unreadable_not_claimed_readable(store):
    """Clause: 已到期源详情标明不可直接查看，不声称原文仍可查."""
    ids = _seed_corrected_claim(store)
    # C_old's source detail expires; the C_new source stays readable.
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{ids['old_event']['event_id']}#1@0",
        class_="completed_task",
        business_anchor="2020-01-01T00:00:00Z",
    )
    swept = sweep(store, now="2026-10-30T00:00:00Z")
    assert swept["event_parts"] == 1, swept

    source = None
    from knowledge_weaver.store import fetch_source

    source = fetch_source(store, ids["old_event"]["event_id"], 1, 0)
    assert source["availability"] == "expired", source
    assert source["text"] is None, source

    history = _history(store, query="SQLite")
    old_hits = [item for item in history["items"] if item["revision_id"] == ids["old_revision"]]
    assert old_hits, history["items"]
    old_item = old_hits[0]
    assert old_item["source_detail_expired"] is True, old_item
    assert old_item["sources"], old_item
    assert all(source_link["availability"] == "expired" for source_link in old_item["sources"]), old_item

    # The current truth still resolves through its own (readable) source.
    current = _current(store)
    current_item = [item for item in current["items"] if item["claim_id"] == ids["claim_id"]][0]
    assert current_item["source_detail_expired"] is False, current_item
    assert all(link["availability"] == "full" for link in current_item["sources"]), current_item


def test_correction_requires_exact_user_evidence_and_rejects_invalid_sources(store):
    """Actions clause: 更正引用精确 user 原始证据 revision/part/span/hash；拒绝
    助手/无关/撤回来源；同时比较 projection generation."""
    old_event = make_event(native_id="message:ac08-ev-old", text=OLD_TEXT, event_time=T1)
    receive_batch(store, make_batch([old_event]))
    _extract_claim(store, old_event, value="SQLite")
    item = current_items(store, workspace_id="owner:demo")[0]
    claim_id, old_revision = item["claim_id"], item["revision_id"]
    generation = get_current(store, claim_id)["projection_generation"]

    new_event = create_user_event(store, NEW_TEXT, native_id="message:ac08-ev-new")
    # The applied correction cites an exact revision/part/span/hash.
    evidence = evidence_ref(new_event, start_byte=0, end_byte=len(NEW_TEXT.encode("utf-8")))
    result = apply_correction(store, _correction(claim_id, old_revision, generation, evidence))
    assert result["status"] == "applied", result
    with store.read() as conn:
        link = conn.execute(
            "SELECT * FROM evidence_links WHERE revision_id = ?", (result["revision_id"],)
        ).fetchone()
    assert link["event_id"] == new_event["event_id"]
    assert link["part_index"] == 0
    assert link["start_byte"] == 0 and link["end_byte"] == len(NEW_TEXT.encode("utf-8"))
    assert link["body_sha256"] == new_event["body"]["sha256"]
    applied_generation = result["projection_generation"]
    assert applied_generation >= generation
    expected_revision = result["revision_id"]

    # Assistant evidence is not native user evidence: pending_evidence, no write,
    # generation unchanged.
    assistant = make_event(
        native_id="message:ac08-assistant",
        text=NEW_TEXT,
        role="assistant",
        origin_kind="assistant_reply",
    )
    receive_batch(store, make_batch([assistant]))
    rejected = apply_correction(
        store,
        _correction(
            claim_id,
            expected_revision,
            applied_generation,
            evidence_ref(assistant),
            operation_id="op:ac08:assistant",
        ),
    )
    assert rejected["status"] == "pending_evidence", rejected
    assert rejected["revision_id"] is None
    assert get_current(store, claim_id)["projection_generation"] == applied_generation

    # A quote that does not support the proposed replacement is rejected.
    unrelated = create_user_event(store, "今天天气不错", native_id="message:ac08-unrelated")
    rejected = apply_correction(
        store,
        _correction(
            claim_id,
            expected_revision,
            applied_generation,
            evidence_ref(unrelated),
            operation_id="op:ac08:unrelated",
        ),
    )
    assert rejected["status"] == "pending_evidence", rejected

    # A retracted source can never be used as correction evidence.
    retracted = create_user_event(store, "用于更正但随后被撤回的证据", native_id="message:ac08-retract")
    retract_event = make_event(
        native_id="message:ac08-retract",
        revision=2,
        operation="retract",
        parent_revision=1,
        text="",
    )
    retract_event["body"]["sha256"] = contracts.EMPTY_BODY_SHA256
    receive_batch(store, make_batch([retract_event], batch_id="batch:ac08-retract"))
    rejected = apply_correction(
        store,
        _correction(
            claim_id,
            expected_revision,
            applied_generation,
            evidence_ref(retracted),
            operation_id="op:ac08:retracted",
        ),
    )
    assert rejected["status"] == "pending_evidence", rejected
    assert get_current(store, claim_id)["projection_generation"] == applied_generation

    # A stale compare-and-swap generation is an explicit conflict, not a write.
    stale = apply_correction(
        store,
        _correction(
            claim_id,
            expected_revision,
            applied_generation - 1,
            evidence_ref(new_event),
            operation_id="op:ac08:stale",
        ),
    )
    assert stale["status"] == "conflict", stale
    assert get_current(store, claim_id)["projection_generation"] == applied_generation


@pytest.mark.parametrize("late_source", ["same_event", "sibling_event"])
def test_late_async_extraction_cannot_override_corrected_current(store, late_source):
    """Clause: 更正优先于异步提炼 (and the summary must not show C_old current).

    The race is real: the extraction job for the old source exists (or is
    re-created) while the user correction is applied; the worker then finishes
    later. Expected: the corrected C_new stays current, generation/history stay
    explainable, and the late extraction revision never becomes current.
    """
    old_event = make_event(native_id="message:ac08-race-old", text=OLD_TEXT, event_time=T1)
    receive_batch(store, make_batch([old_event]))
    _extract_claim(store, old_event, value="SQLite")
    item = current_items(store, workspace_id="owner:demo")[0]
    claim_id, old_revision = item["claim_id"], item["revision_id"]
    generation_before = get_current(store, claim_id)["projection_generation"]

    # A second old-value source whose extraction is still pending when the user
    # corrects; or force a fresh (idempotent-versioned) extraction of the same
    # source to represent a retried/slow worker.
    if late_source == "sibling_event":
        late_event = make_event(native_id="message:ac08-race-sibling", text=OLD_TEXT, event_time=T1)
        receive_batch(store, make_batch([late_event]))
        late_group = (late_event["event_id"], 1)
    else:
        # A retried/duplicated extraction job for the already-processed old
        # source: same input key, new job generation, still pending when the
        # correction is applied.
        from knowledge_weaver.jobs import enqueue as enqueue_job

        enqueue_job(store, "extract", f"{old_event['event_id']}@1", input_version=2)
        late_group = (old_event["event_id"], 1)

    new_event = create_user_event(store, NEW_TEXT, native_id="message:ac08-race-new")
    new_event["event_time"] = T2
    correction = apply_correction(
        store, _correction(claim_id, old_revision, generation_before, evidence_ref(new_event))
    )
    assert correction["status"] == "applied", correction
    generation_after = correction["projection_generation"]
    assert generation_after > generation_before

    # The late asynchronous extraction/summarisation pass now completes.
    part = sanitized_parts(load_event_group(store, late_group[0], late_group[1]))[0]
    late_proposal = _proposal(
        _claim(
            part,
            predicate="project.db",
            facet="desired",
            value="SQLite",
            display_text=OLD_TEXT,
            assertion_kind="explicit_statement",
            quote=part.text,
        )
    )
    late_claims = 0
    late_results = []
    for _ in range(4):
        late = run_worker_once(store, MockExtractor(late_proposal))
        late_results.append(late)
        if not late.get("claimed"):
            break
        late_claims += int(late.get("claims_written") or 0)
    assert late_claims == 1, late_results

    current = get_current(store, claim_id)
    assert current["state"] == "current", current
    assert current["selected_revision_ids"] == [correction["revision_id"]], current
    assert current["projection_generation"] == generation_after, current

    current_view = _current(store)
    assert [i for i in current_view["items"] if i["claim_id"] == claim_id], current_view["items"]
    assert all(
        i["value"] != "SQLite" for i in current_view["items"] if i["claim_id"] == claim_id
    ), current_view["items"]

    # History stays explainable: the late extraction revision is retained but is
    # not current, and the correction is ordered after the old revision.
    with store.read() as conn:
        revisions = conn.execute(
            "SELECT revision_id, revision_number, status, display_text FROM claim_revisions "
            "WHERE claim_id = ? ORDER BY revision_number",
            (claim_id,),
        ).fetchall()
    statuses = {row["revision_number"]: row["status"] for row in revisions}
    assert len(revisions) == 3, [dict(row) for row in revisions]
    assert statuses[1] == "superseded", statuses
    assert statuses[2] == "current", statuses
    assert statuses[3] in ("superseded", "uncertain"), statuses
    assert statuses[3] != "current", statuses

    # History intent returns the revisions matched by the query (existing
    # semantics, same as test_history_pagination_covers_superseded_revisions).
    # Querying the old body text surfaces both old-value revisions, and neither
    # may be current; the corrected body text reaches the correction revision.
    old_history = _history(store, query="SQLite")
    old_items = {item["revision_id"]: item for item in old_history["items"]}
    ids_old = {row["revision_id"] for row in revisions if row["revision_number"] in (1, 3)}
    assert ids_old
    assert ids_old <= set(old_items), (ids_old, old_items)
    for revision_id in ids_old:
        assert old_items[revision_id]["state"] != "current", old_items[revision_id]
    correction_history = _history(store, query="PostgreSQL")
    correction_ids = {item["revision_id"] for item in correction_history["items"]}
    assert correction["revision_id"] in correction_ids, correction_history["items"]


def test_correction_replay_is_idempotent(store):
    """AC-08 evidence clause: operation replay keeps one correction revision and
    the same projection generation (compare-and-swap semantics)."""
    ids = _seed_corrected_claim(store)
    replay = _correction(
        ids["claim_id"],
        ids["old_revision"],
        ids["generation_before"],
        evidence_ref(ids["new_event"]),
    )
    second = apply_correction(store, replay)
    assert second["status"] == "duplicate", second
    assert second["revision_id"] == ids["correction_revision"]
    with store.read() as conn:
        revisions = conn.execute(
            "SELECT revision_number, status FROM claim_revisions WHERE claim_id = ? "
            "ORDER BY revision_number",
            (ids["claim_id"],),
        ).fetchall()
    assert [(row["revision_number"], row["status"]) for row in revisions] == [
        (1, "superseded"),
        (2, "current"),
    ]
    assert get_current(store, ids["claim_id"])["projection_generation"] == ids["generation_after"]
