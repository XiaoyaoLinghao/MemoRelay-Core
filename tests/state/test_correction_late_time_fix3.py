"""T20 round-4 regression: AC-08 / AC-35 (late older business time must not
undo a later explicit user correction).

Codex round-3 counterexamples:

* ``test_followups.py::test_AC08_actual_worker_late_march_does_not_undo_june_correction``:
  January SQLite fact -> explicit June user correction to PostgreSQL -> a late
  real worker extraction of a *March* historical record.  The correction used to
  inherit the January ordering time, so the March record sorted later and
  re-opened the old value (generation 2 -> 3).
* ``test_ac35.py::test_AC35_late_old_port_cannot_reverse_explicit_port_correction``:
  same shape for a port value (4100 -> correction 4200 -> late March 4100).

Round-4 rule: keep the corrected fact's *age* (its original business time is
still the retention/explainability anchor) but give the correction an effective
business time taken from the user's own correction evidence.  A record whose
business time is earlier than the correction stays stale; a genuinely later
native statement may still change the fact (AC-35 "旧记录不反向覆盖", not a
freeze).  No retention clock is refreshed to solve ordering.
"""

from __future__ import annotations

from knowledge_weaver.extraction import (
    load_event_group,
    run_worker_once,
    sanitized_parts,
)
from knowledge_weaver.store import current_items, get_current, receive_batch, rebuild_projection
from knowledge_weaver.store.claims import apply_correction

from ..extraction.test_extraction import MockExtractor, _claim, _proposal
from ..store.conftest import make_batch, make_event
from .conftest import evidence_ref, record

T1 = {"value": "2026-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"}
T2 = {"value": "2026-06-01T00:00:00Z", "precision": "instant", "timezone": "UTC"}
T3 = {"value": "2026-07-01T00:00:00Z", "precision": "instant", "timezone": "UTC"}
OLD_DB_TEXT = "项目使用 SQLite"
NEW_DB_TEXT = "改用本地 PostgreSQL"
WORKSPACE = "owner:demo"


def _event(native, text, when):
    return make_event(
        native_id=native,
        text=text,
        event_time={"value": when, "precision": "instant", "timezone": "UTC"},
    )


def _extract(store, event, value, *, display_text, predicate="project.db"):
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
    assert result["state"] == "succeeded", result
    assert result["claims_written"] == 1, result
    return result


def _correction(claim_id, expected_revision, generation, evidence, *, new_value, display_text):
    return {
        "schema_version": "1.0",
        "operation_id": "op:fix3:ac08:" + expected_revision[-12:],
        "claim_id": claim_id,
        "expected_revision_id": expected_revision,
        "expected_projection_generation": generation,
        "new_value": new_value,
        "display_text": display_text,
        "evidence": [evidence],
        "supersedes_revision_ids": [expected_revision],
        "reason": "user correction",
    }


def _only_current(store, claim_id):
    items = [
        item
        for item in current_items(store, workspace_id=WORKSPACE)
        if item["claim_id"] == claim_id and item["state"] == "current"
    ]
    assert len(items) == 1, items
    return items[0]


def _drain_extraction(store):
    while True:
        result = run_worker_once(store, MockExtractor({"schema_version": "1.0", "claims": [], "excluded_event_ids": [], "warnings": []}))
        if not result.get("claimed"):
            return


# ---------------------------------------------------------------------------
# Original Codex counterexamples (real worker / real correction path)
# ---------------------------------------------------------------------------


def test_AC08_actual_worker_late_march_does_not_undo_june_correction(store):
    """Verbatim semantics of the Codex follow-up counterexample."""
    old = _event("message:fix3-ac08-jan", OLD_DB_TEXT, T1["value"])
    receive_batch(store, make_batch([old], batch_id="batch:fix3-ac08-jan"))
    _extract(store, old, "SQLite", display_text=OLD_DB_TEXT)
    initial = current_items(store, workspace_id=WORKSPACE)[0]
    current = get_current(store, initial["claim_id"])

    change = _event("message:fix3-ac08-june", NEW_DB_TEXT, T2["value"])
    receive_batch(store, make_batch([change], batch_id="batch:fix3-ac08-june"))
    applied = apply_correction(
        store,
        _correction(
            initial["claim_id"],
            initial["revision_id"],
            current["projection_generation"],
            evidence_ref(change),
            new_value="PostgreSQL",
            display_text=NEW_DB_TEXT,
        ),
    )
    assert applied["status"] == "applied", applied
    _drain_extraction(store)
    generation_after = get_current(store, initial["claim_id"])["projection_generation"]

    late = _event("message:fix3-ac08-march", OLD_DB_TEXT, "2026-03-01T00:00:00Z")
    receive_batch(store, make_batch([late], batch_id="batch:fix3-ac08-march"))
    _extract(store, late, "SQLite", display_text=OLD_DB_TEXT)

    after = get_current(store, initial["claim_id"])
    assert after["selected_revision_ids"] == [applied["revision_id"]], {
        "current": after,
        "visible_values": [x["value"] for x in current_items(store, workspace_id=WORKSPACE)],
    }
    assert after["projection_generation"] == generation_after, after
    assert after["state"] == "current", after
    assert _only_current(store, initial["claim_id"])["value"] == "PostgreSQL"


def test_AC35_late_old_port_cannot_reverse_explicit_port_correction(store):
    """Verbatim semantics of the Codex AC-35 counterexample."""
    old = _event("message:fix3-port-jan", "端口4100", T1["value"])
    receive_batch(store, make_batch([old], batch_id="batch:fix3-port-jan"))
    claimed = record(
        store,
        display_text="端口4100",
        value=4100,
        evidence_event=old,
        event_time=old["event_time"],
    )
    correction_event = _event("message:fix3-port-june", "端口改为4200", T2["value"])
    receive_batch(store, make_batch([correction_event], batch_id="batch:fix3-port-june"))
    before = get_current(store, claimed["claim_id"])
    applied = apply_correction(
        store,
        _correction(
            claimed["claim_id"],
            claimed["revision_id"],
            before["projection_generation"],
            evidence_ref(correction_event),
            new_value=4200,
            display_text="端口改为4200",
        ),
    )
    assert applied["status"] == "applied", applied

    late = _event("message:fix3-port-march", "端口4100", "2026-03-01T00:00:00Z")
    receive_batch(store, make_batch([late], batch_id="batch:fix3-port-march"))
    record(
        store,
        display_text="端口4100",
        value=4100,
        evidence_event=late,
        event_time=late["event_time"],
    )

    items = current_items(store, workspace_id=WORKSPACE)
    assert any(
        item["claim_id"] == claimed["claim_id"]
        and item["value"] == 4200
        and item["state"] == "current"
        for item in items
    ), items


# ---------------------------------------------------------------------------
# Adjacent scenarios
# ---------------------------------------------------------------------------


def _seed_corrected_claim(store):
    old = _event("message:fix3-ac08-seed-old", OLD_DB_TEXT, T1["value"])
    receive_batch(store, make_batch([old], batch_id="batch:fix3-ac08-seed-old"))
    old_claim = record(
        store,
        display_text=OLD_DB_TEXT,
        value="SQLite",
        predicate="project.db",
        evidence_event=old,
    )
    generation_before = get_current(store, old_claim["claim_id"])["projection_generation"]
    change = _event("message:fix3-ac08-seed-new", NEW_DB_TEXT, T2["value"])
    receive_batch(store, make_batch([change], batch_id="batch:fix3-ac08-seed-new"))
    applied = apply_correction(
        store,
        _correction(
            old_claim["claim_id"],
            old_claim["revision_id"],
            generation_before,
            evidence_ref(change),
            new_value="PostgreSQL",
            display_text=NEW_DB_TEXT,
        ),
    )
    assert applied["status"] == "applied", applied
    return {
        "claim_id": old_claim["claim_id"],
        "old_revision": old_claim["revision_id"],
        "correction_revision": applied["revision_id"],
        "generation_after": get_current(store, old_claim["claim_id"])["projection_generation"],
        "old_event": old,
        "change_event": change,
    }


def _late_sibling(store, ids, *, when, native_id, value="SQLite", event_time=None):
    """A late record of the old truth from another source, business time ``when``."""
    sibling = _event(native_id, OLD_DB_TEXT, when)
    receive_batch(store, make_batch([sibling], batch_id=f"batch:{native_id}"))
    return record(
        store,
        display_text=OLD_DB_TEXT,
        value=value,
        predicate="project.db",
        event_time=event_time or sibling["event_time"],
        evidence_event=sibling,
    )


def test_AC08_late_intermediate_time_from_another_source_stays_stale(store):
    """The March record loses to the June correction with non-worker writes too."""
    ids = _seed_corrected_claim(store)
    late = _late_sibling(
        store, ids, when="2026-03-01T00:00:00Z", native_id="message:fix3-ac08-late-march"
    )
    current = get_current(store, ids["claim_id"])
    assert current["state"] == "current", current
    assert current["selected_revision_ids"] == [ids["correction_revision"]], current
    assert current["projection_generation"] == ids["generation_after"], current
    with store.read() as conn:
        rows = conn.execute(
            "SELECT revision_number, status FROM claim_revisions WHERE claim_id = ? "
            "ORDER BY revision_number",
            (ids["claim_id"],),
        ).fetchall()
    assert {row["revision_number"]: row["status"] for row in rows} == {
        1: "superseded",
        2: "current",
        3: "superseded",
    }, [dict(row) for row in rows]
    assert late["revision_id"] not in current["selected_revision_ids"]


def test_AC08_genuinely_later_business_can_still_change_after_correction(store):
    """AC-35: an older record cannot reverse the correction; real later truth can."""
    ids = _seed_corrected_claim(store)
    newer = _late_sibling(
        store,
        ids,
        when=T3["value"],
        native_id="message:fix3-ac08-newer-july",
        value="MySQL",
    )
    # The sibling helper writes the old display text; a genuinely newer native
    # statement carries the new truth, so write it explicitly as well.
    current = get_current(store, ids["claim_id"])
    assert current["state"] == "current", current
    assert current["selected_revision_ids"] == [newer["revision_id"]], current
    assert current["projection_generation"] > ids["generation_after"], current
    with store.read() as conn:
        statuses = {
            row["revision_number"]: row["status"]
            for row in conn.execute(
                "SELECT revision_number, status FROM claim_revisions WHERE claim_id = ?",
                (ids["claim_id"],),
            ).fetchall()
        }
    assert statuses[2] == "superseded", statuses
    assert statuses[newer["revision_number"]] == "current", statuses


def test_AC08_equal_business_time_record_stays_behind_correction(store):
    """An equal-business-time record is not "genuinely newer" (AC-08 same-time rule)."""
    ids = _seed_corrected_claim(store)
    late = _late_sibling(
        store,
        ids,
        when=T2["value"],
        native_id="message:fix3-ac08-equal-time",
    )
    current = get_current(store, ids["claim_id"])
    assert current["selected_revision_ids"] == [ids["correction_revision"]], current
    assert current["projection_generation"] == ids["generation_after"], current
    assert late["revision_id"] not in current["selected_revision_ids"]


def test_AC08_same_business_time_replay_still_stays_non_current(store):
    """Round-3 protection preserved: exact/same-time replays never re-open C_old."""
    ids = _seed_corrected_claim(store)
    late = record(
        store,
        display_text=OLD_DB_TEXT,
        value="SQLite",
        predicate="project.db",
        event_time=T1,
        evidence_event=ids["old_event"],
    )
    current = get_current(store, ids["claim_id"])
    assert current["selected_revision_ids"] == [ids["correction_revision"]], current
    assert current["projection_generation"] == ids["generation_after"], current
    assert late["revision_id"] not in current["selected_revision_ids"]


def test_AC08_correction_keeps_original_age_and_projection_rebuilds_deterministically(store):
    """The fact's age stays the original business time; only ordering uses the
    correction's effective time, and a rebuild yields the same answer."""
    ids = _seed_corrected_claim(store)
    late = _late_sibling(
        store, ids, when="2026-03-01T00:00:00Z", native_id="message:fix3-ac08-age"
    )
    with store.read() as conn:
        row = conn.execute(
            "SELECT event_time_value, event_time_precision, correction_of, assertion_kind "
            "FROM claim_revisions WHERE revision_id = ?",
            (ids["correction_revision"],),
        ).fetchone()
    assert row["event_time_value"] == T1["value"], dict(row)
    assert row["event_time_precision"] == "instant", dict(row)
    assert row["assertion_kind"] == "explicit_correction", dict(row)
    assert row["correction_of"] == ids["old_revision"], dict(row)

    rebuild_projection(store)
    rebuilt = get_current(store, ids["claim_id"])
    assert rebuilt["selected_revision_ids"] == [ids["correction_revision"]], rebuilt
    assert rebuilt["projection_generation"] == ids["generation_after"], rebuilt
    assert late["revision_id"] not in rebuilt["selected_revision_ids"]
