"""T20 round-3 regressions: F02 (redact after expired/retracted source) and F05
(abnormal-future high authority must not hide a trusted current).

F02: an explicit ``redact`` must purge the derived claim payloads (display text,
value, keyword tokens, vectors) for *every* affected old source lineage, not
only links that are still ``availability='full'``. A source detail that expired
(``expired``) or was retracted (``retracted``) before the redaction used to be
skipped, so the newly classified secret stayed visible in ordinary recall and in
the logical export.

F05: an abnormal far-future business time is a data anomaly. It must be isolated
*before* authority ranking, so a ``user_confirmed`` 2099 record cannot remove a
trusted ``tool_verified`` current from the projection; the anomaly itself stays
visible/explainable as ``uncertain``.

Adjacent invariants covered here: a retract-then-redact chain, availability
truth preserved (expired stays expired, retracted stays retracted, full becomes
withheld), non-secret lineage kept explainable (revision row, authority,
business time, evidence hash/span), unrelated events/claims never scrubbed,
within-tolerance future times keep normal authority semantics and unknown-time
semantics stay unchanged.

Only synthetic canaries, tmp_path databases and the hash test embedder are used;
no runtime mirror path, real HOME data, model download or non-loopback network.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.embeddings.index import (
    HashEmbeddingProvider,
    configure_model,
    enqueue_embed_in_transaction,
    load_vectors,
    run_embed_worker_once,
)
from knowledge_weaver.lifecycle import register_lifecycle, sweep
from knowledge_weaver.portability import export_snapshot
from knowledge_weaver.portability.export import ExportRefused
from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.store import (
    current_items,
    get_current,
    rebuild_projection,
    receive_batch,
)
from knowledge_weaver.store.claims import apply_correction, invalidate_withheld_evidence

from ..retrieval.conftest import recall_request
from ..store.conftest import make_batch, make_event
from .conftest import create_user_event, evidence_ref, record

CANARY = "T20FIX2_NEWLY_CLASSIFIED_CANARY_71823"
WORKSPACE = "owner:demo"
PROJECT = "project:songta"
BINDING = {
    "client_id": "client:test",
    "workspace_id": WORKSPACE,
    "project_ids": [PROJECT],
    "actor_ids": ["*"],
}
EMBEDDING_MODEL = "test:hash-embedding"
EMBEDDING_DIMENSION = 4
EXPIRED_AT = "2026-10-30T00:00:00Z"
SOURCE_EXPIRY_ANCHOR = "2020-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def _reset_embedding_config():
    configure_model(None)
    yield
    configure_model(None)


# ---------------------------------------------------------------------------
# Builders / queries
# ---------------------------------------------------------------------------


def _redaction_event(event, *, revision=2, text="历史配置 [REDACTED:PASSWORD_ASSIGNMENT]"):
    return make_event(
        native_id=event["source"]["native_event_id"],
        revision=revision,
        operation="redact",
        parent_revision=revision - 1,
        text=text,
        content_revision=event["body"]["content_revision"],
        event_time=event["event_time"],
    )


def _retraction_event(event, *, revision=2):
    retract = make_event(
        native_id=event["source"]["native_event_id"],
        revision=revision,
        operation="retract",
        parent_revision=revision - 1,
        text="",
        content_revision=event["body"]["content_revision"],
        event_time=event["event_time"],
    )
    retract["body"]["sha256"] = contracts.EMPTY_BODY_SHA256
    return retract


def _embed(store, revision_id):
    configure_model(EMBEDDING_MODEL, dimension=EMBEDDING_DIMENSION)
    try:
        with store.transaction() as conn:
            enqueue_embed_in_transaction(conn, revision_id)
        result = run_embed_worker_once(
            store,
            HashEmbeddingProvider(
                dimension=EMBEDDING_DIMENSION, model_id=EMBEDDING_MODEL
            ),
        )
        assert result["claimed"] == 1, result
        assert load_vectors(
            store,
            {revision_id},
            model_id=EMBEDDING_MODEL,
            dimension=EMBEDDING_DIMENSION,
        )
    finally:
        configure_model(None)


def _expire_source_part(store, event, *, revision=1):
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#{revision}@0",
        class_="completed_task",
        business_anchor=SOURCE_EXPIRY_ANCHOR,
    )
    return sweep(store, now=EXPIRED_AT)


def _secret_claim(store, native_id, *, embed=True):
    event = create_user_event(store, "历史配置 密码: " + CANARY, native_id=native_id)
    claimed = record(
        store,
        display_text=event["body"]["text"],
        value=CANARY,
        evidence_event=event,
    )
    if embed:
        _embed(store, claimed["revision_id"])
    return event, claimed


def _one(store, sql, params=()):
    with store.read() as conn:
        return conn.execute(sql, params).fetchone()


def _claim_revision(store, revision_id):
    return _one(
        store,
        "SELECT * FROM claim_revisions WHERE revision_id = ?",
        (revision_id,),
    )


def _links(store, revision_id):
    with store.read() as conn:
        return conn.execute(
            "SELECT * FROM evidence_links WHERE revision_id = ? "
            "ORDER BY event_id, event_revision, part_index, start_byte",
            (revision_id,),
        ).fetchall()


def _fts_rows(store, revision_id):
    return _one(
        store,
        "SELECT COUNT(*) AS c FROM claim_fts WHERE revision_id = ?",
        (revision_id,),
    )["c"]


def _vector_rows(store, revision_id):
    return _one(
        store,
        "SELECT COUNT(*) AS c FROM claim_vectors WHERE revision_id = ?",
        (revision_id,),
    )["c"]


def _revision_statuses(store, claim_id):
    with store.read() as conn:
        rows = conn.execute(
            "SELECT revision_number, status FROM claim_revisions WHERE claim_id = ? "
            "ORDER BY revision_number",
            (claim_id,),
        ).fetchall()
    return {row["revision_number"]: row["status"] for row in rows}


def _assert_export_without_canary(store, destination):
    try:
        export_snapshot(store, destination)
    except ExportRefused as exc:  # defence-in-depth refusal also proves a leak
        pytest.fail(
            "logical export still carries the newly classified secret "
            f"({exc.code}): {exc.message}"
        )
    leaks = [
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
        and CANARY in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert leaks == [], leaks


def _recall_json(store, **overrides):
    response = execute_recall(store, recall_request(**overrides), binding=BINDING)
    return response, json.dumps(response, ensure_ascii=False)


def _assert_no_canary_bytes_on_disk(store):
    """SEC-02 at-rest check: secure_delete/checkpoint leaves no secret bytes behind."""
    needle = CANARY.encode("utf-8")
    candidates = [
        store.db_path,
        Path(f"{store.db_path}-wal"),
        Path(f"{store.db_path}-shm"),
    ]
    residue = {
        path.name: needle in path.read_bytes() for path in candidates if path.exists()
    }
    assert not any(residue.values()), residue


def _anomalous_time(value="2099-01-01T00:00:00Z"):
    return {"value": value, "precision": "instant", "timezone": "UTC"}


# ---------------------------------------------------------------------------
# F02: redact after the source detail already expired
# ---------------------------------------------------------------------------


def test_F02_redact_after_source_expired_scrubs_recall_and_export(store, tmp_path):
    """Original Codex counterexample: expiry first, redact second, no residual leak."""
    event, claimed = _secret_claim(store, "message:fix2-f02-expired")
    revision_id = claimed["revision_id"]
    _expire_source_part(store, event)

    # Precondition: the detail is gone, yet the retained conclusion still derives
    # from it and still carries the old payload (retention keeps conclusions).
    assert {link["availability"] for link in _links(store, revision_id)} == {"expired"}
    assert _claim_revision(store, revision_id)["display_text"] == event["body"]["text"]
    assert _vector_rows(store, revision_id) == 1

    result = receive_batch(
        store,
        make_batch([_redaction_event(event)], batch_id="batch:fix2-f02-expired"),
    )
    assert result["receipts"][0]["disposition"] == "accepted", result

    _, payload = _recall_json(store, query="历史配置")
    assert CANARY not in payload, "ordinary recall still leaks the redacted derived payload"
    _assert_export_without_canary(store, tmp_path / "export-f02-expired")


def test_F02_redact_after_source_expired_scrubs_derived_but_keeps_lineage(store):
    """The scrub clears payloads; revision row, authority, time and evidence hashes stay."""
    event, claimed = _secret_claim(store, "message:fix2-f02-lineage")
    revision_id = claimed["revision_id"]
    _expire_source_part(store, event)

    result = receive_batch(
        store,
        make_batch([_redaction_event(event)], batch_id="batch:fix2-f02-lineage"),
    )
    assert result["receipts"][0]["disposition"] == "accepted", result

    row = _claim_revision(store, revision_id)
    assert row["display_text"] == ""
    assert row["value_json"] == "null"
    assert _fts_rows(store, revision_id) == 0
    assert _vector_rows(store, revision_id) == 0

    # Availability truth is preserved: expiry is not rewritten as withheld.
    links = _links(store, revision_id)
    assert {link["availability"] for link in links} == {"expired"}

    # Non-secret lineage is still explainable.
    assert row["claim_id"] == claimed["claim_id"]
    assert row["revision_number"] == 1
    assert row["authority"] == "user_confirmed"
    assert row["assertion_kind"] == "explicit_statement"
    assert row["event_time_value"] == event["event_time"]["value"]
    assert row["event_time_precision"] == "instant"
    link = links[0]
    assert (link["event_id"], link["event_revision"], link["part_index"]) == (
        event["event_id"],
        1,
        0,
    )
    assert link["body_sha256"] == event["body"]["sha256"]
    assert link["start_byte"] == 0
    assert link["end_byte"] == len(event["body"]["text"].encode("utf-8"))

    current = get_current(store, claimed["claim_id"])
    assert current["state"] == "uncertain"
    assert current["selected_revision_ids"] == []

    items = current_items(store, workspace_id=WORKSPACE)
    item = next(item for item in items if item["claim_id"] == claimed["claim_id"])
    assert item["text"] == ""
    assert item["value"] is None
    assert item["state"] == "uncertain"
    assert item["source_detail_expired"] is True
    assert item["sources"][0]["body_sha256"] == event["body"]["sha256"]
    assert item["event_time"]["value"] == event["event_time"]["value"]

    _assert_no_canary_bytes_on_disk(store)


def test_F02_redact_after_retract_scrubs_derived(store, tmp_path):
    """A retracted link is still part of the secret lineage and must be scrubbed."""
    event, claimed = _secret_claim(store, "message:fix2-f02-retracted")
    revision_id = claimed["revision_id"]

    result = receive_batch(
        store,
        make_batch([_retraction_event(event)], batch_id="batch:fix2-f02-retract"),
    )
    assert result["receipts"][0]["disposition"] == "accepted", result
    assert {link["availability"] for link in _links(store, revision_id)} == {"retracted"}

    result = receive_batch(
        store,
        make_batch(
            [_redaction_event(event, revision=3)],
            batch_id="batch:fix2-f02-retract-redact",
        ),
    )
    assert result["receipts"][0]["disposition"] == "accepted", result

    row = _claim_revision(store, revision_id)
    assert row["display_text"] == ""
    assert row["value_json"] == "null"
    assert _fts_rows(store, revision_id) == 0
    assert _vector_rows(store, revision_id) == 0
    # Availability truth is preserved: retracted is not rewritten either.
    assert {link["availability"] for link in _links(store, revision_id)} == {"retracted"}
    # Non-secret lineage survives.
    assert row["claim_id"] == claimed["claim_id"]
    assert row["event_time_value"] == event["event_time"]["value"]

    _, payload = _recall_json(store)
    assert CANARY not in payload
    _assert_export_without_canary(store, tmp_path / "export-f02-retracted")
    _assert_no_canary_bytes_on_disk(store)


def test_F02_redact_does_not_scrub_unrelated_lineage(store):
    """Only old lineages of the redacted event are scrubbed, and the cleanup is idempotent."""
    secret_event, secret_claim = _secret_claim(store, "message:fix2-f02-scope-secret")
    clean_event = create_user_event(store, "当前端口4200", native_id="message:fix2-f02-scope-clean")
    clean_claim = record(
        store,
        display_text="当前端口4200",
        value=4200,
        evidence_event=clean_event,
        subject_id="subject:clean-server",
    )
    _embed(store, clean_claim["revision_id"])

    _expire_source_part(store, secret_event)
    redact = _redaction_event(secret_event)
    result = receive_batch(
        store,
        make_batch([redact], batch_id="batch:fix2-f02-scope"),
    )
    assert result["receipts"][0]["disposition"] == "accepted", result

    # The affected secret lineage is scrubbed.
    assert _claim_revision(store, secret_claim["revision_id"])["display_text"] == ""

    # A claim derived from another event is untouched.
    clean_row = _claim_revision(store, clean_claim["revision_id"])
    assert clean_row["display_text"] == "当前端口4200"
    assert json.loads(clean_row["value_json"]) == 4200
    assert _fts_rows(store, clean_claim["revision_id"]) > 0
    assert _vector_rows(store, clean_claim["revision_id"]) == 1
    assert {link["availability"] for link in _links(store, clean_claim["revision_id"])} == {
        "full"
    }
    assert get_current(store, clean_claim["claim_id"])["state"] == "current"

    # A claim derived from the sanitized redaction revision of the *same* event is
    # not old secret lineage; a repeated cleanup pass must not scrub it.
    post_redaction = record(
        store,
        display_text=redact["body"]["text"],
        value="[REDACTED:PASSWORD_ASSIGNMENT]",
        evidence_event=redact,
        subject_id="subject:post-redaction",
    )
    _embed(store, post_redaction["revision_id"])
    with store.transaction() as conn:
        invalidate_withheld_evidence(
            conn, secret_event["event_id"], redact["event_revision"]
        )

    post_row = _claim_revision(store, post_redaction["revision_id"])
    assert post_row["display_text"] == redact["body"]["text"]
    assert json.loads(post_row["value_json"]) == "[REDACTED:PASSWORD_ASSIGNMENT]"
    assert _fts_rows(store, post_redaction["revision_id"]) > 0
    assert _vector_rows(store, post_redaction["revision_id"]) == 1
    assert {link["availability"] for link in _links(store, post_redaction["revision_id"])} == {
        "full"
    }
    assert get_current(store, post_redaction["claim_id"])["state"] == "current"


# ---------------------------------------------------------------------------
# F05: abnormal-future revisions must not dominate the trusted current
# ---------------------------------------------------------------------------


def test_F05_anomalous_high_authority_does_not_hide_known_current(store):
    """Original Codex counterexample: user_confirmed 2099 must not hide tool_verified current."""
    known = record(
        store,
        display_text="端口4100",
        value=4100,
        authority="tool_verified",
    )
    anomalous = record(
        store,
        display_text="端口4300",
        value=4300,
        authority="user_confirmed",
        event_time=_anomalous_time(),
    )

    current = get_current(store, known["claim_id"])
    assert current["state"] == "current", current
    assert current["selected_revision_ids"] == [known["revision_id"]]

    statuses = _revision_statuses(store, known["claim_id"])
    assert statuses[1] == "current", statuses
    assert statuses[2] == "uncertain", statuses

    items = current_items(store, workspace_id=WORKSPACE)
    assert [item["value"] for item in items if item["state"] == "current"] == [4100], items
    assert not any(item["value"] == 4300 and item["state"] == "current" for item in items)

    # The anomaly stays visible and explainable in history, labelled not-current.
    history = execute_recall(
        store, recall_request(intent="history", query="端口"), binding=BINDING
    )
    anomalous_items = [
        item for item in history["items"] if item["revision_id"] == anomalous["revision_id"]
    ]
    assert len(anomalous_items) == 1, history["items"]
    assert anomalous_items[0]["state"] == "uncertain"
    assert anomalous_items[0]["value"] == 4300
    assert anomalous_items[0]["event_time"] == _anomalous_time()


def test_F05_anomalous_only_candidates_stay_uncertain_without_current(store):
    first = record(
        store,
        display_text="端口4300",
        value=4300,
        authority="user_confirmed",
        event_time=_anomalous_time(),
    )
    current = get_current(store, first["claim_id"])
    assert current["state"] == "uncertain"
    assert current["selected_revision_ids"] == []
    assert _revision_statuses(store, first["claim_id"]) == {1: "uncertain"}
    assert not any(item["state"] == "current" for item in current_items(store, workspace_id=WORKSPACE))

    record(
        store,
        display_text="端口4400",
        value=4400,
        authority="user_confirmed",
        event_time=_anomalous_time("2098-01-01T00:00:00Z"),
    )
    current = get_current(store, first["claim_id"])
    assert current["state"] == "uncertain"
    assert current["selected_revision_ids"] == []
    assert _revision_statuses(store, first["claim_id"]) == {1: "uncertain", 2: "uncertain"}


def test_F05_within_tolerance_future_time_keeps_normal_authority_semantics(store):
    normal = record(
        store,
        display_text="端口4100",
        value=4100,
        authority="tool_verified",
        event_time={
            "value": "2026-01-02T00:00:00Z",
            "precision": "instant",
            "timezone": "UTC",
        },
    )
    soon = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=30)
    planned = record(
        store,
        display_text="端口4200",
        value=4200,
        authority="user_confirmed",
        event_time={
            "value": soon.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "precision": "instant",
            "timezone": "UTC",
        },
    )

    current = get_current(store, normal["claim_id"])
    assert current["state"] == "current", current
    assert current["selected_revision_ids"] == [planned["revision_id"]]
    # Within the tolerance window time ordering is unchanged: the later normal
    # future business time wins, and nothing is isolated as anomalous.
    assert _revision_statuses(store, normal["claim_id"]) == {
        1: "superseded",
        2: "current",
    }


def test_F05_unknown_time_semantics_unchanged(store):
    known = record(
        store,
        display_text="端口4100",
        value=4100,
        authority="tool_verified",
    )
    record(
        store,
        display_text="端口4200（时间未知）",
        value=4200,
        authority="tool_verified",
        event_time={"value": None, "precision": "unknown", "timezone": "UTC"},
    )

    current = get_current(store, known["claim_id"])
    assert current["state"] == "current"
    assert current["selected_revision_ids"] == [known["revision_id"]]
    assert _revision_statuses(store, known["claim_id"]) == {1: "current", 2: "uncertain"}

    solo = record(
        store,
        display_text="备份周期每周（时间未知）",
        value="weekly",
        predicate="backup_interval",
        event_time={"value": None, "precision": "unknown", "timezone": "UTC"},
    )
    solo_current = get_current(store, solo["claim_id"])
    assert solo_current["state"] == "uncertain"
    assert solo_current["selected_revision_ids"] == []


# ---------------------------------------------------------------------------
# AC-08: an explicit user correction outranks late asynchronous extraction
# ---------------------------------------------------------------------------

CORRECTION_T1 = {
    "value": "2026-01-01T00:00:00Z",
    "precision": "instant",
    "timezone": "UTC",
}
LATER_T2 = {
    "value": "2026-06-01T00:00:00Z",
    "precision": "instant",
    "timezone": "UTC",
}
#: Strictly later than the correction evidence time (``LATER_T2``). With the
#: round-4 AC08 fix a correction is ordered by the business time of its own
#: user evidence, so "genuinely newer" must be strictly later than that time;
#: an equal-time record stays behind the correction (correction priority) and is
#: covered by ``test_correction_late_time_fix3.py``.
LATER_T3 = {
    "value": "2026-07-01T00:00:00Z",
    "precision": "instant",
    "timezone": "UTC",
}
OLD_DB_TEXT = "项目使用 SQLite"
NEW_DB_TEXT = "改用本地 PostgreSQL"


def _correction_payload(claim_id, expected_revision, generation, evidence, *, operation_id):
    return {
        "schema_version": "1.0",
        "operation_id": operation_id,
        "claim_id": claim_id,
        "expected_revision_id": expected_revision,
        "expected_projection_generation": generation,
        "new_value": "PostgreSQL",
        "display_text": NEW_DB_TEXT,
        "evidence": [evidence],
        "supersedes_revision_ids": [expected_revision],
        "reason": "用户明确更正",
    }


def _seed_corrected_db_claim(store):
    """C_old rev1@t1 -> user correction rev2 (inherits t1); returns the ids."""
    old_event = make_event(
        native_id="message:fix2-ac08-old", text=OLD_DB_TEXT, event_time=CORRECTION_T1
    )
    receive_batch(store, make_batch([old_event], batch_id="batch:fix2-ac08-old"))
    old = record(
        store,
        display_text=OLD_DB_TEXT,
        value="SQLite",
        predicate="project.db",
        evidence_event=old_event,
    )
    generation_before = get_current(store, old["claim_id"])["projection_generation"]
    new_event = make_event(
        native_id="message:fix2-ac08-new", text=NEW_DB_TEXT, event_time=LATER_T2
    )
    receive_batch(store, make_batch([new_event], batch_id="batch:fix2-ac08-new"))
    applied = apply_correction(
        store,
        _correction_payload(
            old["claim_id"],
            old["revision_id"],
            generation_before,
            evidence_ref(new_event),
            operation_id="op:fix2:ac08:1",
        ),
    )
    assert applied["status"] == "applied", applied
    generation_after = get_current(store, old["claim_id"])["projection_generation"]
    assert generation_after > generation_before
    return {
        "claim_id": old["claim_id"],
        "old_revision": old["revision_id"],
        "old_event": old_event,
        "new_event": new_event,
        "correction_revision": applied["revision_id"],
        "generation_before": generation_before,
        "generation_after": generation_after,
    }


def _late_db_revision(store, ids, *, evidence_event, native_id):
    """A late extraction of the corrected C_old statement (same business time)."""
    if evidence_event is None:
        evidence_event = make_event(
            native_id=native_id, text=OLD_DB_TEXT, event_time=CORRECTION_T1
        )
        receive_batch(
            store, make_batch([evidence_event], batch_id=f"batch:{native_id}")
        )
    return record(
        store,
        display_text=OLD_DB_TEXT,
        value="SQLite",
        predicate="project.db",
        event_time=CORRECTION_T1,
        evidence_event=evidence_event,
    )


def _assert_correction_still_current(store, ids, late):
    current = get_current(store, ids["claim_id"])
    assert current["state"] == "current", current
    assert current["selected_revision_ids"] == [ids["correction_revision"]], current
    # A late extraction must not bump the projection generation either.
    assert current["projection_generation"] == ids["generation_after"], current

    statuses = _revision_statuses(store, ids["claim_id"])
    assert statuses == {
        1: "superseded",
        2: "current",
        3: "superseded",
    }, statuses

    items = [item for item in current_items(store, workspace_id=WORKSPACE) if item["claim_id"] == ids["claim_id"]]
    assert [(item["value"], item["state"]) for item in items] == [("PostgreSQL", "current")], items

    # History stays explainable through each revision's own text. The history
    # intent keeps its existing contract (only query-matched revisions are
    # returned): C_old and the late replay are visible as superseded, C_new as
    # current, and every revision row is retained.
    old_history = execute_recall(
        store, recall_request(query="SQLite", intent="history"), binding=BINDING
    )
    old_states = {item["revision_id"]: item["state"] for item in old_history["items"]}
    assert old_states.get(ids["old_revision"]) == "superseded", old_history["items"]
    assert old_states.get(late["revision_id"]) == "superseded", old_history["items"]
    assert all(item["state"] != "current" for item in old_history["items"]), old_history["items"]
    new_history = execute_recall(
        store, recall_request(query="PostgreSQL", intent="history"), binding=BINDING
    )
    new_states = {item["revision_id"]: item["state"] for item in new_history["items"]}
    assert new_states.get(ids["correction_revision"]) == "current", new_history["items"]
    with store.read() as conn:
        retained = {
            row["revision_id"]
            for row in conn.execute(
                "SELECT revision_id FROM claim_revisions WHERE claim_id = ?",
                (ids["claim_id"],),
            ).fetchall()
        }
    assert retained == {
        ids["old_revision"],
        ids["correction_revision"],
        late["revision_id"],
    }, retained

    # The projection is rebuildable to the same answer (deterministic, no clock).
    rebuild_projection(store)
    rebuilt = get_current(store, ids["claim_id"])
    assert rebuilt["selected_revision_ids"] == [ids["correction_revision"]], rebuilt
    assert rebuilt["projection_generation"] == ids["generation_after"], rebuilt


def test_AC08_correction_dominates_late_replay_of_corrected_evidence(store):
    """A re-extraction replaying the corrected evidence span cannot re-open C_old."""
    ids = _seed_corrected_db_claim(store)
    late = _late_db_revision(
        store, ids, evidence_event=ids["old_event"], native_id="message:fix2-ac08-replay"
    )
    _assert_correction_still_current(store, ids, late)


def test_AC08_correction_dominates_late_extraction_from_another_source(store):
    """Old C_old knowledge arriving from a sibling source at the same time is stale."""
    ids = _seed_corrected_db_claim(store)
    late = _late_db_revision(
        store, ids, evidence_event=None, native_id="message:fix2-ac08-sibling"
    )
    _assert_correction_still_current(store, ids, late)


def test_AC08_correction_reusing_the_corrected_span_is_not_excluded(store):
    """The correction itself keeps competing even when it reuses the old span."""
    old_text = "端口先用4100，改为4200"
    old_event = make_event(
        native_id="message:fix2-ac08-span", text=old_text, event_time=CORRECTION_T1
    )
    receive_batch(store, make_batch([old_event], batch_id="batch:fix2-ac08-span"))
    old = record(
        store,
        display_text="端口4100",
        value=4100,
        predicate="server.port",
        evidence_event=old_event,
    )
    generation = get_current(store, old["claim_id"])["projection_generation"]
    applied = apply_correction(
        store,
        {
            "schema_version": "1.0",
            "operation_id": "op:fix2:ac08:span",
            "claim_id": old["claim_id"],
            "expected_revision_id": old["revision_id"],
            "expected_projection_generation": generation,
            "new_value": 4200,
            "display_text": "端口4200",
            "evidence": [evidence_ref(old_event)],
            "supersedes_revision_ids": [old["revision_id"]],
            "reason": "用户明确更正",
        },
    )
    assert applied["status"] == "applied", applied
    current = get_current(store, old["claim_id"])
    assert current["state"] == "current", current
    assert current["selected_revision_ids"] == [applied["revision_id"]], current

    # A replay of the same span after the correction is still non-current.
    late = record(
        store,
        display_text="端口4100",
        value=4100,
        predicate="server.port",
        event_time=CORRECTION_T1,
        evidence_event=old_event,
    )
    current = get_current(store, old["claim_id"])
    assert current["selected_revision_ids"] == [applied["revision_id"]], current
    assert _revision_statuses(store, old["claim_id"])[late["revision_number"]] == "superseded"


def test_AC08_new_user_evidence_after_correction_can_supersede_again(store):
    """Only genuinely newer native user evidence may replace the corrected truth.

    Round-4 fixture note: the newer native record is strictly *later* than the
    correction's own evidence business time (``LATER_T3`` > ``LATER_T2``). An
    equal-business-time record is not "genuinely newer"; the correction keeps
    priority there (AC-08 same-time rule), which the round-4 regression module
    ``test_correction_late_time_fix3.py`` asserts explicitly.
    """
    ids = _seed_corrected_db_claim(store)
    newer_event = make_event(
        native_id="message:fix2-ac08-newer",
        text="数据库最终确认改用 MySQL",
        event_time=LATER_T3,
    )
    receive_batch(store, make_batch([newer_event], batch_id="batch:fix2-ac08-newer"))
    newer = record(
        store,
        display_text="数据库最终确认改用 MySQL",
        value="MySQL",
        predicate="project.db",
        event_time=LATER_T3,
        evidence_event=newer_event,
    )
    current = get_current(store, ids["claim_id"])
    assert current["state"] == "current", current
    assert current["selected_revision_ids"] == [newer["revision_id"]], current
    assert current["projection_generation"] > ids["generation_after"], current
    statuses = _revision_statuses(store, ids["claim_id"])
    assert statuses[2] == "superseded", statuses
    assert statuses[newer["revision_number"]] == "current", statuses
