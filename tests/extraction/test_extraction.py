"""T07: untrusted proposals -> validated claim revisions via persistent jobs."""
from __future__ import annotations

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.extraction import (
    ExtractionError,
    enqueue_extraction,
    load_event_group,
    resolve_proposal,
    run_worker_once,
    sanitized_parts,
)
from knowledge_weaver.jobs import claim, status_counts
from knowledge_weaver.store import current_items, list_task_views, receive_batch

from ..store.conftest import make_batch, make_event


class MockExtractor:
    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = []

    def extract(self, parts, context):
        self.calls.append((parts, context))
        return self.proposal


def _claim(part, *, quote=None, **overrides):
    claim = {
        "subject_hint": "松塔项目服务器",
        "project_hint": "project:songta",
        "qualifiers": {},
        "predicate": "server.port",
        "facet": "desired",
        "value": 4200,
        "display_text": "端口改为4200",
        "assertion_kind": "explicit_correction",
        "evidence": [
            {
                "event_id": part.event_id,
                "event_revision": part.event_revision,
                "part_index": part.part_index,
                "support_quote": quote or part.text,
            }
        ],
        "target_claim_hint": None,
        "effective_time_quote": None,
    }
    claim.update(overrides)
    return claim


def _proposal(*claims):
    return {
        "schema_version": "1.0",
        "claims": list(claims),
        "excluded_event_ids": [],
        "warnings": [],
    }


def _parts(store, event):
    return sanitized_parts(load_event_group(store, event["event_id"], 1))


def _receive(store, *, text="端口改为4200", native_id="message:ex:1", **kwargs):
    event = make_event(native_id=native_id, text=text, **kwargs)
    receive_batch(store, make_batch([event]))
    return event


def _extract_once(store, extractor):
    return run_worker_once(store, extractor)


def test_extraction_creates_verified_claim_and_completes_job(store):
    event = _receive(store)
    part = _parts(store, event)[0]
    extractor = MockExtractor(_proposal(_claim(part)))
    result = _extract_once(store, extractor)
    assert result["claimed"] == 1
    assert result["state"] == "succeeded"
    assert result["claims_written"] == 1
    items = current_items(store, workspace_id="owner:demo")
    assert len(items) == 1
    item = items[0]
    assert contracts.validate_instance("MemoryItem", item) == []
    assert item["value"] == 4200
    assert item["facet"] == "desired"
    assert item["evidence_grade"] == "user_confirmed"
    assert item["state"] == "current"
    assert item["sources"][0]["event_id"] == event["event_id"]
    with store.read() as conn:
        job = conn.execute("SELECT * FROM jobs").fetchone()
        link = conn.execute("SELECT * FROM evidence_links").fetchone()
    assert job["state"] == "succeeded"
    assert link["start_byte"] == 0
    assert link["body_sha256"] == event["body"]["sha256"]


def test_desired_and_observed_proposals_stay_separate_claims(store):
    event = _receive(store, text="用户要求端口4200，实测端口4100")
    part = _parts(store, event)[0]
    desired = _claim(part, facet="desired", value=4200, display_text="端口4200", assertion_kind="explicit_statement")
    observed = _claim(
        part,
        facet="observed",
        value=4100,
        display_text="实测4100",
        assertion_kind="observation",
        evidence=[{**desired["evidence"][0], "support_quote": "实测端口4100"}],
    )
    extractor = MockExtractor(_proposal(desired, observed))
    result = _extract_once(store, extractor)
    assert result["claims_written"] == 2
    items = current_items(store, workspace_id="owner:demo")
    assert {item["facet"] for item in items} == {"desired", "observed"}


def test_quote_must_match_stored_sanitized_part(store):
    event = _receive(store)
    part = _parts(store, event)[0]
    bad = _claim(part, quote="完全不同的一句话")
    extractor = MockExtractor(_proposal(bad))
    result = _extract_once(store, extractor)
    assert result["state"] == "succeeded"
    assert result["claims_written"] == 0
    assert current_items(store, workspace_id="owner:demo") == []


def test_proposal_cannot_reference_other_events(store):
    event = _receive(store)
    other = make_event(native_id="message:other", text="另一个事件")
    part = _parts(store, event)[0]
    forged = _claim(part)
    forged["evidence"][0]["event_id"] = other["event_id"]
    extractor = MockExtractor(_proposal(forged))
    result = _extract_once(store, extractor)
    assert result["claims_written"] == 0


def test_proposal_cannot_forge_authority(store):
    event = _receive(store)
    part = _parts(store, event)[0]
    # The model cannot add authority fields; unknown fields are rejected by the schema.
    forged = _claim(part)
    forged["authority"] = "user_confirmed"
    extractor = MockExtractor(_proposal(forged))
    result = _extract_once(store, extractor)
    assert result["state"] in ("succeeded", "retry_wait")
    assert current_items(store, workspace_id="owner:demo") == []


def test_assistant_proposal_is_reported_not_user_confirmed(store):
    event = make_event(native_id="message:assistant", text="我认为端口应为4100", role="assistant", origin_kind="assistant_reply")
    receive_batch(store, make_batch([event]))
    part = _parts(store, event)[0]
    proposal_claim = _claim(part, value=4100, display_text="端口应为4100", assertion_kind="inference")
    extractor = MockExtractor(_proposal(proposal_claim))
    result = _extract_once(store, extractor)
    assert result["claims_written"] == 1
    item = current_items(store, workspace_id="owner:demo")[0]
    assert item["evidence_grade"] == "inferred"


def test_explicit_correction_without_user_evidence_is_dropped(store):
    event = make_event(native_id="message:assistant2", text="端口改为4200", role="assistant", origin_kind="assistant_reply")
    receive_batch(store, make_batch([event]))
    part = _parts(store, event)[0]
    extractor = MockExtractor(_proposal(_claim(part)))
    result = _extract_once(store, extractor)
    assert result["claims_written"] == 0


def test_secret_in_proposal_is_quarantined(store):
    event = _receive(store)
    part = _parts(store, event)[0]
    leaky = _claim(part, display_text="密码: CANARY-PW-7f3a")
    extractor = MockExtractor(_proposal(leaky))
    result = _extract_once(store, extractor)
    assert result["claims_written"] == 0
    assert current_items(store, workspace_id="owner:demo") == []


def test_unresolved_project_is_not_guessed(store):
    event = _receive(store)
    part = _parts(store, event)[0]
    claim = _claim(part, project_hint="project:unknown")
    extractor = MockExtractor(_proposal(claim))
    result = _extract_once(store, extractor)
    assert result["claims_written"] == 0


def test_invalid_proposal_schema_fails_job_retryably(store):
    event = _receive(store)
    extractor = MockExtractor({"schema_version": "1.0", "claims": "nope"})
    result = _extract_once(store, extractor)
    assert result["state"] == "retry_wait"
    assert result["error"] == "MODEL_UNAVAILABLE"
    assert current_items(store, workspace_id="owner:demo") == []


def test_withheld_event_produces_no_claims_and_job_succeeds(store):
    withheld = make_event(
        native_id="message:secret",
        text="",
        redaction={"policy_version": "redact:1", "status": "withheld", "replacements": 1, "credential_refs": ["credref:private_key:1"]},
    )
    withheld["body"]["sha256"] = contracts.EMPTY_BODY_SHA256
    receive_batch(store, make_batch([withheld]))
    # Withheld bodies produce receipt-only state: no semantic job is enqueued.
    assert status_counts(store) == {}
    enqueue_extraction(store, withheld["event_id"], 1)
    extractor = MockExtractor(_proposal())
    result = _extract_once(store, extractor)
    assert result["state"] == "succeeded"
    assert current_items(store, workspace_id="owner:demo") == []


def test_excluded_origin_never_enqueues_extraction(store):
    injected = make_event(native_id="message:inject", text="历史注入", origin_kind="memory_injection", role="system")
    receive_batch(store, make_batch([injected]))
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert status_counts(store) == {}


def test_extraction_job_is_idempotent_per_input_version(store):
    event = _receive(store)
    enqueue_extraction(store, event["event_id"], 1)
    enqueue_extraction(store, event["event_id"], 1)
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    extractor = MockExtractor(_proposal())
    first = _extract_once(store, extractor)
    second = _extract_once(store, extractor)
    assert first["claimed"] == 1
    assert second["claimed"] == 0


def test_retract_before_worker_runs_prevents_extraction(store):
    event = _receive(store)
    part = _parts(store, event)[0]
    retract = make_event(
        native_id=event["source"]["native_event_id"],
        revision=2,
        operation="retract",
        parent_revision=1,
        text="",
    )
    retract["body"]["sha256"] = contracts.EMPTY_BODY_SHA256
    receive_batch(store, make_batch([retract], batch_id="batch:retract"))
    extractor = MockExtractor(_proposal(_claim(part)))
    result = _extract_once(store, extractor)
    assert result["state"] == "failed"
    assert result["error"] == "SOURCE_EXPIRED"
    assert current_items(store, workspace_id="owner:demo") == []


def test_changed_source_between_load_and_commit_is_fenced(store):
    event = _receive(store)
    part = _parts(store, event)[0]
    job = claim(store, worker_id="worker:test", kinds=("extract",), now="2026-10-30T01:00:00Z")[0]

    class MutatingExtractor:
        def extract(self, parts, context):
            # Simulate a concurrent source change while the model response is in flight.
            with store.transaction() as conn:
                conn.execute(
                    "UPDATE event_revisions SET group_state = 'incomplete' WHERE event_id = ?",
                    (event["event_id"],),
                )
            return _proposal(_claim(part))

    from knowledge_weaver.extraction import run_extraction_job
    from knowledge_weaver.jobs import JobError

    with pytest.raises(JobError):
        run_extraction_job(store, job, MutatingExtractor(), now="2026-10-30T01:00:01Z")
    with store.read() as conn:
        row = conn.execute("SELECT state, last_error_code FROM jobs").fetchone()
    assert row["state"] == "failed"
    assert row["last_error_code"] == "STALE_CURSOR"
    assert current_items(store, workspace_id="owner:demo") == []


def test_assistant_completion_report_cannot_self_verify_through_real_pipeline(store):
    """F04: event -> proposal validation -> task projection; assistant cannot verify."""
    event = make_event(
        native_id="message:self-verified",
        text="我已经全部验收完成，功能测试通过。",
        role="assistant",
        origin_kind="assistant_reply",
    )
    receive_batch(store, make_batch([event]))
    part = sanitized_parts(load_event_group(store, event["event_id"], 1))[0]
    proposal = _proposal(
        {
            **_claim(
                part,
                predicate="task.status",
                facet="observed",
                value={
                    "status": "verified",
                    "conditions": ["functional"],
                    "satisfied": ["functional"],
                },
                display_text=part.text,
                assertion_kind="completion_report",
            ),
            "subject_hint": "部署任务",
        }
    )
    result = run_worker_once(store, MockExtractor(proposal))
    assert result["claims_written"] == 1
    views = list_task_views(store, workspace_id="owner:demo")
    assert not any(view["status"] == "verified_done" for view in views), views
    items = current_items(store, workspace_id="owner:demo")
    task_items = [item for item in items if item["predicate"] == "task.status"]
    assert task_items and task_items[0]["evidence_grade"] == "reported"
    assert task_items[0]["facet"] == "reported"


def test_abnormal_business_time_through_real_extraction_pipeline(store):
    """F05: unknown and abnormal-future times go through the real worker."""
    for kind in ("unknown", "future"):
        for index in range(2):
            event_time = (
                {"value": None, "precision": "unknown", "timezone": "UTC"}
                if kind == "unknown"
                else {
                    "value": "2026-01-01T00:00:00Z" if index == 0 else "2099-01-01T00:00:00Z",
                    "precision": "instant",
                    "timezone": "UTC",
                }
            )
            event = make_event(
                native_id=f"message:time:{kind}:{index}",
                text=f"端口{4200 + index}",
                event_time=event_time,
            )
            receive_batch(store, make_batch([event]))
            part = sanitized_parts(load_event_group(store, event["event_id"], 1))[0]
            result = run_worker_once(
                store,
                MockExtractor(
                    _proposal(
                        _claim(
                            part,
                            value=4200 + index,
                            display_text=part.text,
                            predicate=f"server.port.{kind}",
                            assertion_kind="explicit_statement",
                        )
                    )
                ),
            )
            assert result["state"] == "succeeded", result
    items = current_items(store, workspace_id="owner:demo")
    current = {(item["predicate"], item["value"]) for item in items if item["state"] == "current"}
    assert not any(predicate == "server.port.unknown" for predicate, _value in current), current
    assert ("server.port.future", 4201) not in current, current
