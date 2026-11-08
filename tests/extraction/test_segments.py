"""Synthetic tests for bounded, checkpointed long-group extraction segments."""

from __future__ import annotations

import hashlib
import json

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.extraction.proposals import ExtractionError, SanitizedPart
from knowledge_weaver.extraction.segments import (
    SEGMENT_CACHE_TABLE,
    CheckpointingExtractor,
)


def _event_id(label: str) -> str:
    return "evt:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _part(
    text: str,
    *,
    label: str = "part",
    event_id: str | None = None,
    part_index: int = 0,
) -> SanitizedPart:
    return SanitizedPart(
        event_id=event_id or _event_id(label),
        event_revision=1,
        part_index=part_index,
        text=text,
        body_sha256=contracts.body_digest(text),
        role="user",
        origin_kind="user_input",
        actor_id="actor:synthetic",
        event_time={"value": "2026-11-05T09:00:00Z", "precision": "instant", "timezone": "UTC"},
        project_ids=("project:synthetic",),
        workspace_id="owner:synthetic",
    )


def _context(**overrides) -> dict:
    base = {
        "event_id": _event_id("context"),
        "event_revision": 1,
        "project_ids": ["project:synthetic"],
        "model_version": "model:test-a",
        "prompt_version": "prompt:test",
        "known_claims": [],
    }
    base.update(overrides)
    return base


def _claim(
    part: SanitizedPart,
    *,
    predicate: str = "fact.synthetic",
    value=None,
    display_text: str = "合成提取",
    quote: str | None = None,
) -> dict:
    return {
        "subject_hint": "合成主题",
        "project_hint": "project:synthetic",
        "qualifiers": {},
        "predicate": predicate,
        "facet": "reported",
        "value": {} if value is None else value,
        "display_text": display_text,
        "assertion_kind": "inference",
        "evidence": [
            {
                "event_id": part.event_id,
                "event_revision": part.event_revision,
                "part_index": part.part_index,
                "support_quote": part.text if quote is None else quote,
            }
        ],
        "target_claim_hint": None,
        "effective_time_quote": None,
    }


def _proposal(claims=(), *, excluded=(), warnings=()) -> dict:
    return {
        "schema_version": "1.0",
        "claims": list(claims),
        "excluded_event_ids": list(excluded),
        "warnings": list(warnings),
    }


class _StubExtractor:
    def __init__(self, responder):
        self.responder = responder
        self.calls: list[tuple[list[SanitizedPart], dict]] = []

    def extract(self, parts, context):
        self.calls.append((list(parts), context))
        return self.responder(list(parts), context, len(self.calls))


class _OutputLengthDiagnostic(Exception):
    category = "output_length"


def _result_rows(store) -> int:
    with store.read() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS c FROM {SEGMENT_CACHE_TABLE} WHERE kind = 'result'"
        ).fetchone()
    return int(row["c"])


def _kind_rows(store) -> dict[str, int]:
    with store.read() as conn:
        rows = conn.execute(
            f"SELECT kind, COUNT(*) AS c FROM {SEGMENT_CACHE_TABLE} GROUP BY kind"
        ).fetchall()
    return {row["kind"]: int(row["c"]) for row in rows}


def test_utf8_windows_cover_chinese_emoji_and_tail(store):
    text = "松塔项目😀" * 240
    part = _part(text, label="chinese-emoji")
    seen: list[list[dict]] = []

    def responder(parts, context, call):
        seen.append(context["segment_ranges"])
        return _proposal(excluded=[part.event_id])

    inner = _StubExtractor(responder)
    wrapper = CheckpointingExtractor(store, inner, max_bytes=256, overlap_bytes=32)
    result = wrapper.extract([part], _context())

    assert contracts.validate_extraction_output(result) == []
    assert result["excluded_event_ids"] == [part.event_id]
    assert len(inner.calls) > 1

    data = text.encode("utf-8")
    total = len(data)
    coverage = wrapper.coverage
    assert coverage["total_bytes"] == total
    assert coverage["covered_bytes"] == total
    assert coverage["complete"] is True
    assert coverage["delegated_unchanged"] is False
    assert "😀" not in json.dumps(coverage, ensure_ascii=False)

    windows = sorted(coverage["windows"], key=lambda item: item["start_byte"])
    assert windows[0]["start_byte"] == 0
    assert windows[-1]["end_byte"] == total
    cursor = 0
    for item in windows:
        assert 0 <= item["start_byte"] < item["end_byte"] <= total
        assert item["bytes"] <= 256
        assert data[item["start_byte"]:item["end_byte"]].decode("utf-8")
        assert item["start_byte"] <= cursor
        cursor = max(cursor, item["end_byte"])
    assert cursor == total

    assert seen and seen[-1][-1]["end_byte"] == total
    for ranges in seen:
        for item in ranges:
            assert item["event_id"] == part.event_id
            assert 0 <= item["start_byte"] < item["end_byte"] <= total


def test_middle_failure_resumes_and_skips_cached_successes(store):
    parts = [
        _part("甲" * 90, label="one"),
        _part("乙" * 90, label="two"),
        _part("丙" * 90, label="three"),
    ]
    predicates = {"甲": "fact.one", "乙": "fact.two", "丙": "fact.three"}
    state = {"failed": False}

    def responder(parts, context, call):
        key = parts[0].text[0]
        if key == "乙" and not state["failed"]:
            state["failed"] = True
            raise ExtractionError("MODEL_UNAVAILABLE", "synthetic transport interrupted", retryable=True)
        return _proposal([_claim(parts[0], predicate=predicates[key])])

    inner = _StubExtractor(responder)
    wrapper = CheckpointingExtractor(store, inner, max_bytes=300, overlap_bytes=32)
    context = _context()

    with pytest.raises(ExtractionError) as interrupted:
        wrapper.extract(parts, context)
    assert interrupted.value.retryable is True
    assert _result_rows(store) == 1

    result = wrapper.extract(parts, context)
    assert contracts.validate_extraction_output(result) == []
    assert [claim["predicate"] for claim in result["claims"]] == [
        "fact.one",
        "fact.two",
        "fact.three",
    ]
    assert _result_rows(store) == 3

    event_calls = [call[0][0].event_id for call in inner.calls]
    assert event_calls.count(parts[0].event_id) == 1
    assert event_calls.count(parts[1].event_id) == 2
    assert event_calls.count(parts[2].event_id) == 1


def test_output_length_bisection_resumes_persisted_tree(store):
    part_a = _part("A" * 1000, label="long-a")
    part_b = _part("B" * 100, label="small-b")
    state = {"transport_failed": False}

    def responder(parts, context, call):
        window = parts[0]
        item = context["segment_ranges"][0]
        size = item["end_byte"] - item["start_byte"]
        if window.event_id == part_a.event_id and size > 512:
            raise _OutputLengthDiagnostic("synthetic output length diagnostic")
        if (
            window.event_id == part_a.event_id
            and item["start_byte"] > 0
            and not state["transport_failed"]
        ):
            state["transport_failed"] = True
            raise ExtractionError("MODEL_UNAVAILABLE", "synthetic transport interrupted", retryable=True)
        if window.event_id == part_a.event_id:
            predicate = "fact.a.left" if item["start_byte"] == 0 else "fact.a.right"
        else:
            predicate = "fact.b"
        return _proposal([_claim(window, predicate=predicate)])

    inner = _StubExtractor(responder)
    wrapper = CheckpointingExtractor(store, inner, max_bytes=1024, overlap_bytes=128)
    context = _context()

    with pytest.raises(ExtractionError) as interrupted:
        wrapper.extract([part_a, part_b], context)
    assert interrupted.value.retryable is True
    assert _kind_rows(store) == {"result": 1, "split": 1}

    result = wrapper.extract([part_a, part_b], context)
    assert contracts.validate_extraction_output(result) == []
    assert [claim["predicate"] for claim in result["claims"]] == [
        "fact.a.left",
        "fact.a.right",
        "fact.b",
    ]

    parent_calls = [
        call
        for call in inner.calls
        if call[0][0].event_id == part_a.event_id
        and call[1]["segment_ranges"][0]["end_byte"]
        - call[1]["segment_ranges"][0]["start_byte"]
        == 1000
    ]
    assert len(parent_calls) == 1
    left_calls = [
        call
        for call in inner.calls
        if call[0][0].event_id == part_a.event_id
        and call[1]["segment_ranges"][0]["start_byte"] == 0
        and call[1]["segment_ranges"][0]["end_byte"] == 500
    ]
    assert len(left_calls) == 1


def test_cache_context_and_version_invalidate(store):
    parts = [_part("a" * 300, label="cache-a"), _part("b" * 300, label="cache-b")]

    def responder(parts, context, call):
        return _proposal([_claim(parts[0], predicate=f"fact.{parts[0].text[0]}")])

    inner = _StubExtractor(responder)
    context = _context()
    wrapper = CheckpointingExtractor(store, inner, max_bytes=350, overlap_bytes=32)
    first = wrapper.extract(parts, context)
    assert [claim["predicate"] for claim in first["claims"]] == ["fact.a", "fact.b"]
    assert len(inner.calls) == 2

    cached_inner = _StubExtractor(responder)
    cached_wrapper = CheckpointingExtractor(store, cached_inner, max_bytes=350, overlap_bytes=32)
    cached = cached_wrapper.extract(parts, context)
    assert cached_inner.calls == []
    assert [claim["predicate"] for claim in cached["claims"]] == ["fact.a", "fact.b"]

    changed_inner = _StubExtractor(responder)
    changed = CheckpointingExtractor(store, changed_inner, max_bytes=350, overlap_bytes=32)
    changed_context = {**context, "model_version": "model:test-b"}
    changed.extract(parts, changed_context)
    assert len(changed_inner.calls) == 2

    previous_inner = _StubExtractor(responder)
    previous = CheckpointingExtractor(store, previous_inner, max_bytes=350, overlap_bytes=32)
    previous_context = {**context, "previous_validation_error": "synthetic previous failure"}
    previous.extract(parts, previous_context)
    assert len(previous_inner.calls) == 2

    claims_inner = _StubExtractor(responder)
    claims = CheckpointingExtractor(store, claims_inner, max_bytes=350, overlap_bytes=32)
    claims_context = {**context, "known_claims": [{"claim_id": "claim:synthetic"}]}
    claims.extract(parts, claims_context)
    assert len(claims_inner.calls) == 2


def test_secret_proposal_is_not_cached_and_is_retried(store):
    parts = [_part("a" * 300, label="secret-a"), _part("b" * 300, label="secret-b")]
    secret = "sk-" + "A" * 24
    state = {"leak": True}

    def responder(parts, context, call):
        if state["leak"]:
            state["leak"] = False
            return _proposal(
                [
                    _claim(
                        parts[0],
                        predicate="fact.secret",
                        display_text=f"合成提取 {secret}",
                    )
                ]
            )
        return _proposal([_claim(parts[0], predicate=f"fact.{parts[0].text[0]}")])

    inner = _StubExtractor(responder)
    wrapper = CheckpointingExtractor(store, inner, max_bytes=350, overlap_bytes=32)
    context = _context()

    with pytest.raises(ExtractionError) as rejected:
        wrapper.extract(parts, context)
    assert rejected.value.code == "MODEL_UNAVAILABLE"
    assert rejected.value.retryable is True
    assert rejected.value.message == "segment proposal contains secret content"
    assert secret not in rejected.value.message
    assert _kind_rows(store) == {}

    result = wrapper.extract(parts, context)
    assert [claim["predicate"] for claim in result["claims"]] == ["fact.a", "fact.b"]
    assert _result_rows(store) == 2
    assert len(inner.calls) == 3
    assert sum(1 for call in inner.calls if call[0][0].event_id == parts[0].event_id) == 2
    with store.read() as conn:
        payloads = [row["payload_json"] for row in conn.execute(
            f"SELECT payload_json FROM {SEGMENT_CACHE_TABLE}"
        )]
    assert payloads
    for payload in payloads:
        assert secret not in payload
        assert contracts.secret_kinds(payload) == []


def test_warnings_are_preserved_when_merging(store):
    parts = [_part("a" * 300, label="warn-a"), _part("b" * 300, label="warn-b")]

    def responder(parts, context, call):
        return _proposal(
            excluded=[parts[0].event_id],
            warnings=[f"synthetic warning:{parts[0].text[0]}"],
        )

    inner = _StubExtractor(responder)
    wrapper = CheckpointingExtractor(store, inner, max_bytes=350, overlap_bytes=32)
    result = wrapper.extract(parts, _context())

    assert result["claims"] == []
    assert result["warnings"] == ["synthetic warning:a", "synthetic warning:b"]
    assert result["excluded_event_ids"] == sorted(
        [parts[0].event_id, parts[1].event_id]
    )
    assert contracts.validate_extraction_output(result) == []
    assert _result_rows(store) == 2


def test_mixed_exclusion_and_claims_does_not_falsely_exclude(store):
    event_a = _event_id("mixed-a")
    event_b = _event_id("mixed-b")
    part_a0 = _part("a" * 200, event_id=event_a, part_index=0, label="mixed-a-0")
    part_a1 = _part("b" * 200, event_id=event_a, part_index=1, label="mixed-a-1")
    part_b = _part("c" * 200, event_id=event_b, part_index=0, label="mixed-b")

    def responder(parts, context, call):
        window = parts[0]
        if window.event_id == event_a and window.part_index == 0:
            return _proposal([_claim(window, predicate="fact.mixed.a")])
        if window.event_id == event_a:
            return _proposal(excluded=[event_a])
        return _proposal(excluded=[event_b])

    inner = _StubExtractor(responder)
    wrapper = CheckpointingExtractor(store, inner, max_bytes=250, overlap_bytes=32)
    result = wrapper.extract([part_a0, part_a1, part_b], _context())

    assert contracts.validate_extraction_output(result) == []
    assert [claim["predicate"] for claim in result["claims"]] == ["fact.mixed.a"]
    assert event_a not in result["excluded_event_ids"]
    assert result["excluded_event_ids"] == [event_b]


def test_call_budget_stops_safely_and_resumes(store):
    parts = [_part("a" * 200, label=f"budget-{index}") for index in range(4)]
    predicates = {part.event_id: f"fact.budget{index}" for index, part in enumerate(parts)}

    def responder(parts, context, call):
        return _proposal([_claim(parts[0], predicate=predicates[parts[0].event_id])])

    inner = _StubExtractor(responder)
    wrapper = CheckpointingExtractor(
        store, inner, max_bytes=250, overlap_bytes=32, max_calls=2
    )
    context = _context()

    with pytest.raises(ExtractionError) as stopped:
        wrapper.extract(parts, context)
    assert stopped.value.code == "MODEL_UNAVAILABLE"
    assert stopped.value.retryable is True
    assert stopped.value.message == "segment extraction call budget exhausted"
    assert _result_rows(store) == 2
    assert len(inner.calls) == 2

    resumed = CheckpointingExtractor(
        store, inner, max_bytes=250, overlap_bytes=32, max_calls=64
    )
    result = resumed.extract(parts, context)
    assert contracts.validate_extraction_output(result) == []
    assert [claim["predicate"] for claim in result["claims"]] == [
        "fact.budget0",
        "fact.budget1",
        "fact.budget2",
        "fact.budget3",
    ]
    assert len(inner.calls) == 4


def test_short_group_delegates_unchanged_once(store):
    part = _part("a" * 100, label="short")
    context = _context()
    proposal = _proposal([_claim(part, predicate="fact.short")])
    inner = _StubExtractor(lambda parts, context, call: proposal)
    wrapper = CheckpointingExtractor(store, inner, max_bytes=200, overlap_bytes=32)

    result = wrapper.extract([part], context)
    assert result is proposal
    assert len(inner.calls) == 1
    assert inner.calls[0][1] is context
    assert wrapper.coverage["complete"] is True
    assert wrapper.coverage["delegated_unchanged"] is True


def test_zero_claim_leaf_without_exclusion_is_rejected(store):
    parts = [_part("a" * 300, label="silent-a"), _part("b" * 300, label="silent-b")]
    inner = _StubExtractor(lambda parts, context, call: _proposal())
    wrapper = CheckpointingExtractor(store, inner, max_bytes=350, overlap_bytes=32)

    with pytest.raises(ExtractionError) as rejected:
        wrapper.extract(parts, _context())
    assert rejected.value.retryable is True
    assert rejected.value.message == "segment extraction leaf has no claims or explicit exclusions"
    assert _kind_rows(store) == {}


def test_output_length_bisection_respects_max_depth(store):
    part_a = _part("A" * 1000, label="depth-a")
    part_b = _part("B" * 100, label="depth-b")

    def responder(parts, context, call):
        if parts[0].event_id == part_a.event_id:
            raise _OutputLengthDiagnostic("synthetic output length diagnostic")
        return _proposal([_claim(parts[0], predicate="fact.depth.b")])

    inner = _StubExtractor(responder)
    wrapper = CheckpointingExtractor(
        store, inner, max_bytes=1024, overlap_bytes=128, max_depth=1
    )

    with pytest.raises(ExtractionError) as stopped:
        wrapper.extract([part_a, part_b], _context())
    assert stopped.value.code == "MODEL_UNAVAILABLE"
    assert stopped.value.retryable is True
    assert stopped.value.message == "segment extraction exceeded its output length bound"
    assert _kind_rows(store) == {"split": 1}
