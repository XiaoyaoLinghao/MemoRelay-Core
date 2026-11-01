"""T01: ID-01 identity, digests, retract and part-group semantics."""
from __future__ import annotations

import copy
import json

import pytest

from knowledge_weaver import contracts as c

BASE = {
    "schema_version": "1.0",
    "event_id": c.canonical_event_id("owner:demo", "instance:hermes-test", "message:17"),
    "event_revision": 1,
    "workspace_id": "owner:demo",
    "operation": "create",
    "source": {
        "instance_id": "instance:hermes-test",
        "host": "hermes",
        "actor_id": "jun-test",
        "session_id": "session:test",
        "native_event_id": "message:17",
        "native_schema": "hermes-snapshot-20261030",
        "role": "user",
        "origin_kind": "user_input",
        "locator": "hermes://session:test/message:17",
    },
    "event_time": {"value": "2026-10-30T09:00:00+08:00", "precision": "instant", "timezone": "Asia/Shanghai"},
    "body": {
        "text": "松塔项目端口改为4200。",
        "sha256": c.body_digest("松塔项目端口改为4200。"),
        "part_index": 0,
        "part_count": 1,
        "content_revision": "native:1",
        "format": "text/plain",
    },
    "redaction": {"policy_version": "redact:1", "status": "passed", "replacements": 0, "credential_refs": []},
    "project_ids": ["project:songta"],
    "parent_revision": None,
    "derived_from": [],
}


def event(**overrides):
    value = copy.deepcopy(BASE)
    b_patch = overrides.pop("body", None)
    if b_patch:
        value["body"].update(b_patch)
    value.update(overrides)
    return value


def test_manifest_event_id_matches_baseline_contract():
    # Baseline sample in docs/baseline-20261030/contracts/valid-examples.json
    assert BASE["event_id"] == "evt:4ccc6ba8b5f8693f4cdb5421453404b7da8c44f94c2a99c6de2638506abb3005"


def test_identity_is_workspace_instance_native_namespace():
    base = c.canonical_event_id("owner:demo", "instance:hermes-test", "message:17")
    assert c.canonical_event_id("owner:other", "instance:hermes-test", "message:17") != base
    assert c.canonical_event_id("owner:demo", "instance:other", "message:17") != base
    assert c.canonical_event_id("owner:demo", "instance:hermes-test", "message:18") != base
    # Same native id on two instances must not collide globally (AC-25).
    a = c.canonical_event_id("owner:demo", "instance:a", "message:17")
    b = c.canonical_event_id("owner:demo", "instance:b", "message:17")
    assert a != b
    assert c.EVENT_ID_RE.match(a)


def test_canonical_json_is_compact_and_unescaped():
    payload = c.compact_json(["owner:demo", "实例:1", "消息:17"])
    assert payload == '["owner:demo","实例:1","消息:17"]'
    assert " " not in payload
    expected = "evt:" + c.sha256_hex(payload)
    assert c.canonical_event_id("owner:demo", "实例:1", "消息:17") == expected


def test_backend_recomputes_and_rejects_forged_id():
    forged = event(event_id="evt:" + "0" * 64)
    issues = c.validate_event(forged)
    assert any("ID-01" in issue.message for issue in issues)


def test_bare_native_id_is_not_a_valid_event_id():
    issues = c.validate_instance("Event", event(event_id="evt:message-17"))
    assert issues


def test_body_digest_is_after_redaction():
    original = "API key sk-synthetic-1234"
    redacted = "API key [REDACTED:API_KEY]"
    assert c.body_digest(redacted) != c.body_digest(original)
    issues = c.validate_event_semantics(event(body={"sha256": c.body_digest(original)}))
    assert any("body digest" in issue.message for issue in issues)


def test_utf8_byte_spans_not_character_offsets():
    text = "端口改为4200"
    span = c.utf8_span(text, "4200")
    assert span == (len("端口改为".encode("utf-8")), len("端口改为4200".encode("utf-8")))
    assert c.span_matches_hash(text, span[0], span[1], c.body_digest("4200"))
    # Cutting through a multi-byte character must never match.
    assert not c.span_matches_hash(text, 1, 4, c.body_digest(text[0:1]))


def test_utf8_span_missing_quote_returns_none():
    assert c.utf8_span("abc", "zzz") is None
    assert c.utf8_span("abc", "") is None


def test_create_must_be_first_revision_without_parent():
    assert c.validate_event_semantics(event(event_revision=2, parent_revision=1))
    assert c.validate_event_semantics(event(parent_revision=1))


def test_edit_requires_next_revision_of_a_parent():
    edited = event(event_revision=2, parent_revision=1, operation="edit")
    assert c.validate_event_semantics(edited) == []
    assert c.validate_event_semantics(event(event_revision=3, parent_revision=1, operation="edit"))
    assert c.validate_event_semantics(event(event_revision=2, parent_revision=None, operation="edit"))


def test_retract_is_one_empty_part_targeting_parent():
    retract = event(
        operation="retract",
        event_revision=2,
        parent_revision=1,
        body={"text": "", "sha256": c.EMPTY_BODY_SHA256, "part_index": 0, "part_count": 1},
    )
    assert c.validate_event(retract) == []
    bad = copy.deepcopy(retract)
    bad["body"]["text"] = "retract please"
    bad["body"]["sha256"] = c.body_digest("retract please")
    assert c.validate_event_semantics(bad)
    bad2 = copy.deepcopy(retract)
    bad2["body"]["part_count"] = 2
    assert c.validate_event_semantics(bad2)


def test_withheld_body_must_not_carry_content():
    withheld = event(redaction={"policy_version": "redact:1", "status": "withheld", "replacements": 0, "credential_refs": []})
    assert c.validate_event_semantics(withheld)
    clean = copy.deepcopy(withheld)
    clean["body"].update({"text": "", "sha256": c.EMPTY_BODY_SHA256})
    assert c.validate_event_semantics(clean) == []


def test_unsupported_redaction_policy_and_schema_version():
    assert c.validate_event_semantics(event(redaction={"policy_version": "redact:9", "status": "passed", "replacements": 0, "credential_refs": []}))
    assert c.validate_event_semantics(event(schema_version="99.0"))


def test_part_size_limits_are_not_truncated():
    too_many_chars = "a" * (c.MAX_PART_CHARS + 1)
    issues = c.validate_event_semantics(event(body={"text": too_many_chars, "sha256": c.body_digest(too_many_chars)}))
    assert any("characters" in issue.message for issue in issues)
    # The UTF-8 byte cap is a transport backstop; with the 250k-character cap
    # it cannot be reached by well-formed text, so exercise it with a tightened
    # limit to prove the guard is live and never silently truncates.
    import pytest

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(c, "MAX_PART_BYTES", 10)
        cjk = "中" * 4  # 12 UTF-8 bytes
        issues = c.validate_event_semantics(
            event(body={"text": cjk, "sha256": c.body_digest(cjk)})
        )
        assert any("UTF-8 bytes" in issue.message for issue in issues)
    finally:
        monkeypatch.undo()


def _group_parts(count=2, **overrides):
    parts = []
    for index in range(count):
        text = f"第{index}片：长消息内容"
        part = event(
            body={
                "text": text,
                "sha256": c.body_digest(text),
                "part_index": index,
                "part_count": count,
                "content_revision": "native:1",
            }
        )
        part.update(overrides)
        parts.append(part)
    return parts


def test_part_group_completeness_and_metadata_agreement():
    parts = _group_parts()
    assert c.validate_batch_semantics({"schema_version": "1.0", "batch_id": "batch:1", "events": parts}) == []
    # SRC-03: metadata may not disagree across parts.
    broken = _group_parts()
    broken[1] = copy.deepcopy(broken[1])
    broken[1]["project_ids"] = ["project:other"]
    assert c.validate_batch_semantics({"schema_version": "1.0", "batch_id": "batch:1", "events": broken})
    # Missing part: group must be visibly incomplete, never silently processed.
    assert c.validate_batch_semantics({"schema_version": "1.0", "batch_id": "batch:1", "events": parts[:1]})
    # Role/origin disagreement across parts.
    role_broken = _group_parts()
    role_broken[1]["source"]["role"] = "assistant"
    assert c.validate_batch_semantics({"schema_version": "1.0", "batch_id": "batch:1", "events": role_broken})


def test_duplicate_part_keys_in_one_batch_are_rejected():
    parts = _group_parts()
    issues = c.validate_batch_semantics({"schema_version": "1.0", "batch_id": "batch:1", "events": parts + [copy.deepcopy(parts[0])]})
    assert any("duplicate" in issue.message for issue in issues)


def test_batch_size_limits():
    parts = _group_parts()
    many = {"schema_version": "1.0", "batch_id": "batch:1", "events": (parts * 60)[: c.MAX_BATCH_EVENTS + 1]}
    assert any("events" in issue.message for issue in c.validate_batch_semantics(many))
    assert any("bytes" in issue.message for issue in c.validate_batch_semantics({"schema_version": "1.0", "batch_id": "batch:1", "events": parts}, encoded_bytes=c.MAX_BATCH_BYTES + 1))
