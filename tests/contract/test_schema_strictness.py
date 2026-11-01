"""T01: strict unknown-field and version rejection; schema health."""
from __future__ import annotations

import copy
import json

import pytest

from knowledge_weaver import contracts as c


def test_every_object_schema_with_properties_is_strict():
    for name, schema in c.component_schemas().items():
        if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
            assert schema.get("additionalProperties") is False, name


def test_unknown_fields_rejected_at_every_level():
    event = copy.deepcopy(c.valid_examples()["Event"])
    top = dict(event, totally_unknown=1)
    assert any("unknown property" in str(issue) for issue in c.validate_instance("Event", top))
    body = copy.deepcopy(event)
    body["body"]["secret_field"] = "x"
    assert c.validate_instance("Event", body)
    source = copy.deepcopy(event)
    source["source"]["password"] = "synthetic-only"
    assert c.validate_instance("Event", source)
    batch = copy.deepcopy(c.valid_examples()["Batch"])
    batch["events"][0]["extra"] = True
    assert c.validate_instance("Batch", batch)


def test_unknown_version_is_rejected_not_guessed():
    event = copy.deepcopy(c.valid_examples()["Event"])
    event["schema_version"] = "99.0"
    issues = c.validate_instance("Event", event)
    assert issues
    for name in ("RecallRequest", "Health", "Correction", "Batch"):
        value = copy.deepcopy(c.valid_examples()[name])
        value["schema_version"] = "2.0"
        assert c.validate_instance(name, value), name


def test_time_precision_rules():
    event = copy.deepcopy(c.valid_examples()["Event"])
    event["event_time"] = {"value": "2026-10-30T09:00:00", "precision": "instant", "timezone": "Asia/Shanghai"}
    assert c.validate_instance("Event", event)
    event["event_time"] = {"value": None, "precision": "unknown", "timezone": "Asia/Shanghai"}
    assert c.validate_instance("Event", event) == []
    event["event_time"] = {"value": "2026-10-30", "precision": "day", "timezone": "Asia/Shanghai"}
    assert c.validate_instance("Event", event) == []
    event["event_time"] = {"value": "2026-10-30T09:00:00+08:00", "precision": "day", "timezone": "Asia/Shanghai"}
    assert c.validate_instance("Event", event)


def test_recall_budget_and_limit_are_bounded():
    request = copy.deepcopy(c.valid_examples()["RecallRequest"])
    request["limit"] = 100000
    assert c.validate_instance("RecallRequest", request)
    request = copy.deepcopy(c.valid_examples()["RecallRequest"])
    request["budget_tokens"] = 8_001
    assert c.validate_instance("RecallRequest", request)
    request = copy.deepcopy(c.valid_examples()["RecallRequest"])
    request["deadline_ms"] = 7001
    assert c.validate_instance("RecallRequest", request)


def test_unsupported_schema_name_raises():
    with pytest.raises(c.ContractError):
        c.validate_instance("NoSuchSchema", {})


def test_unknown_format_would_be_rejected_by_contract_health():
    schema = {"type": "object", "additionalProperties": False, "properties": {"x": {"type": "string", "format": "ipv4"}}}
    issues = c._check_schema_node(schema, schema, "schema:Test", ref_ok=False)
    assert any("format" in issue.message for issue in issues)


def test_object_schema_without_strictness_is_flagged():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    issues = c._check_schema_node(schema, schema, "schema:Test", ref_ok=False)
    assert any("additionalProperties" in issue.message for issue in issues)
