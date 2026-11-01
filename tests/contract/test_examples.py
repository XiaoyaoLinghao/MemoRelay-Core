"""T01: both repositories must accept every shared example and reject every bad one."""
from __future__ import annotations

import copy
import json

import pytest

from knowledge_weaver import contracts as c


def test_valid_examples_pass():
    for name, value in c.valid_examples().items():
        issues = c.validate_instance(name, value)
        assert issues == [], f"valid example {name}: {[str(i) for i in issues]}"


def test_invalid_examples_are_rejected():
    valid = c.valid_examples()
    for entry in c.invalid_examples():
        patched = copy.deepcopy(valid[entry["schema"]])
        patched.update(entry["patch"])
        issues = c.validate_instance(entry["schema"], patched)
        assert issues, f"invalid example {entry['reason']!r} was accepted"
        assert c.validate_instance(entry["schema"], patched)


def test_valid_event_also_passes_semantic_checks():
    assert c.validate_event(c.valid_examples()["Event"]) == []


def test_valid_batch_also_passes_semantic_checks():
    assert c.validate_batch(c.valid_examples()["Batch"]) == []


def test_extraction_output_schema_accepts_conforming_proposal():
    event = c.valid_examples()["Event"]
    proposal = {
        "schema_version": "1.0",
        "claims": [
            {
                "subject_hint": "松塔项目服务器",
                "project_hint": "project:songta",
                "qualifiers": {"environment": "prod"},
                "predicate": "server.port",
                "facet": "desired",
                "value": 4200,
                "display_text": "松塔项目端口改为4200。",
                "assertion_kind": "explicit_correction",
                "evidence": [
                    {
                        "event_id": event["event_id"],
                        "event_revision": 1,
                        "part_index": 0,
                        "support_quote": "松塔项目端口改为4200。",
                    }
                ],
                "target_claim_hint": None,
                "effective_time_quote": None,
            }
        ],
        "excluded_event_ids": [],
        "warnings": [],
    }
    assert c.validate_extraction_output(proposal) == []
    # An empty claims list is a valid "nothing extracted" proposal (the
    # schema bounds only maxItems; exclusion receipts carry that state).
    assert c.validate_extraction_output({**proposal, "claims": []}) == []
    for invalid in (
        {**proposal, "unknown_top": 1},
        {**proposal, "claims": "not-a-list"},
        {
            **proposal,
            "claims": [{**proposal["claims"][0], "facet": "guess"}],
        },
        {
            **proposal,
            "claims": [{**proposal["claims"][0], "evidence": []}],
        },
    ):
        assert c.validate_extraction_output(invalid), invalid


def test_every_component_schema_is_covered_by_structure_check():
    names = set(c.component_schemas())
    report = c.contract_health()["checks"]
    checked = {item["check"][len("schema:"):] for item in report if item["check"].startswith("schema:")}
    assert checked == names
