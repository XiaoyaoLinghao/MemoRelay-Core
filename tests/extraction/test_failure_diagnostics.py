"""Failure diagnostics: closed categories and bounded, leak-free details."""

from __future__ import annotations

import json

import pytest

from knowledge_weaver.extraction.diagnostics import (
    CATEGORIES,
    EXTERNAL_REQUEST_OUTCOME_UNKNOWN,
    FIELD_IDENTITY_COLLISION,
    FIELD_IDENTITY_COLLISION_PREFIX,
    classify_failure,
    failure_detail,
)

SECRET = "sk-live-CANARY-9f3a7c2e"
URL = "https://user:pass@example.invalid/v1/chat?token=canary"


def test_category_vocabulary_is_closed():
    assert set(CATEGORIES) == {
        "field_identity_collision",
        "model_review_required",
        "output_length",
        "transport",
        "invalid_response",
        "schema_invalid",
        "insufficient_metadata",
        "secret_content", "missing_disposition", "call_budget", "coverage_incomplete", "evidence_invalid", "proposal_invalid",
    }
    assert len(CATEGORIES) == len(set(CATEGORIES))


def test_field_identity_collision_prefix_wins_over_model_unavailable():
    detail = (
        FIELD_IDENTITY_COLLISION_PREFIX
        + " null target_claim_hint collides with claim:abc; "
        + SECRET
        + " "
        + URL
    )
    category = classify_failure("MODEL_UNAVAILABLE", detail)
    assert category == "field_identity_collision"
    assert SECRET not in category
    assert URL not in category


def test_field_identity_collision_constant_alone_is_classified():
    assert classify_failure(FIELD_IDENTITY_COLLISION) == "field_identity_collision"
    assert classify_failure(FIELD_IDENTITY_COLLISION_PREFIX + " text") == "field_identity_collision"


def test_field_identity_collision_in_structured_metadata():
    detail = {
        "error_code": "MODEL_UNAVAILABLE",
        "message": FIELD_IDENTITY_COLLISION_PREFIX + " collision " + SECRET,
    }
    assert classify_failure("MODEL_UNAVAILABLE", detail) == "field_identity_collision"

    nested = {"error": {"message": FIELD_IDENTITY_COLLISION_PREFIX + " " + URL}}
    assert classify_failure("MODEL_UNAVAILABLE", json.dumps(nested)) == "field_identity_collision"


def test_model_review_required_stays_distinct_from_collision():
    assert classify_failure("MODEL_REVIEW_REQUIRED", "model proposal requires review") == (
        "model_review_required"
    )
    assert classify_failure("MODEL_UNAVAILABLE", "model proposal requires review") == (
        "model_review_required"
    )
    # A constant collision marker is more specific than the generic legacy code.
    assert classify_failure(
        "MODEL_UNAVAILABLE", FIELD_IDENTITY_COLLISION_PREFIX + " collision"
    ) == "field_identity_collision"


def test_output_length_from_code_and_safe_metadata():
    assert classify_failure("OUTPUT_LENGTH") == "output_length"
    assert classify_failure("MAX_TOKENS_EXCEEDED") == "output_length"
    assert classify_failure(
        "MODEL_UNAVAILABLE", json.dumps({"finish_reason": "length"})
    ) == "output_length"
    assert classify_failure(
        "MODEL_UNAVAILABLE", {"finish_reason": "max_tokens"}
    ) == "output_length"
    assert classify_failure(
        "MODEL_UNAVAILABLE", {"error": {"code": "context_length_exceeded"}}
    ) == "output_length"
    assert classify_failure(
        "MODEL_UNAVAILABLE", {"choices": [{"finish_reason": "length"}]}
    ) == "output_length"


def test_transport_from_legacy_messages_and_safe_status():
    assert classify_failure("TRANSPORT_ERROR") == "transport"
    assert classify_failure(
        "MODEL_UNAVAILABLE", "model endpoint unreachable (URLError)"
    ) == "transport"
    assert classify_failure(
        "MODEL_UNAVAILABLE", "external extractor timed out"
    ) == "transport"
    assert classify_failure(
        "MODEL_UNAVAILABLE", "MODEL_UNAVAILABLE: model endpoint unreachable"
    ) == "transport"
    assert classify_failure(
        "MODEL_UNAVAILABLE", json.dumps({"http_status": 503})
    ) == "transport"
    assert classify_failure("503") == "transport"


def test_invalid_response_from_legacy_messages_and_metadata():
    invalid_messages = (
        "model endpoint returned invalid JSON",
        "model content is not a JSON proposal",
        "model proposal is not an object",
        "model response has no content",
        "external extractor returned invalid JSON",
        "external extractor returned a non-object proposal",
    )
    for message in invalid_messages:
        assert classify_failure("MODEL_UNAVAILABLE", message) == "invalid_response"
    assert classify_failure("INVALID_RESPONSE") == "invalid_response"
    assert classify_failure(
        "MODEL_UNAVAILABLE", {"error": {"code": "invalid_json"}}
    ) == "invalid_response"
    assert classify_failure(
        "MODEL_UNAVAILABLE", {"http_status": 200}
    ) == "insufficient_metadata"
    assert classify_failure("MODEL_UNAVAILABLE", {"http_status":200,"finish_reason":"stop"}) == "insufficient_metadata"


def test_schema_invalid_from_legacy_pipeline_message():
    assert classify_failure("SCHEMA_INVALID") == "schema_invalid"
    assert classify_failure(
        "MODEL_UNAVAILABLE",
        "extraction proposal failed schema at claims[0].value: expected object",
    ) == "schema_invalid"
    assert classify_failure(
        "MODEL_UNAVAILABLE", {"error": {"code": "schema_validation_error"}}
    ) == "schema_invalid"


def test_legacy_unknown_model_unavailable_has_insufficient_metadata():
    assert classify_failure(None) == "insufficient_metadata"
    assert classify_failure("MODEL_UNAVAILABLE") == "insufficient_metadata"
    assert classify_failure("SOMETHING_ELSE", "opaque provider text") == (
        "insufficient_metadata"
    )
    legacy_detail = json.dumps(
        {
            "billing_outcome": "unknown",
            "provider_usage": None,
            "reason": EXTERNAL_REQUEST_OUTCOME_UNKNOWN,
        }
    )
    assert classify_failure("MODEL_UNAVAILABLE", legacy_detail) == (
        "insufficient_metadata"
    )


def test_classification_never_echoes_malicious_detail():
    detail = json.dumps(
        {
            "message": f"provider failed at {URL} with Bearer {SECRET}",
            "error": {
                "message": f"response body contained {SECRET}",
                "code": f"unknown-{SECRET}",
            },
            "body": f"{URL} {SECRET}",
        }
    )
    category = classify_failure("MODEL_UNAVAILABLE", detail)
    assert category in CATEGORIES
    assert category == "insufficient_metadata"
    assert SECRET not in category
    assert URL not in category

    category = classify_failure(SECRET, URL + " " + SECRET)
    assert category in CATEGORIES
    assert SECRET not in category
    assert URL not in category


def test_failure_detail_preserves_legacy_unknown_outcome_shape():
    payload = json.loads(failure_detail("transport"))
    assert payload == {
        "category": "transport",
        "usage_present": False,
        "billing_outcome": "unknown",
        "provider_usage": None,
        "reason": EXTERNAL_REQUEST_OUTCOME_UNKNOWN,
    }
    assert payload["reason"] == "external_request_outcome_unknown"

    insufficient = json.loads(failure_detail("insufficient_metadata"))
    assert insufficient["billing_outcome"] == "unknown"
    assert insufficient["provider_usage"] is None
    assert insufficient["reason"] == EXTERNAL_REQUEST_OUTCOME_UNKNOWN


def test_field_collision_with_received_response_is_not_transport_unknown():
    payload = json.loads(failure_detail("field_identity_collision"))
    assert payload["category"] == "field_identity_collision"
    assert "reason" not in payload
    assert "billing_outcome" not in payload
    assert "provider_usage" not in payload

    for category in (
        "model_review_required",
        "output_length",
        "invalid_response",
        "schema_invalid",
    ):
        received = json.loads(failure_detail(category))
        assert received["category"] == category
        assert "reason" not in received


def test_failure_detail_allows_safe_response_metadata():
    payload = json.loads(
        failure_detail(
            "output_length",
            http_status=200,
            finish_reason="length",
            usage_present=True,
            billing_outcome="billed",
        )
    )
    assert payload == {
        "category": "output_length",
        "usage_present": True,
        "http_status": 200,
        "finish_reason": "length",
        "billing_outcome": "billed",
        "provider_usage": None,
    }


def test_failure_detail_accepts_known_billing_without_unknown_reason():
    payload = json.loads(
        failure_detail("transport", billing_outcome="not_billed")
    )
    assert payload["billing_outcome"] == "not_billed"
    assert payload["provider_usage"] is None
    assert "reason" not in payload


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [(200, 200), ("200", 200), (503, 503), ("503", 503), (100, 100), ("599", 599)],
)
def test_failure_detail_keeps_only_safe_http_codes(raw_status, expected):
    payload = json.loads(failure_detail("invalid_response", http_status=raw_status))
    assert payload["http_status"] == expected


@pytest.mark.parametrize(
    "bad_status",
    [
        URL,
        SECRET,
        True,
        False,
        99,
        600,
        -1,
        "200 OK",
        "050",
        "20",
        "200000",
        "<script>alert(1)</script>",
        object(),
    ],
)
def test_failure_detail_drops_unsafe_http_codes(bad_status):
    text = failure_detail("invalid_response", http_status=bad_status)
    payload = json.loads(text)
    assert "http_status" not in payload
    assert SECRET not in text
    assert URL not in text


@pytest.mark.parametrize(
    "finish_reason",
    ["stop", "length", "content_filter", "tool_calls", "end_turn", "max_tokens"],
)
def test_failure_detail_keeps_allowlisted_finish_reasons(finish_reason):
    payload = json.loads(failure_detail("output_length", finish_reason=finish_reason))
    assert payload["finish_reason"] == finish_reason


@pytest.mark.parametrize(
    "bad_reason",
    [SECRET, URL, "<script>alert(1)</script>", "x" * 100, 123, object(), "safe\ninjected"],
)
def test_failure_detail_drops_unsafe_finish_reasons(bad_reason):
    text = failure_detail("invalid_response", finish_reason=bad_reason)
    payload = json.loads(text)
    assert "finish_reason" not in payload
    assert SECRET not in text
    assert URL not in text


def test_failure_detail_sanitizes_malicious_billing_and_category():
    text = failure_detail(
        URL + " " + SECRET,
        http_status=URL,
        finish_reason=SECRET,
        billing_outcome=URL,
    )
    payload = json.loads(text)
    assert payload["category"] == "insufficient_metadata"
    assert payload["billing_outcome"] == "unknown"
    assert payload["provider_usage"] is None
    assert payload["reason"] == EXTERNAL_REQUEST_OUTCOME_UNKNOWN
    assert SECRET not in text
    assert URL not in text


def test_failure_detail_legacy_code_is_normalized_without_echo():
    text = failure_detail("MODEL_REVIEW_REQUIRED", http_status="201")
    payload = json.loads(text)
    assert payload["category"] == "model_review_required"
    assert payload["http_status"] == 201
    assert "reason" not in payload
    assert "MODEL_REVIEW_REQUIRED" not in text


def test_failure_detail_is_deterministic_json():
    first = failure_detail("transport", http_status=503, usage_present=False)
    second = failure_detail("transport", http_status=503, usage_present=False)
    assert first == second
    assert json.loads(first)["category"] == "transport"
