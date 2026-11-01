"""T01: stable error catalog semantics."""
from __future__ import annotations

import pytest

from knowledge_weaver import contracts as c


def test_catalog_required_codes():
    codes = {entry["code"] for entry in c.error_catalog()["entries"]}
    assert c.REQUIRED_ERROR_CODES <= codes


def test_unknown_code_is_a_contract_error():
    with pytest.raises(c.ContractError):
        c.is_retryable("SOMETHING_NEW")
    with pytest.raises(c.ContractError):
        c.error_http_status("SOMETHING_NEW")


def test_retryability_and_status_mapping():
    assert c.is_retryable("STORE_UNAVAILABLE") is True
    assert c.error_http_status("STORE_UNAVAILABLE") == 503
    assert c.is_retryable("RESTORE_IN_PROGRESS") is True
    assert c.is_retryable("MODEL_UNAVAILABLE") is True
    assert c.is_retryable("PARENT_PENDING") is True
    assert c.is_retryable("EVENT_CONFLICT") is False
    assert c.error_http_status("EVENT_CONFLICT") == 409
    assert c.error_http_status("UNREGISTERED_CLIENT") == 403
    assert c.error_http_status("SCOPE_DENIED") == 403
    assert c.error_http_status("PAYLOAD_TOO_LARGE") == 413
    assert c.error_http_status("INVALID_REQUEST") == 400
    assert c.error_http_status("UNSUPPORTED_SCHEMA") == 400


def test_error_entry_fields_and_token_codes():
    seen = set()
    for entry in c.error_catalog()["entries"]:
        assert c.TOKEN_RE.match(entry["code"])
        assert entry["code"] not in seen
        seen.add(entry["code"])
        assert 400 <= entry["http_status"] <= 599
        assert isinstance(entry["retryable"], bool)
        assert entry["description"] and entry["client_action"]
        assert set(entry["scope"]) <= {"http", "receipt"}
    assert c.error_entry("REDACTION_REQUIRED")["scope"] == ["receipt"]
