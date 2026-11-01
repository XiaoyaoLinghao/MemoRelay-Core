"""Stable API errors for the KW service.

Every code comes from the frozen `contracts/v1/error-catalog.json`; the service
never invents codes. Response/log messages are human text and must never echo
credentials, request bodies or raw backend errors.
"""

from __future__ import annotations

import re

from knowledge_weaver import contracts

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_MESSAGE = 300


def safe_message(message: str | None) -> str:
    """Clamp an error message so it cannot leak bodies or control sequences."""
    if not message:
        return ""
    cleaned = _CONTROL_RE.sub(" ", str(message))
    cleaned = " ".join(cleaned.split())
    return cleaned[:_MAX_MESSAGE]


class ApiError(Exception):
    """An error code from the frozen catalog with its HTTP mapping."""

    def __init__(self, code: str, message: str = "", *, request_id: str | None = None) -> None:
        if contracts.error_entry(code) is None:  # never invent codes
            code = "INTERNAL_ERROR"
            message = "internal error"
        super().__init__(message or code)
        self.code = code
        self.message = safe_message(message) or code
        self.http_status = contracts.error_http_status(code)
        self.retryable = contracts.is_retryable(code)
        self.request_id = request_id


def error_payload(code: str, message: str, request_id: str) -> dict:
    return {
        "error": {
            "code": code,
            "message": safe_message(message) or code,
            "retryable": contracts.is_retryable(code),
            "request_id": request_id,
        }
    }
