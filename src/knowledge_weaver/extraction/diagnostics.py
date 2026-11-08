"""Stable, bounded failure diagnostics for extraction model calls.

The extraction pipeline talks to untrusted external model endpoints. A failure
there may be a transport problem, a truncated response, malformed JSON, a
schema-invalid proposal, a review requirement, or a field-identity collision
found by backend validation. Operators need to distinguish those classes
without persisting provider messages, response bodies, URLs, credentials or
other arbitrary exception text.

This module is deliberately small and pure (stdlib only, no I/O):

* :func:`classify_failure` maps an external/legacy error code plus optional
  already-safe detail metadata to one closed, stable category. The legacy job
  error codes themselves (for example ``MODEL_UNAVAILABLE`` and
  ``MODEL_REVIEW_REQUIRED``) are left unchanged by callers; this function only
  derives the diagnostic class. A detail beginning with the constant
  ``FIELD_IDENTITY_COLLISION:`` prefix always classifies as a field collision,
  even when the external code is the generic ``MODEL_UNAVAILABLE``.
* :func:`failure_detail` serializes only bounded, allowlisted metadata
  (category, safe HTTP status, safe finish reason, usage presence and a safe
  billing outcome). It never accepts or emits a provider message. For outcomes
  that are genuinely unknown (transport/insufficient metadata) it preserves the
  legacy ``billing_outcome``/``provider_usage: null``/
  ``external_request_outcome_unknown`` shape. A field-identity collision means a
  model response was already received, so it is not reported as an unknown
  external request outcome.

Structured JSON details are supported, but only known metadata fields are
inspected. Unknown strings, messages, URLs and nested content are ignored and
never copied into either return value.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

CATEGORY_FIELD_IDENTITY_COLLISION = "field_identity_collision"
CATEGORY_MODEL_REVIEW_REQUIRED = "model_review_required"
CATEGORY_OUTPUT_LENGTH = "output_length"
CATEGORY_TRANSPORT = "transport"
CATEGORY_INVALID_RESPONSE = "invalid_response"
CATEGORY_SCHEMA_INVALID = "schema_invalid"
CATEGORY_INSUFFICIENT_METADATA = "insufficient_metadata"

#: The closed set of stable diagnostic categories. Order is part of the public
#: documentation only; callers must not treat it as a precedence order.
CATEGORIES = (
    CATEGORY_FIELD_IDENTITY_COLLISION,
    CATEGORY_MODEL_REVIEW_REQUIRED,
    CATEGORY_OUTPUT_LENGTH,
    CATEGORY_TRANSPORT,
    CATEGORY_INVALID_RESPONSE,
    CATEGORY_SCHEMA_INVALID,
    CATEGORY_INSUFFICIENT_METADATA,
    "secret_content", "missing_disposition", "call_budget", "coverage_incomplete", "evidence_invalid", "proposal_invalid",
)
_CATEGORY_SET = frozenset(CATEGORIES)
#: Compatibility alias for callers that describe the same closed vocabulary as
#: failure categories.
FAILURE_CATEGORIES = CATEGORIES

#: Marker used by ``proposals.resolve_proposal`` for a received model response
#: whose null target did not match an existing scoped field. The constant
#: prefix is part of the backend contract; arbitrary text after it is never
#: emitted by this module.
FIELD_IDENTITY_COLLISION = "FIELD_IDENTITY_COLLISION"
FIELD_IDENTITY_COLLISION_PREFIX = FIELD_IDENTITY_COLLISION + ":"

#: Legacy detail reason written for a model request whose external outcome is
#: unknown. This is kept for older consumers when diagnostics have no evidence
#: that a response was received.
EXTERNAL_REQUEST_OUTCOME_UNKNOWN = "external_request_outcome_unknown"
EXTERNAL_REQUEST_OUTCOME_UNKNOWN_REASON = EXTERNAL_REQUEST_OUTCOME_UNKNOWN

#: External error codes from the pre-diagnostics pipeline. They remain the
#: public job codes; this module only classifies them and never rewrites them.
LEGACY_ERROR_CODES = frozenset(
    {
        "MODEL_UNAVAILABLE",
        "MODEL_REVIEW_REQUIRED",
        "UNSUPPORTED_PROMPT_VERSION",
    }
)

#: Bound how much untrusted detail text is parsed. Diagnostics never need a
#: large provider payload; refusing oversized structured details keeps this
#: helper allocation-bounded and avoids treating arbitrary content as metadata.
_MAX_DETAIL_LENGTH = 65536
_MAX_CODE_LENGTH = 128

#: Codes that only say "something failed" and must be refined from safe detail
#: metadata. ``MODEL_UNAVAILABLE`` is the legacy generic provider code; the
#: others are generic or insufficient-by-construction. Without supporting
#: detail they degrade to ``insufficient_metadata`` rather than being guessed
#: as transport.
_GENERIC_CODES = frozenset(
    {
        "MODEL_UNAVAILABLE",
        "UNKNOWN",
        "UNSPECIFIED",
        "NOT_AVAILABLE",
        "INSUFFICIENT_METADATA",
    }
)

#: Direct (non-generic) code mappings.
_DIRECT_CODE_CATEGORIES = {
    # Explicit diagnostic/legacy marker codes.
    "FIELD_IDENTITY_COLLISION": CATEGORY_FIELD_IDENTITY_COLLISION,
    "MODEL_REVIEW_REQUIRED": CATEGORY_MODEL_REVIEW_REQUIRED,
    "REVIEW_REQUIRED": CATEGORY_MODEL_REVIEW_REQUIRED,
    # Output truncation.
    "OUTPUT_LENGTH": CATEGORY_OUTPUT_LENGTH,
    "OUTPUT_LENGTH_EXCEEDED": CATEGORY_OUTPUT_LENGTH,
    "OUTPUT_TOO_LONG": CATEGORY_OUTPUT_LENGTH,
    "OUTPUT_TRUNCATED": CATEGORY_OUTPUT_LENGTH,
    "MODEL_OUTPUT_LENGTH": CATEGORY_OUTPUT_LENGTH,
    "MODEL_OUTPUT_TRUNCATED": CATEGORY_OUTPUT_LENGTH,
    "LENGTH": CATEGORY_OUTPUT_LENGTH,
    "MAX_TOKENS": CATEGORY_OUTPUT_LENGTH,
    "MAX_TOKENS_EXCEEDED": CATEGORY_OUTPUT_LENGTH,
    "MAX_OUTPUT_TOKENS": CATEGORY_OUTPUT_LENGTH,
    "TOKEN_LIMIT": CATEGORY_OUTPUT_LENGTH,
    "TOKEN_LIMIT_EXCEEDED": CATEGORY_OUTPUT_LENGTH,
    "CONTEXT_LENGTH": CATEGORY_OUTPUT_LENGTH,
    "CONTEXT_LENGTH_EXCEEDED": CATEGORY_OUTPUT_LENGTH,
    "CONTEXT_WINDOW_EXCEEDED": CATEGORY_OUTPUT_LENGTH,
    "MODEL_CONTEXT_LENGTH_EXCEEDED": CATEGORY_OUTPUT_LENGTH,
    "TRUNCATED": CATEGORY_OUTPUT_LENGTH,
    # Response was received but could not be used as a proposal.
    "INVALID_RESPONSE": CATEGORY_INVALID_RESPONSE,
    "MODEL_INVALID_RESPONSE": CATEGORY_INVALID_RESPONSE,
    "INVALID_JSON": CATEGORY_INVALID_RESPONSE,
    "MODEL_INVALID_JSON": CATEGORY_INVALID_RESPONSE,
    "MALFORMED_JSON": CATEGORY_INVALID_RESPONSE,
    "MALFORMED_RESPONSE": CATEGORY_INVALID_RESPONSE,
    "JSON_DECODE_ERROR": CATEGORY_INVALID_RESPONSE,
    "JSON_DECODING_ERROR": CATEGORY_INVALID_RESPONSE,
    "RESPONSE_NOT_JSON": CATEGORY_INVALID_RESPONSE,
    "NO_CONTENT": CATEGORY_INVALID_RESPONSE,
    "EMPTY_RESPONSE": CATEGORY_INVALID_RESPONSE,
    "EMPTY_CONTENT": CATEGORY_INVALID_RESPONSE,
    "MODEL_NO_CONTENT": CATEGORY_INVALID_RESPONSE,
    "NON_OBJECT_PROPOSAL": CATEGORY_INVALID_RESPONSE,
    "PROPOSAL_NOT_OBJECT": CATEGORY_INVALID_RESPONSE,
    # JSON parsed but violates the extraction schema.
    "SCHEMA_INVALID": CATEGORY_SCHEMA_INVALID,
    "INVALID_SCHEMA": CATEGORY_SCHEMA_INVALID,
    "MODEL_SCHEMA_INVALID": CATEGORY_SCHEMA_INVALID,
    "PROPOSAL_SCHEMA_INVALID": CATEGORY_SCHEMA_INVALID,
    "SCHEMA_VALIDATION": CATEGORY_SCHEMA_INVALID,
    "SCHEMA_VALIDATION_ERROR": CATEGORY_SCHEMA_INVALID,
    "SCHEMA_MISMATCH": CATEGORY_SCHEMA_INVALID,
    # No usable response arrived (network/HTTP/service failure).
    "TRANSPORT": CATEGORY_TRANSPORT,
    "TRANSPORT_ERROR": CATEGORY_TRANSPORT,
    "MODEL_TRANSPORT_ERROR": CATEGORY_TRANSPORT,
    "NETWORK": CATEGORY_TRANSPORT,
    "NETWORK_ERROR": CATEGORY_TRANSPORT,
    "CONNECTION_ERROR": CATEGORY_TRANSPORT,
    "CONNECTION_REFUSED": CATEGORY_TRANSPORT,
    "CONNECTION_RESET": CATEGORY_TRANSPORT,
    "DNS_ERROR": CATEGORY_TRANSPORT,
    "TIMEOUT": CATEGORY_TRANSPORT,
    "TIMED_OUT": CATEGORY_TRANSPORT,
    "MODEL_TIMEOUT": CATEGORY_TRANSPORT,
    "UNREACHABLE": CATEGORY_TRANSPORT,
    "ENDPOINT_UNREACHABLE": CATEGORY_TRANSPORT,
    "MODEL_ENDPOINT_UNREACHABLE": CATEGORY_TRANSPORT,
    "HTTP_ERROR": CATEGORY_TRANSPORT,
    "BAD_STATUS": CATEGORY_TRANSPORT,
    "BAD_RESPONSE_STATUS_CODE": CATEGORY_TRANSPORT,
    "RATE_LIMIT": CATEGORY_TRANSPORT,
    "RATE_LIMITED": CATEGORY_TRANSPORT,
    "RATE_LIMIT_EXCEEDED": CATEGORY_TRANSPORT,
    "RATE_LIMIT_ERROR": CATEGORY_TRANSPORT,
    "SERVER_ERROR": CATEGORY_TRANSPORT,
    "SERVICE_UNAVAILABLE": CATEGORY_TRANSPORT,
    "OVERLOADED_ERROR": CATEGORY_TRANSPORT,
    "AUTHENTICATION_ERROR": CATEGORY_TRANSPORT,
    "PERMISSION_ERROR": CATEGORY_TRANSPORT,
    "API_ERROR": CATEGORY_TRANSPORT,
    "MODEL_ERROR": CATEGORY_TRANSPORT,
    "REQUEST_FAILED": CATEGORY_TRANSPORT,
    # No diagnostic evidence was supplied or recognized.
    "INSUFFICIENT_METADATA": CATEGORY_INSUFFICIENT_METADATA,
    "UNKNOWN": CATEGORY_INSUFFICIENT_METADATA,
    "UNSPECIFIED": CATEGORY_INSUFFICIENT_METADATA,
    "NOT_AVAILABLE": CATEGORY_INSUFFICIENT_METADATA,
    "UNSUPPORTED_PROMPT_VERSION": CATEGORY_INSUFFICIENT_METADATA,
}

#: Fixed legacy messages emitted by the current HTTP/command adapters and the
#: proposal schema check. Matching is prefix-only; the rest of the message may
#: contain arbitrary text and is never returned. Order matters: field collision
#: is checked before the other classes.
_PLAIN_DETAIL_PREFIXES = (
    (FIELD_IDENTITY_COLLISION_PREFIX.lower(), CATEGORY_FIELD_IDENTITY_COLLISION),
    # Schema messages from the pipeline and common provider vocabulary.
    ("extraction proposal failed schema", CATEGORY_SCHEMA_INVALID),
    ("proposal failed schema", CATEGORY_SCHEMA_INVALID),
    ("schema validation failed", CATEGORY_SCHEMA_INVALID),
    ("schema validation error", CATEGORY_SCHEMA_INVALID),
    ("failed schema validation", CATEGORY_SCHEMA_INVALID),
    ("validation failed", CATEGORY_SCHEMA_INVALID),
    ("validation error", CATEGORY_SCHEMA_INVALID),
    ("schema invalid", CATEGORY_SCHEMA_INVALID),
    ("invalid schema", CATEGORY_SCHEMA_INVALID),
    # Review-required messages.
    ("model proposal requires review", CATEGORY_MODEL_REVIEW_REQUIRED),
    ("extraction proposal requires review", CATEGORY_MODEL_REVIEW_REQUIRED),
    ("model review required", CATEGORY_MODEL_REVIEW_REQUIRED),
    ("proposal requires review", CATEGORY_MODEL_REVIEW_REQUIRED),
    # Transport/network failures (no usable response).
    ("model endpoint unreachable", CATEGORY_TRANSPORT),
    ("model endpoint timed out", CATEGORY_TRANSPORT),
    ("model request timed out", CATEGORY_TRANSPORT),
    ("request timed out", CATEGORY_TRANSPORT),
    ("connection refused", CATEGORY_TRANSPORT),
    ("connection reset", CATEGORY_TRANSPORT),
    ("network error", CATEGORY_TRANSPORT),
    ("network is unreachable", CATEGORY_TRANSPORT),
    ("dns error", CATEGORY_TRANSPORT),
    ("ssl error", CATEGORY_TRANSPORT),
    ("tls error", CATEGORY_TRANSPORT),
    ("rate limit", CATEGORY_TRANSPORT),
    ("rate limited", CATEGORY_TRANSPORT),
    ("quota exceeded", CATEGORY_TRANSPORT),
    ("server error", CATEGORY_TRANSPORT),
    ("service unavailable", CATEGORY_TRANSPORT),
    ("bad gateway", CATEGORY_TRANSPORT),
    ("gateway timeout", CATEGORY_TRANSPORT),
    ("http error", CATEGORY_TRANSPORT),
    ("model endpoint returned http", CATEGORY_TRANSPORT),
    ("urlerror", CATEGORY_TRANSPORT),
    ("timeouterror", CATEGORY_TRANSPORT),
    ("connectionerror", CATEGORY_TRANSPORT),
    ("oserror", CATEGORY_TRANSPORT),
    ("external extractor timed out", CATEGORY_TRANSPORT),
    ("external extractor failed to start", CATEGORY_TRANSPORT),
    ("external extractor exited with", CATEGORY_TRANSPORT),
    # Response was received but could not be parsed/used.
    ("model endpoint returned invalid json", CATEGORY_INVALID_RESPONSE),
    ("model returned invalid json", CATEGORY_INVALID_RESPONSE),
    ("model content is not a json proposal", CATEGORY_INVALID_RESPONSE),
    ("model response has no content", CATEGORY_INVALID_RESPONSE),
    ("model proposal is not an object", CATEGORY_INVALID_RESPONSE),
    ("external extractor returned invalid json", CATEGORY_INVALID_RESPONSE),
    ("external extractor returned a non-object proposal", CATEGORY_INVALID_RESPONSE),
    ("invalid json", CATEGORY_INVALID_RESPONSE),
    ("malformed json", CATEGORY_INVALID_RESPONSE),
    ("jsondecodeerror", CATEGORY_INVALID_RESPONSE),
    ("unicodedecodeerror", CATEGORY_INVALID_RESPONSE),
    ("json decode error", CATEGORY_INVALID_RESPONSE),
    ("response is not json", CATEGORY_INVALID_RESPONSE),
    ("response was not json", CATEGORY_INVALID_RESPONSE),
    ("empty response", CATEGORY_INVALID_RESPONSE),
    ("no content", CATEGORY_INVALID_RESPONSE),
    ("non-object proposal", CATEGORY_INVALID_RESPONSE),
    ("proposal is not an object", CATEGORY_INVALID_RESPONSE),
    # Output truncation. Broad "truncated" is deliberately last.
    ("model output was truncated", CATEGORY_OUTPUT_LENGTH),
    ("model output truncated", CATEGORY_OUTPUT_LENGTH),
    ("model response was truncated", CATEGORY_OUTPUT_LENGTH),
    ("model output length", CATEGORY_OUTPUT_LENGTH),
    ("model output token limit", CATEGORY_OUTPUT_LENGTH),
    ("response too long", CATEGORY_OUTPUT_LENGTH),
    ("output length", CATEGORY_OUTPUT_LENGTH),
    ("output truncated", CATEGORY_OUTPUT_LENGTH),
    ("token limit", CATEGORY_OUTPUT_LENGTH),
    ("max tokens", CATEGORY_OUTPUT_LENGTH),
    ("max_tokens", CATEGORY_OUTPUT_LENGTH),
    ("too many tokens", CATEGORY_OUTPUT_LENGTH),
    ("context length", CATEGORY_OUTPUT_LENGTH),
    ("maximum context", CATEGORY_OUTPUT_LENGTH),
    ("finish_reason=length", CATEGORY_OUTPUT_LENGTH),
    ("finish_reason: length", CATEGORY_OUTPUT_LENGTH),
    ("finish reason: length", CATEGORY_OUTPUT_LENGTH),
    ("finish reason length", CATEGORY_OUTPUT_LENGTH),
    ("truncated", CATEGORY_OUTPUT_LENGTH),
)

#: Finish reasons that mean the response was cut off before a usable proposal.
_FINISH_REASON_OUTPUT_LENGTH = frozenset(
    {
        "length",
        "max_tokens",
        "max_output_tokens",
        "output_length",
        "token_limit",
        "truncated",
    }
)

#: Finish reasons that mean the provider withheld the response content.
_FINISH_REASON_INVALID_RESPONSE = frozenset(
    {
        "content_filter",
        "safety",
        "recitation",
        "blocked",
    }
)

#: Values safe to expose from a provider finish reason. Anything else is
#: normalized away instead of being copied into diagnostics.
_SAFE_FINISH_REASONS = frozenset(
    {
        "stop",
        "length",
        "content_filter",
        "tool_calls",
        "function_call",
        "tool_use",
        "end_turn",
        "stop_sequence",
        "max_tokens",
        "max_output_tokens",
        "output_length",
        "token_limit",
        "truncated",
        "completed",
        "safety",
        "recitation",
        "blocked",
        "abort",
        "error",
    }
)

#: Billing outcomes are a closed vocabulary. Unknown caller values degrade to
#: ``unknown`` rather than being treated as free-form text.
_BILLING_OUTCOMES = frozenset(
    {
        "unknown",
        "not_applicable",
        "not_billed",
        "not_charged",
        "billed",
        "charged",
        "estimated",
        "included",
        "free",
    }
)

#: An outcome is genuinely unknown only for classes with no received response.
_UNKNOWN_OUTCOME_CATEGORIES = frozenset(
    {CATEGORY_TRANSPORT, CATEGORY_INSUFFICIENT_METADATA}
)


def _as_text(value: object) -> str:
    """Return bounded text without ever stringifying arbitrary objects."""
    if isinstance(value, str):
        return value[:_MAX_DETAIL_LENGTH]
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value)[:_MAX_DETAIL_LENGTH].decode("utf-8", "replace")
        except (TypeError, ValueError):
            return ""
    return ""


def _code_key(value: object) -> str:
    """Normalize a code for comparison; the original is never emitted."""
    text = _as_text(value).strip().upper()[: _MAX_CODE_LENGTH * 2]
    for character in (":", " ", "-", ".", "/", "\t", "\n", "\r"):
        text = text.replace(character, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _plain_probe(text: str) -> str:
    """Strip a leading legacy code label before fixed-prefix matching."""
    probe = text.strip()
    upper = probe.upper()
    for label in sorted(LEGACY_ERROR_CODES, key=len, reverse=True):
        if upper.startswith(label + ":") or upper.startswith(label + "\t"):
            return probe[len(label) + 1 :].lstrip()
        if upper.startswith(label + " "):
            return probe[len(label) + 1 :].lstrip()
    return probe


def _is_field_marker(value: object) -> bool:
    """True for the exact collision constant or its constant-prefix form."""
    text = _as_text(value).strip()
    if not text:
        return False
    upper = text.upper()
    return upper == FIELD_IDENTITY_COLLISION or upper.startswith(
        FIELD_IDENTITY_COLLISION_PREFIX
    )


def _plain_is_field_collision(text: str) -> bool:
    """True when the untrusted detail begins with the collision marker."""
    return _is_field_marker(_plain_probe(text))


def _structured_metadata(detail: object) -> Mapping[str, object] | None:
    """Return a structured detail mapping, or ``None`` for arbitrary text.

    Only a bounded JSON object is accepted. Lists, scalars, malformed JSON and
    oversized payloads are treated as opaque text; nothing from them is copied
    into a result.
    """
    if isinstance(detail, Mapping):
        return detail
    text = _as_text(detail).strip()
    if not text or len(text) > _MAX_DETAIL_LENGTH:
        return None
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _metadata_sources(meta: Mapping[str, object]):
    """Yield the top-level metadata mapping and known nested metadata objects."""
    yield meta
    for key in ("error", "detail", "metadata", "response"):
        nested = meta.get(key)
        if isinstance(nested, Mapping):
            yield nested


def _metadata_has_field_collision(meta: Mapping[str, object]) -> bool:
    """Detect the collision marker in safe structured metadata fields."""
    for source in _metadata_sources(meta):
        for key in ("category", "error_code", "code", "reason", "marker", "kind"):
            if _is_field_marker(source.get(key)):
                return True
        for key in ("message", "detail", "error", "reason"):
            value = source.get(key)
            if isinstance(value, str) and _plain_is_field_collision(value):
                return True
    return False


def _metadata_category(meta: Mapping[str, object]) -> str | None:
    """Return an explicitly labelled category from structured metadata."""
    for source in _metadata_sources(meta):
        for key in ("category", "failure_category"):
            value = source.get(key)
            if isinstance(value, str):
                candidate = _code_key(value).lower()
                if candidate in _CATEGORY_SET:
                    return candidate
    return None


def _metadata_codes(meta: Mapping[str, object]):
    """Yield candidate error-code values from known metadata fields only."""
    for source in _metadata_sources(meta):
        for key in ("error_code", "code", "reason", "type", "failure_code"):
            value = source.get(key)
            if isinstance(value, str):
                yield value
            elif isinstance(value, int) and not isinstance(value, bool):
                # Some providers report a numeric HTTP/error code here.
                yield str(value)


def _normalized_finish_reason(value: object) -> str:
    """Normalize a finish reason to a bounded lowercase token."""
    text = _as_text(value).strip().lower()
    if len(text) > 64:
        return ""
    return text.replace("-", "_").replace(" ", "_")


def _metadata_finish_reason(meta: Mapping[str, object]) -> str:
    """Return a bounded finish reason from structured metadata."""
    for source in _metadata_sources(meta):
        for key in ("finish_reason", "stop_reason"):
            reason = _normalized_finish_reason(source.get(key))
            if reason:
                return reason
        choices = source.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            reason = _normalized_finish_reason(choices[0].get("finish_reason"))
            if reason:
                return reason
    return ""


def _safe_http_status(value: object) -> int | None:
    """Accept only a real HTTP status integer in the standard range."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        status = value
    elif isinstance(value, str):
        text = value.strip()
        if len(text) != 3 or not text.isdigit():
            return None
        status = int(text)
    else:
        return None
    if 100 <= status <= 599:
        return status
    return None


def _metadata_http_status(meta: Mapping[str, object]) -> int | None:
    """Return a bounded HTTP status from known structured metadata fields."""
    for source in _metadata_sources(meta):
        for key in ("http_status", "status_code", "status"):
            status = _safe_http_status(source.get(key))
            if status is not None:
                return status
    return None


def _finish_reason_category(reason: str) -> str | None:
    """Map an already-normalized finish reason to a diagnostic category."""
    if reason in _FINISH_REASON_OUTPUT_LENGTH:
        return CATEGORY_OUTPUT_LENGTH
    if reason in _FINISH_REASON_INVALID_RESPONSE:
        return CATEGORY_INVALID_RESPONSE
    return None


def _http_status_category(status: int | None) -> str | None:
    """Map an HTTP status to a diagnostic class without exposing the body."""
    if status is None:
        return None
    if 400 <= status <= 599:
        # The request did not produce a usable model response (including
        # authentication, quota, rate limit and server errors).
        return CATEGORY_TRANSPORT
    # HTTP success alone cannot distinguish schema, semantic or parse failure.
    # A specific category requires its own diagnostic evidence.
    return None


def _plain_detail_category(probe: str) -> str | None:
    """Classify only fixed legacy message prefixes from opaque detail text."""
    lowered = _plain_probe(probe).lower()
    if not lowered:
        return None
    for prefix, category in _PLAIN_DETAIL_PREFIXES:
        if lowered.startswith(prefix):
            return category
    return None


def classify_failure(error_code: object, detail: object = None) -> str:
    """Return one stable category for an extraction failure.

    ``error_code`` is the external/legacy job error code (for example
    ``MODEL_UNAVAILABLE`` or ``MODEL_REVIEW_REQUIRED``); it is inspected, not
    rewritten or returned. ``detail`` may be a plain message emitted by the
    existing adapters or a structured JSON object/string containing known
    metadata fields. Arbitrary detail text is ignored, and the result is always
    exactly one of :data:`CATEGORIES` -- never a message, URL or exception
    string.

    Precedence: the constant ``FIELD_IDENTITY_COLLISION:`` marker wins (a
    response was received and failed backend validation); an explicit
    non-generic error code wins next; then safe structured metadata (category,
    nested error codes, finish reason, HTTP status); then fixed legacy message
    prefixes; otherwise the evidence is insufficient. The generic legacy
    ``MODEL_UNAVAILABLE`` code is never guessed to be transport without
    supporting detail.
    """
    plain = _plain_probe(_as_text(detail))
    meta = _structured_metadata(detail)
    key = _code_key(error_code)

    # A field collision is a response that was already received and then
    # failed validation. It must never be grouped with an unknown transport
    # outcome merely because the adapter used the generic MODEL_UNAVAILABLE
    # code for it.
    if _is_field_marker(error_code) or _plain_is_field_collision(plain):
        return CATEGORY_FIELD_IDENTITY_COLLISION
    if meta is not None and _metadata_has_field_collision(meta):
        return CATEGORY_FIELD_IDENTITY_COLLISION

    # Explicit non-generic external codes are authoritative.
    if key and key not in _GENERIC_CODES:
        category = _DIRECT_CODE_CATEGORIES.get(key)
        if category is not None:
            return category

    # Safe structured metadata is more specific than opaque legacy text.
    if meta is not None:
        category = _metadata_category(meta)
        if category is not None:
            return category
        for candidate in _metadata_codes(meta):
            candidate_key = _code_key(candidate)
            if not candidate_key or candidate_key in _GENERIC_CODES:
                continue
            category = _DIRECT_CODE_CATEGORIES.get(candidate_key)
            if category is not None:
                return category
            if len(candidate_key) == 3 and candidate_key.isdigit():
                category = _http_status_category(int(candidate_key))
                if category is not None:
                    return category
        category = _finish_reason_category(_metadata_finish_reason(meta))
        if category is not None:
            return category
        category = _http_status_category(_metadata_http_status(meta))
        if category is not None:
            return category

    # Fixed legacy message prefixes. This intentionally does not inspect the
    # rest of the message, so a secret or URL in an exception string cannot
    # influence or enter the result.
    category = _plain_detail_category(plain)
    if category is not None:
        return category

    if len(key) == 3 and key.isdigit():
        category = _http_status_category(int(key))
        if category is not None:
            return category

    return CATEGORY_INSUFFICIENT_METADATA


def _normalized_category(value: object) -> str:
    """Coerce a category argument to the closed vocabulary.

    Legacy codes and category spellings are accepted for compatibility; any
    unknown value degrades to ``insufficient_metadata`` instead of being
    echoed.
    """
    if isinstance(value, str):
        candidate = value.strip().lower().replace("-", "_").replace(" ", "_")
        if candidate in _CATEGORY_SET:
            return candidate
        classified = classify_failure(value)
        if classified in _CATEGORY_SET:
            return classified
    return CATEGORY_INSUFFICIENT_METADATA


def _normalized_billing_outcome(value: object) -> str:
    """Return a closed, bounded billing outcome."""
    text = _as_text(value).strip().lower()
    if len(text) > 32:
        return "unknown"
    text = text.replace("-", "_").replace(" ", "_")
    return text if text in _BILLING_OUTCOMES else "unknown"


def _safe_finish_reason(value: object) -> str | None:
    """Return a finish reason only when it is in the closed safe vocabulary."""
    reason = _normalized_finish_reason(value)
    if not reason or len(reason) > 32:
        return None
    return reason if reason in _SAFE_FINISH_REASONS else None


def failure_detail(
    category: object,
    *,
    http_status: object = None,
    finish_reason: object = None,
    usage_present: object = False,
    billing_outcome: object = "unknown",
    validation: object = None,
    segment_ranges: object = None,
) -> str:
    """Serialize bounded, allowlisted failure metadata as JSON.

    The payload contains only a validated category, an optional standard HTTP
    status, an optional allowlisted finish reason, a boolean usage-presence
    flag and an optional closed-vocabulary billing outcome. Provider messages,
    response content, URLs, headers and usage quantities are never accepted.

    When the category has no received response (``transport`` or
    ``insufficient_metadata``) and the billing outcome is unknown, the legacy
    compatibility fields are emitted: ``billing_outcome="unknown"``,
    ``provider_usage=null`` and
    ``reason="external_request_outcome_unknown"``. A field-identity collision
    (and every other response-received category) is not reported as an unknown
    external request outcome; it never emits that reason.
    """
    resolved = _normalized_category(category)
    usage_flag = bool(usage_present)
    payload: dict[str, object] = {
        "category": resolved,
        "usage_present": usage_flag,
    }

    status = _safe_http_status(http_status)
    if status is not None:
        payload["http_status"] = status

    reason = _safe_finish_reason(finish_reason)
    if reason is not None:
        payload["finish_reason"] = reason

    billing = _normalized_billing_outcome(billing_outcome)
    if resolved in _UNKNOWN_OUTCOME_CATEGORIES and billing == "unknown":
        # Preserve the legacy detail shape exactly for consumers that already
        # understand an unknown external request outcome.
        payload["billing_outcome"] = "unknown"
        payload["provider_usage"] = None
        payload["reason"] = EXTERNAL_REQUEST_OUTCOME_UNKNOWN
    elif billing != "unknown":
        payload["billing_outcome"] = billing
        payload["provider_usage"] = None
    elif usage_flag:
        # A response was received and usage metadata was observed, but the
        # billing outcome itself is not asserted. Raw usage is still omitted.
        payload["provider_usage"] = None

    # These details originate in our schema validator, never a provider body.
    if isinstance(validation, dict):
        from .page_validation import schema_details
        allowed_codes = {"missing_required","unknown_property","wrong_type","enum","constant","upper_bound","lower_bound","pattern","any_of","one_of","other_validation"}
        import re
        items=[]
        for item in (validation.get("issues") or [])[:8]:
            if isinstance(item,dict) and item.get("code") in allowed_codes and isinstance(item.get("path"),str) and re.fullmatch(r"\$(?:\.[A-Za-z_*][A-Za-z_0-9*-]*|\[\d{1,8}\])*",item["path"]) and len(item["path"])<=200:
                # Re-sanitize property names against the schema, even for callers outside this pipeline.
                from types import SimpleNamespace
                safe=schema_details([SimpleNamespace(path=item["path"],message="")])["issues"][0]
                safe["code"]=item["code"]
                if item.get("comparison") in {"<=",">=","<",">"} and type(item.get("bound")) is int and 0<=item["bound"]<=999999999:
                    safe.update(comparison=item["comparison"],bound=item["bound"])
                if isinstance(item.get("expected_types"),list):
                    safe["expected_types"]=[t for t in item["expected_types"][:7] if t in {"object","array","string","integer","number","boolean","null"}]

                items.append(safe)
        payload["validation"]={"issues":items,"total_issue_count":min(10000,max(0,validation.get("total_issue_count",0))) if type(validation.get("total_issue_count")) is int else len(items)}
    if isinstance(segment_ranges,list):
        payload["segment_ranges"]=[{k:v for k,v in item.items() if k in {"part_index","start_byte","end_byte","window_index"} and type(v) is int and 0<=v<=2**63-1} for item in segment_ranges[:8] if isinstance(item,dict)]
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)
