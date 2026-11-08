"""Prompt:10 exact source-span transport helpers.

The HTTP extractor offers the model a bounded catalog of exact substrings of
each sanitized part window it already receives.  A model may then put an opaque
span marker in ``evidence.support_quote`` instead of retyping escaped JSON,
code or non-ASCII text.  Only a marker from this exact catalog, bound to the
exact supplied ``(event_id, event_revision, part_index)``, is expanded; an
unknown or misbound marker fails closed with category ``evidence_invalid``.

Raw literal support quotes remain accepted only when they are nonempty, at
most :data:`MAX_SPAN_CHARS` characters, and an exact substring of the supplied
part they name.  This module never invents, rewrites or truncates source text,
and it never edits claim values, facets, targets or warnings.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .proposals import ExtractionError

#: A support quote may not exceed this many characters.
MAX_SPAN_CHARS = 2000

#: Modest overlap for conservative splits of a single very long line.
LONG_LINE_OVERLAP_CHARS = 128

#: Opaque span identifiers are a truncated SHA-256 digest (128 bits).
_SPAN_ID_HEX_CHARS = 32

#: Prefer a whitespace/punctuation split point within this lookback distance.
_SOFT_BREAK_LOOKBACK_CHARS = 200

_SOFT_BREAK_CHARS = frozenset(",;、。！？!?")

SPAN_REFERENCE_PREFIX = "[[span:"
SPAN_REFERENCE_SUFFIX = "]]"
SPAN_REFERENCE_PATTERN = re.compile(
    rf"^\[\[span:([0-9a-f]{{{_SPAN_ID_HEX_CHARS}}})\]\]$"
)


def _canonical(value: object) -> str:
    """Canonical JSON used only for deterministic opaque identifiers."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


@dataclass(frozen=True)
class SourceSpan:
    """One exact, bounded substring of a supplied sanitized part window."""

    marker: str
    span_id: str
    event_id: str
    event_revision: int
    part_index: int
    quote: str
    start_char: int
    end_char: int

    def request_record(self) -> dict:
        """The public catalog record sent to the model (never logs)."""
        return {
            "marker": self.marker,
            "event_id": self.event_id,
            "event_revision": self.event_revision,
            "part_index": self.part_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quote": self.quote,
        }


class SpanCatalog:
    """Deterministic marker index for one supplied set of part windows."""

    def __init__(self, spans: Iterable[SourceSpan] = ()) -> None:
        self.spans = tuple(spans)
        self._by_marker = {span.marker: span for span in self.spans}

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(span.marker for span in self.spans)

    def get(self, marker: str) -> SourceSpan | None:
        return self._by_marker.get(marker)

    def request_records(self) -> list[dict]:
        return [span.request_record() for span in self.spans]


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _line_ranges(text: str) -> Iterable[tuple[int, int]]:
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        yield start, end
        start = end
    if start < len(text):
        yield start, len(text)


def _paragraph_ranges(text: str) -> Iterable[tuple[int, int]]:
    """Paragraphs separated by blank lines; each range keeps its line endings."""
    paragraph_start: int | None = None
    for line_start, line_end in _line_ranges(text):
        if paragraph_start is None:
            paragraph_start = line_start
        if not text[line_start:line_end].strip():
            yield paragraph_start, line_end
            paragraph_start = None
    if paragraph_start is not None:
        yield paragraph_start, len(text)


def _long_line_chunks(text: str, start: int, end: int) -> Iterable[tuple[int, int]]:
    """Split a single long line into bounded chunks with modest overlap."""
    while start < end:
        if end - start <= MAX_SPAN_CHARS:
            yield start, end
            return
        hard_end = start + MAX_SPAN_CHARS
        floor = max(
            start + (MAX_SPAN_CHARS // 2),
            hard_end - _SOFT_BREAK_LOOKBACK_CHARS,
        )
        cut = hard_end
        for index in range(hard_end, floor, -1):
            character = text[index - 1]
            if character.isspace() or character in _SOFT_BREAK_CHARS:
                cut = index
                break
        if cut <= start:
            cut = hard_end
        yield start, cut
        next_start = cut - LONG_LINE_OVERLAP_CHARS
        if next_start <= start:
            next_start = start + 1
        start = next_start


def _span_ranges(text: str) -> Iterable[tuple[int, int]]:
    """Every non-whitespace character belongs to at least one yielded span.

    Short paragraphs stay whole.  A long paragraph is packed from line ranges;
    a single line over :data:`MAX_SPAN_CHARS` is split conservatively with a
    fixed overlap.  Whitespace-only ranges are omitted, but no non-whitespace
    source character is skipped or truncated.
    """
    for paragraph_start, paragraph_end in _paragraph_ranges(text):
        start, end = _trimmed(text, paragraph_start, paragraph_end)
        if start >= end:
            continue
        if end - start <= MAX_SPAN_CHARS:
            yield start, end
            continue

        current_start: int | None = None
        current_end = 0
        cursor = paragraph_start
        for line in text[paragraph_start:paragraph_end].splitlines(keepends=True):
            line_start = cursor
            line_end = line_start + len(line)
            cursor = line_end
            line_start, line_end = _trimmed(text, line_start, line_end)
            if line_start >= line_end:
                continue
            if line_end - line_start > MAX_SPAN_CHARS:
                if current_start is not None:
                    yield current_start, current_end
                    current_start = None
                for chunk_start, chunk_end in _long_line_chunks(text, line_start, line_end):
                    chunk_start, chunk_end = _trimmed(text, chunk_start, chunk_end)
                    if chunk_start < chunk_end:
                        yield chunk_start, chunk_end
                continue
            if current_start is None:
                current_start, current_end = line_start, line_end
            elif line_end - current_start <= MAX_SPAN_CHARS:
                current_end = line_end
            else:
                yield current_start, current_end
                current_start, current_end = line_start, line_end
        if current_start is not None:
            yield current_start, current_end


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_span_catalog(parts: Sequence[object]) -> SpanCatalog:
    """Build the deterministic, bounded catalog for the supplied part windows.

    Identity includes the part binding, the part body hash, the supplied window
    text hash, the span offsets and the span text hash.  The opaque marker is a
    truncated SHA-256 digest; it does not expose source content.  Identical text
    in different parts therefore receives different markers.
    """
    spans: list[SourceSpan] = []
    seen: set[str] = set()
    for part in parts:
        text = part.text if isinstance(part.text, str) else ""
        event_id = str(part.event_id)
        event_revision = int(part.event_revision)
        part_index = int(part.part_index)
        body_sha256 = str(getattr(part, "body_sha256", "") or "")
        window_sha256 = _text_sha256(text)
        for start, end in _span_ranges(text):
            quote = text[start:end]
            if not quote or not quote.strip() or len(quote) > MAX_SPAN_CHARS:
                continue
            identity = [
                event_id,
                event_revision,
                part_index,
                body_sha256,
                window_sha256,
                start,
                end,
                _text_sha256(quote),
            ]
            span_id = hashlib.sha256(
                _canonical(identity).encode("utf-8")
            ).hexdigest()[:_SPAN_ID_HEX_CHARS]
            marker = f"{SPAN_REFERENCE_PREFIX}{span_id}{SPAN_REFERENCE_SUFFIX}"
            if marker in seen:
                continue
            seen.add(marker)
            spans.append(
                SourceSpan(
                    marker=marker,
                    span_id=span_id,
                    event_id=event_id,
                    event_revision=event_revision,
                    part_index=part_index,
                    quote=quote,
                    start_char=start,
                    end_char=end,
                )
            )
    return SpanCatalog(spans)


def _exact_key(reference: dict) -> tuple[str, int, int] | None:
    event_id = reference.get("event_id")
    event_revision = reference.get("event_revision")
    part_index = reference.get("part_index")
    if (
        not isinstance(event_id, str)
        or type(event_revision) is not int
        or type(part_index) is not int
    ):
        return None
    return event_id, event_revision, part_index


def _evidence_error() -> ExtractionError:
    # Fixed, source-free message: never echo a quote, marker or model output.
    return ExtractionError(
        "MODEL_UNAVAILABLE",
        "evidence span reference is unknown or not bound to the supplied source part",
        retryable=True,
        category="evidence_invalid",
    )


def resolve_span_references(
    proposal: object, catalog: SpanCatalog, parts: Sequence[object]
) -> object:
    """Resolve recognized markers to their original verbatim span.

    Any marker-like value that is unknown, malformed or bound to a different
    supplied part raises ``evidence_invalid``.  A raw literal quote is accepted
    only when it is exact and bound to a supplied part; all ordinary final
    evidence guards still run later.  No other proposal field is changed.
    """
    if not isinstance(proposal, dict):
        return proposal
    parts_by_key: dict[tuple[str, int, int], object] = {}
    for part in parts:
        if (
            isinstance(part.event_id, str)
            and type(part.event_revision) is int
            and type(part.part_index) is int
        ):
            parts_by_key[
                (part.event_id, part.event_revision, part.part_index)
            ] = part

    claims = proposal.get("claims")
    if not isinstance(claims, list):
        return proposal

    replacements: list[tuple[dict, str]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        references = claim.get("evidence")
        if not isinstance(references, list):
            continue
        for reference in references:
            if not isinstance(reference, dict):
                continue
            quote = reference.get("support_quote")
            if not isinstance(quote, str):
                continue
            if quote.startswith(SPAN_REFERENCE_PREFIX):
                match = SPAN_REFERENCE_PATTERN.fullmatch(quote)
                span = catalog.get(quote) if match is not None else None
                if span is None:
                    raise _evidence_error()
                binding = (span.event_id, span.event_revision, span.part_index)
                if _exact_key(reference) != binding:
                    raise _evidence_error()
                part = parts_by_key.get(binding)
                part_text = part.text if part is not None and isinstance(part.text, str) else ""
                if (
                    not span.quote
                    or len(span.quote) > MAX_SPAN_CHARS
                    or span.quote not in part_text
                ):
                    raise _evidence_error()
                replacements.append((reference, span.quote))
                continue

            # A raw literal is a compatibility path, still governed by the
            # ordinary exact-substring and part-binding requirement.
            binding = _exact_key(reference)
            part = parts_by_key.get(binding) if binding is not None else None
            part_text = part.text if part is not None and isinstance(part.text, str) else ""
            if (
                binding is None
                or part is None
                or not quote
                or len(quote) > MAX_SPAN_CHARS
                or quote not in part_text
            ):
                raise _evidence_error()

    for reference, quote in replacements:
        reference["support_quote"] = quote
    return proposal


__all__ = [
    "LONG_LINE_OVERLAP_CHARS",
    "MAX_SPAN_CHARS",
    "SPAN_REFERENCE_PATTERN",
    "SPAN_REFERENCE_PREFIX",
    "SPAN_REFERENCE_SUFFIX",
    "SourceSpan",
    "SpanCatalog",
    "build_span_catalog",
    "resolve_span_references",
]
