"""Bounded, checkpointed extraction over long sanitized event groups.

The model-facing extractor still sees only sanitized text, but a group larger
than ``max_bytes`` is split into deterministic UTF-8-safe windows.  Each
window is a :class:`~knowledge_weaver.extraction.proposals.SanitizedPart`
produced with :func:`dataclasses.replace`, so the original event/part identity
and ``body_sha256`` are retained and any support quote remains an exact
substring of the full stored part.  The real pipeline later resolves the merged
proposal against the *full* original parts; this wrapper never grants authority
or touches jobs, claims or source lifecycle.

Successful leaves are checkpointed in the local, rebuildable SQLite table
``extraction_segment_cache``.  The cache key covers the original group hashes,
model/prompt/policy versions, the previous validation error, the known-claims
context and the byte range.  Only schema-valid, secret-checked proposals are
stored; raw source text, raw exceptions and credentials are never stored.

If the inner extractor raises a diagnostic with category ``output_length`` (or
the fixed message flag), the wrapper deterministically bisects that range down
to :data:`MIN_BISECT_BYTES` / ``max_depth``.  A persisted split marker means a
retry enters the already-created children instead of calling the failed parent
again.  If the call budget or bisection bounds are exhausted, a retryable
``ExtractionError`` with a safe fixed message is raised; a partial result is
never returned as success.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from knowledge_weaver import contracts
from knowledge_weaver.store.db import utc_now

from .proposals import ExtractionError, SanitizedPart

#: Local rebuildable checkpoint table (not part of the portable snapshot).
SEGMENT_CACHE_TABLE = "extraction_segment_cache"

#: Smallest range bisection may stop at.
MIN_BISECT_BYTES = 512

#: Fixed token used by providers to report an output-length failure.  A
#: provider may also set ``exception.category = "output_length"``.
OUTPUT_LENGTH_FLAG = "output_length"

_CACHE_TABLE_SQL = (
    f"CREATE TABLE IF NOT EXISTS {SEGMENT_CACHE_TABLE} ("
    "cache_key TEXT PRIMARY KEY, group_hash TEXT NOT NULL, "
    "kind TEXT NOT NULL, "
    "payload_json TEXT NOT NULL, "
    "created_at TEXT NOT NULL)"
)

_SAFE_OUTPUT_LENGTH_MESSAGE = "segment extraction exceeded its output length bound"
_SAFE_BUDGET_MESSAGE = "segment extraction call budget exhausted"
_SAFE_DISPOSITION_MESSAGE = "segment extraction leaf has no claims or explicit exclusions"
_SAFE_SCHEMA_MESSAGE = "segment proposal failed schema validation"
_SAFE_SECRET_MESSAGE = "segment proposal contains secret content"
_SAFE_MERGE_MESSAGE = "merged segment proposal is not schema-valid"


def _canonical(value: Any) -> str:
    """Canonical compact JSON used for hashes, keys and claim de-duplication."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _safe_error(message: str) -> ExtractionError:
    categories = {_SAFE_OUTPUT_LENGTH_MESSAGE: "output_length", _SAFE_BUDGET_MESSAGE: "call_budget", _SAFE_DISPOSITION_MESSAGE: "missing_disposition", _SAFE_SECRET_MESSAGE: "secret_content", "segment coverage incomplete": "coverage_incomplete"}
    return ExtractionError("MODEL_UNAVAILABLE", message, retryable=True, category=categories.get(message, "schema_invalid"))


def _is_output_length(exc: BaseException) -> bool:
    """Detect the provider's output-length diagnostic without trusting text."""
    category = getattr(exc, "category", None)
    if isinstance(category, str) and category.strip().lower() == OUTPUT_LENGTH_FLAG:
        return True
    message = str(exc).lower()
    return OUTPUT_LENGTH_FLAG in message or "output length" in message


def _boundary_down(data: bytes, index: int, floor: int = 0) -> int:
    """Largest UTF-8 boundary at or below ``index`` (never below ``floor``)."""
    if index >= len(data):
        return len(data)
    while index > floor and (data[index] & 0xC0) == 0x80:
        index -= 1
    return index


def _next_boundary(data: bytes, index: int, limit: int) -> int:
    """Smallest UTF-8 boundary strictly after ``index`` (at most ``limit``)."""
    index = min(index + 1, limit)
    while index < limit and (data[index] & 0xC0) == 0x80:
        index += 1
    return index


def _utf8_windows(text: str, max_bytes: int, overlap_bytes: int) -> Iterable[tuple[int, int]]:
    """Deterministic, bounded, overlapping windows over ``text`` bytes.

    ``start``/``end`` are UTF-8 byte offsets and always land on character
    boundaries.  Every byte of the original text is in at least one window and
    each window is at most ``max_bytes`` long.  Consecutive windows overlap by
    at least ``overlap_bytes`` where the character geometry permits it.
    """
    data = text.encode("utf-8")
    total = len(data)
    if total <= max_bytes:
        yield 0, total
        return
    start = 0
    while start < total:
        end = _boundary_down(data, min(start + max_bytes, total), floor=start)
        if end <= start:
            end = _next_boundary(data, start, total)
        yield start, end
        if end >= total:
            break
        next_start = _boundary_down(data, end - overlap_bytes, floor=start)
        if next_start <= start:
            next_start = _next_boundary(data, start, total)
        start = next_start


@dataclass(frozen=True)
class _SegmentRange:
    """One window inside one original sanitized part (UTF-8 byte range)."""

    event_id: str
    event_revision: int
    part_index: int
    start_byte: int
    end_byte: int
    window_index: int

    @property
    def size(self) -> int:
        return self.end_byte - self.start_byte

    def record(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_revision": self.event_revision,
            "part_index": self.part_index,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "bytes": self.size,
            "window_index": self.window_index,
        }


@dataclass(frozen=True)
class _Page:
    """A bounded set of ranges sent to the inner extractor in one call."""

    ranges: tuple[_SegmentRange, ...]
    depth: int

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.ranges)


def _part_key(part: SanitizedPart) -> tuple[str, int, int]:
    return (part.event_id, int(part.event_revision), int(part.part_index))


def _plan_pages(parts: list[SanitizedPart], max_bytes: int, overlap_bytes: int) -> list[_Page]:
    ranges_by_part: dict[tuple[str, int, int], list[_SegmentRange]] = {}
    for part in parts:
        key = _part_key(part)
        ranges_by_part[key] = [
            _SegmentRange(
                event_id=part.event_id,
                event_revision=int(part.event_revision),
                part_index=int(part.part_index),
                start_byte=start,
                end_byte=end,
                window_index=index,
            )
            for index, (start, end) in enumerate(
                _utf8_windows(part.text, max_bytes, overlap_bytes)
            )
        ]

    pages: list[_Page] = []
    current: list[_SegmentRange] = []
    current_parts: set[tuple[str, int, int]] = set()
    current_bytes = 0
    for part in parts:
        for item in ranges_by_part[_part_key(part)]:
            if current and (
                _part_key(part) in current_parts or current_bytes + item.size > max_bytes
            ):
                pages.append(_Page(tuple(current), 0))
                current = []
                current_parts = set()
                current_bytes = 0
            current.append(item)
            current_parts.add(_part_key(part))
            current_bytes += item.size
    if current:
        pages.append(_Page(tuple(current), 0))
    return pages


def _full_pages(parts: list[SanitizedPart]) -> list[_Page]:
    """Coverage-only representation of the unchanged short-group delegation."""
    if not parts:
        return []
    return [
        _Page(
            tuple(
                _SegmentRange(
                    event_id=part.event_id,
                    event_revision=int(part.event_revision),
                    part_index=int(part.part_index),
                    start_byte=0,
                    end_byte=len(part.text.encode("utf-8")),
                    window_index=0,
                )
                for part in parts
            ),
            0,
        )
    ]


def _coverage(parts: list[SanitizedPart], pages: list[_Page], *, delegated: bool) -> dict:
    """Source-text-free coverage metadata for tests and diagnostics."""
    total_bytes = sum(len(part.text.encode("utf-8")) for part in parts)
    records: list[dict] = []
    intervals: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    for page_index, page in enumerate(pages):
        for item in page.ranges:
            interval_key = (item.event_id, item.event_revision, item.part_index)
            intervals.setdefault(interval_key, []).append((item.start_byte, item.end_byte))
            record = item.record()
            record["page"] = page_index
            record["depth"] = page.depth
            records.append(record)
    covered = 0
    for values in intervals.values():
        merged_start, merged_end = -1, -1
        for start, end in sorted(values):
            if start > merged_end:
                if merged_end >= 0:
                    covered += merged_end - merged_start
                merged_start, merged_end = start, end
            else:
                merged_end = max(merged_end, end)
        if merged_end >= 0:
            covered += merged_end - merged_start
    return {
        "total_bytes": total_bytes,
        "covered_bytes": covered,
        "complete": covered == total_bytes,
        "delegated_unchanged": delegated,
        "pages": len(pages),
        "windows": records,
        "segments": records,
    }


@dataclass
class _State:
    calls: int = 0
    leaves: list[tuple[_Page, dict]] = field(default_factory=list)


class CheckpointingExtractor:
    """Segment a long sanitized group and checkpoint successful leaves."""

    def __init__(
        self,
        store,
        inner,
        *,
        max_bytes: int = 8192,
        overlap_bytes: int = 512,
        max_depth: int = 4,
        max_calls: int = 64,
    ) -> None:
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if not isinstance(overlap_bytes, int) or overlap_bytes < 0:
            raise ValueError("overlap_bytes must be a non-negative integer")
        if overlap_bytes >= max_bytes:
            raise ValueError("overlap_bytes must be smaller than max_bytes")
        if not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if not isinstance(max_calls, int) or max_calls < 0:
            raise ValueError("max_calls must be a non-negative integer")
        self.store = store
        self.inner = inner
        self.max_bytes = max_bytes
        self.overlap_bytes = overlap_bytes
        self.max_depth = max_depth
        self.max_calls = max_calls
        self._table_ready = False
        self._coverage: dict = _coverage([], [], delegated=True)

    # -- test-facing metadata (never contains source text) -----------------
    @property
    def coverage(self) -> dict:
        """Coverage metadata from the most recent :meth:`extract` call."""
        return self._coverage

    @property
    def last_coverage(self) -> dict:
        return self._coverage

    def coverage_metadata(self) -> dict:
        return self._coverage

    # -- cache ---------------------------------------------------------------
    def _ensure_cache_table(self) -> None:
        if self._table_ready:
            return
        with self.store.transaction() as conn:
            conn.execute(_CACHE_TABLE_SQL)
        self._table_ready = True

    def _cache_get(self, key: str, kind: str) -> dict | None:
        with self.store.read() as conn:
            row = conn.execute(
                f"SELECT kind, payload_json FROM {SEGMENT_CACHE_TABLE} WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None or row["kind"] != kind:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _cache_put(self, key: str, kind: str, payload: dict) -> None:
        encoded = _canonical(payload)
        with self.store.transaction() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {SEGMENT_CACHE_TABLE} "
                "(cache_key, group_hash, kind, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (key, self._group_hash, kind, encoded, utc_now()),
            )

    def _cache_key(self, group_hash: str, context: dict, page: _Page) -> str:
        key_material = {
            "group_hash": group_hash,
            "segmentation_policy": "bytes-v1",
            "max_bytes": self.max_bytes,
            "overlap_bytes": self.overlap_bytes,
            "max_depth": self.max_depth,
            "model_version": context.get("model_version"),
            "prompt_version": context.get("prompt_version"),
            "policy_version": context.get("policy_version")
            or contracts.redaction_policy_version(),
            "previous_validation_error": context.get("previous_validation_error"),
            "known_claims": context.get("known_claims") or [],
            "known_claim_details": context.get("known_claim_details") or [],
            "event_id": context.get("event_id"),
            "event_revision": context.get("event_revision"),
            "project_ids": context.get("project_ids") or [],
            "depth": page.depth,
            "ranges": [
                [item.event_id, item.event_revision, item.part_index, item.start_byte, item.end_byte]
                for item in page.ranges
            ],
        }
        if "source_scope" in context:
            key_material["source_scope"] = context["source_scope"]
        if context.get("prompt_version") in ("prompt:8", "prompt:9", "prompt:10"):
            # Cached pages are revalidated against current anchors before reuse.
            for volatile in ("previous_validation_error", "known_claims", "known_claim_details"):
                key_material.pop(volatile, None)
        return contracts.sha256_hex(_canonical(key_material))

    # -- windows / context ---------------------------------------------------
    def _window_parts(
        self, page: _Page, parts_by_key: dict[tuple[str, int, int], SanitizedPart]
    ) -> list[SanitizedPart]:
        windows: list[SanitizedPart] = []
        for item in page.ranges:
            part = parts_by_key[(item.event_id, item.event_revision, item.part_index)]
            data = part.text.encode("utf-8")
            if item.end_byte > len(data) or item.start_byte < 0:
                raise _safe_error(_SAFE_SCHEMA_MESSAGE)
            text = data[item.start_byte:item.end_byte].decode("utf-8")
            windows.append(dataclasses.replace(part, text=text))
        return windows

    @staticmethod
    def _segment_context(context: dict, page: _Page) -> dict:
        base = dict(context) if isinstance(context, dict) else {}
        ranges = [item.record() for item in page.ranges]
        base["segment"] = {
            "depth": page.depth,
            "bytes": page.total_bytes,
            "ranges": ranges,
        }
        base["segment_ranges"] = ranges
        base["segment_depth"] = page.depth
        return base

    # -- proposal checks -----------------------------------------------------
    @staticmethod
    def _validate_proposal(proposal: object) -> dict:
        if not isinstance(proposal, dict):
            raise _safe_error(_SAFE_SCHEMA_MESSAGE)
        try:
            issues = contracts.validate_extraction_output(proposal)
            serialized = contracts.compact_json(proposal)
        except (TypeError, ValueError):
            raise _safe_error(_SAFE_SCHEMA_MESSAGE) from None
        if issues:
            from .page_validation import validation_error
            raise validation_error("schema_invalid", issues=issues)
        if contracts.secret_kinds(serialized):
            raise _safe_error(_SAFE_SECRET_MESSAGE)
        return proposal

    @staticmethod
    def _has_explicit_disposition(page: _Page, proposal: dict) -> bool:
        claims = proposal.get("claims") or []
        if claims:
            return True
        excluded = {
            value for value in (proposal.get("excluded_event_ids") or []) if isinstance(value, str)
        }
        if not excluded:
            return False
        # A zero-claim leaf must say explicitly that the event(s) it covered
        # carry nothing to remember; silence is not an implicit exclusion.
        page_events = {item.event_id for item in page.ranges}
        return bool(page_events) and all(event in excluded for event in page_events)

    # -- bisection -----------------------------------------------------------
    def _bisect(
        self, page: _Page, parts_by_key: dict[tuple[str, int, int], SanitizedPart]
    ) -> tuple[_Page, ...] | None:
        if page.depth >= self.max_depth or page.total_bytes <= MIN_BISECT_BYTES:
            return None
        if len(page.ranges) >= 2:
            split_at = len(page.ranges) - 1
            accumulated = 0
            for index in range(1, len(page.ranges)):
                accumulated += page.ranges[index - 1].size
                if accumulated * 2 >= page.total_bytes:
                    split_at = index
                    break
            left = tuple(page.ranges[:split_at])
            right = tuple(page.ranges[split_at:])
            if not left or not right:
                return None
            return (
                _Page(left, page.depth + 1),
                _Page(right, page.depth + 1),
            )

        item = page.ranges[0]
        if item.size <= MIN_BISECT_BYTES:
            return None
        part = parts_by_key[(item.event_id, item.event_revision, item.part_index)]
        data = part.text.encode("utf-8")
        midpoint = (item.start_byte + item.end_byte) // 2
        midpoint = _boundary_down(data, midpoint, floor=item.start_byte)
        if midpoint <= item.start_byte:
            midpoint = _next_boundary(data, item.start_byte, item.end_byte)
        if midpoint >= item.end_byte:
            return None
        return (
            _Page((dataclasses.replace(item, end_byte=midpoint),), page.depth + 1),
            _Page((dataclasses.replace(item, start_byte=midpoint),), page.depth + 1),
        )

    @staticmethod
    def _page_record(page: _Page) -> dict:
        return {
            "depth": page.depth,
            "ranges": [
                {
                    "event_id": item.event_id,
                    "event_revision": item.event_revision,
                    "part_index": item.part_index,
                    "start_byte": item.start_byte,
                    "end_byte": item.end_byte,
                    "window_index": item.window_index,
                }
                for item in page.ranges
            ],
        }

    @staticmethod
    def _children_from_marker(payload: dict) -> tuple[_Page, ...] | None:
        records = payload.get("children")
        if not isinstance(records, list) or not records:
            return None
        children: list[_Page] = []
        try:
            for record in records:
                if not isinstance(record, dict):
                    return None
                depth = int(record["depth"])
                raw_ranges = record["ranges"]
                if not isinstance(raw_ranges, list) or not raw_ranges:
                    return None
                ranges = tuple(
                    _SegmentRange(
                        event_id=str(item["event_id"]),
                        event_revision=int(item["event_revision"]),
                        part_index=int(item["part_index"]),
                        start_byte=int(item["start_byte"]),
                        end_byte=int(item["end_byte"]),
                        window_index=int(item.get("window_index", 0)),
                    )
                    for item in raw_ranges
                    if isinstance(item, dict)
                )
                if not ranges:
                    return None
                children.append(_Page(ranges, depth))
        except (KeyError, TypeError, ValueError):
            return None
        return tuple(children)

    # -- one page ------------------------------------------------------------
    def _process_page(
        self,
        page: _Page,
        context: dict,
        parts_by_key: dict[tuple[str, int, int], SanitizedPart],
        group_hash: str,
        state: _State,
    ) -> None:
        key = self._cache_key(group_hash, context, page)

        marker = self._cache_get(key, "split")
        if marker is not None:
            children = self._children_from_marker(marker)
            if children:
                for child in children:
                    self._process_page(child, context, parts_by_key, group_hash, state)
                return

        cached = self._cache_get(key, "result")
        if cached is not None:
            try:
                cached = self._validate_proposal(cached)
                if not self._has_explicit_disposition(page, cached):
                    raise _safe_error(_SAFE_DISPOSITION_MESSAGE)
                if context.get("prompt_version") in ("prompt:8", "prompt:9", "prompt:10"):
                    self.inner.validate(cached, self._window_parts(page, parts_by_key), self._segment_context(context,page))
            except ExtractionError as exc:
                if context.get("prompt_version") not in ("prompt:8", "prompt:9", "prompt:10"):
                    raise
                with self.store.transaction() as conn:
                    conn.execute(f"DELETE FROM {SEGMENT_CACHE_TABLE} WHERE cache_key=?", (key,))
                self.inner.cached_error = exc
            else:
                state.leaves.append((page, cached))
                return

        if state.calls >= self.max_calls:
            raise _safe_error(_SAFE_BUDGET_MESSAGE)
        state.calls += 1
        windows = self._window_parts(page, parts_by_key)
        try:
            proposal = self.inner.extract(windows, self._segment_context(context, page))
        except Exception as exc:
            if not _is_output_length(exc):
                raise
            children = self._bisect(page, parts_by_key)
            if not children:
                raise _safe_error(_SAFE_OUTPUT_LENGTH_MESSAGE) from None
            self._cache_put(
                key,
                "split",
                {"children": [self._page_record(child) for child in children]},
            )
            for child in children:
                self._process_page(child, context, parts_by_key, group_hash, state)
            return

        proposal = self._validate_proposal(proposal)
        if not self._has_explicit_disposition(page, proposal):
            raise _safe_error(_SAFE_DISPOSITION_MESSAGE)
        self._cache_put(key, "result", proposal)
        state.leaves.append((page, proposal))

    # -- merge ---------------------------------------------------------------
    @staticmethod
    def _merge(leaves: list[tuple[_Page, dict]]) -> dict:
        claims: list[dict] = []
        seen_claims: set[str] = set()
        warnings: list[str] = []
        for _page, proposal in leaves:
            warnings.extend(str(item) for item in (proposal.get("warnings") or []))
            for claim in proposal.get("claims") or []:
                canonical = _canonical(claim)
                if canonical in seen_claims:
                    continue
                seen_claims.add(canonical)
                claims.append(claim)
        if len(claims) > 100:
            raise _safe_error(_SAFE_MERGE_MESSAGE)

        claimed_events: set[str] = set()
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            for reference in claim.get("evidence") or []:
                if isinstance(reference, dict) and isinstance(reference.get("event_id"), str):
                    claimed_events.add(reference["event_id"])

        leaves_by_event: dict[str, list[set[str]]] = {}
        for page, proposal in leaves:
            excluded = {
                value
                for value in (proposal.get("excluded_event_ids") or [])
                if isinstance(value, str)
            }
            for item in page.ranges:
                leaves_by_event.setdefault(item.event_id, []).append(excluded)

        excluded_events: set[str] = set()
        for event_id, exclusion_sets in leaves_by_event.items():
            if event_id in claimed_events:
                continue
            if all(event_id in excluded for excluded in exclusion_sets):
                excluded_events.add(event_id)

        return {
            "schema_version": "1.0",
            "claims": claims,
            "excluded_event_ids": sorted(excluded_events),
            "warnings": warnings,
        }

    # -- public interface ----------------------------------------------------
    def discard_proposals(self, proposals):
        """Remove semantically rejected leaves, preserving usable siblings.

        A schema-valid proposal is not necessarily safe to commit. Without this
        invalidation an unchanged repair context can indefinitely replay the same
        rejected page instead of giving a bounded retry a chance to repair it.
        """
        if not self._table_ready:
            return 0
        removed = 0
        with self.store.transaction() as conn:
            for encoded in {_canonical(item) for item in proposals}:
                removed += conn.execute(
                    f"DELETE FROM {SEGMENT_CACHE_TABLE} WHERE group_hash=? "
                    "AND kind='result' AND payload_json=?", (self._group_hash, encoded)
                ).rowcount
        return removed

    def clear_checkpoints_in_transaction(self, conn):
        if self._table_ready:
            conn.execute(f"DELETE FROM {SEGMENT_CACHE_TABLE} WHERE group_hash=?", (self._group_hash,))

    def extract_proposals(self, parts, context):
        """Return schema-valid bounded proposals; the pipeline resolves them atomically.

        This private pipeline interface permits a long source to contain over100
        facts without relaxing the100-claim limit of EACH model proposal.
        """
        parts_list = list(parts)
        total_bytes = sum(len(part.text.encode("utf-8")) for part in parts_list)
        base_context = context if isinstance(context, dict) else {}
        pages = _plan_pages(parts_list, self.max_bytes, self.overlap_bytes)
        self._group_hash = contracts.sha256_hex(_canonical([
            [part.event_id, int(part.event_revision), int(part.part_index), part.body_sha256]
            for part in parts_list]))
        self._coverage = _coverage(parts_list, [], delegated=False)
        self._leaf_results = []
        resume_short = False
        if total_bytes <= self.max_bytes and parts_list:
            with self.store.read() as conn:
                existing = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (SEGMENT_CACHE_TABLE,)).fetchone()
            if existing:
                self._table_ready = True
                page = _full_pages(parts_list)[0]
                key = self._cache_key(self._group_hash, base_context, page)
                resume_short = self._cache_get(key, "split") is not None or self._cache_get(key, "result") is not None
        if total_bytes <= self.max_bytes and not resume_short:
            # Small ordinary records retain the previous single-call path.
            try:
                proposal = self.inner.extract(parts_list, context)
            except Exception as exc:
                if not _is_output_length(exc):
                    raise
                self._ensure_cache_table()
                parts_by_key = {_part_key(part): part for part in parts_list}
                page = _full_pages(parts_list)[0] if parts_list else None
                children = self._bisect(page, parts_by_key) if page else None
                if not children:
                    raise _safe_error(_SAFE_OUTPUT_LENGTH_MESSAGE) from None
                key = self._cache_key(self._group_hash, base_context, page)
                self._cache_put(key, "split", {"children": [self._page_record(child) for child in children]})
                pages = [page]
            else:
                if contracts.validate_extraction_output(proposal):
                    raise _safe_error(_SAFE_SCHEMA_MESSAGE)
                self._coverage = _coverage(parts_list, _full_pages(parts_list), delegated=True)
                self._leaf_results = [(_full_pages(parts_list)[0] if parts_list else _Page((),0), proposal)]
                return [proposal]
        self._ensure_cache_table()
        state = _State()
        parts_by_key = {_part_key(part): part for part in parts_list}
        try:
            for page in pages:
                self._process_page(page, base_context, parts_by_key, self._group_hash, state)
        finally:
            self._coverage = _coverage(parts_list, [page for page,_ in state.leaves], delegated=False)
        if not self._coverage["complete"]:
            raise _safe_error("segment coverage incomplete")
        self._leaf_results = state.leaves
        return [proposal for _,proposal in state.leaves]

    def merged_metadata(self):
        # Exclusion is source-wide only when every leaf excluded that source.
        exclusions = {}
        claimed_events = set()
        warnings = []
        for page,proposal in self._leaf_results:
            warnings.extend(proposal.get("warnings") or [])
            for claim in proposal.get("claims") or []:
                claimed_events.update(ref["event_id"] for ref in claim.get("evidence", []))
            for segment in page.ranges:
                exclusions.setdefault(segment.event_id, []).append(segment.event_id in (proposal.get("excluded_event_ids") or []))
        return {"warnings": warnings, "excluded_event_ids": [eid for eid,values in exclusions.items() if all(values) and eid not in claimed_events]}

    def extract(self, parts, context):
        self.extract_proposals(parts, context)
        if self._coverage["delegated_unchanged"]:
            return self._leaf_results[0][1]
        merged = self._merge(self._leaf_results)
        if contracts.validate_extraction_output(merged):
            raise _safe_error(_SAFE_MERGE_MESSAGE)
        return merged


__all__ = ["CheckpointingExtractor", "MIN_BISECT_BYTES", "OUTPUT_LENGTH_FLAG", "SEGMENT_CACHE_TABLE"]
