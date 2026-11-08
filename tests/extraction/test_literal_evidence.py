"""Prompt:10 exact source-span transport tests (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from knowledge_weaver.extraction import http_client
from knowledge_weaver.extraction.literal_evidence import (
    MAX_SPAN_CHARS,
    SPAN_REFERENCE_PATTERN,
    build_span_catalog,
    resolve_span_references,
)
from knowledge_weaver.extraction.proposals import ExtractionError


def _part(text, *, part_index=0, event_id="event-1", revision=1):
    return SimpleNamespace(
        event_id=event_id,
        event_revision=revision,
        part_index=part_index,
        text=text,
        body_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        role="tool",
        origin_kind="tool_execution",
        actor_id="actor-1",
        event_time={
            "value": "2026-01-01T00:00:00Z",
            "precision": "second",
            "timezone": "UTC",
        },
        project_ids=(),
        workspace_id="workspace-1",
    )


def _proposal(
    support_quote,
    *,
    event_id="event-1",
    revision=1,
    part_index=0,
):
    return {
        "schema_version": "1.0",
        "claims": [
            {
                "project_hint": None,
                "subject_hint": "server.primary",
                "predicate": "server.port",
                "facet": "observed",
                "target_claim_hint": None,
                "qualifiers": {"environment": "staging"},
                "value": {"port": 8080},
                "display_text": "The staging server port is 8080.",
                "assertion_kind": "explicit_statement",
                "evidence": [
                    {
                        "event_id": event_id,
                        "event_revision": revision,
                        "part_index": part_index,
                        "support_quote": support_quote,
                    }
                ],
            }
        ],
        "excluded_event_ids": [],
        "warnings": ["keep: unresolved identity"],
    }


class _FakeResponse:
    def __init__(self, proposal):
        self._body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(proposal, ensure_ascii=False)
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1},
            },
            ensure_ascii=False,
        ).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _call_http(parts, context, proposal):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(proposal)

    extractor = http_client.HttpExtractor(
        "https://model.invalid/v1/chat/completions",
        model="test-model",
        timeout=3.0,
    )
    with mock.patch.object(http_client.urllib.request, "urlopen", fake_urlopen):
        result = extractor.extract(parts, context)
    return extractor, result, captured


class SpanCatalogTests(unittest.TestCase):
    def test_catalog_covers_escaped_json_chinese_and_long_lines(self):
        escaped = r'{"message":"café \"quoted\"","nested":{"ok":true}}'
        chinese = "决策已记录：备份计划为每周日。"
        long_line = "LONGSTART " + ("segment 中文 " * 280) + "LONGEND"
        text = (
            "First paragraph line one.\nSecond line with escaped: "
            + escaped
            + "\n\n"
            + chinese
            + "\n"
            + long_line
            + "\n\nFinal paragraph."
        )
        self.assertGreater(len(long_line), MAX_SPAN_CHARS)

        catalog = build_span_catalog([_part(text)])
        self.assertTrue(catalog.spans)
        for span in catalog.spans:
            self.assertRegex(span.marker, SPAN_REFERENCE_PATTERN)
            self.assertLessEqual(len(span.quote), MAX_SPAN_CHARS)
            self.assertEqual(span.quote, text[span.start_char : span.end_char])
            self.assertIs(catalog.get(span.marker), span)

        covered = [False] * len(text)
        for span in catalog.spans:
            for index in range(span.start_char, span.end_char):
                covered[index] = True
        for index, character in enumerate(text):
            if not character.isspace():
                self.assertTrue(
                    covered[index],
                    f"uncovered non-whitespace character at offset {index}",
                )

        self.assertTrue(any(escaped in span.quote for span in catalog.spans))
        self.assertTrue(any("决策已记录" in span.quote for span in catalog.spans))

        line_start = text.index(long_line)
        line_end = line_start + len(long_line)
        long_spans = [
            span
            for span in catalog.spans
            if span.start_char >= line_start and span.end_char <= line_end
        ]
        self.assertGreater(len(long_spans), 1)

        again = build_span_catalog([_part(text)])
        self.assertEqual(
            [
                (span.marker, span.quote, span.start_char, span.end_char)
                for span in catalog.spans
            ],
            [
                (span.marker, span.quote, span.start_char, span.end_char)
                for span in again.spans
            ],
        )

    def test_same_text_in_different_parts_has_distinct_markers(self):
        text = 'identical escaped \\n "payload" 中文'
        first = _part(text, part_index=0)
        second = _part(text, part_index=1)
        catalog = build_span_catalog([first, second])
        first_span = next(span for span in catalog.spans if span.part_index == 0)
        second_span = next(span for span in catalog.spans if span.part_index == 1)
        self.assertNotEqual(first_span.marker, second_span.marker)
        self.assertEqual(first_span.quote, second_span.quote)


class SpanReferenceTests(unittest.TestCase):
    def test_marker_resolves_only_with_exact_part_binding(self):
        text = "same exact text in two parts"
        first = _part(text, part_index=0)
        second = _part(text, part_index=1)
        catalog = build_span_catalog([first, second])
        span = next(item for item in catalog.spans if item.part_index == 0)

        resolved = resolve_span_references(
            _proposal(span.marker, part_index=0), catalog, [first, second]
        )
        self.assertEqual(
            resolved["claims"][0]["evidence"][0]["support_quote"], span.quote
        )

        misbound = _proposal(span.marker, part_index=1)
        with self.assertRaises(ExtractionError) as caught:
            resolve_span_references(misbound, catalog, [first, second])
        self.assertEqual(caught.exception.category, "evidence_invalid")
        # Fail closed without partially rewriting the rejected proposal.
        self.assertEqual(
            misbound["claims"][0]["evidence"][0]["support_quote"], span.marker
        )

        for overrides in (
            {"event_id": "other-event"},
            {"revision": 2},
        ):
            with self.assertRaises(ExtractionError) as caught:
                resolve_span_references(
                    _proposal(span.marker, **overrides), catalog, [first, second]
                )
            self.assertEqual(caught.exception.category, "evidence_invalid")

    def test_unknown_and_malformed_markers_are_rejected(self):
        text = "exact source text"
        part = _part(text)
        catalog = build_span_catalog([part])
        known = catalog.spans[0]
        for marker in (
            "[[span:" + ("0" * 32) + "]]",
            "[[span:not-a-hex-id]]",
            "[[span:" + known.span_id.upper() + "]]",
            "prefix " + known.marker,
        ):
            with self.assertRaises(ExtractionError) as caught:
                resolve_span_references(_proposal(marker), catalog, [part])
            self.assertEqual(caught.exception.category, "evidence_invalid")

    def test_raw_literal_must_be_exact_bound_and_bounded(self):
        text = 'exact "escaped" 中文\\n first-only tail'
        part = _part(text)
        other = _part("different second-part text", part_index=1)
        catalog = build_span_catalog([part, other])
        resolved = resolve_span_references(_proposal(text), catalog, [part, other])
        self.assertEqual(
            resolved["claims"][0]["evidence"][0]["support_quote"], text
        )

        for bad_quote in (
            'exact "escaped" 中文\n first-only tail',
            'exact "escaped" 中文\\n first-only TAIL',
            "",
        ):
            with self.assertRaises(ExtractionError) as caught:
                resolve_span_references(
                    _proposal(bad_quote), catalog, [part, other]
                )
            self.assertEqual(caught.exception.category, "evidence_invalid")

        # The text is exact only in part 0; naming part 1 cannot pass.
        with self.assertRaises(ExtractionError) as caught:
            resolve_span_references(
                _proposal(text, part_index=1),
                catalog,
                [part, other],
            )
        self.assertEqual(caught.exception.category, "evidence_invalid")

        too_long = "z" * (MAX_SPAN_CHARS + 1)
        long_part = _part(too_long)
        with self.assertRaises(ExtractionError) as caught:
            resolve_span_references(
                _proposal(too_long),
                build_span_catalog([long_part]),
                [long_part],
            )
        self.assertEqual(caught.exception.category, "evidence_invalid")


class HttpPrompt10Tests(unittest.TestCase):
    def test_prompt10_enumerates_spans_and_resolves_marker_only(self):
        text = 'server port = 8080\\n{\\"ok\\":true, \\"note\\":\\"中文\\"}'
        part = _part(text)
        catalog = build_span_catalog([part])
        span = next(item for item in catalog.spans if "server port" in item.quote)
        self.assertIn('\\"ok\\":true', span.quote)
        self.assertIn("中文", span.quote)

        proposal = _proposal(span.marker)
        context = {
            "prompt_version": "prompt:10",
            "event_id": part.event_id,
            "event_revision": part.event_revision,
            "project_ids": [],
            "known_claims": [],
        }
        extractor, result, captured = _call_http([part], context, proposal)

        body = captured["body"]
        user_payload = json.loads(body["messages"][1]["content"])
        self.assertEqual(list(user_payload), ["context", "parts", "spans"])
        self.assertEqual(user_payload["parts"][0]["text"], text)
        self.assertEqual(user_payload["spans"], catalog.request_records())
        self.assertIn("PROMPT 10", body["messages"][0]["content"])

        schema = extractor._request_schema([part], context, catalog)
        evidence_props = schema["properties"]["claims"]["items"]["properties"][
            "evidence"
        ]["items"]["properties"]
        self.assertEqual(
            evidence_props["support_quote"]["enum"], list(catalog.markers)
        )
        self.assertEqual(evidence_props["event_id"]["enum"], [part.event_id])

        result_claim = result["claims"][0]
        original_claim = proposal["claims"][0]
        self.assertEqual(
            result_claim["evidence"][0]["support_quote"], span.quote
        )
        for key in (
            "project_hint",
            "subject_hint",
            "predicate",
            "facet",
            "target_claim_hint",
            "qualifiers",
            "value",
            "display_text",
            "assertion_kind",
        ):
            self.assertEqual(result_claim[key], original_claim[key])
        self.assertEqual(result["warnings"], proposal["warnings"])
        self.assertEqual(
            result["excluded_event_ids"], proposal["excluded_event_ids"]
        )

    def test_http_prompt10_rejects_unknown_and_misbound_markers(self):
        text = "same text in both parts 中文"
        first = _part(text, part_index=0)
        second = _part(text, part_index=1)
        catalog = build_span_catalog([first, second])
        span = next(item for item in catalog.spans if item.part_index == 0)
        context = {
            "prompt_version": "prompt:10",
            "project_ids": [],
            "known_claims": [],
        }

        with self.assertRaises(ExtractionError) as caught:
            _call_http(
                [first, second],
                context,
                _proposal(span.marker, part_index=1),
            )
        self.assertEqual(caught.exception.category, "evidence_invalid")

        with self.assertRaises(ExtractionError) as caught:
            _call_http(
                [first, second],
                context,
                _proposal("[[span:" + ("0" * 32) + "]]"),
            )
        self.assertEqual(caught.exception.category, "evidence_invalid")

    def test_old_prompt_versions_transport_is_unchanged(self):
        text = 'literal "quoted" escaped \\n text'
        part = _part(text)
        expected_prompts = {
            "prompt:3": http_client.EXTRACTION_SYSTEM_PROMPT_V3,
            "prompt:4": http_client.EXTRACTION_SYSTEM_PROMPT_V4,
            "prompt:5": http_client.EXTRACTION_SYSTEM_PROMPT,
            "prompt:6": http_client.EXTRACTION_SYSTEM_PROMPT_V6,
            "prompt:7": http_client.EXTRACTION_SYSTEM_PROMPT_V7,
            "prompt:8": http_client.EXTRACTION_SYSTEM_PROMPT_V8,
            "prompt:9": http_client.EXTRACTION_SYSTEM_PROMPT_V9,
        }
        proposal = _proposal(text)
        for version, system_prompt in expected_prompts.items():
            context = {
                "prompt_version": version,
                "project_ids": [],
                "known_claims": [],
            }
            extractor, result, captured = _call_http([part], context, proposal)
            body = captured["body"]
            self.assertNotIn("spans", body)
            user_payload = json.loads(body["messages"][1]["content"])
            self.assertEqual(list(user_payload), ["context", "parts"])
            self.assertEqual(user_payload["parts"][0]["text"], text)
            self.assertTrue(
                body["messages"][0]["content"].startswith(
                    system_prompt + "\nJSON Schema:\n"
                )
            )
            self.assertEqual(
                result["claims"][0]["evidence"][0]["support_quote"], text
            )
            schema = extractor._request_schema([part], context)
            evidence_props = schema["properties"]["claims"]["items"]["properties"][
                "evidence"
            ]["items"]["properties"]
            self.assertNotIn("enum", evidence_props["support_quote"])


if __name__ == "__main__":
    unittest.main()
