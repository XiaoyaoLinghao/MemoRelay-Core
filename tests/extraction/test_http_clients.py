"""HTTP extractor / embedding provider parsing and failure behavior (stdlib clients)."""
from __future__ import annotations

import io
import json
import urllib.error
from unittest import mock

import pytest

from knowledge_weaver.embeddings.http_provider import EmbeddingUnavailable, HttpEmbeddingProvider
from knowledge_weaver import contracts
from knowledge_weaver.extraction import ExtractionError, HttpExtractor
from knowledge_weaver.extraction.http_client import EXTRACTION_SYSTEM_PROMPT
from knowledge_weaver.extraction.proposals import SanitizedPart


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _part():
    return SanitizedPart(
        event_id="evt:" + "a" * 64,
        event_revision=1,
        part_index=0,
        text="松塔项目端口改为4200",
        body_sha256="b" * 64,
        role="user",
        origin_kind="user_input",
        actor_id="jun-test",
        event_time={"value": "2026-10-30T01:00:00Z", "precision": "instant", "timezone": "UTC"},
        project_ids=("project:songta",),
        workspace_id="owner:demo",
    )


def test_http_extractor_parses_chat_completion_and_sends_system_prompt():
    proposal = {"schema_version": "1.0", "claims": [], "excluded_event_ids": [], "warnings": []}
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            json.dumps({"choices": [{"message": {"content": "```json\n" + json.dumps(proposal) + "\n```"}}]}).encode()
        )

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        extractor = HttpExtractor("http://127.0.0.1:1/v1/chat/completions", model="mock")
        result = extractor.extract([_part()], {"event_id": "evt:x"})
    assert result == proposal
    messages = captured["body"]["messages"]
    instruction, schema_json = messages[0]["content"].split("\nJSON Schema:\n", 1)
    assert instruction == EXTRACTION_SYSTEM_PROMPT
    assert json.loads(schema_json) == contracts.extraction_schema()
    payload = json.loads(messages[1]["content"])
    assert payload["parts"][0]["text"] == "松塔项目端口改为4200"
    assert payload["parts"][0]["role"] == "user"
    assert payload["parts"][0]["origin_kind"] == "user_input"
    assert payload["parts"][0]["event_time"] == _part().event_time


def test_http_extractor_failures_are_retryable_model_errors():
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        extractor = HttpExtractor("http://127.0.0.1:1/v1/chat/completions", model="mock")
        with pytest.raises(ExtractionError) as excinfo:
            extractor.extract([_part()], {})
        assert excinfo.value.retryable is True
        assert excinfo.value.code == "MODEL_UNAVAILABLE"

    def fake_invalid(request, timeout=None):
        return _FakeResponse(b'{"choices":[{"message":{"content":"not json"}}]}')

    with mock.patch("urllib.request.urlopen", fake_invalid):
        with pytest.raises(ExtractionError):
            HttpExtractor("http://127.0.0.1:1/v1/chat/completions", model="mock").extract([_part()], {})


def test_embedding_provider_parses_both_shapes_and_normalizes(monkeypatch):
    monkeypatch.setenv("TEST_EMBED_KEY", "sk-test-invalid")
    captured = {}

    def openai(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["auth"] = request.headers.get("Authorization")
        return _FakeResponse(json.dumps({"data": [{"embedding": [3.0, 0.0, 4.0, 0.0]}]}).encode())

    with mock.patch("urllib.request.urlopen", openai):
        provider = HttpEmbeddingProvider(
            "http://127.0.0.1:1/v1/embeddings",
            model_id="bge-m3",
            dimension=4,
            api_key_env="TEST_EMBED_KEY",
        )
        vectors = provider.embed(["端口4200"])
    assert vectors == [[0.6, 0.0, 0.8, 0.0]]
    assert captured["body"] == {"model": "bge-m3", "input": ["端口4200"]}
    assert captured["auth"].startswith("Bearer sk-test-invalid")

    def sentence_transformers(request, timeout=None):
        return _FakeResponse(json.dumps({"embeddings": [[0.0, 1.0, 0.0, 0.0]]}).encode())

    with mock.patch("urllib.request.urlopen", sentence_transformers):
        provider = HttpEmbeddingProvider("http://x/v1/embeddings", model_id="bge", dimension=4)
        assert provider.embed(["x"]) == [[0.0, 1.0, 0.0, 0.0]]


def test_embedding_provider_dimension_and_transport_failures_degrade():
    def wrong_dimension(request, timeout=None):
        return _FakeResponse(json.dumps({"data": [{"embedding": [1.0, 0.0]}]}).encode())

    with mock.patch("urllib.request.urlopen", wrong_dimension):
        provider = HttpEmbeddingProvider("http://x/v1/embeddings", model_id="bge", dimension=4)
        with pytest.raises(EmbeddingUnavailable):
            provider.embed(["x"])
    with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("slow")):
        provider = HttpEmbeddingProvider("http://x/v1/embeddings", model_id="bge", dimension=4)
        with pytest.raises(EmbeddingUnavailable) as excinfo:
            provider.embed(["x"])
        assert "TimeoutError" in str(excinfo.value)

@pytest.mark.parametrize("thinking,expected",[(None,None),(False,"disabled"),(True,"enabled")])
def test_explicit_provider_thinking_option_preserves_generic_default(thinking,expected):
 captured={}
 def response(request,timeout=None):
  captured.update(json.loads(request.data));return _FakeResponse(json.dumps({"choices":[{"message":{"content":json.dumps({"schema_version":"1.0","claims":[],"excluded_event_ids":[],"warnings":[]})}}]}).encode())
 with mock.patch("urllib.request.urlopen",response):HttpExtractor("http://unused",model="mock",thinking=thinking).extract([_part()],{})
 assert captured.get("thinking")==({"type":expected} if expected else None)

@pytest.mark.parametrize("mode,expected",[("provider-default",None),("off",False),("on",True)])
def test_cli_passes_thinking_option_and_lease_covers_model_timeout(tmp_path,monkeypatch,mode,expected):
 from knowledge_weaver import cli
 from knowledge_weaver.store import Store
 import knowledge_weaver.extraction as extraction
 db=tmp_path/"knowledge.db";Store(db);seen={}
 class Client:
  def __init__(self,*a,**kw):seen.update(kw)
 def run(store,extractor,**kw):seen.update(kw);return {"claimed":0}
 monkeypatch.setattr(extraction,"HttpExtractor",Client);monkeypatch.setattr(extraction,"run_worker_once",run)
 assert cli.main(["extract-worker","--db",str(db),"--url","http://localhost","--model","mock","--thinking",mode,"--timeout","120"])==0
 assert seen["thinking"] is expected and seen["lease_seconds"]>=150


@pytest.mark.parametrize("version",["prompt:3","prompt:4","prompt:5"])
def test_http_prompt_version_matches_the_recorded_job_processor(version):
 import hashlib
 from knowledge_weaver.extraction.http_client import EXTRACTION_SYSTEM_PROMPT_V3, EXTRACTION_SYSTEM_PROMPT_V4
 captured={}
 def response(request,timeout=None):
  captured.update(json.loads(request.data));return _FakeResponse(json.dumps({"choices":[{"message":{"content":json.dumps({"schema_version":"1.0","claims":[],"excluded_event_ids":[],"warnings":[]})}}]}).encode())
 with mock.patch("urllib.request.urlopen",response):HttpExtractor("http://unused",model="mock").extract([_part()],{"prompt_version":version})
 instruction=captured['messages'][0]['content'].split('\nJSON Schema:\n',1)[0]
 if version=='prompt:3':
  # Frozen actual v3 transport bytes, not a newly labelled v4 prompt.
  assert instruction==EXTRACTION_SYSTEM_PROMPT_V3
  assert hashlib.sha256(instruction.encode()).hexdigest()=='49da6c47634913c8a02463b5f62175d5dd78bdc0922eb852588abc7a79034be7'
 elif version=='prompt:4':
  assert instruction==EXTRACTION_SYSTEM_PROMPT_V4
  assert hashlib.sha256(instruction.encode()).hexdigest()=='556a05c71d934fcbf8418bdfd6150aadb96c050cdfaa9b2a4671477b318ae4e0'
 else:
  assert instruction==EXTRACTION_SYSTEM_PROMPT and instruction!=EXTRACTION_SYSTEM_PROMPT_V4
  assert "Partial progress is NOT a completion report" in instruction
  assert "not a fresh measurement" in instruction

def test_unknown_prompt_version_refuses_before_a_model_request():
 with mock.patch('urllib.request.urlopen') as outgoing:
  with pytest.raises(ExtractionError) as excinfo:HttpExtractor('http://unused',model='mock').extract([_part()],{'prompt_version':'prompt:unknown'})
 assert excinfo.value.code=='UNSUPPORTED_PROMPT_VERSION' and not excinfo.value.retryable
 outgoing.assert_not_called()
