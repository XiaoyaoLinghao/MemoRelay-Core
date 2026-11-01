"""T08: scope-first hybrid recall, RRF, budgets, pagination, degradation."""
from __future__ import annotations

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.embeddings.index import (
    configure_model,
    load_vectors,
    run_embed_worker_once,
)
from knowledge_weaver.retrieval import (
    RecallCursorError,
    RecallScopeDenied,
    execute_recall,
)

from .conftest import ConceptEmbeddingProvider, FailingEmbeddingProvider, binding, recall_request, record


def _seed_songta(store):
    port = record(store, display_text="松塔项目服务器端口为4200", value=4200, predicate="server.port")
    backup = record(store, display_text="松塔项目备份周期为每周", value="weekly", predicate="backup_interval")
    return port, backup


def test_recall_returns_contract_valid_scoped_items(store, binding):
    port, backup = _seed_songta(store)
    record(store, display_text="其他项目端口为80", value=80, predicate="server.port", project_id="project:other")
    result = execute_recall(store, recall_request(), binding=binding)
    assert contracts.validate_instance("RecallResult", result) == []
    assert result["status"] in ("ok", "degraded")
    assert len(result["items"]) == 1
    item = result["items"][0]
    assert item["claim_id"] == port["claim_id"]
    assert item["project_id"] == "project:songta"
    assert item["relevance_score"] > 0
    assert backup["claim_id"] != item["claim_id"]


def test_recall_is_degraded_when_no_embedder(store, binding):
    _seed_songta(store)
    result = execute_recall(store, recall_request(), binding=binding)
    assert "embedding_unavailable" in result["incomplete_reasons"]
    assert result["status"] == "degraded"
    assert result["items"]  # keyword results still usable


def test_recall_scope_unresolved_returns_candidates(store):
    _seed_songta(store)
    wide_binding = {"client_id": "c", "workspace_id": "owner:demo", "project_ids": [], "actor_ids": ["*"]}
    result = execute_recall(store, recall_request(project_ids=[]), binding=wide_binding)
    assert result["status"] == "degraded"
    assert "scope_unresolved" in result["incomplete_reasons"]
    assert result["items"] == []
    assert {candidate["project_id"] for candidate in result["scope_candidates"]} == {"project:songta"}


def test_all_projects_intent_searches_authorized_scope(store):
    _seed_songta(store)
    record(store, display_text="其他项目端口为80", value=80, predicate="server.port", project_id="project:other")
    wide_binding = {"client_id": "c", "workspace_id": "owner:demo", "project_ids": [], "actor_ids": ["*"]}
    result = execute_recall(
        store,
        recall_request(project_ids=[], intent="all_projects"),
        binding=wide_binding,
    )
    assert {item["project_id"] for item in result["items"]} == {"project:songta", "project:other"}


def test_unauthorized_project_scope_is_denied(store, binding):
    with pytest.raises(RecallScopeDenied):
        execute_recall(store, recall_request(project_ids=["project:other"]), binding=binding)


def test_conflict_item_exposes_candidates(store, binding):
    same_time = {"value": "2026-10-30T10:00:00+00:00", "precision": "instant", "timezone": "UTC"}
    first = record(store, display_text="端口4200", value=4200, event_time=same_time)
    record(store, display_text="端口4100", value=4100, event_time=same_time)
    result = execute_recall(store, recall_request(), binding=binding)
    assert result["items"]
    item = result["items"][0]
    assert item["state"] == "conflict"
    assert item["claim_id"] == first["claim_id"]
    assert item["conflict_revision_ids"]


def test_limit_and_cursor_pagination(store, binding):
    for index in range(3):
        record(
            store,
            display_text=f"松塔项目服务器端口说明第{index}条",
            value=4200 + index,
            predicate=f"server.port.{index}",
        )
    first = execute_recall(store, recall_request(limit=1), binding=binding)
    assert len(first["items"]) == 1
    assert first["truncated"] is True
    assert first["next_cursor"]
    second = execute_recall(
        store, recall_request(limit=1, cursor=first["next_cursor"]), binding=binding
    )
    assert len(second["items"]) == 1
    assert second["items"][0]["revision_id"] != first["items"][0]["revision_id"]


def test_cursor_bound_to_query_and_watermark(store, binding):
    for index in range(2):
        record(store, display_text=f"端口说明第{index}条", value=4200 + index, predicate=f"server.port.{index}")
    first = execute_recall(store, recall_request(limit=1), binding=binding)
    assert first["next_cursor"]
    with pytest.raises(RecallCursorError):
        execute_recall(store, recall_request(limit=1, query="备份", cursor=first["next_cursor"]), binding=binding)


def test_history_intent_includes_superseded_revision(store, binding):
    first = record(store, display_text="端口4100", value=4100)
    second = record(store, display_text="端口改为4200", value=4200, supersedes=[first["revision_id"]])
    current = execute_recall(store, recall_request(), binding=binding)
    assert current["items"][0]["revision_id"] == second["revision_id"]
    history = execute_recall(store, recall_request(intent="history"), binding=binding)
    states = {item["state"] for item in history["items"]}
    assert "superseded" in states and "current" in states


def test_embedding_jobs_fill_vectors_and_semantic_query_works(store, binding):
    provider = ConceptEmbeddingProvider()
    configure_model(provider.model_id, model_revision=provider.model_revision, dimension=provider.dimension)
    record(store, display_text="服务监听4200", value=4200, predicate="server.port")
    assert run_embed_worker_once(store, provider)["claimed"] == 1
    assert run_embed_worker_once(store, provider)["claimed"] == 0
    with store.read() as conn:
        revision_id = conn.execute("SELECT revision_id FROM claim_revisions").fetchone()["revision_id"]
    vectors = load_vectors(store, {revision_id}, model_id=provider.model_id, dimension=provider.dimension)
    assert revision_id in vectors
    # The query "port" has no literal Chinese overlap with the stored text: only
    # the vector ranking can surface the item.
    result = execute_recall(store, recall_request(query="port"), binding=binding, embedder=provider)
    assert result["items"]
    assert result["items"][0]["revision_id"] == revision_id
    assert "embedding_unavailable" not in result["incomplete_reasons"]


def test_embedding_outage_degrades_to_keyword(store, binding):
    _seed_songta(store)
    result = execute_recall(
        store, recall_request(query="端口"), binding=binding, embedder=FailingEmbeddingProvider()
    )
    assert "embedding_unavailable" in result["incomplete_reasons"]
    assert result["status"] == "degraded"
    assert result["items"]


def test_empty_recall_is_not_unavailable(store, binding):
    result = execute_recall(store, recall_request(query="完全不存在的内容"), binding=binding)
    assert result["status"] == "empty"
    assert result["items"] == []
    assert result["total_hits"] == 0


# ---------------------------------------------------------------------------
# F08: page windows, cursor progress and the token budget
# ---------------------------------------------------------------------------


def test_pages_cover_every_result_exactly_once(store, binding):
    for index in range(14):
        record(
            store,
            display_text=f"端口配置第{index}台",
            value=index,
            predicate=f"server.port.{index}",
        )
    cursor = None
    seen: list[str] = []
    for _page in range(20):
        result = execute_recall(store, recall_request(limit=3, cursor=cursor), binding=binding)
        seen.extend(item["revision_id"] for item in result["items"])
        next_cursor = result["next_cursor"]
        assert next_cursor != cursor, {"returned": len(seen), "total": result["total_hits"]}
        if not next_cursor:
            break
        cursor = next_cursor
    assert len(seen) == len(set(seen)) == 14, seen


def test_oversized_first_item_does_not_bypass_token_budget(store, binding):
    from knowledge_weaver.retrieval.tokenizer import estimate_tokens

    text = "端口配置" + ("中文内容" * 600)
    record(store, display_text=text, value=4200)
    assert estimate_tokens(text) > 128
    result = execute_recall(store, recall_request(budget_tokens=128), binding=binding)
    assert result["used_tokens"] <= 128, result
    assert result["truncated"] is True
    assert result["next_cursor"], "a skipped oversized item must still advance the cursor"
    # The follow-up page terminates instead of repeating the same cursor.
    follow = execute_recall(
        store, recall_request(budget_tokens=128, cursor=result["next_cursor"]), binding=binding
    )
    assert follow["items"] == []
    assert follow["next_cursor"] is None


def test_budget_truncation_respects_budget_and_loses_nothing(store, binding):
    from knowledge_weaver.retrieval.tokenizer import estimate_tokens

    texts = [f"端口配置说明第{index}条" + "内容" * 20 for index in range(6)]
    for index, text in enumerate(texts):
        record(store, display_text=text, value=index, predicate=f"server.port.{index}")
    per_item = estimate_tokens(texts[0])
    budget = max(128, per_item * 2)  # contract minimum budget is 128
    cursor = None
    seen: list[str] = []
    for _page in range(20):
        result = execute_recall(
            store, recall_request(limit=8, budget_tokens=budget, cursor=cursor), binding=binding
        )
        assert result["used_tokens"] <= budget, result
        seen.extend(item["revision_id"] for item in result["items"])
        if not result["next_cursor"]:
            break
        assert result["next_cursor"] != cursor
        cursor = result["next_cursor"]
    assert len(seen) == len(set(seen)) == 6, seen


def test_history_pagination_covers_superseded_revisions(store, binding):
    revision_ids = []
    previous = None
    for index in range(9):
        recorded = record(
            store,
            display_text=f"端口历史第{index}次调整",
            value=4200 + index,
            supersedes=[previous["revision_id"]] if previous else [],
        )
        revision_ids.append(recorded["revision_id"])
        previous = recorded
    cursor = None
    seen: list[str] = []
    for _page in range(20):
        result = execute_recall(
            store,
            recall_request(intent="history", limit=2, cursor=cursor),
            binding=binding,
        )
        seen.extend(item["revision_id"] for item in result["items"])
        if not result["next_cursor"]:
            break
        assert result["next_cursor"] != cursor
        cursor = result["next_cursor"]
    assert set(seen) == set(revision_ids), {"seen": seen, "expected": revision_ids}
    assert len(seen) == len(set(seen))
