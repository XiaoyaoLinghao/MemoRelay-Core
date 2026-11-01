"""AC-48 coverage gap: complete RRF/scope/cursor/budget behaviour.

Codex round-2: "AC48没有完整RRF/同义词组合断言". The previous mapping only covered
paging and budgets; this module adds the full frozen-expected combination:

* RRF相关性不被importance重排 - a low-authority but relevant item beats a
  high-authority but less relevant one; there is no importance input in the
  ranking path at all;
* 范围过滤在top-k前 - many out-of-scope matching items cannot consume the
  authorized page;
* 固定游标稳定无重漏 - multi-page current/history walks return every item once,
  the same cursor is stable, and a changed query/scope/world is a stale cursor;
* 预算截断可见 - a small token budget is reported explicitly (``truncated``,
  ``incomplete_reasons=budget``, non-null cursor) rather than silently dropping;
* Chinese synonym / importance combination - an English synonym query reaches a
  Chinese claim through the semantic half of RRF, and an irrelevant high-authority
  claim is neither surfaced nor able to reorder relevance.

Frozen expected (docs/baseline-20261030/acceptance-cases.json AC-48):
"RRF相关性不被importance重排；范围过滤在top-k前；固定游标稳定无重漏；预算截断可见。"
"""

from __future__ import annotations

import pytest

from tests.retrieval.conftest import ConceptEmbeddingProvider, recall_request
from tests.state.conftest import record

from knowledge_weaver.retrieval import RecallCursorError, RecallScopeDenied, execute_recall
from knowledge_weaver.store import Store

BINDING = {
    "client_id": "client:test",
    "workspace_id": "owner:demo",
    "project_ids": ["project:songta"],
    "actor_ids": ["*"],
}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")


@pytest.fixture
def embedder():
    return ConceptEmbeddingProvider()


def _recall(store, *, query, embedder=None, intent="current", cursor=None, limit=8, budget=2000, project_ids=None):
    request = recall_request(
        query=query,
        intent=intent,
        cursor=cursor,
        limit=limit,
        budget_tokens=budget,
        project_ids=project_ids if project_ids is not None else ["project:songta"],
    )
    return execute_recall(store, request, binding=BINDING, embedder=embedder)


def _seed(
    store,
    count,
    *,
    prefix="端口",
    native_prefix="seed",
    predicate="server.port",
    project_id="project:songta",
):
    recorded = []
    for index in range(count):
        recorded.append(
            record(
                store,
                display_text=f"{prefix}{4200 + index}",
                value=4200 + index,
                predicate=predicate,
                project_id=project_id,
                subject_id=f"subject:{native_prefix}:{index}",
                native_id=f"message:ac48:{native_prefix}:{index}",
            )
        )
    return recorded


def test_rrf_relevance_beats_importance_like_authority(store, embedder):
    """Clause: RRF相关性不被importance重排."""
    # The low-authority item is the genuinely relevant one: it matches both
    # query concepts in FTS and in the semantic half. The high-authority item
    # only matches one of them, so no importance/authority input may promote it.
    relevant_low = record(
        store,
        display_text="端口 备份 4200 监听",
        value=4200,
        authority="reported",
        subject_id="subject:relevant",
        native_id="message:ac48:relevant",
    )
    relevant_high = record(
        store,
        display_text="端口 4100",
        value=4100,
        predicate="server.port.alt",
        authority="user_confirmed",
        subject_id="subject:relevant2",
        native_id="message:ac48:relevant2",
    )
    irrelevant_high = record(
        store,
        display_text="硬件维护计划 2027",
        value="planned",
        predicate="maintenance.plan",
        authority="user_confirmed",
        subject_id="subject:irrelevant",
        native_id="message:ac48:irrelevant",
    )
    result = _recall(store, query="端口 备份", embedder=embedder)
    values = [item["value"] for item in result["items"]]
    assert values[0] == 4200, result["items"]
    assert set(values) == {4200, 4100}, result["items"]
    assert irrelevant_high["revision_id"] not in {
        item["revision_id"] for item in result["items"]
    }, result["items"]
    scores = [item["relevance_score"] for item in result["items"]]
    assert scores[0] > scores[1], scores
    # The high-authority item is still returned (not hidden), just not promoted.
    assert result["items"][1]["revision_id"] == relevant_high["revision_id"], result["items"]
    assert result["items"][0]["revision_id"] == relevant_low["revision_id"], result["items"]


def test_scope_filter_happens_before_top_k(store):
    """Clause: 范围过滤在top-k前."""
    _seed(store, 10, prefix="外项目端口", native_prefix="other", project_id="project:other")
    in_scope = _seed(store, 2, native_prefix="in-scope")
    result = _recall(store, query="端口", limit=2, project_ids=["project:songta"])
    assert result["total_hits"] == 2, result
    assert len(result["items"]) == 2, result
    assert {item["project_id"] for item in result["items"]} == {"project:songta"}, result["items"]
    assert {item["revision_id"] for item in result["items"]} == {
        item["revision_id"] for item in in_scope
    }
    # An unauthorized scope is denied before any ranking happens.
    with pytest.raises(RecallScopeDenied):
        _recall(store, query="端口", project_ids=["project:other"])


@pytest.mark.parametrize("intent", ["current", "history"])
def test_fixed_cursor_is_stable_without_duplicates_or_gaps(store, intent):
    """Clause: 固定游标稳定无重漏."""
    if intent == "current":
        expected_total = 29
        _seed(store, expected_total, prefix="分页端口", native_prefix="page")
    else:
        # 4 claims -> each superseded once => 8 revision rows in history.
        for index in range(4):
            first = record(
                store,
                display_text=f"历史端口{4100 + index}",
                value=4100 + index,
                predicate=f"history.port.{index}",
                subject_id=f"subject:history:{index}",
                native_id=f"message:ac48:hist:{index}",
            )
            record(
                store,
                display_text=f"历史端口改为{4200 + index}",
                value=4200 + index,
                predicate=f"history.port.{index}",
                subject_id=f"subject:history:{index}",
                supersedes=[first["revision_id"]],
                event_time={"value": "2026-10-30T11:00:00Z", "precision": "instant", "timezone": "UTC"},
                native_id=f"message:ac48:hist2:{index}",
            )
        expected_total = 8

    seen = []
    cursor = None
    first_page_cursor = None
    for _ in range(40):
        page = _recall(store, query="端口", intent=intent, cursor=cursor, limit=3)
        seen.extend(item["revision_id"] for item in page["items"])
        if cursor is None:
            first_page_cursor = page["next_cursor"]
        if not page["next_cursor"]:
            break
        assert page["next_cursor"] != cursor, page["next_cursor"]
        cursor = page["next_cursor"]
    assert len(seen) == expected_total, (len(seen), expected_total)
    assert len(set(seen)) == expected_total, "cursor walk must not duplicate items"

    # Re-requesting the same fixed cursor is stable: same page, same order.
    first_page = _recall(store, query="端口", intent=intent, limit=3)
    repeat = _recall(store, query="端口", intent=intent, cursor=first_page["next_cursor"], limit=3)
    repeat_again = _recall(
        store, query="端口", intent=intent, cursor=first_page["next_cursor"], limit=3
    )
    assert [item["revision_id"] for item in repeat["items"]] == [
        item["revision_id"] for item in repeat_again["items"]
    ], "a fixed cursor must return a stable page"


def test_cursor_is_bound_to_query_scope_and_watermark(store):
    """Clause: 固定游标稳定无重漏 (a changed world yields stale cursor, not silent
    skips/duplicates)."""
    _seed(store, 10, prefix="游标端口", native_prefix="cursor")
    first = _recall(store, query="端口", limit=2)
    cursor = first["next_cursor"]
    assert cursor, first

    with pytest.raises(RecallCursorError) as error:
        _recall(store, query="备份端口", cursor=cursor, limit=2)
    assert error.value.code == "STALE_CURSOR"
    with pytest.raises(RecallCursorError):
        _recall(store, query="端口", intent="history", cursor=cursor, limit=2)
    # A write advances the watermark: the old cursor no longer describes the
    # same result set.
    record(
        store,
        display_text="游标端口新增",
        value=9000,
        predicate="cursor.new",
        subject_id="subject:cursor:new",
        native_id="message:ac48:cursor:new",
    )
    with pytest.raises(RecallCursorError):
        _recall(store, query="端口", cursor=cursor, limit=2)


def test_budget_truncation_is_explicit_and_paginates_without_repeats(store):
    """Clause: 预算截断可见."""
    # Each item costs ~61 CJK tokens; budget 130 fits exactly one per page.
    long_text = "端口" + "中文内容" * 30
    for index in range(3):
        record(
            store,
            display_text=long_text + str(index),
            value=4200 + index,
            predicate="budget.port",
            subject_id=f"subject:budget:{index}",
            native_id=f"message:ac48:budget:{index}",
        )
    first = _recall(store, query="端口", limit=8, budget=130)
    assert first["truncated"] is True, first
    assert "budget" in first["incomplete_reasons"], first
    assert first["used_tokens"] <= 130, first
    assert first["next_cursor"], first
    assert len(first["items"]) == 1, first["items"]

    seen = [item["revision_id"] for item in first["items"]]
    cursor = first["next_cursor"]
    while cursor:
        page = _recall(store, query="端口", limit=8, budget=130, cursor=cursor)
        seen.extend(item["revision_id"] for item in page["items"])
        assert page["used_tokens"] <= 130, page
        cursor = page["next_cursor"]
    assert len(seen) == 3 and len(set(seen)) == 3, seen
    assert "budget" in first["incomplete_reasons"]


def test_chinese_synonym_and_importance_combination(store, embedder):
    """Combination clause from the Codex basis: 中文同义词/importance 组合."""
    chinese = record(
        store,
        display_text="服务器监听端口 4200",
        value=4200,
        authority="reported",
        subject_id="subject:cn",
        native_id="message:ac48:cn",
    )
    english = record(
        store,
        display_text="port listening 4300",
        value=4300,
        authority="reported",
        subject_id="subject:en",
        native_id="message:ac48:en",
    )
    irrelevant = record(
        store,
        display_text="每周备份 NAS 存储",
        value="weekly",
        predicate="backup.interval",
        authority="user_confirmed",
        subject_id="subject:backup",
        native_id="message:ac48:backup",
    )

    mixed = _recall(store, query="端口 port", embedder=embedder)
    ids = {item["revision_id"] for item in mixed["items"]}
    assert chinese["revision_id"] in ids, mixed["items"]
    assert english["revision_id"] in ids, mixed["items"]
    assert irrelevant["revision_id"] not in ids, mixed["items"]
    assert all(item["relevance_score"] > 0 for item in mixed["items"]), mixed["items"]

    # English synonym reaches the Chinese claim through the semantic half of RRF
    # (keyword FTS alone cannot match "listening" against the Chinese text).
    synonym = _recall(store, query="listening", embedder=embedder)
    synonym_ids = {item["revision_id"] for item in synonym["items"]}
    assert chinese["revision_id"] in synonym_ids, synonym
    assert irrelevant["revision_id"] not in synonym_ids, synonym
    assert synonym["status"] == "ok", synonym
    assert "embedding_unavailable" not in synonym["incomplete_reasons"], synonym
