"""T08: dependency-free Chinese tokenization and FTS keyword ranking."""
from __future__ import annotations

from knowledge_weaver.retrieval import estimate_tokens, fts_query, tokenize
from knowledge_weaver.retrieval.keyword import keyword_ranking

from .conftest import record


def test_cjk_bigrams_and_latin_tokens():
    tokens = tokenize("松塔项目端口改为4200，Port=4200")
    assert "松塔" in tokens and "端口" in tokens and "4200" in tokens and "port" in tokens
    assert "松" not in tokens  # multi-char runs use bigrams only


def test_single_cjk_char_and_query_expression():
    assert tokenize("端口") == ["端口"]
    assert tokenize("端") == ["端"]
    assert fts_query("端口 4200").startswith('"端口" OR')
    assert fts_query("") is None


def test_estimate_tokens_is_conservative_for_cjk():
    assert estimate_tokens("端口") >= 2
    assert estimate_tokens("port") >= 1


def test_keyword_ranking_respects_candidate_scope(store):
    claim_a = record(store, display_text="松塔项目服务器端口4200", value=4200, predicate="server.port")
    claim_b = record(
        store,
        display_text="松塔项目服务器端口80",
        value=80,
        predicate="server.port",
        project_id="project:other",
    )
    with store.read() as conn:
        revision_ids = {
            row["revision_id"] for row in conn.execute("SELECT revision_id FROM claim_revisions")
        }
    candidates = {claim_a["revision_id"]}
    ranked = keyword_ranking(store, candidate_revision_ids=candidates, query="端口")
    assert ranked == [claim_a["revision_id"]]
    assert claim_b["revision_id"] not in ranked
    all_ranked = keyword_ranking(store, candidate_revision_ids=revision_ids, query="服务器")
    assert set(all_ranked) == revision_ids


def test_keyword_ranking_orders_better_match_first(store):
    weak = record(store, display_text="松塔项目备份周期", value="weekly", predicate="backup_interval")
    strong = record(
        store,
        display_text="松塔项目服务器端口4200，端口说明",
        value=4200,
        predicate="server.port",
    )
    with store.read() as conn:
        revision_ids = {
            row["revision_id"] for row in conn.execute("SELECT revision_id FROM claim_revisions")
        }
    ranked = keyword_ranking(store, candidate_revision_ids=revision_ids, query="端口")
    assert ranked[0] == strong["revision_id"]
    assert weak["revision_id"] not in ranked
