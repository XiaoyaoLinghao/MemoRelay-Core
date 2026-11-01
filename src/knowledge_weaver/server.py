"""Knowledge Weaver MCP Server entry point.

Creates a FastMCP server that registers 7 tools (6 knowledge + 1 consolidate),
3 resources, reads environment variables for configuration, and supports CLI subcommands.

Usage:
    python -m knowledge_weaver.server           # serve via stdio (default)
    python -m knowledge_weaver.server serve      # explicit serve
    python -m knowledge_weaver.server consolidate  # run consolidation
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

def _default_db_path() -> str:
    home = os.environ.get("HOME", os.path.expanduser("~"))
    return os.path.join(home, ".openclaw", "knowledge", "knowledge.db")


def _default_memory_dir() -> str:
    home = os.environ.get("HOME", os.path.expanduser("~"))
    return os.path.join(home, ".openclaw", "workspace", "memory")


DB_PATH = os.environ.get("KNOWLEDGE_WEAVER_DB_PATH") or _default_db_path()
MEMORY_DIR = os.environ.get("KNOWLEDGE_WEAVER_MEMORY_DIR") or _default_memory_dir()
LOG_LEVEL = os.environ.get(
    "KNOWLEDGE_WEAVER_LOG_LEVEL",
    "INFO",
).upper()


def _parse_memory_dirs() -> list[tuple[str, str]]:
    """Parse memory dirs config into [(name, path), ...].

    Priority:
    1. KNOWLEDGE_WEAVER_MEMORY_DIRS (colon-separated, optional name= prefix)
    2. KNOWLEDGE_WEAVER_MEMORY_DIR (singular, backwards compat)
    3. Default _default_memory_dir() under source name "default"
    """
    multi = os.environ.get("KNOWLEDGE_WEAVER_MEMORY_DIRS", "").strip()
    if multi:
        result = []
        for entry in multi.split(":"):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                name, path = entry.split("=", 1)
                result.append((name.strip(), os.path.expanduser(path.strip())))
            else:
                path = os.path.expanduser(entry)
                name = os.path.basename(os.path.normpath(path)) or "default"
                result.append((name, path))
        return result

    single = os.environ.get("KNOWLEDGE_WEAVER_MEMORY_DIR", "").strip()
    if single:
        path = os.path.expanduser(single)
        name = os.path.basename(os.path.normpath(path)) or "default"
        return [(name, path)]

    return [("default", _default_memory_dir())]


MEMORY_DIRS = _parse_memory_dirs()


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def _get_conn() -> sqlite3.Connection:
    """Get a fresh DB connection for tool calls."""
    from knowledge_weaver.db import init_db
    return init_db(DB_PATH)


def get_embedder():
    """Lazy embedder factory — returns EmbeddingClient or None."""
    from knowledge_weaver.embedder import get_embedder as _get
    return _get()


def _clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp an integer value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# ---------------------------------------------------------------------------
# MCP Server creation
# ---------------------------------------------------------------------------

def create_server() -> FastMCP:
    """Create and configure the MCP server with all tools and resources."""
    mcp = FastMCP("knowledge-weaver")

    # --- Tool 1: knowledge_search ---
    @mcp.tool()
    async def knowledge_search(
        query: str,
        entity_type: str = "",
        source: str = "",
        max_results: int = 10,
        min_score: float = 0.0,
        offset: int = 0,
    ) -> str:
        """Search the structured knowledge base for entities by keyword or semantic similarity.

        Use this tool when you need to find specific decisions, projects, risks, preferences,
        or technical concepts that have been extracted from past conversations and daily memory
        archives. This is NOT a general memory search — it searches a curated knowledge graph,
        not raw text chunks. For exploring how topics connect to each other, use knowledge_trace
        instead. For listing current projects, use active_projects.

        Args:
            query: The search query text. Can be a keyword, phrase, or natural language question.
            entity_type: Optional filter by entity type: project, decision, risk, preference, task, idea, tech, fact.
            source: Optional filter by source agent name (e.g., "openclaw", "hermes"). Empty string = no filter.
            max_results: Maximum number of results to return (1-100, default 10).
            min_score: Minimum importance score threshold (default 0.0, no filtering).
                       Raise to filter out low-importance entities; fact/idea types tend to
                       have low importance and will be hidden if min_score > 0.05.
            offset: Number of results to skip for pagination (default 0).
        """
        max_results = _clamp(max_results, 1, 100)
        offset = max(0, offset)
        from knowledge_weaver.tools import knowledge_search as _search
        conn = _get_conn()
        try:
            result = _search(
                conn,
                query=query,
                entity_type=entity_type or None,
                source=source or None,
                max_results=max_results,
                min_score=min_score,
                embedder=get_embedder(),
            )
            if offset > 0 and "results" in result:
                result["results"] = result["results"][offset:]
        except Exception as exc:
            return json.dumps({"error": str(exc), "isError": True}, ensure_ascii=False)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False)

    # --- Tool 2: knowledge_trace ---
    @mcp.tool()
    async def knowledge_trace(topic: str, max_depth: int = 2) -> str:
        """Trace a topic's relationship graph — how decisions, projects, risks, and preferences connect to each other.

        Use this when you need to understand the CONTEXT around a topic: what decisions were made,
        what they depend on, what risks they carry, and how they relate to other topics. This follows
        entity relationships (RELATES_TO, CONTINUES, DEPENDS_ON, CONTRADICTS) to build a connected view.
        For a simple keyword search, use knowledge_search instead.

        Args:
            topic: The topic to trace. Can be an entity name, partial name, or keyword.
            max_depth: How many hops to follow in the relationship graph (1-5, default 2). Depth 1 shows direct connections; depth 2+ reveals indirect relationships.
        """
        max_depth = _clamp(max_depth, 1, 5)
        from knowledge_weaver.tools import knowledge_trace as _trace
        conn = _get_conn()
        try:
            result = _trace(conn, topic=topic, max_depth=max_depth)
        except Exception as exc:
            return json.dumps({"error": str(exc), "isError": True}, ensure_ascii=False)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False)

    # --- Tool 3: active_projects ---
    @mcp.tool()
    async def active_projects(lookback_days: int = 14) -> str:
        """List currently active projects — projects seen in recent daily memory archives.

        Use this to get a high-level overview of what the user has been working on recently.
        Each project includes its activity status and related open tasks. For detailed project
        relationships, use knowledge_trace with the project name.

        Args:
            lookback_days: How many days back to consider a project active (1-90, default 14).
        """
        lookback_days = _clamp(lookback_days, 1, 90)
        from knowledge_weaver.tools import active_projects as _active
        conn = _get_conn()
        try:
            result = _active(conn, lookback_days=lookback_days)
        except Exception as exc:
            return json.dumps({"error": str(exc), "isError": True}, ensure_ascii=False)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False)

    # --- Tool 4: preference_lookup ---
    @mcp.tool()
    async def preference_lookup(topic: str = "", domain: str = "") -> str:
        """Look up user preferences, habits, and stated preferences from past conversations.

        Use this when you need to understand the user's preferred tools, coding style, workflow
        habits, or any explicitly stated preferences. For decisions and their rationale, use
        decision_history instead.

        Args:
            topic: Optional topic filter (e.g., "editor", "language", "workflow").
            domain: Optional domain filter.
        """
        from knowledge_weaver.tools import preference_lookup as _lookup
        conn = _get_conn()
        try:
            result = _lookup(conn, topic=topic or None, domain=domain or None)
        except Exception as exc:
            return json.dumps({"error": str(exc), "isError": True}, ensure_ascii=False)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False)

    # --- Tool 5: decision_history ---
    @mcp.tool()
    async def decision_history(
        topic: str,
        include_risk: bool = True,
        offset: int = 0,
    ) -> str:
        """Query historical decisions and their rationale on a specific topic.

        Use this when you need to understand WHY something was decided, what alternatives were
        considered, and what risks were identified. This returns the decision timeline with
        rationale and optionally linked risks. For finding decisions by keyword without a specific
        topic, use knowledge_search with entity_type="decision".

        Args:
            topic: The topic to search decisions for.
            include_risk: Whether to include related risks (default True).
            offset: Number of results to skip for pagination (default 0).
        """
        offset = max(0, offset)
        from knowledge_weaver.tools import decision_history as _history
        conn = _get_conn()
        try:
            result = _history(conn, topic=topic, include_risk=include_risk)
            if offset > 0 and "decisions" in result:
                result["decisions"] = result["decisions"][offset:]
        except Exception as exc:
            return json.dumps({"error": str(exc), "isError": True}, ensure_ascii=False)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False)

    # --- Tool 6: knowledge_stats ---
    @mcp.tool()
    async def knowledge_stats() -> str:
        """Knowledge base health check — entity counts by type, relation counts, indexed days, quality metrics.

        Use this to diagnose whether the knowledge base is functioning properly, check data freshness,
        or understand the distribution of knowledge types. Not needed for normal queries."""
        from knowledge_weaver.tools import knowledge_stats as _stats
        conn = _get_conn()
        try:
            result = _stats(conn)
        except Exception as exc:
            return json.dumps({"error": str(exc), "isError": True}, ensure_ascii=False)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False)

    # --- Tool 7: knowledge_consolidate ---
    @mcp.tool()
    async def knowledge_consolidate() -> str:
        """Manually trigger knowledge consolidation from DMA daily memory files.

        Normally runs via cron. This tool allows on-demand consolidation.
        Requires EMBEDDING_BASE_URL and EMBEDDING_API_KEY to be configured for embeddings.
        """
        from knowledge_weaver.pipeline import run_consolidation
        embedder = get_embedder()
        if embedder is None:
            return json.dumps({
                "error": "Embedding not configured. Set EMBEDDING_BASE_URL, EMBEDDING_API_KEY, and EMBEDDING_MODEL.",
                "isError": True,
            }, ensure_ascii=False)

        try:
            result = run_consolidation(DB_PATH, memory_dirs=MEMORY_DIRS, embedder=embedder)
        except Exception as exc:
            return json.dumps({
                "error": str(exc),
                "isError": True,
            }, ensure_ascii=False)

        return json.dumps({
            "status": result.status,
            "files_processed": result.files_processed,
            "files_skipped": result.files_skipped,
            "entities_created": result.entities_created,
            "entities_updated": result.entities_updated,
            "errors": result.errors,
        }, ensure_ascii=False)

    # --- Resources ---

    @mcp.resource("knowledge://stats")
    def stats_resource() -> str:
        """System status as a resource."""
        from knowledge_weaver.tools import knowledge_stats as _stats
        conn = _get_conn()
        try:
            result = _stats(conn)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.resource("knowledge://entity/{entity_id}")
    def entity_resource(entity_id: str) -> str:
        """Individual entity details by ID."""
        from knowledge_weaver.db import get_entity, get_relations_for_entity
        conn = _get_conn()
        try:
            entity = get_entity(conn, entity_id)
            if entity is None:
                return json.dumps({"error": "Entity not found"}, ensure_ascii=False)
            result = dict(entity)
            rels = get_relations_for_entity(conn, entity_id)
            result["relations"] = [dict(r) for r in rels]
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    @mcp.resource("knowledge://days/{date}")
    def day_resource(date: str) -> str:
        """Daily manifest entry by date (YYYY-MM-DD)."""
        from knowledge_weaver.db import get_manifest, list_entities_by_type
        conn = _get_conn()
        try:
            manifest = get_manifest(conn, date)
            if manifest is None:
                return json.dumps({"error": "No manifest for this date"}, ensure_ascii=False)
            result = dict(manifest)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    # --- W1: entity-resolution review queue ---
    @mcp.tool()
    async def kw_review_pending(limit: int = 50) -> str:
        """List pending entity-merge / registry-purge items awaiting your confirmation.

        The resolver auto-merges only high-confidence duplicates; uncertain
        candidates (and projects removed from the registry) are queued here
        instead of being merged/deleted silently. Review them, then call
        kw_resolve. Pending items never block ingestion.

        Args:
            limit: Max items to return (default 50).
        """
        from knowledge_weaver.tools import review_pending as _rp
        conn = _get_conn()
        try:
            result = _rp(conn, limit=_clamp(limit, 1, 200))
        except Exception as exc:
            return json.dumps({"error": str(exc), "isError": True}, ensure_ascii=False)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False)

    @mcp.tool()
    async def kw_resolve(review_id: int, action: str) -> str:
        """Resolve one review item from kw_review_pending.

        Args:
            review_id: The review_id from kw_review_pending.
            action: For a merge item: "merge" to combine the two entities, or
                "reject" to keep them separate. For a purge item: "purge" to
                delete the entity, or "keep" to keep it. Merges and purges are
                logged and can be rolled back.
        """
        from knowledge_weaver.tools import resolve_review as _rr
        conn = _get_conn()
        try:
            result = _rr(conn, review_id=review_id, action=action)
        except Exception as exc:
            return json.dumps({"error": str(exc), "isError": True}, ensure_ascii=False)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False)

    @mcp.tool()
    async def kw_reconcile_registry() -> str:
        """Detect projects removed from the MEMORY.md registry and reconcile them.

        Diffs the current registry against the last snapshot; low-value removed
        entities are auto-pruned, established ones are queued in kw_review_pending
        for confirmation. The first run only establishes the baseline.
        """
        from knowledge_weaver.tools import reconcile_registry_deletions as _rc
        conn = _get_conn()
        try:
            result = _rc(conn)
        except Exception as exc:
            return json.dumps({"error": str(exc), "isError": True}, ensure_ascii=False)
        finally:
            conn.close()
        return json.dumps(result, ensure_ascii=False)

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_consolidation_cli() -> int:
    """Run consolidation from the command line."""
    from knowledge_weaver.pipeline import run_consolidation as _run

    embedder = get_embedder()
    if embedder is None:
        print("WARNING: Embedding not configured. Set EMBEDDING_BASE_URL, EMBEDDING_API_KEY, and EMBEDDING_MODEL.")
        print("Running consolidation without embeddings (vector search will be unavailable).")

    # W6: detect a silent embedding model/dimension change (corrupts vectors).
    if embedder is not None:
        try:
            from knowledge_weaver.db import check_embedding_drift, init_db
            _c = init_db(DB_PATH)
            try:
                d = check_embedding_drift(_c, model=embedder.model, dimension=embedder.dimension)
                if d.get("drift"):
                    print(f"  ⚠️ EMBEDDING DRIFT: {d['changed']} changed "
                          f"({d['stored']} -> {d['current']}). Vector search may be "
                          f"inconsistent — run scripts/re_embed.py, then the meta will update.")
            finally:
                _c.close()
        except Exception as exc:
            print(f"  embedding drift check skipped: {exc}")

    result = _run(DB_PATH, memory_dirs=MEMORY_DIRS, embedder=embedder)

    # W1/B: reconcile registry deletions (snapshot diff) on the production path.
    try:
        from knowledge_weaver.db import init_db
        from knowledge_weaver.tools import (
            ensure_registered_project_entities,
            reconcile_registry_deletions,
        )
        _conn = init_db(DB_PATH)
        try:
            rc = reconcile_registry_deletions(_conn)
            if rc["deleted_detected"]:
                print(f"  Registry reconcile: auto-pruned {len(rc['auto_pruned'])}, "
                      f"queued {len(rc['queued_for_review'])} for review")
            created = ensure_registered_project_entities(_conn)
            if created:
                print(f"  Created {len(created)} registered-project node(s)")
        finally:
            _conn.close()
    except Exception as exc:
        print(f"  Registry reconcile skipped: {exc}")

    # W3.4 online edge typing: type any newly-created RELATES_TO edges into the
    # knowledge-graph vocabulary (env-gated; type_relations only touches the
    # still-generic edges, i.e. the new ones).
    _rel_url = os.environ.get("KW_REL_API_URL")
    if _rel_url and os.environ.get("KW_REL_API_KEY") and os.environ.get("KW_REL_MODEL"):
        try:
            from knowledge_weaver.typed_relations import llm_type_pairs, type_relations
            tr = type_relations(DB_PATH, lambda c: llm_type_pairs(
                c, api_url=_rel_url, api_key=os.environ["KW_REL_API_KEY"],
                model=os.environ["KW_REL_MODEL"]))
            if tr["candidates"]:
                print(f"  Online edge typing: typed {tr['typed']}, pruned {tr['pruned']}")
        except Exception as exc:
            print(f"  Online edge typing skipped: {exc}")

    # Second-pass semantic dedup: catch same-concept-different-name entities that
    # ingest-time resolution never compared (env-gated; default off). "review" =
    # queue every candidate for human approval (safe for unattended cron);
    # "auto" = also auto-merge the HIGH band.
    _dedupe_mode = (os.environ.get("KW_SEMANTIC_DEDUPE") or "").strip().lower()
    if _dedupe_mode in ("review", "auto"):
        _dc = None
        try:
            from knowledge_weaver.db import init_db as _init
            from knowledge_weaver.semantic_dedupe import semantic_dedupe
            _dc = _init(DB_PATH)
            sd = semantic_dedupe(_dc, auto_merge=(_dedupe_mode == "auto"))
            print(f"  Semantic dedupe ({_dedupe_mode}): merged {sd['merged']}, "
                  f"queued {sd['queued']} (high={sd['candidates_high']}, "
                  f"mid={sd['candidates_mid']})")
        except Exception as exc:
            print(f"  Semantic dedupe skipped: {exc}")
        finally:
            if _dc is not None:
                _dc.close()  # always release the connection/lock before the next pass

    # T1: LLM tie-breaker over the merge-review queue — resolves identifier-like
    # pairs the cheap heuristics demoted to review (env-gated; needs KW_REL_* +
    # KW_TIEBREAK_REVIEWS=1, default off). Auto-merges 'same', dismisses 'different'.
    if (os.environ.get("KW_TIEBREAK_REVIEWS") or "").strip() in ("1", "true", "yes"):
        _tb_url = os.environ.get("KW_REL_API_URL")
        if _tb_url and os.environ.get("KW_REL_API_KEY") and os.environ.get("KW_REL_MODEL"):
            _tc = None
            try:
                from knowledge_weaver.db import init_db as _init
                from knowledge_weaver.tiebreak import llm_judge_pairs, tiebreak_reviews
                _tc = _init(DB_PATH)
                tb = tiebreak_reviews(_tc, lambda p: llm_judge_pairs(
                    p, api_url=_tb_url, api_key=os.environ["KW_REL_API_KEY"],
                    model=os.environ["KW_REL_MODEL"]))
                if tb["candidates"]:
                    print(f"  Review tie-break: merged {tb['merged']}, "
                          f"dismissed {tb['dismissed']}, kept {tb['kept']}")
            except Exception as exc:
                print(f"  Review tie-break skipped: {exc}")
            finally:
                if _tc is not None:
                    _tc.close()

    print(f"Consolidation: {result.status}")
    print(f"  Sources: {len(MEMORY_DIRS)}")
    for name, path in MEMORY_DIRS:
        print(f"    - {name}: {path}")
    print(f"  Files processed: {result.files_processed}")
    print(f"  Files skipped:   {result.files_skipped}")
    print(f"  Entities created: {result.entities_created}")
    print(f"  Entities updated: {result.entities_updated}")
    if result.errors:
        for err in result.errors:
            print(f"  ERROR: {err}")
    return 0 if result.status == 'ok' else 1


def main() -> int:
    """CLI entry point — dispatches to serve or consolidate."""
    _configure_logging()

    if len(sys.argv) > 1:
        subcommand = sys.argv[1]
        if subcommand == "consolidate":
            return run_consolidation_cli()
        elif subcommand == "serve":
            mcp = create_server()
            mcp.run()
            return 0
        else:
            print(f"Unknown subcommand: {subcommand}", file=sys.stderr)
            print("Usage: python -m knowledge_weaver.server [serve|consolidate]", file=sys.stderr)
            return 1
    else:
        # Default: serve MCP via stdio
        mcp = create_server()
        mcp.run()
        return 0


if __name__ == "__main__":
    sys.exit(main())
