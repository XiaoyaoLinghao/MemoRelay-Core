"""FTS5 keyword index and search over claim revision text."""

from __future__ import annotations

from knowledge_weaver.store.db import Store

from .tokenizer import fts_query, fts_text


def index_revision_in_transaction(conn, revision_id: str, text: str) -> None:
    """Insert/replace the keyword row for one revision inside a transaction."""
    conn.execute("DELETE FROM claim_fts WHERE revision_id = ?", (revision_id,))
    conn.execute(
        "INSERT INTO claim_fts (revision_id, tokens) VALUES (?, ?)",
        (revision_id, fts_text(text)),
    )


def keyword_ranking(
    store: Store,
    *,
    candidate_revision_ids: set[str],
    query: str,
    max_scan: int = 1000,
) -> list[str]:
    """Rank candidate revisions for the query by FTS5 bm25 (best first)."""
    expression = fts_query(query)
    if not expression or not candidate_revision_ids:
        return []
    with store.read() as conn:
        rows = conn.execute(
            "SELECT revision_id FROM claim_fts WHERE claim_fts MATCH ? "
            "ORDER BY bm25(claim_fts), revision_id LIMIT ?",
            (expression, max_scan),
        ).fetchall()
    return [row["revision_id"] for row in rows if row["revision_id"] in candidate_revision_ids]
