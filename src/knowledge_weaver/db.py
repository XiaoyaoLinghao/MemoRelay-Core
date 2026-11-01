"""SQLite schema and CRUD operations for Knowledge Weaver."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import struct
from typing import Optional

logger = logging.getLogger(__name__)

# Recurrence Gate — A-soft project entity promotion thresholds
PROJECT_MIN_DAYS = int(os.getenv("KNOWLEDGE_WEAVER_PROJECT_MIN_DAYS", "2"))
PROJECT_GRACE_DAYS = int(os.getenv("KNOWLEDGE_WEAVER_PROJECT_GRACE_DAYS", "14"))

# fix: unified from embedder module — production vectors are 1024-dim
from knowledge_weaver.embedder import DEFAULT_DIMENSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    name        TEXT NOT NULL,
    summary     TEXT NOT NULL,
    importance  REAL NOT NULL DEFAULT 0.0,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    day_count   INTEGER NOT NULL DEFAULT 1,
    source_lines TEXT NOT NULL DEFAULT '[]',
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_importance ON entities(importance DESC);
CREATE INDEX IF NOT EXISTS idx_entities_last_seen ON entities(last_seen);
CREATE INDEX IF NOT EXISTS idx_entities_first_seen ON entities(first_seen);

CREATE TABLE IF NOT EXISTS relations (
    id          TEXT PRIMARY KEY,
    from_entity TEXT NOT NULL REFERENCES entities(id),
    to_entity   TEXT NOT NULL REFERENCES entities(id),
    rel_type    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.5,
    evidence    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_entity);
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_entity);
CREATE INDEX IF NOT EXISTS idx_relations_from_to ON relations(from_entity, to_entity);
CREATE INDEX IF NOT EXISTS idx_relations_to_from ON relations(to_entity, from_entity);

CREATE TABLE IF NOT EXISTS daily_manifest (
    date        TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'default',
    file_path   TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    entity_count INTEGER NOT NULL DEFAULT 0,
    processed_at TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ok',
    PRIMARY KEY (date, source)
);

CREATE TABLE IF NOT EXISTS access_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   TEXT NOT NULL REFERENCES entities(id),
    tool        TEXT NOT NULL,
    query       TEXT NOT NULL,
    accessed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_access_log_entity ON access_log(entity_id);
CREATE INDEX IF NOT EXISTS idx_access_log_time ON access_log(accessed_at);

-- W1 Entity Resolution: human-review queue for the uncertain "middle band"
-- (and registry-deletion purge candidates). Never blocks ingestion.
CREATE TABLE IF NOT EXISTS merge_review (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL DEFAULT 'merge',   -- 'merge' | 'purge'
    new_entity_id TEXT NOT NULL,                  -- the freshly-ingested entity (merge) / target (purge)
    candidate_id TEXT,                            -- the existing entity it might merge into (merge only)
    entity_type  TEXT,
    score        REAL,
    reason       TEXT,
    status       TEXT NOT NULL DEFAULT 'pending', -- pending | merged | rejected | purged | kept | dismissed(LLM tie-break) | stale(orphaned by a merge)
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_merge_review_status ON merge_review(status);

-- W1: rollback log for executed merges (and purges). Stores the merged-away
-- entity snapshot so a merge can be undone.
CREATE TABLE IF NOT EXISTS merge_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    merged_from_id TEXT NOT NULL,
    merged_into_id TEXT,                          -- NULL for a purge
    reason         TEXT,
    score          REAL,
    from_snapshot  TEXT,                          -- JSON of the removed entity (for rollback)
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- W1/B: snapshot of registered project slugs, to detect registry deletions
-- by diffing the previous snapshot against the current registry on load.
CREATE TABLE IF NOT EXISTS registry_snapshot (
    slug        TEXT PRIMARY KEY,
    recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- W6: provenance of the embedding model used to build entity_vectors, so a
-- silent provider/model/dimension change (which corrupts vector search) is
-- detected instead of failing quietly.
CREATE TABLE IF NOT EXISTS embedding_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

VECTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_vectors (
    entity_id  TEXT PRIMARY KEY,
    embedding  TEXT NOT NULL,
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);
"""

# Per-connection sqlite-vec availability (keyed by id(conn), cleared on close)
_vec_loaded_conns: dict[int, bool] = {}


def _mark_vec_loaded(conn: sqlite3.Connection) -> None:
    _vec_loaded_conns[id(conn)] = True


def _is_vec_loaded(conn: sqlite3.Connection) -> bool:
    return _vec_loaded_conns.get(id(conn), False)


def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database with all tables and indexes. Returns connection."""
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()

    # Always create the plain-text fallback table
    conn.execute(VECTOR_SCHEMA)
    conn.commit()

    # Try to enable sqlite-vec extension and create virtual table
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        _init_vec_virtual_table(conn)
        _mark_vec_loaded(conn)
        _migrate_vectors_to_vec(conn)
    except Exception:
        # Check if entity_vec already exists (from a previous successful init)
        try:
            conn.execute("SELECT count(*) FROM entity_vec LIMIT 1")
            _mark_vec_loaded(conn)
        except Exception:
            logger.warning(
                "sqlite-vec extension not available; using Python-based vector search. "
                "Install sqlite-vec for native vector search support."
            )

    # Migrate daily_manifest if old single-PK schema exists
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(daily_manifest)")]
        if cols and "source" not in cols:
            conn.execute("ALTER TABLE daily_manifest ADD COLUMN source TEXT NOT NULL DEFAULT 'default'")
            conn.commit()
            logger.info("Migrated daily_manifest: added source column")
    except sqlite3.OperationalError:
        pass

    # Create FTS5 virtual table and rebuild index
    _init_fts_table(conn)

    return conn


def _init_vec_virtual_table(conn: sqlite3.Connection) -> None:
    """Create the vec0 virtual table for native vector search."""
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS entity_vec "
        f"USING vec0(entity_id TEXT PRIMARY KEY, embedding float[{DEFAULT_DIMENSION}])"
    )
    conn.commit()


def _migrate_vectors_to_vec(conn: sqlite3.Connection) -> None:
    """Migrate existing JSON vectors from entity_vectors to entity_vec virtual table."""
    # Check if entity_vec already has data
    count = conn.execute("SELECT count(*) FROM entity_vec").fetchone()[0]
    if count > 0:
        return
    # Migrate from plain-text table
    rows = conn.execute("SELECT entity_id, embedding FROM entity_vectors").fetchall()
    for row in rows:
        try:
            vec = json.loads(row[1])
            vec_bytes = struct.pack(f"{len(vec)}f", *vec)
            conn.execute(
                "INSERT OR REPLACE INTO entity_vec(entity_id, embedding) VALUES (?, ?)",
                (row[0], vec_bytes),
            )
        except (json.JSONDecodeError, struct.error):
            continue
    conn.commit()
    logger.info("Migrated %d vectors to sqlite-vec virtual table", len(rows))


# --- Chinese-aware FTS tokenization (jieba) -------------------------------- #
# fts5's unicode61 tokenizer does NOT segment Chinese, so a CJK query matches
# poorly. We pre-segment indexed text AND queries with jieba (space-joined),
# which unicode61 then splits on — mirroring the TDAM approach.
_jieba_mod = None
_jieba_tried = False


def _get_jieba():
    global _jieba_mod, _jieba_tried
    if not _jieba_tried:
        _jieba_tried = True
        try:
            import jieba  # noqa: PLC0415
            jieba.setLogLevel(60)  # silence jieba's logging
            _jieba_mod = jieba
        except Exception:
            _jieba_mod = None
    return _jieba_mod


def jieba_tokenize(text: Optional[str]) -> str:
    """Space-join jieba search tokens (falls back to original text)."""
    if not text:
        return text or ""
    j = _get_jieba()
    if j is None:
        return text
    try:
        return " ".join(t for t in j.cut_for_search(text) if t.strip())
    except Exception:
        return text


def build_fts_match(query: str) -> str:
    """Build an FTS5 MATCH expression from a query (jieba tokens OR-joined)."""
    j = _get_jieba()
    if j is not None:
        try:
            toks = [t.strip() for t in j.cut_for_search(query) if t.strip()]
        except Exception:
            toks = re.findall(r"[\w]+", query)
    else:
        toks = re.findall(r"[\w]+", query)
    toks = list(dict.fromkeys(toks))  # dedup, preserve order
    if not toks:
        return query
    return " OR ".join('"' + t.replace('"', "") + '"' for t in toks)


def _aliases_from_metadata(metadata: Optional[str]) -> list[str]:
    """Extract the alias surface-forms stored in an entity's metadata JSON."""
    try:
        meta = json.loads(metadata or "{}")
    except Exception:
        return []
    al = meta.get("aliases", []) if isinstance(meta, dict) else []
    return [a for a in al if isinstance(a, str) and a.strip()]


def fts_name_value(name: Optional[str], aliases: Optional[list[str]] = None) -> str:
    """jieba-tokenized name PLUS its alias surface-forms, so a query using a
    merged-away ('wrong'/variant) name still matches the surviving entity
    lexically. Without this, aliases live only in metadata JSON, invisible to FTS."""
    parts = [name or ""] + [a for a in (aliases or []) if a and a != name]
    return jieba_tokenize(" ".join(p for p in parts if p))


def rebuild_fts(conn: sqlite3.Connection, auto_commit: bool = True) -> int:
    """Re-index entity_fts with jieba-segmented name(+aliases)/summary (one-off migration)."""
    conn.execute("DELETE FROM entity_fts")
    rows = conn.execute("SELECT id, name, summary, type, metadata FROM entities").fetchall()
    for r in rows:
        conn.execute(
            "INSERT INTO entity_fts(entity_id, name, summary, type) VALUES (?, ?, ?, ?)",
            (r["id"], fts_name_value(r["name"], _aliases_from_metadata(r["metadata"])),
             jieba_tokenize(r["summary"]), r["type"]),
        )
    if auto_commit:
        conn.commit()
    return len(rows)


def _init_fts_table(conn: sqlite3.Connection) -> None:
    """Create FTS5 virtual table and rebuild index from existing entities."""
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5("
        "entity_id UNINDEXED, name, summary, type, tokenize='unicode61')"
    )
    # Rebuild only if FTS table is empty
    count = conn.execute("SELECT count(*) FROM entity_fts").fetchone()[0]
    if count == 0:
        conn.execute(
            "INSERT INTO entity_fts(entity_id, name, summary, type) "
            "SELECT id, name, summary, type FROM entities"
        )
        conn.commit()
        logger.info("FTS5 index rebuilt from %d entities", count)


def is_provisional_project(row: dict, registered_slugs: set[str] | None = None) -> bool:
    """A project with day_count below threshold is 'provisional' (hidden from queries).

    Non-project types always return False.
    Explicitly registered projects (via registered_slugs) bypass the gate.
    Accepts both dict and sqlite3.Row objects.
    """
    # Support both dict.get() and sqlite3.Row dict-style access
    try:
        row_type = row.get("type")
    except AttributeError:
        row_type = row["type"] if "type" in row.keys() else None
    if row_type != "project":
        return False
    if registered_slugs:
        try:
            row_id = row.get("id")
        except AttributeError:
            row_id = row["id"] if "id" in row.keys() else None
        if row_id in registered_slugs:
            return False
    try:
        day_count = row.get("day_count", 0)
    except AttributeError:
        day_count = row["day_count"] if "day_count" in row.keys() else 0
    return day_count < PROJECT_MIN_DAYS


# --- Entity operations ---


def insert_entity(conn: sqlite3.Connection, entity: dict, auto_commit: bool = True) -> None:
    """Insert or UPSERT an entity record."""
    defaults = {
        "day_count": 1,
        "source_lines": "[]",
        "metadata": "{}",
    }
    e = {**defaults, **entity}
    conn.execute(
        """INSERT INTO entities (id, type, name, summary, importance, first_seen, last_seen,
           day_count, source_lines, metadata, updated_at)
           VALUES (:id, :type, :name, :summary, :importance, :first_seen, :last_seen,
           :day_count, :source_lines, :metadata, datetime('now'))
           ON CONFLICT(id) DO UPDATE SET
           type=excluded.type, name=excluded.name, summary=excluded.summary,
           importance=excluded.importance, last_seen=excluded.last_seen,
           day_count=excluded.day_count, source_lines=excluded.source_lines,
           metadata=excluded.metadata, updated_at=datetime('now')""",
        e,
    )
    # Sync to FTS5 index (name+aliases so variant names stay lexically searchable)
    try:
        conn.execute("DELETE FROM entity_fts WHERE entity_id=?", (e["id"],))
        conn.execute(
            "INSERT INTO entity_fts(entity_id, name, summary, type) VALUES (?, ?, ?, ?)",
            (e["id"], fts_name_value(e["name"], _aliases_from_metadata(e.get("metadata"))),
             jieba_tokenize(e["summary"]), e["type"]),
        )
    except Exception:
        pass  # FTS table may not exist yet
    if auto_commit:
        conn.commit()


def get_entity(conn: sqlite3.Connection, entity_id: str) -> Optional[sqlite3.Row]:
    """Get entity by ID."""
    return conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()


def list_entities_by_type(conn: sqlite3.Connection, entity_type: str) -> list[sqlite3.Row]:
    """List entities of a given type, ordered by importance DESC."""
    return conn.execute(
        "SELECT * FROM entities WHERE type=? ORDER BY importance DESC",
        (entity_type,),
    ).fetchall()


def list_all_entities(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """List all entities ordered by importance DESC."""
    return conn.execute("SELECT * FROM entities ORDER BY importance DESC").fetchall()


# ---------------------------------------------------------------------------
# W1 Entity Resolution helpers: stored-vector lookup, review queue, merge log.
# ---------------------------------------------------------------------------
def get_entity_vector(conn: sqlite3.Connection, entity_id: str) -> Optional[list[float]]:
    """Return an entity's stored embedding (for consistent cosine scoring).

    Reads from entity_vectors (JSON) so the score uses the same cosine metric
    regardless of the vec0 table's (default L2) distance metric.
    """
    row = conn.execute(
        "SELECT embedding FROM entity_vectors WHERE entity_id=?", (entity_id,)
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None


def insert_review(conn: sqlite3.Connection, *, kind: str, new_entity_id: str,
                  candidate_id: Optional[str], entity_type: str, score: float,
                  reason: str, auto_commit: bool = True) -> int:
    """Queue a merge/purge candidate for human review (idempotent per pair).

    Idempotent against every *decided-distinct* status — pending (in queue),
    dismissed (LLM tie-breaker) AND rejected (human via kw_resolve) — and
    direction-agnostic (a→b == b→a). Without this the self-maintenance loop would
    re-surface and re-judge (LLM cost) the same distinct pair every consolidation
    cycle. ``merged``/``purged`` are deliberately NOT skipped: a merge rolled back
    via merge_log should be re-evaluable.
    """
    existing = conn.execute(
        "SELECT id FROM merge_review WHERE status IN ('pending','dismissed','rejected') "
        "AND kind=? AND ("
        "(new_entity_id=? AND IFNULL(candidate_id,'')=IFNULL(?,'')) OR "
        "(new_entity_id=? AND IFNULL(candidate_id,'')=IFNULL(?,'')))",
        (kind, new_entity_id, candidate_id, candidate_id, new_entity_id),
    ).fetchone()
    if existing:
        return existing[0]
    cur = conn.execute(
        "INSERT INTO merge_review(kind, new_entity_id, candidate_id, entity_type, "
        "score, reason) VALUES (?, ?, ?, ?, ?, ?)",
        (kind, new_entity_id, candidate_id, entity_type, score, reason),
    )
    if auto_commit:
        conn.commit()
    return int(cur.lastrowid)


def count_pending_reviews(conn: sqlite3.Connection) -> int:
    """Number of pending review items (for the HEARTBEAT.md nudge)."""
    return conn.execute(
        "SELECT COUNT(*) FROM merge_review WHERE status='pending'"
    ).fetchone()[0]


def list_pending_reviews(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    """Pending review items, strongest signal first."""
    return conn.execute(
        "SELECT * FROM merge_review WHERE status='pending' "
        "ORDER BY score DESC, created_at ASC LIMIT ?",
        (limit,),
    ).fetchall()


def set_review_status(conn: sqlite3.Connection, review_id: int, status: str,
                      auto_commit: bool = True) -> None:
    conn.execute(
        "UPDATE merge_review SET status=?, resolved_at=datetime('now') WHERE id=?",
        (status, review_id),
    )
    if auto_commit:
        conn.commit()


def record_merge(conn: sqlite3.Connection, *, merged_from_id: str,
                 merged_into_id: Optional[str], reason: str, score: float,
                 from_snapshot: str, auto_commit: bool = True) -> None:
    """Log an executed merge/purge so it can be rolled back."""
    conn.execute(
        "INSERT INTO merge_log(merged_from_id, merged_into_id, reason, score, "
        "from_snapshot) VALUES (?, ?, ?, ?, ?)",
        (merged_from_id, merged_into_id, reason, score, from_snapshot),
    )
    if auto_commit:
        conn.commit()


def get_review(conn: sqlite3.Connection, review_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM merge_review WHERE id=?", (review_id,)).fetchone()


def clean_entity_indexes(conn: sqlite3.Connection, entity_id: str) -> None:
    """Remove an entity's FTS + vector index rows (entity row handled separately)."""
    for sql in (
        "DELETE FROM entity_fts WHERE entity_id=?",
        "DELETE FROM entity_vectors WHERE entity_id=?",
        "DELETE FROM entity_vec WHERE entity_id=?",
    ):
        try:
            conn.execute(sql, (entity_id,))
        except Exception:
            pass


def merge_existing_entities(conn: sqlite3.Connection, from_id: str, into_id: str, *,
                            reason: str = "manual", score: float = 0.0,
                            auto_commit: bool = True) -> bool:
    """Merge an existing entity ``from_id`` INTO ``into_id`` (into stays canonical).

    Unions source_lines + aliases, repoints relations, removes from_id's row +
    indexes, and writes a rollback snapshot (entity + its relations) to merge_log.
    """
    src = get_entity(conn, from_id)
    dst = get_entity(conn, into_id)
    if src is None or dst is None or from_id == into_id:
        return False

    rels = [dict(r) for r in get_relations_for_entity(conn, from_id)]
    # K4: also snapshot dst's PRE-merge accumulators so rollback can restore them
    # (the merge bumps day_count / unions source_lines / MINs first_seen / etc.).
    snapshot = json.dumps({
        "entity": dict(src),
        "relations": rels,
        "dst_pre": {k: dst[k] for k in
                    ("day_count", "source_lines", "first_seen", "importance", "metadata")},
    }, ensure_ascii=False)

    src_sources = json.loads(src["source_lines"] or "[]")
    dst_sources = json.loads(dst["source_lines"] or "[]")
    merged_sources = list(dict.fromkeys(dst_sources + src_sources))

    dst_meta = json.loads(dst["metadata"] or "{}")
    src_meta = json.loads(src["metadata"] or "{}")
    aliases = list(dst_meta.get("aliases", []))
    for a in [src["name"]] + list(src_meta.get("aliases", [])):
        if a and a != dst["name"] and a not in aliases:
            aliases.append(a)
    if aliases:
        dst_meta["aliases"] = aliases

    conn.execute(
        "UPDATE entities SET source_lines=?, metadata=?, importance=?, "
        "first_seen=MIN(first_seen, ?), day_count=day_count+?, "
        "updated_at=datetime('now') WHERE id=?",
        (json.dumps(merged_sources, ensure_ascii=False),
         json.dumps(dst_meta, ensure_ascii=False),
         max(src["importance"], dst["importance"]),
         src["first_seen"], src["day_count"], into_id),
    )

    # Refresh dst's FTS row so the merged-away name (now an alias) is searchable.
    try:
        conn.execute("DELETE FROM entity_fts WHERE entity_id=?", (into_id,))
        conn.execute(
            "INSERT INTO entity_fts(entity_id, name, summary, type) VALUES (?, ?, ?, ?)",
            (into_id, fts_name_value(dst["name"], aliases),
             jieba_tokenize(dst["summary"]), dst["type"]),
        )
    except Exception:
        pass

    from knowledge_weaver.linker import generate_relation_id
    for r in rels:
        nf = into_id if r["from_entity"] == from_id else r["from_entity"]
        nt = into_id if r["to_entity"] == from_id else r["to_entity"]
        if nf == nt:
            continue
        # K3: INSERT OR IGNORE (not REPLACE) — if the repointed edge collides with
        # an edge dst already has, keep dst's existing weight/evidence instead of
        # clobbering it with from's (which rollback could not restore).
        conn.execute(
            "INSERT OR IGNORE INTO relations "
            "(id, from_entity, to_entity, rel_type, weight, evidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (generate_relation_id(nf, nt, r["rel_type"]), nf, nt, r["rel_type"],
             r.get("weight", 1.0), r.get("evidence", "")),
        )

    delete_entity(conn, from_id, auto_commit=False)  # also drops from_id's relations
    clean_entity_indexes(conn, from_id)
    # Any OTHER pending review referencing the now-deleted entity is dangling —
    # mark it 'stale' so it stops being counted/re-surfaced (the review that drove
    # this merge, if any, is set to 'merged' by its caller afterward).
    conn.execute(
        "UPDATE merge_review SET status='stale', resolved_at=datetime('now') "
        "WHERE status='pending' AND (new_entity_id=? OR candidate_id=?)",
        (from_id, from_id),
    )
    record_merge(conn, merged_from_id=from_id, merged_into_id=into_id,
                 reason=reason, score=score, from_snapshot=snapshot, auto_commit=False)
    if auto_commit:
        conn.commit()
    return True


def rollback_merge(conn: sqlite3.Connection, log_id: int,
                   auto_commit: bool = True) -> bool:
    """Undo a merge: recreate the merged-away entity + relations from snapshot,
    and drop its alias from the surviving entity. (Vector needs re-embed.)"""
    row = conn.execute(
        "SELECT merged_into_id, from_snapshot FROM merge_log WHERE id=?", (log_id,)
    ).fetchone()
    if not row or not row[1]:
        return False
    snap = json.loads(row[1])
    ent = snap.get("entity")
    if not ent:
        return False
    insert_entity(conn, {k: ent[k] for k in (
        "id", "type", "name", "summary", "importance", "first_seen",
        "last_seen", "day_count", "source_lines", "metadata")}, auto_commit=False)
    for r in snap.get("relations", []):
        insert_relation(conn, r, auto_commit=False)
    into_id = row[0]
    dst_pre = snap.get("dst_pre")
    if into_id:
        dst = get_entity(conn, into_id)
        if dst is not None and dst_pre:
            # K4: restore dst's full pre-merge accumulators (day_count / source_lines
            # / first_seen / importance / metadata) — not just strip one alias.
            conn.execute(
                "UPDATE entities SET day_count=?, source_lines=?, first_seen=?, "
                "importance=?, metadata=?, updated_at=datetime('now') WHERE id=?",
                (dst_pre["day_count"], dst_pre["source_lines"], dst_pre["first_seen"],
                 dst_pre["importance"], dst_pre["metadata"], into_id),
            )
            try:
                meta = json.loads(dst_pre["metadata"] or "{}")
                conn.execute("DELETE FROM entity_fts WHERE entity_id=?", (into_id,))
                conn.execute(
                    "INSERT INTO entity_fts(entity_id, name, summary, type) VALUES (?, ?, ?, ?)",
                    (into_id, fts_name_value(dst["name"], meta.get("aliases", [])),
                     jieba_tokenize(dst["summary"]), dst["type"]),
                )
            except Exception:
                pass
        elif dst is not None:
            # legacy snapshot without dst_pre: best-effort alias strip (old behavior)
            meta = json.loads(dst["metadata"] or "{}")
            aliases = [a for a in meta.get("aliases", []) if a != ent.get("name")]
            if aliases:
                meta["aliases"] = aliases
            else:
                meta.pop("aliases", None)
            conn.execute("UPDATE entities SET metadata=? WHERE id=?",
                         (json.dumps(meta, ensure_ascii=False), into_id))
    # K4: restore reviews this merge orphaned to 'stale' — the entity is back.
    conn.execute(
        "UPDATE merge_review SET status='pending', resolved_at=NULL "
        "WHERE status='stale' AND (new_entity_id=? OR candidate_id=?)",
        (ent["id"], ent["id"]),
    )
    if auto_commit:
        conn.commit()
    return True


def previous_registered_slugs(conn: sqlite3.Connection) -> set[str]:
    """Slugs recorded in the last registry snapshot (empty on first run)."""
    return {r[0] for r in conn.execute("SELECT slug FROM registry_snapshot").fetchall()}


def snapshot_registered_slugs(conn: sqlite3.Connection, slugs: set[str],
                              auto_commit: bool = True) -> None:
    """Replace the registry snapshot with the current registered slug set."""
    conn.execute("DELETE FROM registry_snapshot")
    conn.executemany(
        "INSERT OR IGNORE INTO registry_snapshot(slug) VALUES (?)",
        [(s,) for s in slugs],
    )
    if auto_commit:
        conn.commit()


# --- W6: embedding-model drift detection ----------------------------------- #
def get_embedding_meta(conn: sqlite3.Connection) -> dict:
    return {r[0]: r[1] for r in conn.execute("SELECT key, value FROM embedding_meta").fetchall()}


def set_embedding_meta(conn: sqlite3.Connection, *, model: str, dimension: int,
                       provider: str = "openai-compatible", auto_commit: bool = True) -> None:
    for k, v in (("model", str(model)), ("dimension", str(dimension)),
                 ("provider", str(provider))):
        conn.execute(
            "INSERT INTO embedding_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, v))
    if auto_commit:
        conn.commit()


def check_embedding_drift(conn: sqlite3.Connection, *, model: str, dimension: int,
                          provider: str = "openai-compatible") -> dict:
    """Compare the current embedder identity against the recorded one.

    Records current on first run (no drift). On a later model/dimension change,
    returns drift=True so the caller can warn and trigger a re-embed (a silent
    change otherwise corrupts vector search)."""
    stored = get_embedding_meta(conn)
    current = {"model": str(model), "dimension": str(dimension), "provider": str(provider)}
    if not stored:
        set_embedding_meta(conn, model=model, dimension=dimension, provider=provider)
        return {"drift": False, "first_record": True, "current": current}
    changed = [k for k in ("model", "dimension") if stored.get(k) != current[k]]
    return {"drift": bool(changed), "changed": changed, "stored": stored, "current": current}


def search_entities_fts(
    conn: sqlite3.Connection, query: str, limit: int = 10
) -> list[sqlite3.Row]:
    """Full-text search using FTS5, with LIKE fallback."""
    try:
        rows = conn.execute(
            """SELECT e.* FROM entity_fts f
               JOIN entities e ON f.entity_id = e.id
               WHERE entity_fts MATCH ?
               ORDER BY f.rank LIMIT ?""",
            (build_fts_match(query), limit),
        ).fetchall()
        if rows:
            return rows
    except Exception:
        pass
    # Fallback to LIKE if FTS5 unavailable or no results
    return conn.execute(
        """SELECT * FROM entities WHERE name LIKE ? OR summary LIKE ?
           ORDER BY importance DESC LIMIT ?""",
        (f"%{query}%", f"%{query}%", limit),
    ).fetchall()


def delete_entity(conn: sqlite3.Connection, entity_id: str, auto_commit: bool = True) -> None:
    """Delete entity and its relations."""
    conn.execute("DELETE FROM relations WHERE from_entity=? OR to_entity=?", (entity_id, entity_id))
    conn.execute("DELETE FROM entities WHERE id=?", (entity_id,))
    if auto_commit:
        conn.commit()


# --- Relation operations ---


def insert_relation(conn: sqlite3.Connection, rel: dict, auto_commit: bool = True) -> None:
    """Insert or REPLACE a relation. Self-loops are silently rejected."""
    if rel.get("from_entity") == rel.get("to_entity"):
        return
    conn.execute(
        """INSERT OR REPLACE INTO relations (id, from_entity, to_entity, rel_type, weight, evidence)
           VALUES (:id, :from_entity, :to_entity, :rel_type, :weight, :evidence)""",
        rel,
    )
    if auto_commit:
        conn.commit()


def get_relations_for_entity(
    conn: sqlite3.Connection, entity_id: str
) -> list[sqlite3.Row]:
    """Get all relations where entity appears as from_entity or to_entity."""
    return conn.execute(
        "SELECT * FROM relations WHERE from_entity=? OR to_entity=? ORDER BY weight DESC",
        (entity_id, entity_id),
    ).fetchall()


def get_relations_for_entities(
    conn: sqlite3.Connection, entity_ids: list[str]
) -> list[sqlite3.Row]:
    """Get all relations where any of the given entities appear as from or to."""
    if not entity_ids:
        return []
    placeholders = ",".join(["?" for _ in entity_ids])
    return conn.execute(
        f"SELECT * FROM relations WHERE from_entity IN ({placeholders}) OR to_entity IN ({placeholders}) ORDER BY weight DESC",
        entity_ids + entity_ids,
    ).fetchall()


def get_entities_by_ids(
    conn: sqlite3.Connection, entity_ids: list[str]
) -> dict[str, sqlite3.Row]:
    """Get multiple entities by ID. Returns {id: Row} dict."""
    if not entity_ids:
        return {}
    placeholders = ",".join(["?" for _ in entity_ids])
    rows = conn.execute(
        f"SELECT * FROM entities WHERE id IN ({placeholders})",
        entity_ids,
    ).fetchall()
    return {r["id"]: r for r in rows}


# --- daily_manifest operations ---


def get_manifest(conn: sqlite3.Connection, date: str, source: str = "default") -> Optional[sqlite3.Row]:
    """Get manifest entry for a given date and source."""
    return conn.execute(
        "SELECT * FROM daily_manifest WHERE date=? AND source=?", (date, source)
    ).fetchone()


def upsert_manifest(conn: sqlite3.Connection, entry: dict, auto_commit: bool = True) -> None:
    """Insert or update a daily manifest entry."""
    entry = {"source": "default", **entry}
    conn.execute(
        """INSERT INTO daily_manifest (date, source, file_path, file_hash, entity_count, processed_at, status)
           VALUES (:date, :source, :file_path, :file_hash, :entity_count, datetime('now'), :status)
           ON CONFLICT(date, source) DO UPDATE SET
           file_path=excluded.file_path, file_hash=excluded.file_hash,
           entity_count=excluded.entity_count, processed_at=datetime('now'),
           status=excluded.status""",
        entry,
    )
    if auto_commit:
        conn.commit()


def list_all_manifest(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """List all manifest entries ordered by date."""
    return conn.execute(
        "SELECT * FROM daily_manifest ORDER BY date"
    ).fetchall()


# --- access_log operations ---


def log_access(
    conn: sqlite3.Connection, entity_id: str, tool: str, query: str,
    auto_commit: bool = True,
) -> None:
    """Log an access event for an entity."""
    conn.execute(
        """INSERT INTO access_log (entity_id, tool, query)
           VALUES (?, ?, ?)""",
        (entity_id, tool, query),
    )
    if auto_commit:
        conn.commit()


def get_access_count(conn: sqlite3.Connection, entity_id: str) -> int:
    """Get the number of access events for an entity."""
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM access_log WHERE entity_id=?",
        (entity_id,),
    ).fetchone()
    return row["cnt"] if row else 0


def _can_use_vec(conn: sqlite3.Connection) -> bool:
    """Check if sqlite-vec virtual table is usable with this connection.

    Results are tracked per-connection — a connection that fails to load
    the extension will not retry, and a connection that succeeds will
    not need to re-load.
    """
    if _is_vec_loaded(conn):
        return True
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        conn.execute("SELECT count(*) FROM entity_vec LIMIT 1")
        _mark_vec_loaded(conn)
        return True
    except Exception:
        return False


def upsert_entity_vector(conn: sqlite3.Connection, entity_id: str, embedding: list[float], auto_commit: bool = True) -> None:
    """Store embedding in both plain-text table and sqlite-vec virtual table (if available)."""
    # Always write to plain-text fallback
    conn.execute(
        "INSERT OR REPLACE INTO entity_vectors(entity_id, embedding) VALUES (?, ?)",
        (entity_id, json.dumps(embedding)),
    )
    # Also write to sqlite-vec virtual table if available
    if _can_use_vec(conn):
        try:
            vec_bytes = struct.pack(f"{len(embedding)}f", *embedding)
            conn.execute(
                "INSERT OR REPLACE INTO entity_vec(entity_id, embedding) VALUES (?, ?)",
                (entity_id, vec_bytes),
            )
        except Exception:
            logger.debug("Failed to insert into entity_vec virtual table")
    if auto_commit:
        conn.commit()


def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_entity_vectors(conn: sqlite3.Connection, query_vec: list[float],
                          limit: int = 10) -> list[sqlite3.Row]:
    """Search entities by cosine similarity.

    Uses sqlite-vec virtual table when available (O(log N)),
    falls back to Python O(N) scan otherwise.
    """
    if _can_use_vec(conn):
        try:
            return _search_entity_vectors_vec(conn, query_vec, limit)
        except Exception:
            logger.debug("sqlite-vec search failed, falling back to Python scan")

    return _search_entity_vectors_python(conn, query_vec, limit)


def _search_entity_vectors_vec(conn: sqlite3.Connection, query_vec: list[float],
                                limit: int = 10) -> list[sqlite3.Row]:
    """Native vector search using sqlite-vec virtual table."""
    vec_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)
    rows = conn.execute(
        "SELECT entity_id, distance FROM entity_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (vec_bytes, limit),
    ).fetchall()
    top_ids = [r[0] for r in rows]
    if not top_ids:
        return []
    placeholders = ",".join(["?" for _ in top_ids])
    fetched = conn.execute(
        f"SELECT * FROM entities WHERE id IN ({placeholders})",
        top_ids,
    ).fetchall()
    # K1: `IN (...)` returns rows in arbitrary order — restore the distance order
    # from the vec0 query so callers (RRF fusion) get nearest-first. (Only the
    # sqlite-vec path was affected; the Python fallback already sorts by score.)
    by_id = {r["id"]: r for r in fetched}
    return [by_id[i] for i in top_ids if i in by_id]


def _search_entity_vectors_python(conn: sqlite3.Connection, query_vec: list[float],
                                   limit: int = 10) -> list[sqlite3.Row]:
    """Fallback: Python O(N) cosine similarity scan."""
    rows = conn.execute(
        "SELECT v.entity_id, v.embedding FROM entity_vectors v"
    ).fetchall()
    scored = []
    for r in rows:
        try:
            vec = json.loads(r[1])
            sim = _cosine(query_vec, vec)
            scored.append((sim, r[0]))
        except (json.JSONDecodeError, TypeError, IndexError):
            pass
    scored.sort(key=lambda x: x[0], reverse=True)
    top_ids = [eid for _, eid in scored[:limit]]
    if not top_ids:
        return []
    placeholders = ",".join(["?" for _ in top_ids])
    return conn.execute(
        f"SELECT * FROM entities WHERE id IN ({placeholders})",
        top_ids,
    ).fetchall()
