#!/usr/bin/env python3
"""Re-embed all existing entities using the configured embedder.

Replaces all vectors in entity_vectors and entity_vec tables
with fresh embeddings from the current embedding model.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import struct
import sys

# Allow imports from project src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import init_db
from knowledge_weaver.embedder import DEFAULT_DIMENSION, get_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def re_embed_all(db_path: str) -> None:
    embedder = get_embedder()
    if embedder is None:
        logger.error("No embedder configured. Set EMBEDDING_BASE_URL, EMBEDDING_MODEL env vars.")
        sys.exit(1)

    logger.info("Embedder: model=%s base_url=%s dim=%s", embedder.model, embedder.base_url, embedder.dimension)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Reinitialize DB to create entity_vec with correct dimension
    conn = init_db(db_path)

    # Check if sqlite-vec is available
    can_use_vec: bool = False
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        # Drop and recreate entity_vec with correct dimension
        conn.execute("DROP TABLE IF EXISTS entity_vec")
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS entity_vec "
            f"USING vec0(entity_id TEXT PRIMARY KEY, embedding float[{embedder.dimension}])"
        )
        conn.commit()
        can_use_vec = True
        logger.info("sqlite-vec virtual table recreated with dim=%d", embedder.dimension)
    except Exception:
        logger.warning("sqlite-vec not available, using plain-text fallback only")

    # Clear old vectors
    conn.execute("DELETE FROM entity_vectors")
    conn.commit()

    # Fetch all entities
    rows = conn.execute("SELECT id, name, summary FROM entities").fetchall()
    total = len(rows)
    logger.info("Re-embedding %d entities...", total)

    texts: list[str] = []
    ids: list[str] = []
    vec_count = 0
    batch_size = embedder.MAX_BATCH_SIZE

    for i, row in enumerate(rows):
        # Build embedding text: name + summary
        text = f"{row['name']} {row['summary']}"[:1000]
        texts.append(text)
        ids.append(row["id"])

        if len(texts) >= batch_size or i == total - 1:
            # Embed batch
            try:
                vectors = embedder.embed_batch(texts)
            except Exception as e:
                logger.warning("Embed batch failed at %d: %s", i, e)
                vectors = [[] for _ in texts]

            for eid, vec in zip(ids, vectors):
                if not vec:
                    logger.debug("Empty vector for %s", eid)
                    continue
                vec_list = list(vec)
                # Plain-text fallback
                conn.execute(
                    "INSERT INTO entity_vectors(entity_id, embedding) VALUES (?, ?)",
                    (eid, __import__("json").dumps(vec_list)),
                )
                # sqlite-vec
                if can_use_vec:
                    try:
                        vec_bytes = struct.pack(f"{len(vec_list)}f", *vec_list)
                        conn.execute(
                            "INSERT OR REPLACE INTO entity_vec(entity_id, embedding) VALUES (?, ?)",
                            (eid, vec_bytes),
                        )
                    except Exception:
                        pass
                vec_count += 1

            conn.commit()
            progress = min(i + 1, total)
            logger.info("Progress: %d/%d (%.1f%%) — vectors stored: %d",
                        progress, total, 100 * progress / total, vec_count)

            texts.clear()
            ids.clear()

    # W6: record the embedding identity so drift detection has a fresh baseline.
    try:
        from knowledge_weaver.db import set_embedding_meta
        set_embedding_meta(conn, model=embedder.model, dimension=embedder.dimension)
    except Exception:
        pass

    conn.close()
    logger.info("Done. %d/%d entities embedded.", vec_count, total)


if __name__ == "__main__":
    db_path = os.environ.get("KNOWLEDGE_WEAVER_DB_PATH", "/home/openclaw/.openclaw/knowledge/knowledge.db")
    re_embed_all(db_path)
