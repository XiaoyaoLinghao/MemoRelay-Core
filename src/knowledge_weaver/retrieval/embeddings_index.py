"""Bounded inline vector collection for the recall path (workers normally fill it).

Inline collection runs while recall is already holding candidate revisions, so a
provider call may return after the revision was redacted, retracted or expired by
another request. Writes therefore go through the same late-commit guard as the
embed worker:

* the guard is evaluated **inside** the write transaction immediately before each
  ``INSERT OR REPLACE`` (F11), never against the pre-embed snapshot;
* a rejected candidate is simply not cached (recall degrades for it), while the
  other valid candidates of the same inline batch are still written;
* a *retained* conclusion whose cited source detail expired under retention is
  not a rejected candidate: its own body is still valid, so its vector can be
  rebuilt after index loss or a model-generation switch (F11-R3/AC-40);
* the vector cache key (model id/revision/dimension/normalization + text hash)
  stays exactly the F10 generation key.
"""

from __future__ import annotations

from knowledge_weaver.embeddings.index import (
    claim_vector_commit_guard,
    embedding_text,
    load_vectors,
    store_vector_in_transaction,
    text_hash,
)

INLINE_EMBED_LIMIT = 200


def collect_vectors(store, candidate_ids: set[str], embedder) -> dict[str, list[float]]:
    """Load cached vectors, filling a bounded number of missing ones synchronously.

    A missing vector means semantic search is not ready yet; the caller degrades to
    keyword-only if collection fails. Vector writes are cached by model/dimension
    so the next turn does not re-embed the same revision. A late result whose
    revision became invalid during the provider call is never written (F11).
    """
    if not candidate_ids:
        return {}
    vectors = load_vectors(
        store,
        candidate_ids,
        model_id=embedder.model_id,
        dimension=int(embedder.dimension),
        model_revision=str(getattr(embedder, "model_revision", "1")),
        normalization=str(getattr(embedder, "normalization", "l2")),
    )
    missing = [revision_id for revision_id in candidate_ids if revision_id not in vectors]
    if not missing or len(missing) > INLINE_EMBED_LIMIT:
        return vectors
    placeholders = ",".join("?" for _ in missing)
    with store.read() as conn:
        rows = conn.execute(
            f"SELECT revision_id, display_text FROM claim_revisions "
            f"WHERE revision_id IN ({placeholders}) ORDER BY revision_id",
            missing,
        ).fetchall()
    if not rows:
        return vectors
    embedded = embedder.embed([embedding_text(row["display_text"]) for row in rows])
    # F11: re-read and re-validate each candidate inside the write transaction.
    # The pre-embed snapshot is only used for the vector input and its text hash.
    with store.transaction() as conn:
        for row, vector in zip(rows, embedded):
            ok, _reason = claim_vector_commit_guard(
                conn,
                row["revision_id"],
                expected_text_hash=text_hash(row["display_text"]),
            )
            if not ok:
                continue
            store_vector_in_transaction(
                conn,
                revision_id=row["revision_id"],
                provider=embedder,
                vector=vector,
                display_text=row["display_text"],
            )
            vectors[row["revision_id"]] = vector
    return vectors
