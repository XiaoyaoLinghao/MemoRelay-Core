"""Semantic vector cache and embed jobs (optional capability, no hard deps).

The core never imports a vector database: vectors are float32 blobs in the local
SQLite file, keyed by revision + normalized text hash + model id/revision +
dimension + normalization. Switching model means a new index generation; tokens
with a missing/incompatible vector degrade recall to keyword-only and say so
(incomplete_reasons=embedding_unavailable), never mix indexes.
"""

from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from typing import Protocol

from knowledge_weaver import contracts
from knowledge_weaver.jobs import JobError, commit, enqueue_in_transaction
from knowledge_weaver.store.db import Store, utc_now

_ACTIVE: dict = {
    "model_id": None,
    "model_revision": None,
    "dimension": None,
    "normalization": "l2",
}


def configure_model(
    model_id: str | None,
    *,
    model_revision: str = "1",
    dimension: int | None = None,
    normalization: str = "l2",
) -> None:
    """Enable embed-job enqueueing for claim writes (T17 wires this from config)."""
    _ACTIVE.update(
        {
            "model_id": model_id,
            "model_revision": model_revision if model_id else None,
            "dimension": dimension,
            "normalization": normalization,
        }
    )


def active_model() -> dict | None:
    return dict(_ACTIVE) if _ACTIVE["model_id"] else None


def embedding_text(display_text: str) -> str:
    return unicodedata.normalize("NFKC", display_text or "").strip()


def text_hash(display_text: str) -> str:
    return hashlib.sha256(embedding_text(display_text).encode("utf-8")).hexdigest()


class EmbeddingProvider(Protocol):
    model_id: str
    model_revision: str
    dimension: int
    normalization: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic local provider used by tests and offline smoke checks."""

    def __init__(self, dimension: int = 64, model_id: str = "fake:hash-embedding") -> None:
        self.dimension = dimension
        self.model_id = model_id
        self.model_revision = "1"
        self.normalization = "l2"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(embedding_text(text).encode("utf-8")).digest()
            values = [((byte / 255.0) * 2 - 1) for byte in digest]
            while len(values) < self.dimension:
                digest = hashlib.sha256(digest).digest()
                values.extend(((byte / 255.0) * 2 - 1) for byte in digest)
            values = values[: self.dimension]
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append([value / norm for value in values])
        return vectors


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _generation_input_version(model_id: str, model_revision: str, dimension, normalization: str) -> int:
    """Stable job input-version for one vector generation (F10).

    Embedding jobs are idempotent per ``(kind, input_key, input_version)``; folding
    the model id/revision/dimension/normalization into the version means a model
    or normalization switch gets its own job instead of silently reusing the old
    generation's deduplicated job.
    """
    payload = contracts.compact_json(
        [str(model_id), str(model_revision), int(dimension or 0), str(normalization)]
    )
    return 1 + int(contracts.sha256_hex(payload)[:12], 16)


def enqueue_embed_in_transaction(
    conn,
    revision_id: str,
    *,
    model_id: str | None = None,
    model_revision: str | None = None,
) -> dict:
    model = active_model() or {}
    resolved_model = model_id or model.get("model_id")
    if not resolved_model:
        raise JobError("INVALID_REQUEST", "no embedding model configured")
    resolved_revision = str(model_revision or model.get("model_revision") or "1")
    dimension = model.get("dimension") or 0
    normalization = str(model.get("normalization") or "l2")
    return enqueue_in_transaction(
        conn,
        "embed",
        revision_id,
        input_version=_generation_input_version(
            resolved_model, resolved_revision, dimension, normalization
        ),
        model_version=f"{resolved_model}@{resolved_revision}",
        policy_version=contracts.redaction_policy_version(),
    )


def store_vector(store: Store, *, revision_id: str, provider, vector: list[float], display_text: str) -> None:
    with store.transaction() as conn:
        store_vector_in_transaction(
            conn, revision_id=revision_id, provider=provider, vector=vector, display_text=display_text
        )


def store_vector_in_transaction(conn, *, revision_id: str, provider, vector: list[float], display_text: str) -> None:
    if len(vector) != int(provider.dimension):
        raise ValueError("embedding dimension does not match provider")
    conn.execute(
        "INSERT OR REPLACE INTO claim_vectors "
        "(revision_id, text_hash, model_id, model_revision, dimension, normalization, vector, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            revision_id,
            text_hash(display_text),
            provider.model_id,
            provider.model_revision,
            int(provider.dimension),
            getattr(provider, "normalization", "l2"),
            sqlite_blob(vector),
            utc_now(),
        ),
    )


def sqlite_blob(vector: list[float]):
    from sqlite3 import Binary

    return Binary(_pack(vector))


def claim_vector_commit_guard(
    conn, revision_id: str, *, expected_text_hash: str
) -> tuple[bool, str]:
    """F11: the one late-commit guard shared by every vector write path.

    A vector may only be written while the revision that produced it is still
    valid input evidence. Row existence alone is not enough: the revision may
    have expired/been revoked/superseded (with a newer deletion generation), its
    text may have been scrubbed or corrected while the provider was running, a
    retention tombstone may have landed, or its source evidence may have been
    withheld/retracted. The embed worker and the recall inline collection both
    call this *inside* their write transaction, immediately before the vector
    insert.

    F11-R3/AC-40: *source-detail expiry* is deliberately different from
    *revision invalidation*. When retention purges the cited source detail, the
    still-valid conclusion keeps its own retained body (retraction/expiry of the
    revision itself scrubs it), so its vector is a rebuildable projection of that
    body: index loss, a model-generation switch or a restore may rebuild it.
    Only retention writes ``availability = 'expired'`` (see
    ``lifecycle.retention._expire_event_part`` / ``_expire_claim_revision``); a
    redacted source is ``withheld`` and a retracted one is ``retracted``, and
    both keep rejecting late results.
    """
    current = conn.execute(
        "SELECT display_text, status FROM claim_revisions WHERE revision_id = ?",
        (revision_id,),
    ).fetchone()
    if current is None:
        return False, "claim revision removed"
    if current["status"] in ("expired", "revoked", "superseded"):
        return False, "claim revision is no longer current evidence state"
    if text_hash(current["display_text"]) != expected_text_hash:
        return False, "claim revision text changed since embedding started"
    tombstone = conn.execute(
        "SELECT MAX(generation) AS g FROM tombstones "
        "WHERE subject_kind = 'claim_revision' AND subject_id = ?",
        (revision_id,),
    ).fetchone()
    if tombstone is not None and tombstone["g"] is not None:
        return False, "claim revision is tombstoned"
    links = conn.execute(
        "SELECT availability FROM evidence_links WHERE revision_id = ?", (revision_id,)
    ).fetchall()
    if not links:
        return False, "source evidence is no longer available"
    availabilities = {link["availability"] for link in links}
    if not availabilities <= {"full", "expired"}:
        # Withheld (redaction) or retracted lineage is invalid evidence forever;
        # only retention expiry may be tolerated for a retained conclusion.
        return False, "source evidence was withheld or retracted"
    if not embedding_text(current["display_text"]):
        # A scrubbed revision (redaction, expiry, tombstone) has no retained body
        # to derive a vector from even when its availability row still says
        # ``expired`` (redaction keeps the earlier retention truth).
        return False, "claim revision body is no longer retained"
    return True, "ok"


def _vector_generation_key(provider) -> dict:
    """The full cache key of the active vector generation (F10).

    Query and write both use model id + model revision + dimension +
    normalization + per-row text hash. Any component change means a different
    generation: the old vectors are ignored (never mixed) and rebuilt.
    """
    return {
        "model_id": str(getattr(provider, "model_id", "")),
        "model_revision": str(getattr(provider, "model_revision", "1")),
        "dimension": int(getattr(provider, "dimension", 0)),
        "normalization": str(getattr(provider, "normalization", "l2")),
    }


def run_embed_job(store: Store, job: dict, provider) -> dict:
    revision_id = job["input_key"]
    with store.read() as conn:
        row = conn.execute(
            "SELECT revision_id, display_text FROM claim_revisions WHERE revision_id = ?",
            (revision_id,),
        ).fetchone()
    if row is None:
        raise JobError("SOURCE_NOT_FOUND", "claim revision no longer exists")
    display_text = row["display_text"]
    source_hash = text_hash(display_text)
    vector = provider.embed([embedding_text(display_text)])[0]

    def mutate(conn):
        store_vector_in_transaction(
            conn, revision_id=revision_id, provider=provider, vector=vector, display_text=display_text
        )

    def fence(conn):
        # F11: the worker and the recall inline path share the same guard.
        return claim_vector_commit_guard(
            conn, revision_id, expected_text_hash=source_hash
        )

    result = commit(
        store,
        job["job_id"],
        job["lease_token"],
        expected_input_version=job["input_version"],
        mutate=mutate,
        fence_check=fence,
    )
    result["revision_id"] = revision_id
    return result


def run_embed_worker_once(
    store: Store,
    provider,
    *,
    worker_id: str = "worker:embed",
    lease_seconds: int = 60,
    now: str | None = None,
) -> dict:
    from knowledge_weaver.jobs import claim as claim_jobs

    claimed = claim_jobs(
        store,
        worker_id=worker_id,
        kinds=("embed",),
        lease_seconds=lease_seconds,
        limit=1,
        now=now,
    )
    if not claimed:
        return {"claimed": 0}
    job = claimed[0]
    try:
        result = run_embed_job(store, job, provider)
        return {"claimed": 1, "job_id": job["job_id"], "state": "succeeded"}
    except JobError as exc:
        if exc.code == "STALE_CURSOR":
            # Fencing/lease-loss outcomes are already persisted by jobs.commit.
            return {"claimed": 1, "job_id": job["job_id"], "state": "stale", "error": exc.code}
        # Pre-commit job errors (e.g. revision removed) must still be recorded so
        # the leased job cannot stay "running" forever.
        from knowledge_weaver.jobs import fail as fail_job

        try:
            outcome = fail_job(
                store, job["job_id"], job["lease_token"], exc.code, retryable=False, now=now
            )
        except JobError:
            return {"claimed": 1, "job_id": job["job_id"], "state": "stale", "error": exc.code}
        return {
            "claimed": 1,
            "job_id": job["job_id"],
            "state": outcome["state"],
            "error": exc.code,
        }
    except Exception as exc:  # noqa: BLE001 - real providers raise RuntimeError/transport errors
        # F12: real embedding failures (EmbeddingUnavailable/transport/shape) must
        # land in the retryable failed branch instead of escaping the worker.
        from knowledge_weaver.jobs import fail as fail_job

        code = str(getattr(exc, "code", None) or "EMBEDDING_MISMATCH")
        retryable = getattr(exc, "retryable", None)
        if retryable is None:
            try:
                from knowledge_weaver.contracts import is_retryable

                retryable = bool(is_retryable(code))
            except Exception:  # noqa: BLE001 - unknown code in a custom provider
                retryable = True
        try:
            outcome = fail_job(
                store,
                job["job_id"],
                job["lease_token"],
                code,
                retryable=bool(retryable),
                now=now,
            )
        except JobError:
            # The lease was already lost/reclaimed; a second fail() would mask
            # the original error, and the job is fenced anyway.
            return {"claimed": 1, "job_id": job["job_id"], "state": "stale", "error": code}
        return {
            "claimed": 1,
            "job_id": job["job_id"],
            "state": outcome["state"],
            "error": code,
        }


def load_vectors(
    store: Store,
    revision_ids: set[str],
    *,
    model_id: str,
    dimension: int,
    model_revision: str | None = None,
    normalization: str | None = None,
) -> dict[str, list[float]]:
    """Load cached vectors for exactly the active generation (F10).

    The query filters model id/revision/dimension/normalization and then verifies
    each row's ``text_hash`` against the *current* revision text, so a changed
    text, model revision or normalization can never reuse a stale vector.
    """
    if not revision_ids:
        return {}
    placeholders = ",".join("?" for _ in revision_ids)
    clauses = ["v.model_id = ?", "v.dimension = ?", f"v.revision_id IN ({placeholders})"]
    params: list = [model_id, int(dimension), *revision_ids]
    if model_revision is not None:
        clauses.append("v.model_revision = ?")
        params.append(str(model_revision))
    if normalization is not None:
        clauses.append("v.normalization = ?")
        params.append(str(normalization))
    with store.read() as conn:
        rows = conn.execute(
            f"SELECT v.revision_id, v.text_hash, v.vector, r.display_text "
            f"FROM claim_vectors v JOIN claim_revisions r ON r.revision_id = v.revision_id "
            f"WHERE {' AND '.join(clauses)}",
            params,
        ).fetchall()
    cached: dict[str, list[float]] = {}
    for row in rows:
        if row["text_hash"] != text_hash(row["display_text"]):
            # The revision text changed after this vector was cached: never mix
            # a stale embedding with the new text.
            continue
        cached[row["revision_id"]] = _unpack(row["vector"])
    return cached


def vector_ranking(
    *,
    query_vector: list[float],
    vectors: dict[str, list[float]],
) -> list[str]:
    scored = [
        (revision_id, cosine(query_vector, vector))
        for revision_id, vector in vectors.items()
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [revision_id for revision_id, score in scored if score > 0]
