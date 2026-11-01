-- 0004_retrieval: keyword index and vector cache for hybrid recall.
-- KW core does not depend on jieba/sqlite-vec: Chinese text is indexed with a
-- deterministic CJK bigram expansion, and vectors are stored as float32 blobs
-- keyed by (revision, text hash, model, dimension, normalization).

CREATE VIRTUAL TABLE IF NOT EXISTS claim_fts USING fts5(
    revision_id UNINDEXED,
    tokens,
    tokenize = 'unicode61'
);

CREATE TABLE IF NOT EXISTS claim_vectors (
    revision_id TEXT PRIMARY KEY REFERENCES claim_revisions (revision_id),
    text_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    normalization TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claim_vectors_model ON claim_vectors (model_id, dimension);
