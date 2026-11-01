-- 0005_lifecycle: per-fact retention anchors, expiry and minimal tombstones.
-- Business anchors are source business time / status-change time, never access,
-- summary, import, backup, restore or maintenance time (TTL-01/TTL-02).

ALTER TABLE claim_revisions ADD COLUMN status_changed_at TEXT NULL;

CREATE TABLE IF NOT EXISTS lifecycle_records (
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    class TEXT NOT NULL,
    business_anchor TEXT NULL,
    expires_at TEXT NULL,
    state TEXT NOT NULL DEFAULT 'active',
    generation INTEGER NOT NULL DEFAULT 0,
    reason TEXT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (subject_kind, subject_id)
);

CREATE INDEX IF NOT EXISTS idx_lifecycle_due ON lifecycle_records (state, expires_at);

CREATE TABLE IF NOT EXISTS tombstones (
    tombstone_id TEXT PRIMARY KEY,
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    identity_hash TEXT NOT NULL,
    generation INTEGER NOT NULL,
    class TEXT NOT NULL,
    reason TEXT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (subject_kind, subject_id, generation)
);

CREATE INDEX IF NOT EXISTS idx_tombstones_subject ON tombstones (subject_kind, subject_id);
