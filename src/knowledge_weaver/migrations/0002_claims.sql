-- 0002_claims: fact-level claims, revisions, evidence links, current projection.
-- One claim key = (workspace, project, subject, predicate, facet, qualifiers);
-- desired/observed/reported/preference/decision/capability are separate facets.
-- Revisions are append-only; the current projection is rebuildable and carries
-- a generation so concurrent corrections can fail compare-and-swap cleanly.

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    project_id TEXT NULL,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    facet TEXT NOT NULL,
    qualifiers_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (workspace_id, project_id, subject_id, predicate, facet, qualifiers_json)
);

CREATE TABLE IF NOT EXISTS claim_revisions (
    revision_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims (claim_id),
    revision_number INTEGER NOT NULL,
    value_json TEXT NOT NULL,
    display_text TEXT NOT NULL,
    authority TEXT NOT NULL,
    assertion_kind TEXT NOT NULL,
    event_time_value TEXT NULL,
    event_time_precision TEXT NOT NULL DEFAULT 'unknown',
    event_time_timezone TEXT NOT NULL DEFAULT 'UTC',
    status TEXT NOT NULL DEFAULT 'candidate',
    supersedes_json TEXT NOT NULL DEFAULT '[]',
    correction_of TEXT NULL,
    source_actors_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE (claim_id, revision_number)
);

CREATE INDEX IF NOT EXISTS idx_claim_revisions_claim ON claim_revisions (claim_id, revision_number);

CREATE TABLE IF NOT EXISTS evidence_links (
    revision_id TEXT NOT NULL REFERENCES claim_revisions (revision_id),
    event_id TEXT NOT NULL,
    event_revision INTEGER NOT NULL,
    part_index INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    body_sha256 TEXT NOT NULL,
    quote_sha256 TEXT NOT NULL,
    availability TEXT NOT NULL DEFAULT 'full',
    created_at TEXT NOT NULL,
    PRIMARY KEY (revision_id, event_id, event_revision, part_index, start_byte, end_byte)
);

CREATE INDEX IF NOT EXISTS idx_evidence_event ON evidence_links (event_id, event_revision);

CREATE TABLE IF NOT EXISTS current_claims (
    claim_id TEXT PRIMARY KEY REFERENCES claims (claim_id),
    selected_revision_ids_json TEXT NOT NULL,
    state TEXT NOT NULL,
    projection_generation INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS correction_operations (
    operation_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
