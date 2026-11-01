-- 0001_initial: source evidence envelope, revisions, parts, receipts, watermarks.
-- Logical constraints follow docs/baseline-20261030/contracts and engineering §4-5.
-- Bodies are always stored after adapter redaction; withheld bodies are never stored.

CREATE TABLE IF NOT EXISTS backend_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    backend_instance_id TEXT NOT NULL,
    restore_epoch TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    restore_in_progress INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clients (
    client_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    host TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    actor_ids_json TEXT NOT NULL,
    project_ids_json TEXT NOT NULL DEFAULT '[]',
    binding_version INTEGER NOT NULL DEFAULT 1,
    registered_at TEXT NOT NULL,
    revoked_at TEXT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    host TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    native_event_id TEXT NOT NULL,
    native_schema TEXT NOT NULL,
    role TEXT NOT NULL,
    origin_kind TEXT NOT NULL,
    turn_id TEXT NULL,
    parent_event_id TEXT NULL,
    locator TEXT NOT NULL DEFAULT '',
    aliases_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE (workspace_id, instance_id, native_event_id)
);

CREATE INDEX IF NOT EXISTS idx_events_workspace ON events (workspace_id, instance_id);

CREATE TABLE IF NOT EXISTS event_revisions (
    event_id TEXT NOT NULL REFERENCES events (event_id),
    revision INTEGER NOT NULL,
    operation TEXT NOT NULL,
    parent_revision INTEGER NULL,
    event_time_value TEXT NULL,
    event_time_precision TEXT NOT NULL,
    event_time_timezone TEXT NOT NULL,
    project_ids_json TEXT NOT NULL,
    redaction_policy TEXT NOT NULL,
    redaction_status TEXT NOT NULL,
    redaction_replacements INTEGER NOT NULL,
    credential_refs_json TEXT NOT NULL,
    content_revision TEXT NOT NULL,
    part_count INTEGER NOT NULL,
    derived_from_json TEXT NOT NULL DEFAULT '[]',
    group_state TEXT NOT NULL DEFAULT 'incomplete',
    lifecycle TEXT NOT NULL DEFAULT 'active',
    sequence INTEGER NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY (event_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_revisions_state ON event_revisions (group_state, lifecycle);
CREATE INDEX IF NOT EXISTS idx_revisions_sequence ON event_revisions (sequence);

CREATE TABLE IF NOT EXISTS event_parts (
    event_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    part_index INTEGER NOT NULL,
    body_text TEXT NULL,
    body_sha256 TEXT NOT NULL,
    body_bytes INTEGER NOT NULL DEFAULT 0,
    format TEXT NOT NULL,
    withheld INTEGER NOT NULL DEFAULT 0,
    availability TEXT NOT NULL DEFAULT 'full',
    payload_sha256 TEXT NOT NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY (event_id, revision, part_index),
    FOREIGN KEY (event_id, revision) REFERENCES event_revisions (event_id, revision)
);

CREATE TABLE IF NOT EXISTS receipts (
    event_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    part_index INTEGER NOT NULL,
    disposition TEXT NOT NULL,
    durable_remote INTEGER NOT NULL,
    processing TEXT NOT NULL,
    searchability TEXT NOT NULL,
    error_code TEXT NULL,
    receipt_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (event_id, revision, part_index)
);

CREATE TABLE IF NOT EXISTS watermark_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    ref TEXT NOT NULL,
    created_at TEXT NOT NULL
);
