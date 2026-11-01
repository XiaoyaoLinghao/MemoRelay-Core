-- 0007_tasks_entities: task verification views and reversible entity aliases.
-- A task is represented by claims with predicate "task.status": reported
-- completion, tool-verified observation and failure are separate facets and
-- revisions. The task view exposes them separately and only allows
-- "verified_done" when every declared acceptance condition has verified
-- evidence. Entity merge is an alias (canonical subject), reversible without
-- rewriting historical claims.

CREATE TABLE IF NOT EXISTS task_views (
    task_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    project_id TEXT NULL,
    status TEXT NOT NULL,
    reported_revision_id TEXT NULL,
    verified_revision_id TEXT NULL,
    failed_revision_id TEXT NULL,
    verification_conditions_json TEXT NOT NULL DEFAULT '[]',
    satisfied_conditions_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL,
    projection_generation INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_task_views_scope ON task_views (workspace_id, project_id, status);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    project_id TEXT NULL,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    project_id TEXT NULL,
    source_subject_id TEXT NOT NULL,
    canonical_subject_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    removed_at TEXT NULL,
    UNIQUE (workspace_id, project_id, source_subject_id, canonical_subject_id, state)
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_source ON entity_aliases (workspace_id, project_id, source_subject_id);
