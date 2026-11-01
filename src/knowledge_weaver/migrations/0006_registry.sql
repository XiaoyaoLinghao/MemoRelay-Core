-- 0006_registry: non-secret capability/resource references (T15).
-- Only references and non-secret metadata are stored; credential values never
-- enter this database and there is no arbitrary execution/read-keychain API.

CREATE TABLE IF NOT EXISTS capabilities (
    capability_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    name TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    install_locator TEXT NOT NULL,
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    state TEXT NOT NULL,
    verified_at TEXT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_capabilities_workspace ON capabilities (workspace_id, actor_id);

CREATE TABLE IF NOT EXISTS resources (
    resource_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    project_id TEXT NULL,
    name TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    account_label TEXT NOT NULL,
    credential_ref TEXT NOT NULL,
    recipe_id TEXT NOT NULL,
    last_verified_at TEXT NULL,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_resources_workspace ON resources (workspace_id, project_id);
