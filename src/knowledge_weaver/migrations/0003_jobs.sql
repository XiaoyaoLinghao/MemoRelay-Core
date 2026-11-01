-- 0003_jobs: persistent work queue with leases, heartbeats and fencing.
-- A lease heartbeat is not proof of failure (JOB-01); a stale worker's commit
-- is rejected by lease token + input version + generation fence. Unconfirmed
-- external results stay visible as retry/failed states after a restart.

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    input_key TEXT NOT NULL,
    input_version INTEGER NOT NULL DEFAULT 1,
    state TEXT NOT NULL DEFAULT 'pending',
    run_after TEXT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_token TEXT NULL,
    lease_owner TEXT NULL,
    heartbeat_at TEXT NULL,
    lease_expires_at TEXT NULL,
    last_error_code TEXT NULL,
    model_version TEXT NULL,
    prompt_version TEXT NULL,
    policy_version TEXT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (kind, input_key, input_version)
);

CREATE INDEX IF NOT EXISTS idx_jobs_ready ON jobs (state, run_after);
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs (lease_expires_at);

CREATE TABLE IF NOT EXISTS job_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    lease_token TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NULL,
    outcome TEXT NULL,
    error_code TEXT NULL,
    detail TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_attempts_job ON job_attempts (job_id);
